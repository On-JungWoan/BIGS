import os
import sys
import json
import os.path as op

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

import bigs.datasets.arctic.common.data_utils as data_utils
import bigs.datasets.arctic.common.rot as rot
import bigs.datasets.arctic.common.transforms as tf
import bigs.datasets.arctic.src.datasets.dataset_utils as dataset_utils
from bigs.datasets.arctic.common.data_utils import read_img
from bigs.datasets.arctic.common.object_tensors import ObjectTensors
from bigs.datasets.arctic.src.datasets.dataset_utils import get_valid, pad_jts2d

import cv2
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from pytorch3d.transforms import so3_relative_angle, axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix, so3_exponential_map, so3_log_map
from bigs.utils.graphics import get_projection_matrix

import trimesh
from glob import glob
from submodules.hold.common.rot import matrix_to_axis_angle
from submodules.hold.code.src.utils.io.ours import load_data
from submodules.hold.code.src.datasets.utils import load_K_Rt_from_P
from aitviewer.utils.so3 import aa2rot_numpy
from submodules.hold.common.xdict import xdict
from submodules.hold.code.src.utils.io.ours import map_deform2eval, map_deform2eval_torch
from bigs.utils.general import find_outlier, replace_outlier

import torch.nn.functional as F

def map_deform2eval_batch_torch(verts, inverse_scale, normalize_shift):
    return torch.cat(
        [
            map_deform2eval_torch(v, inverse_scale, normalize_shift)
            for v in verts
        ]
    )

def map_deform2eval_batch(verts, inverse_scale, normalize_shift):
    return np.array(
        [
            map_deform2eval(verts, inverse_scale, normalize_shift)
            for verts in verts.cpu().detach().numpy()
        ]
    )

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def preprocess_image(img):
    w, h, _ = img.shape

    margin = w-h
    margin_list = [0]*2 + [abs(margin)//2]*2
    margin_list.reverse() if margin < 0 else margin_list

    img = cv2.copyMakeBorder(img, *margin_list, cv2.BORDER_CONSTANT)
    img = Image.fromarray(img.astype(np.uint8))
    
    return img

class ArcticDataset(Dataset):
    def __init__(self, args, split, seq=None, device=None):
        assert device is not None
        
        self.device = device
        # self.use_misc = True if args.side != 'object' else False
        self.use_misc = True
        self._load_data(args, split, seq)
        logger.info(
            f"ImageDataset Loaded {self.split} split, num samples {len(self.imgnames)}"
        )


    def __len__(self):
        return len(self.imgnames)
    

    def __getitem__(self, index):
        try:
            imgname = self.imgnames[index]
        except:
            imgname = self.imgnames[-1]
        data = self.getitem(imgname, index)
        return data


    def _process_imgnames(self, seq, split):
        imgnames = self.imgnames
        if seq is not None:
            for seq in seq.split('/'):
                imgnames = [imgname for imgname in imgnames if "/" + seq + "/" in imgname]
                logger.info(f"Only /{seq}/ images are selected which have {len(imgnames)} images.")
        assert len(imgnames) == len(set(imgnames))
        imgnames = dataset_utils.downsample(imgnames, split)
        self.imgnames = imgnames


    def _load_data(self, args, split, seq):
        #* SETUP
        is_one_hand = True if args.name == 'ho3d' else False
        
        if args.name not in args.seq:
            args.seq = f'{args.name}_' + args.seq.replace('/', '_')
        data_root = op.join(args.dataset_path, 'images', args.seq)
        prcessed_root = f'./preprocess/outputs/{args.seq}'
        log_root = f'logs/{args.seq}'
        
        self.args = args
        self.split = split
        self.aug_data = split.endswith("train")
        
        self.imgnames = [op.join(data_root, r) for r in sorted(os.listdir(data_root))]
        self.mask_dir = op.join(prcessed_root, 'masks')
        self.tgt_dir = op.join(prcessed_root, 'mano_tgt')
        f_num = len(self.imgnames)
        

        # * LOAD INITIAL VALUES * #
        self.cached_data = {}
        np_data = np.load(f'{prcessed_root}/data.npy', allow_pickle=True).item()
        
        camera_data = np_data['cameras']
        cached_data = np_data['entities']
        
        hand = ['left', 'right']
        if is_one_hand:
            hand = ['right']
        
        # load hand/object poses
        for side in hand:
            self.cached_data[side] = {
                'global_orient' : cached_data[side]['hand_poses'][:, :3],
                'hand_pose': cached_data[side]['hand_poses'][:, 3:],
                'betas': cached_data[side]['mean_shape'][None, :].repeat(f_num, 0),
                'transl' : cached_data[side]['hand_trans'],
                'scale' : np.ones([1,1]).repeat(f_num, 0),
            }
        self.cached_data['object'] = {
            'global_orient' : cached_data['object']['object_poses'][:, :3],
            'hand_pose': None,
            'betas': None,
            'transl' : cached_data['object']['object_poses'][:, 3:],
            'scale' : np.ones([1,1]).repeat(f_num, 0),
        }
        
        # load cam params
        self.projection_mat = (camera_data['world_mat_0'] @ camera_data['scale_mat_0'])
        P = self.projection_mat[:3, :4]
        intrinsics, extrinsics = load_K_Rt_from_P(None, P)
        self.my_intris_mat = intrinsics
        # self.my_world2cam = extrinsics
            
        # set extrinsics
        self.my_world2cam = np.eye(4)
                
        
        #* USE HOLD INIT VALUE
        self.misc = xdict()
        if self.use_misc:
            if args.log_name is not None:
                log_dir = op.join(log_root, args.log_name)
                assert op.isdir(log_dir), logger.error("custom log dir is not exist!!")
            else:
                log_dir = [d for d in sorted(glob(op.join(log_root, '*'))) if d.split('/')[-1][0] != '_'][0]
                log_name = log_dir.split('/')[-1]
                assert log_name[0] != '_'            
            
            # load ckpt & make misc
            self.misc, params = load_data(f'{log_dir}/checkpoints/last.ckpt')
            
            # if args.name == 'arctic':
            #     assert args.seq == self.misc['fnames'][0].split('/')[2]
            
            for param_k, param_v in params.items():
                for param_v_k, param_v_v in param_v.items():
                    if param_v_k == 'scene_scale':
                        continue
                    repr_key = param_v_k.split('.')[-2]
                    self.cached_data[param_k][repr_key] = param_v_v.cpu().numpy()
            
            # load misc
            misc_p = sorted(glob(f'{log_dir}/misc/*.npy'))
            _misc = np.load(misc_p[-1], allow_pickle=True).item()
            
            # load cano mash
            normalize_shift = np_data['normalize_shift']
            scale = torch.tensor([_misc['scale']]).to(self.device)
            # for side in ['right', 'left']:
            #     # find not valid frames
            #     if iqr_w is not None and iqr_w['weights'] is not None:
            #         is_valid, _ = find_outlier(self.misc[f'root.{side}'].float(), iqr_w['weights'])
            #         for key in repr_keys:
            #             self.cached_data[side][key] = replace_outlier(self.cached_data[side][key], is_valid)                
            
            assert params['right']['scene_scale'] == params['object']['scene_scale'] == scale.float()
            if not np.allclose(self.misc['K'].numpy(), self.my_intris_mat[:-1, :-1]):
                logger.warning("misc cam param and preprocessed cam param are different!!")
                # self.misc.overwrite('K', torch.from_numpy(self.my_intris_mat[:-1, :-1]))
            
            # make cano mesh
            try:
                mesh_cano_obj = _misc['object_cano']
            except:
                mesh_cano_obj = _misc['mesh_c_o']

        
        #* ONLY USE SfM/HaMeR VALUE
        else:
            scale = np_data['entities']['object']['obj_scale']
            scale = torch.tensor([scale]).to(self.device)
            normalize_shift = np_data['normalize_shift']
            mesh_cano_obj = np_data['entities']['object']['pts.cano']
        
        # store
        self.misc['norm_mat'] = np_data['entities']['object']['norm_mat']
        # self.misc['norm_mat'] = torch.from_numpy(np_data['entities']['object']['norm_mat'])
        self.misc['scale'] = scale
        self.misc['inverse_scale'] = float(1.0 / scale[0])
        self.misc['cano_mesh.object'] = mesh_cano_obj # trimesh.load_mesh(f'{log_dir}/mesh_cano/mesh_cano_object_step_misc.obj')
        self.misc['normalize_shift'] = normalize_shift
        self.misc['cached_data'] = self.cached_data        
        
        
        #* DROP NOT-VALID FRAMES
        self.misc['skip_frame'] = []
        if args.side == 'object':            
            # find not valid frames
            transl_thold = 2.0
            transl = self.misc['cached_data']['object']['transl']        
            
            mean_depth = transl[..., -1].mean()
            is_valid_transl = abs(transl[..., -1] - mean_depth) < transl_thold
            skip_frame = (~is_valid_transl).nonzero()[0]
            logger.info(f"[LERP this frame] {skip_frame}")
                        
            self.misc['is_valid'] = is_valid_transl
            skip_frame = set(self.misc['skip_frame']) | set(skip_frame)
            self.misc.overwrite('skip_frame', skip_frame)   


            # just skip
            if op.isfile('prior.json'):
                with open(f'prior.json') as f:
                    iqr_w_dict = json.load(f)
                seq = args.seq.replace('arctic_', '').replace('ho3d_', '')
                iqr_w = iqr_w_dict[seq] if seq in iqr_w_dict.keys() else None
                
                if args.drop_not_valid_frame and iqr_w is not None:
                    if 'skip_frame' in iqr_w.keys():
                        self.misc.overwrite('skip_frame', iqr_w['skip_frame'])
                        
                    if 'using_frame' in iqr_w.keys():
                        using_f = iqr_w['using_frame']
                        skip_f = [idx for idx in range(len(self.imgnames)+1) if idx not in using_f]
                        
                        self.misc.overwrite('skip_frame', skip_f)
        
        logger.info(f"[SKIP THIS FRAMES] {self.misc['skip_frame']}")
        
        #* SMOOTHING
        if args.smt_mode == 'all':
            _tgt_list = ['global_orient', 'transl', 'hand_pose']
        elif args.smt_mode == 'pose_rot':
            _tgt_list = ['global_orient', 'hand_pose']
        elif args.smt_mode == 'pose':
            _tgt_list = ['hand_pose']
        else:
            raise Exception("Not implemented yet!")
        
        try:
            with open(f"misc/valid/{args.seq}.json", 'r') as f:
                valid = json.load(f)
                
            for side in hand:
                valid_0 = valid[side]['th_0']
                valid_4500 = valid[side]['th_4500']
                
                for idx, v in enumerate(valid_0):
                    if idx == 0:
                        continue
                    
                    if not v:
                        tgt_list = _tgt_list
                    elif not valid_4500[idx]:
                        # tgt_list = ['hand_pose']
                        tgt_list = _tgt_list
                    else:
                        continue
                    
                    for tgt_key in tgt_list:
                        self.misc['cached_data'][side][tgt_key][idx] = self.misc['cached_data'][side][tgt_key][idx-1]
                        self.cached_data[side][tgt_key][idx] = self.cached_data[side][tgt_key][idx-1]                    
        except:
            pass

    
    def getitem(self, imgname, index=None, load_rgb=True):
        cv_img, _ = read_img(imgname, (2800, 2000, 3))
        H, W = cv_img.shape[:-1]

        targets = {}
        
        ## MASK ##
        mask_t_hold = 100
        mask = {'right':None, 'left':None, 'object':None}
        for k in mask:
            mask_path = op.join(self.mask_dir, k)
            if op.isdir(mask_path):
                tgt_path = op.join(mask_path, imgname.split('/')[-1].replace('png', 'jpg'))
                if not op.isfile(tgt_path):
                    iname = int(imgname.split('/')[-1].split('.')[0])
                    tgt_path = op.join(mask_path, f"{iname:04}.png")
                cv_mask = cv2.imread(tgt_path)
                cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2GRAY)
                
                cv_mask[cv_mask<mask_t_hold] = 0.
                cv_mask[cv_mask>mask_t_hold] = 1.
                
                cv_mask = torch.FloatTensor(cv_mask)
                if self.args.name == 'ho3d':
                    cv_mask = F.interpolate(torch.FloatTensor(cv_mask)[None][None], size=(H, W))[0,0, ...]
                mask[k] = cv_mask
        
        mask = dict((k,v) for k,v in mask.items() if v is not None)
        targets.update({
            "rgb":torch.FloatTensor(cv_img/255).permute(2, 0, 1).contiguous(),
            "mask":mask,
            "bbox":None,
        })
        
        
        ## CAM ##
        K = self.my_intris_mat
        world_view_transform = torch.FloatTensor(self.my_world2cam) # torch.eye(4)
        height, width = cv_img.shape[:2]
        
        if self.args.use_gt:
            height, width = 2000, 2800
        
        if len(K.shape) == 3:
            _intr = np.eye(4)
            _intr[:-1, :-1] = K[index]
            K = _intr
            
            # world_view_transform = world_view_transform[img_idx]
        
        fovx = 2 * np.arctan(width / (2 * K[0, 0]))
        fovy = 2 * np.arctan(height / (2 * K[1, 1]))
        zfar = 1e+12 # max(zfar, 100.0)
        znear = 1e-12 # min(znear, 0.01)
        
        projection_matrix = get_projection_matrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy).transpose(0,1)
        # projection_matrix = torch.from_numpy(self.projection_mat).float()
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        
        # FIXME: FIX ME!
        camera_center = torch.FloatTensor([height / 2., width / 2., 0.]) # world_view_transform.inverse()[3, :3]
        cam_intrinsics = torch.FloatTensor(K)

        targets.update({
            "fovx": fovx,
            "fovy": fovy,
            "image_height": height,
            "image_width": width,
            "world_view_transform": world_view_transform,
            "full_proj_transform": full_proj_transform,
            "camera_center": camera_center,
            "cam_intrinsics": cam_intrinsics,
            
            "projection_matrix": projection_matrix,
        })

        return targets