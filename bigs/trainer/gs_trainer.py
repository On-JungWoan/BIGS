#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import cv2
import glob
import json
import wandb
import shutil
import torch
import itertools
import torchvision
import numpy as np
from PIL import Image
from tqdm import tqdm
from lpips import LPIPS
from loguru import logger
from torch.nn import functional as F

import trimesh
from bigs.datasets.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params, 
    get_static_camera
)
from bigs.losses.utils import ssim
from bigs.losses.loss import HandObjectLoss
from bigs.models.bigs_trimlp import BIGS_TRIMLP
from bigs.utils.init_opt import optimize_init
from bigs.renderer.gs_renderer import render_human_scene
from bigs.utils.vis import save_ply
from bigs.utils.image import psnr, save_image, crop_and_center_image, rgb_to_grayscale
from bigs.utils.general import (
    RandomIndexIterator, load_human_ckpt, save_images, create_video,
    make_and_export_mesh, custom_to, save_img, grad_off, draw_j2d, apply_transformation
)
from submodules.hold.common.transforms import transform_points_batch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix
import time
import bigs.datasets.arctic.common.transforms as tf

import submodules.hold.code.src.utils.io.gt_arctic as gt
import submodules.hold.code.src.utils.io.gt as gt_common
from submodules.hold.code.src.utils.eval_modules import compute_bounding_box_centers, compute_bounding_box_centers_torch
from submodules.hold.code.src.utils.eval_modules_arctic import eval_cd_f_hand_arctic
from submodules.hold.code.src.utils.eval_modules import (
    eval_mpjpe_right, eval_icp_first_frame, eval_cd_f_right
)
from common.xdict import xdict
from submodules.hold.code.src.arctic.extraction.keys import alt_keys
from submodules.hold.code.src.arctic.extraction.keys import keys as ext_keys

import subprocess

import sys
sys.path = ['./submodules/gtu/'] + sys.path
from submodules.gtu.diffusion_inversion.pidi import PidiNetDetector

def get_train_dataset(cfg, mode, split='train'):
    logger.info(f'Loading ARCTIC dataset {cfg.dataset.seq}-{split}')
    
    import sys; sys.path = ['bigs/datasets/arctic'] + sys.path
    from bigs.datasets.arctic.src.factory import fetch_dataloader
    dataset = fetch_dataloader(cfg.dataset, split, device=cfg.device)
    dataset.cached_data = dict((k if k!='hand_pose' else 'body_pose', torch.from_numpy(v).to(cfg.device) if v is not None else None) for k,v in dataset.cached_data[mode].items())
    
    return dataset


class Gaussians:
    def __init__(self, *modes, cfg=None):
        self.mode = modes
        self.cfg = cfg
        for mode in modes:
            self._set_gaussian(None, mode)
            self._set_dataset(None, mode)
        self.eval_metrics = {}
        
    def _set_gaussian(self, gaussian, mode):
        self.__setattr__(f'{mode}_gs', gaussian)
        
    def _set_dataset(self, dataset, mode):
        self.__setattr__(f'{mode}_dataset', dataset)
    
    def _get_gaussian(self, mode):
        try:
            return self.__getattribute__(f'{mode}_gs')
        except:
            return None
    
    def _get_dataset(self, mode):
        try:
            return self.__getattribute__(f'{mode}_dataset')
        except:
            return None
    
    def save_ckpt(self, iter=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        extra_params = ['body_pose', 'global_orient', 'betas', 'transl', 'scale']
        
        for mode in self.mode:
            gaussian = self._get_gaussian(mode)
            
            state_dict = gaussian.state_dict()
            for p_key in extra_params:
                if hasattr(gaussian, p_key):
                    state_dict[p_key] = gaussian.__getattribute__(p_key)
            torch.save(state_dict, f'{self.cfg.logdir_ckpt}/human_{mode}_{iter_s}.pth')
            logger.info(f'Saved checkpoint {iter_s}')
    
    def anim_frame(self, human_gs_outs, data, gs, t_iter, frame_i, dist=5.0, num_frame=100, keep_images=False, device=None):
        render_mode = self.cfg.mode
        os.makedirs(f'{self.cfg.logdir}/anim_canon', exist_ok=True)
        bg_color = torch.ones(3, dtype=torch.float32, device="cuda")
        
        backup = human_gs_outs['xyz'].clone()
        pbar = enumerate(tqdm(range(num_frame), desc="Canon Anim"))
        for idx, c_num in pbar:
            human_gs_outs['xyz'] = backup.clone()
            
            rot_cam = get_rotating_camera(dist=dist, nframes=num_frame)[c_num]
            world_view_transform = rot_cam['world_view_transform']
            full_proj_transform = rot_cam['full_proj_transform']
            for k,v in rot_cam.items():
                if k not in ['image_height', 'image_width', 'camera_center']:
                    data[k] = v
            
            Ts = []
            num_hand = human_gs_outs['joint3d'].shape[0]
            n_hand_gs = gs[0]._xyz.shape[0]
            n_obj_gs = gs[-1]._xyz.shape[0]
            
            T = compute_bounding_box_centers(human_gs_outs['xyz'][-n_obj_gs: ,:].unsqueeze(0).detach().cpu().numpy())[0]
            T = torch.from_numpy(T).to(device)
            human_gs_outs['xyz'] -= T.unsqueeze(0).cuda()
            
            data['world_view_transform'] = world_view_transform
            data['full_proj_transform'] = full_proj_transform
            
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_outs, 
                scene_gs_out=None, 
                bg_color=bg_color,
                render_mode=render_mode,
                t_iter=t_iter
            )
            
            image = render_pkg['render']
            image = torch.from_numpy(cv2.flip(image.cpu().detach().permute(1,2,0).numpy(), 1)).permute(2,0,1)
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim_canon/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_canon_frame_{frame_i}_iter_{t_iter}.mp4'
        create_video(f'{self.cfg.logdir}/anim_canon/', video_fname, fps=20)
        if not keep_images:
            try:
                shutil.rmtree(f'{self.cfg.logdir}/anim_canon/')
            except:
                pass
    
    def train(self):
        pass

    def eval(self):
        pass

class GaussianTrainer():
    def __init__(self, cfg, mode, refine_step) -> None:
        self.mode = mode
        self.cfg = cfg
        self.refine_step = refine_step
        
        # get dataset
        self.train_dataset = get_train_dataset(cfg, mode, split='train')
        
        self.eval_metrics = {}
        self.lpips = LPIPS(net="alex", pretrained=True).to(cfg.device)
        # get models
        self.gs_model, self.scene_gs = None, None
        gs_ckpt = getattr(cfg.obj, f'{mode}_ckpt')
        cfg.obj.ckpt = gs_ckpt
        
        if cfg.mode in ['human', 'human_scene']:
            init_betas = self.train_dataset.cached_data['betas'].nanmean(dim=0).unsqueeze(0) if mode != 'object' else None
            self.gs_model = BIGS_TRIMLP(
                sh_degree=cfg.obj.sh_degree, 
                n_subdivision=cfg.obj.n_subdivision if mode != 'object' else cfg.obj.min_obj_n_gaussians,  
                use_surface=cfg.obj.use_surface,
                init_2d=cfg.obj.init_2d,
                rotate_sh=cfg.obj.rotate_sh,
                isotropic=cfg.obj.isotropic,
                init_scale_multiplier=cfg.obj.init_scale_multiplier,
                n_features=32,
                use_deformer=cfg.obj.use_deformer if mode != 'object' else False,
                disable_posedirs=cfg.obj.disable_posedirs,
                triplane_res=cfg.obj.triplane_res,
                betas=init_betas[0] if init_betas is not None else init_betas,
                is_rhand=mode=='right',
                device=cfg.device,
                misc=self.train_dataset.misc,
                optim_rotations=cfg.obj.optim_rotations,
                optim_opacity=cfg.obj.optim_opacity,
                mode = mode,
                refine_step = refine_step
            )
            if init_betas is not None:
                self.gs_model.create_betas(init_betas[0], cfg.obj.optim_betas)
            if not cfg.eval:
                self.gs_model.initialize()
                if gs_ckpt is None and not os.path.isfile(os.path.join(cfg.logdir_ckpt, 'human_final.pth')) and not cfg.train.joint_training:
                    # FIXME: this is temporal
                    if self.mode != 'object' and self.cfg.dataset.name != 'ho3d':
                        self.gs_model.load_state_dict(torch.load('misc/human_final.pth'), cfg=cfg.obj.lr, load_extra_param=False)
                    else:
                        pth_path = f'{cfg.logdir_ckpt}/human_gs.pth'
                        if os.path.isfile(pth_path) and not cfg.train.overwrite_init:
                            self.gs_model.load_state_dict(torch.load(pth_path), cfg=cfg.obj.lr)
                        else:
                            self.gs_model = optimize_init(self.gs_model, pth_path, num_steps=7500, cfg=cfg)
            
        # setup the optimizers
        if self.gs_model:
            self.gs_model.setup_optimizer(cfg=cfg.obj.lr)
            logger.info(self.gs_model)
            if gs_ckpt:
                # load_human_ckpt(self.gs_model, cfg.obj.ckpt)
                self.gs_model.load_state_dict(torch.load(gs_ckpt), cfg=cfg.obj.lr)
                logger.info(f'Loaded human model from {gs_ckpt}')
            else:
                ckpt_files = sorted(glob.glob(f'{cfg.logdir_ckpt}/*final.pth'))
                if len(ckpt_files) > 0:
                    if cfg.obj.ckpt is None:
                        cfg.obj.ckpt = ckpt_files[-1]
                    
                    ckpt = torch.load(ckpt_files[-1])
                    self.gs_model.load_state_dict(ckpt, cfg=cfg.obj.lr)
                    logger.info(f'Loaded human model from {ckpt_files[-1]}')
                    
            # * load prev params
            if not self.cfg.train.joint_training and self.cfg.obj.num_refine != 1:
                self.gs_model.load_params(cfg.logdir_ckpt, mode, cfg.device, cfg=cfg)
            # * load prev params

            if not cfg.eval and cfg.obj.ckpt is None:
                f_num = len(self.train_dataset)
                
                init_smpl_global_orient = torch.zeros([f_num, 3]).to(cfg.device) if self.train_dataset.cached_data['global_orient'] is not None else None
                init_smpl_body_pose = torch.zeros([f_num, 45]).to(cfg.device) if self.train_dataset.cached_data['body_pose'] is not None else None
                init_smpl_trans = torch.zeros([f_num, 3]).to(cfg.device) if self.train_dataset.cached_data['transl'] is not None else None
                init_betas = self.train_dataset.cached_data['betas'].nanmean(dim=0).unsqueeze(0) if init_betas is not None else None
                init_scale = self.train_dataset.cached_data['scale']
                init_eps_offsets = torch.zeros((len(self.train_dataset), self.gs_model.n_gs, 3), 
                                            dtype=torch.float32, device="cuda")

                if init_betas is not None:
                    self.gs_model.create_betas(init_betas[0], cfg.obj.optim_betas)
                if init_smpl_body_pose is not None:
                    self.gs_model.create_hand_pose(init_smpl_body_pose, cfg.obj.optim_pose)
                if init_smpl_global_orient is not None:
                    self.gs_model.create_global_orient(init_smpl_global_orient, cfg.obj.optim_pose)
                if init_smpl_trans is not None:
                    self.gs_model.create_transl(init_smpl_trans, cfg.obj.optim_trans)
                if init_scale is not None:
                    self.gs_model.create_scale(init_scale, cfg.obj.optim_scale)                    
                
                self.gs_model.setup_optimizer(cfg=cfg.obj.lr)
        
        bg_color = cfg.bg_color
        if bg_color == 'white':
            self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        else:
            raise ValueError(f"Unknown background color {bg_color}")
        
        self.loss_fn = HandObjectLoss(
            **cfg.obj.loss, bg_color=self.bg_color,
        )
        
        self.canon_camera_params = get_rotating_camera(
            dist=5.0, img_size=512, 
            nframes=cfg.obj.canon_nframes, device='cuda',
            angle_limit=2*torch.pi,
        )
        if self.gs_model.use_deformer:
            betas = self.gs_model.betas.detach() if hasattr(self.gs_model, 'betas') else self.train_dataset.betas[0]
            self.static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.obj.canon_pose_type
            )
                
        if cfg.obj.dgm.use and self.mode == 'object' and not cfg.train.joint_training:
            from bigs.models.sds import setup_for_sds_loss
            cfg.obj.dgm.logdir = cfg.logdir
            cfg.obj.dgm.seq = '_'.join(cfg.dataset.seq.split('_')[1:])
            cfg.obj.dgm.db_name = cfg.dataset.name
            self.DGM = setup_for_sds_loss(cfg.obj.dgm)

    def joint_forward(self, rand_idx_iter, modes, dbs, gs, t_iter, device):
        human_gs_outs = None
        
        while True:
            rnd_idx = next(rand_idx_iter)
            if rnd_idx in self.train_dataset.misc['skip_frame']:
                continue
            break

        for mode, db, g in zip(modes, dbs, gs):
            if hasattr(g, 'update_learning_rate') and g.use_deformer:
                g.update_learning_rate(t_iter, self.cfg.obj.lr.name)
            else:
                for param_group in g.optimizer.param_groups:
                    param_group['lr'] = 0.0
    
            data = db[rnd_idx]
            data = dict((k,v[0] if isinstance(v, list) else v) for k,v in data.items())
            
            valid_test = sum([v.sum() for k,v in data.items() if mode in k and 'valid' not in k])
            
            if valid_test != valid_test or data['mask'][mode].sum() == 0: #FIXME: FIX ME!!!!!!
                continue
            
            data = custom_to(data, 'cuda')
            human_gs_out, scene_gs_out = None, None
            
            human_gs_out = g.forward(
                smpl_scale=None, #self.train_dataset.scale.cuda(),
                dataset_idx=rnd_idx,
                is_train=True,
                ext_tfs=None,
                transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
                misc=db.misc,
                device=device,
                is_flip=self.cfg.obj.num_refine > 1
            )
            g.init_values['edges'] = g.edges
            # human_gs_out['xyz'] = data_gt[f'v3d_c.{mode}'][rnd_idx].cuda()
            
            if 'body_pose' in human_gs_out.keys():
                human_gs_out.pop('body_pose')            
            
            if human_gs_outs is None:
                human_gs_outs = human_gs_out
            else:
                for (ts_k, ts_v), (t_k, t_v) in zip(human_gs_outs.items(), human_gs_out.items()):
                    if t_v is None or ts_k == 'v_posed': continue
                    
                    if isinstance(ts_v, torch.Tensor):
                        human_gs_outs[ts_k] = torch.cat([ts_v, t_v], dim=0)
                    else:
                        human_gs_outs[ts_k] = t_v
                        
        return human_gs_outs, data, modes, dbs, gs, rnd_idx


    def origin_forward(self, t_iter, rand_idx_iter, device):
        if hasattr(self.gs_model, 'update_learning_rate'):
            self.gs_model.update_learning_rate(t_iter, self.cfg.obj.lr.name)
    
        # FIXME: FIX ME!!!!
        while True:
            rnd_idx = next(rand_idx_iter)
            if rnd_idx in self.train_dataset.misc['skip_frame']:
                continue
            
            data = self.train_dataset[rnd_idx]
            data = dict((k,v[0] if isinstance(v, list) else v) for k,v in data.items())
            
            if self.mode == 'object':
                break
            
            valid_test = sum([v.sum() for k,v in data.items() if self.mode in k and 'valid' not in k])
            
            if (valid_test == valid_test and data['mask'][self.mode].sum() > self.cfg.dataset.is_valid_thold) or self.cfg.dataset.name == 'ho3d': #FIXME: FIX ME!!!!!!
                break
        
        data = custom_to(data, device)
        human_gs_out = self.gs_model.forward(
            smpl_scale=None, #self.train_dataset.scale.cuda(),
            dataset_idx=rnd_idx,
            is_train=True,
            ext_tfs=None,
            transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
            misc=self.train_dataset.misc,
            device=device,
            is_flip=self.cfg.obj.num_refine > 1
        )
        self.gs_model.init_values['edges'] = self.gs_model.edges
        
        return human_gs_out, custom_to(data, device, cast_type=True), rnd_idx


    def train(self, joint_train=False, debug_idx = 100):
        if self.cfg.obj.use_small_scale:
            self.gs_model.scaling_multiplier *= 1.75e-4
        
        torch.cuda.set_device(self.cfg.gpus)
        if self.gs_model:
            self.gs_model.train()

        pbar = tqdm(range(self.cfg.train.num_steps+1), desc="Training")
        rand_idx_iter = RandomIndexIterator(len(self.train_dataset))
        self.stored_joint_3d = None
        
        if joint_train:
            gaussians = self.gs_model
            modes = gaussians.mode
            dbs = [gaussians._get_dataset(m) for m in gaussians.mode]
            gs = [gaussians._get_gaussian(m) for m in gaussians.mode]
            
            with torch.no_grad():
                logger.info('Store output !!')
                self.stored_joint_3d = {
                    'right':torch.zeros(len(self.train_dataset), *[21,3]).to(self.cfg.device),
                    'left':torch.zeros(len(self.train_dataset), *[21,3]).to(self.cfg.device),
                    'object':torch.zeros(len(self.train_dataset), *[3]).to(self.cfg.device)
                }
                
                for tmp_ in tqdm(range(len(self.train_dataset))):
                    for idx, g in enumerate(gs):
                        comp = gaussians.mode[idx]
                        gs_out = g.forward(
                                            smpl_scale=None, #self.train_dataset.scale.cuda(),
                                            dataset_idx=tmp_,
                                            is_train=True,
                                            ext_tfs=None,
                                            transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
                                            misc=self.train_dataset.misc,
                                            device=self.cfg.device
                                        )
                        
                        if comp != 'object':
                            self.stored_joint_3d[comp][tmp_] = gs_out['joint3d'][0]
                        else:
                            T = compute_bounding_box_centers(gs_out['xyz'].unsqueeze(0).cpu().numpy())[0]
                            self.stored_joint_3d[comp][tmp_] = torch.from_numpy(T)
        else:
            with torch.no_grad():
                if self.mode != 'object':
                    logger.info('Store output !!')
                    trans_dim = [21, 3] if self.mode != 'object' else [3]
                    self.stored_joint_3d = torch.zeros(len(self.train_dataset), *trans_dim).to(self.cfg.device)
                    
                    for tmp_ in range(len(self.train_dataset)):
                        gs_out = self.gs_model.forward(
                                                    smpl_scale=None, #self.train_dataset.scale.cuda(),
                                                    dataset_idx=tmp_,
                                                    is_train=True,
                                                    ext_tfs=None,
                                                    transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
                                                    misc=self.train_dataset.misc,
                                                    device=self.cfg.device
                                                )
                        #? for hand
                        self.stored_joint_3d[tmp_] = gs_out['joint3d'][0]
                        
                        #? for obj
                        # T = compute_bounding_box_centers(gs_out['xyz'].unsqueeze(0).cpu().numpy())[0]
                        # self.stored_joint_3d[tmp_] = torch.from_numpy(T)
        # data_gt = gt.load_data_hugs(self.cfg.dataset)
            
        # * FOR FINE-TUNE OPTIONS: MUST BE CHECKED
        if joint_train:
            for idx, g in enumerate(gs):
                gs[idx] = grad_off(g, self.cfg.obj.lr, idx)
            assert not gs[-1].use_deformer
            logger.warning('Only hand params are updated.')
            gs[-1].global_orient.requires_grad = False
            # gs[-1].transl.requires_grad = False
            gs[-1].scale.requires_grad = False                            
        else:
            self.gs_model = grad_off(self.gs_model, self.cfg.obj.lr)
        # * FOR FINE-TUNE OPTIONS: MUST BE CHECKED
        
        sds_items = None
        for t_iter in range(self.cfg.train.num_steps+1):
            if sds_items is not None:
                del sds_items
                torch.cuda.empty_cache()
                sds_items = None
            
            render_mode = self.cfg.mode
            
            if joint_train:
                human_gs_out, data, modes, dbs, gs, rnd_idx = self.joint_forward(rand_idx_iter, modes, dbs, gs, t_iter, self.cfg.device)
            else:
                human_gs_out, data, rnd_idx = self.origin_forward(t_iter, rand_idx_iter, self.cfg.device)
            
            bg_color = torch.rand(3, dtype=torch.float32, device=self.cfg.device)
            human_bg_color = None
            render_human_separate = False

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=None, 
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                render_mode=render_mode,
                render_human_separate=render_human_separate,
                t_iter=t_iter,
                device=self.cfg.device
            )
                
            loss, loss_dict, loss_extras = self.loss_fn(
                data,
                render_pkg,
                human_gs_out,
                render_mode=render_mode,
                human_gs_init_values=None,
                bg_color=bg_color,
                human_bg_color=human_bg_color,
                gs=gs if joint_train else self.gs_model, #self.gs_model.smpl_template.v_template,
                mode=self.mode if not joint_train else modes[0],
                joint_train=joint_train,
                misc=self.train_dataset.misc,
                data_idx=rnd_idx,
                sds_items=sds_items,
                stored_j3d=self.stored_joint_3d,
            )
            
            if self.cfg.obj.dgm.use:
                lambda_rgb = self.cfg.obj.dgm.lambda_rgb_loss
                rgb_scaler = 1.
                
                if (not (t_iter >= self.cfg.obj.dgm.dgm_start_iter)):
                    rgb_scaler = 1 / lambda_rgb
                else:
                    dgm_step, max_noise_ratio = self.DGM.get_noise_level()
                    rgb_scaler = max_noise_ratio ** 2
                lambda_rgb = lambda_rgb * rgb_scaler
                loss = loss * lambda_rgb            
            
            loss.backward()
            
            loss_dict['loss'] = loss
            
            if t_iter % 10 == 0:
                postfix_dict = {}
                for k, v in loss_dict.items():
                    postfix_dict["l_"+k] = f"{v.item():.4f}"
                        
                pbar.set_postfix(postfix_dict)
                pbar.update(10)
                
            if t_iter == self.cfg.train.num_steps:
                pbar.close()

            if (t_iter % 1500) == 0:
                with torch.no_grad():
                    pred_img = loss_extras['pred_img']
                    gt_img = loss_extras['gt_img']
                    log_pred_img = (pred_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_gt_img = (gt_img.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    log_img = np.concatenate([log_gt_img, log_pred_img], axis=1)
                    save_images(log_img, f'{self.cfg.logdir}/train/{t_iter:06d}.png')
                    
                    if joint_train:
                        gaussians.anim_frame(human_gs_out, data, gs, t_iter, rnd_idx, device=self.cfg.device)
                
            if not joint_train and 'fine_tune' not in self.cfg.cfg_file:
                render_pkg['human_viewspace_points'] = render_pkg['viewspace_points'][:human_gs_out['xyz'].shape[0]]
                render_pkg['human_viewspace_points'].grad = render_pkg['viewspace_points'].grad[:human_gs_out['xyz'].shape[0]]
                
                # gs_w = render_pkg['gs_w'] if 'gs_w' in render_pkg.keys() else self.gs_model.get_xyz.grad.mean(dim=1)
                gs_w = None
                
                with torch.no_grad():
                    if not joint_train and self.mode == 'object':
                        self.human_densification(
                            human_gs_out=human_gs_out,
                            visibility_filter=render_pkg['human_visibility_filter'],
                            radii=render_pkg['human_radii'],
                            viewspace_point_tensor=render_pkg['human_viewspace_points'],
                            iteration=t_iter+1,
                            gs_w = gs_w
                        )
            
            if self.gs_model:
                if joint_train:
                    for g in gs:
                        g.optimizer.step()
                        g.optimizer.zero_grad(set_to_none=True)
                else:
                    self.gs_model.optimizer.step()
                    self.gs_model.optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
            
            # get sds loss
            if t_iter >= self.cfg.obj.dgm.dgm_start_iter and not joint_train and self.cfg.obj.dgm.use:
                dg_loss = self.get_gdm_items(t_iter=t_iter, device=self.cfg.device)
                
                if dg_loss is not None:
                    _loss = dg_loss * self.cfg.obj.dgm.dgm_loss_weight
                    _loss.backward()

                    # loss += _loss.detach()
                    self.gs_model.optimizer.step()
                    
                self.gs_model.optimizer.zero_grad(set_to_none=True)
                
                torch.cuda.empty_cache()
            
            if self.cfg.use_wandb and t_iter % 150 == 0:
                wandb.log(loss_dict, step=t_iter)
                
            # save checkpoint
            if (t_iter % self.cfg.train.save_ckpt_interval == 0 and t_iter > 0) or \
                (t_iter == self.cfg.train.num_steps and t_iter > 0):
                if joint_train:
                    self.gs_model.save_ckpt(t_iter)
                else:
                    if self.cfg.obj.num_refine > 1:
                        break
                    self.save_ckpt(t_iter)
            
            if t_iter == 0:
                if joint_train:
                    self.stored_gs_render_canonical(t_iter, nframes=self.cfg.obj.canon_nframes, device=self.cfg.device)
                else:
                    self.render_canonical(t_iter, nframes=self.cfg.obj.canon_nframes, device=self.cfg.device)
                    if self.cfg.check_mode:
                        self.animate(t_iter, device=self.cfg.device, check_mode=self.cfg.check_mode, keep_images=True)
                        import sys; sys.exit(0)

            if t_iter % self.cfg.train.anim_interval == 0 and self.cfg.train.anim_interval > 0:
                if joint_train:
                    self.stored_gs_animate(t_iter, device=self.cfg.device)
                else:
                    if self.mode == 'object':
                        do_animate = True if t_iter > 0 else False
                    else:
                        do_animate = True
                        
                    if 'pre_smt' in self.cfg.logdir:
                        self.animate(t_iter, check_mode=True, keep_images=True, device=self.cfg.device)
                    else:
                        if do_animate:
                            self.animate(t_iter, device=self.cfg.device)
                    self.render_canonical(t_iter, nframes=self.cfg.obj.canon_nframes, device=self.cfg.device)
            
            if t_iter % 1000 == 0 and t_iter > 0:
                if joint_train:
                    for g in gs:
                        g.oneupSHdegree()
                else:
                    self.gs_model.oneupSHdegree()
                
            if self.cfg.train.save_progress_images and t_iter % self.cfg.train.progress_save_interval == 0 and self.cfg.mode in ['human', 'human_scene']:
                assert not joint_train, "Not implemented yet!"
                self.render_canonical(t_iter, nframes=2, is_train_progress=True, device=self.cfg.device)
                
            with torch.no_grad():
                if joint_train:
                    assert not gs[-1].use_deformer
                    
                    for idx, g in enumerate(gs):
                        comp = gaussians.mode[idx]
                        gs_out = g.forward(
                                            smpl_scale=None, #self.train_dataset.scale.cuda(),
                                            dataset_idx=tmp_,
                                            is_train=True,
                                            ext_tfs=None,
                                            transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
                                            misc=self.train_dataset.misc,
                                            device=self.cfg.device
                                        )
                        
                        if comp != 'object':
                            self.stored_joint_3d[comp][rnd_idx] = gs_out['joint3d'][0].detach().clone()
                        else:
                            T = compute_bounding_box_centers(gs_out['xyz'].unsqueeze(0).cpu().numpy())[0]
                            self.stored_joint_3d[comp][rnd_idx] = torch.from_numpy(T).detach().clone()
                else:
                    if self.mode != 'object':
                        gs_out = self.gs_model.forward(
                                smpl_scale=None, #self.train_dataset.scale.cuda(),
                                dataset_idx=rnd_idx,
                                is_train=True,
                                ext_tfs=None,
                                transl=None, # self.train_dataset.misc['root.object'][rnd_idx].cuda(),
                                misc=self.train_dataset.misc,
                                device=self.cfg.device
                            )
                
                        self.stored_joint_3d[rnd_idx] = gs_out['joint3d'][0].detach().clone()
        
        # train progress images
        if self.cfg.train.save_progress_images:
            assert not joint_train, "Not implemented yet!"
            video_fname = f'{self.cfg.logdir}/train_{self.cfg.dataset.name}_{self.cfg.dataset.seq}.mp4'
            create_video(f'{self.cfg.logdir}/train_progress/', video_fname, fps=10)
            shutil.rmtree(f'{self.cfg.logdir}/train_progress/')

            
    def save_ckpt(self, iter=None):
        
        iter_s = 'final' if iter is None else f'{iter:06d}'
        
        extra_params = ['body_pose', 'global_orient', 'betas', 'transl', 'scale']
        state_dict = self.gs_model.state_dict()  
        
        for p_key in extra_params:
            if hasattr(self.gs_model, p_key):
                state_dict[p_key] = self.gs_model.__getattribute__(p_key)
        torch.save(state_dict, f'{self.cfg.logdir_ckpt}/human_{iter_s}.pth')
        
        if self.cfg.obj.num_refine > 1:
            torch.save(state_dict, f'{self.cfg.logdir_ckpt}/human_final_{self.refine_step}.pth')
            
        logger.info(f'Saved checkpoint {iter_s}')
    
    def human_densification(self, human_gs_out, visibility_filter, radii, viewspace_point_tensor, iteration, gs_w=None):
        self.gs_model.opacity_tmp = human_gs_out['opacity']
        self.gs_model.scales_tmp = human_gs_out['scales_canon']
        self.gs_model.rotmat_tmp = human_gs_out['rotmat_canon']
        
        # # Keep track of max weight of each GS for pruning
        # self.gs_model.max_weight[visibility_filter] = torch.max(self.gs_model.max_weight[visibility_filter],
        #                                                     gs_w[visibility_filter].float())
        
        # Densification
        if iteration < self.cfg.obj.densify_until_iter:
            # Keep track of max radii in image-space for pruning
            self.gs_model.max_radii2D[visibility_filter] = torch.max(
                self.gs_model.max_radii2D[visibility_filter], 
                radii[visibility_filter]
            )
            self.gs_model.add_densification_stats(viewspace_point_tensor, visibility_filter)            
            
            # if iteration == 1 and self.cfg.obj.init_prune:
            #     self.gs_model.initial_prune(human_gs_out)

            if iteration > self.cfg.obj.densify_from_iter and iteration % self.cfg.obj.densification_interval == 0 and iteration != 1:
                logger.info(f'[iter {iteration}] Do densify and prune')
                # size_threshold = 20 if iteration > self.cfg.obj.opacity_reset_interval else None
                size_threshold = self.cfg.obj.size_threshold
                # self.gs_model.densify_and_prune(
                #     human_gs_out=human_gs_out, max_grad = self.cfg.obj.densify_grad_threshold, max_grad_abs = self.cfg.obj.densify_grad_abs_threshold,
                #     min_opacity=0.005, extent=self.cfg.obj.densify_extent, max_screen_size=size_threshold,
                #     max_n_gs=self.cfg.obj.max_n_gaussians,
                # )
                self.gs_model.densify_and_prune_origin(
                    human_gs_out=human_gs_out, max_grad = self.cfg.obj.densify_grad_threshold,
                    min_opacity=0.0, extent=self.cfg.obj.densify_extent, max_screen_size=size_threshold,
                    max_n_gs=self.cfg.obj.max_n_gaussians,
                    # min_opacity=0.005, extent=self.cfg.obj.densify_extent, max_screen_size=size_threshold,
                )

            # # if iteration % self.cfg.obj.opacity_reduce_interval == 0 and self.cfg.obj.use_reduce:
            # #     self.gs_model.reduce_opacity()

            # # # if iteration % self.cfg.obj.opacity_reset_interval == 0 or (dataset.white_background and iteration == self.cfg.obj.densify_from_iter):
            # # if iteration % self.cfg.obj.opacity_reset_interval == 0 or (True and iteration == self.cfg.obj.densify_from_iter):
            # #     self.gs_model.reset_opacity()
        
        if iteration > self.cfg.obj.densify_from_iter and iteration < self.cfg.obj.prune_until_iter:
            if iteration % self.cfg.obj.opacity_reduce_interval == 0:
                logger.info(f'[iter {iteration}] Prune low opacity gs')
                self.gs_model.prune_opacity(self.cfg.obj.prune_min_opacity)

        if iteration > self.cfg.obj.densify_from_iter and iteration < self.cfg.obj.prune_until_iter and self.cfg.obj.use_prune_weight: #self.cfg.obj.use_prune_weight:
            # if iteration % img_num / img_num_modifier == 0 and iteration % self.cfg.obj.opacity_reset_interval > img_num / img_num_modifier:
            if iteration % self.cfg.obj.opacity_reset_interval == 0:
                prune_mask = (self.gs_model.max_weight < self.cfg.obj.min_weight).squeeze()
                
                num_prune = sum(prune_mask)
                if (self.gs_model.get_xyz.shape[0] - num_prune) < 1e4:
                    tgt_num = (self.gs_model.get_xyz.shape[0] - 1e4) - 1
                    if tgt_num <= 0:
                        logger.warning("There are no saturated points!")
                        return
                    new_mask_idx = torch.where(prune_mask == True)[0][torch.randperm(num_prune)[:int((num_prune - tgt_num))]]
                    prune_mask[new_mask_idx] = False
                if num_prune == 0:
                    return                    
                
                self.gs_model.prune_points(prune_mask)
                self.gs_model.max_weight *= 0
    
    @torch.no_grad()
    def animate(self, iter=None, keep_images=False, check_mode=False, device=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s = f'final_{self.refine_step}' if self.cfg.obj.num_refine > 1 else iter_s
        
        if self.gs_model:
            self.gs_model.eval()
        
        if self.mode != 'object':
            os.makedirs(f'{self.cfg.logdir}/kp2d/', exist_ok=True)
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        if check_mode:
            os.makedirs(f'{self.cfg.logdir}/anim_bg/', exist_ok=True)
            os.makedirs(f'{self.cfg.logdir}/anim_mask/', exist_ok=True)
        
        pabr = tqdm(range(len(self.train_dataset)), desc="Animation")
        
        for idx in pabr:
            if idx in self.train_dataset.misc['skip_frame']:
                continue            
            
            data = self.train_dataset[idx]
            data = dict((k,v[0] if isinstance(v, list) else v) for k,v in data.items())
            
            # if data['is_valid'] == 0:
            #     pass
            
            # else:
            data = custom_to(data, device)
            human_gs_out, scene_gs_out = None, None
            
            if self.gs_model:
                human_gs_out = self.gs_model.forward(
                    global_orient=None,
                    body_pose=None,
                    betas=None,
                    transl=None,
                    smpl_scale=None, #self.train_dataset.scale.cuda(),
                    dataset_idx=idx,
                    is_train=False,
                    misc=self.train_dataset.misc,
                    device=device,
                    is_flip=self.cfg.obj.num_refine > 1
                    # ext_tfs=ext_tfs,
                )
                    
            
            if self.scene_gs:
                scene_gs_out = self.scene_gs.forward()
            
            # human_gs_out['xyz'] = self.train_dataset.misc['v3d_c.object'][idx].to('cuda:0').float()
            
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
                device=device,
            )
            
            image = render_pkg["render"]
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim/{idx:05d}.png')
            
            if self.mode != 'object':
                K = data['cam_intrinsics'][:-1, :-1].unsqueeze(0)
                joint3d = human_gs_out['joint3d']
                draw_j2d(joint3d, K, data['rgb'], f'{self.cfg.logdir}/kp2d/{idx:05d}.png')
            
            if check_mode:
                mask = data['mask'][self.mode]
                m_image = image * mask + torch.FloatTensor([0]).cuda()[:, None, None] * (1. - mask)
                bg_image = image * (1. - mask) + torch.FloatTensor([0]).cuda()[:, None, None] * mask
                
                torchvision.utils.save_image(bg_image, f'{self.cfg.logdir}/anim_bg/{idx:05d}.png')
                torchvision.utils.save_image(m_image, f'{self.cfg.logdir}/anim_mask/{idx:05d}.png')
            
        video_fname = f'{self.cfg.logdir}/anim_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/anim/', video_fname, fps=20)
        
        if self.mode != 'object':
            video_fname = f'{self.cfg.logdir}/kp2d_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
            create_video(f'{self.cfg.logdir}/kp2d/', video_fname, fps=20)
        if not keep_images:
            try:
                shutil.rmtree(f'{self.cfg.logdir}/anim/')
                shutil.rmtree(f'{self.cfg.logdir}/kp2d/')
            except:
                pass

    def get_gdm_items(self, t_iter, img_size=1024, mask_thold=0.99, device=None):
        # if self.gs_model:
        #     self.gs_model.eval()
        
        os.makedirs(f'{self.cfg.logdir}/sds/', exist_ok=True)
        sample_idx = -1
        
        # make random camera
        nframes=50
        dist = 1.5
        camera_params = get_rotating_camera(
            dist=dist, img_size=img_size, 
            nframes=nframes, device=device,
            angle_limit=2*torch.pi,
        )
        
        frame_idx = np.random.randint(0, 50, 1)
        # dist = torch.from_numpy(np.random.random(1) + 1.5).float()
        
        # make random rotation
        random_rot = matrix_to_axis_angle(rotation_6d_to_matrix(torch.randn(1, 6)))[0]
        static_smpl_params = {
            'global_orient' : random_rot,
            'body_pose' : None,
            'betas' : None,
            'transl' : torch.zeros(3).to(device),
            'smpl_scale' : torch.ones(1).to(device),
        }

        # forward
        data = static_smpl_params
        human_gs_out = self.gs_model.forward(
            global_orient=data['global_orient'],
            body_pose=data['body_pose'],
            betas=data['betas'],
            transl=data['transl'],
            smpl_scale=data['smpl_scale'],
            dataset_idx=sample_idx,
            is_train=False,
            ext_tfs=None,
            misc=self.train_dataset.misc,
            device=device,
            render_canon=True,
        )
        
        # avoid in-place op.
        T = compute_bounding_box_centers_torch(human_gs_out['xyz'])
        human_gs_out['xyz'] = human_gs_out['xyz'] - T

        # rendering
        cam_p = camera_params[frame_idx[0]]
        data = dict(static_smpl_params, **cam_p)
        data = custom_to(data, device, cast_type=True)
        render_pkg = render_human_scene(
            data=data, 
            human_gs_out=human_gs_out, 
            scene_gs_out=None, 
            bg_color=self.bg_color,
            render_mode='human',
            device=device
        )

        image = render_pkg["render"]
        _image = image.detach().clone()
        
        # centering 
        mask = crop_and_center_image(_image, img_size=img_size, threshold_value=mask_thold, return_mask=True)
        # cv2.imwrite('test.png', mask.unsqueeze(0).permute(1,2,0).numpy()*255)
        
        if mask.sum() == 0:
            logger.warning('Something went wrong! The sum of mask is zero!!')
            return None

        # make pidi bound
        with torch.no_grad():
            predictor = PidiNetDetector()
            cond_img = predictor((_image.permute(1,2,0).cpu().detach().numpy()*255).astype(np.uint8))
            # cond_img = predictor(cv2.imread('img_pred.png')); cv2.imwrite('test.png', cond_img[..., None])
            dgm_cond = [Image.fromarray(cond_img)]
            dgm_cond[0] = dgm_cond[0].convert('RGB')

        dg_loss, step_ratio, guid_loss_dict = self.DGM.get_loss(
                                        image[None], 
                                        '0',
                                        vers=None, 
                                        hors=None, 
                                        radii=None, 
                                        iteration=t_iter, 
                                        cond_image=dgm_cond, 
                                        additional_prompt=None,
                                        mask=mask,
                                        ddim_num_step_inferences=20,
                                        ddim_fixed_max_time_step=True,
                                        save_intermediate=True,
                                        img_description="",
                                        minimum_mask_thrs=0.02,
                                        cfg_rescale_weight=self.cfg.obj.dgm.cfg_rescale_weight
                                        )
        
        return dg_loss
    
            
    def get_sds_items(self, t_iter, nframes=25, img_size=1024, mask_thold=0.99, device=None, resize_img=False, save_progress=True):
        if self.gs_model:
            self.gs_model.eval()
        
        os.makedirs(f'{self.cfg.logdir}/sds/', exist_ok=True)
        seq_name = self.train_dataset.misc['full_seq_name']
        
        if seq_name in dist_dict.keys():
            dist = dist_dict[seq_name]
        else:
            dist = 1.5
            
        if seq_name in idx_dict.keys():
            sample_idx = idx_dict[seq_name]
        else:
            sample_idx = -1
        
        camera_params = get_rotating_camera(
            dist=dist, img_size=img_size, 
            nframes=nframes, device=device,
            angle_limit=2*torch.pi,
        )

        static_smpl_params = {
            'global_orient' : None, # torch.zeros(3).to(device),
            'body_pose' : None,
            'betas' : None,
            'transl' : torch.zeros(3).to(device), #-self.train_dataset.misc['root.object'][-1].to(device), # torch.zeros(3).to(device),
            'smpl_scale' : torch.ones(1).to(device),
        }

        data = static_smpl_params
        human_gs_out = self.gs_model.forward(
            global_orient=data['global_orient'],
            body_pose=data['body_pose'],
            betas=data['betas'],
            transl=data['transl'],
            smpl_scale=data['smpl_scale'],
            dataset_idx=sample_idx,
            is_train=False,
            ext_tfs=None,
            misc=self.train_dataset.misc,
            device=device,
            render_canon=True,
        )
        T = compute_bounding_box_centers(human_gs_out['xyz'].unsqueeze(0).detach().cpu().numpy())[0]
        human_gs_out['xyz'] -= torch.from_numpy(T).unsqueeze(0).to(device)

        pbar = range(nframes)
        
        pred_images = None
        for idx in pbar:
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)
                
            data = custom_to(data, device, cast_type=True)

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=None, 
                bg_color=self.bg_color,
                render_mode='human',
                device=device
            )
            
            image = render_pkg["render"]
            image = crop_and_center_image(image, img_size=img_size, threshold_value=mask_thold)
            
            with torch.no_grad():
                if save_progress:
                    torchvision.utils.save_image(image, f'{self.cfg.logdir}/sds/{idx:05d}.png')

            if idx == 0:
                with torch.no_grad():
                    if False:
                    # if t_iter % self.cfg.obj.sds_interval == 0:
                        img_name = f"{self.cfg.logdir}/{seq_name}"
                        img_name += f"_CUDA_{os.getenv('CUDA_VISIBLE_DEVICES')}.png"
                        torchvision.utils.save_image(image, img_name)
                        
                        launch_vivid123(img_name, self.cfg.obj.sds_num_steps, self.cfg.obj.sds_use_xl)
                    
                    if self.cfg.obj.sds_use_xl:
                        frame_name = 'xl_frames'
                    else:
                        frame_name = 'base_frames'
                    
                    vivid_dir = f"submodules/vivid123/outputs/{seq_name}"
                    # vivid_dir += f"_CUDA_{os.getenv('CUDA_VISIBLE_DEVICES')}/{frame_name}/*"
                    vivid_dir += f"/{frame_name}/*"
                    
                    gt_images = sorted(glob.glob(vivid_dir))
                    gt_images = [
                                    (torch.tensor(np.array(Image.open(img)))/255.).permute(2,0,1).to(device) \
                                    for img in gt_images
                                ]
                    gt_images = torch.cat([crop_and_center_image(gt, img_size=256, threshold_value=mask_thold, resize_img=resize_img).unsqueeze(0) for gt in gt_images])
                    gt_masks = torch.cat([rgb_to_grayscale(gt).unsqueeze(0) for gt in gt_images]) > mask_thold
                # create_video(f'{self.cfg.logdir}/sds', f'{self.cfg.logdir}/before.mp4', fps=10)

            if resize_img:
                image = crop_and_center_image(image, img_size=256, threshold_value=mask_thold, resize_img=resize_img).unsqueeze(0)
            else:
                image = F.interpolate(image.unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=False)
            # save_img(image[0]*255)

            if pred_images is None:
                pred_images = image
            else:
                pred_images = torch.cat([pred_images, image], dim=0)
    
        # idx = 0
        # save_img(gt_images[idx]*255, 'img_gt.png')
        # save_img(pred_images[idx]*255, 'img_pred.png')
        return pred_images, gt_images, gt_masks
    
    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None, device=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''
        iter_s = f'final_{self.refine_step}' if self.cfg.obj.num_refine > 1 else iter_s
        
        if self.gs_model:
            self.gs_model.eval()
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        dist = 1.5
        
        camera_params = get_rotating_camera(
            dist=dist, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device=device,
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        try:
            betas = self.gs_model.betas.detach() if hasattr(self.gs_model, 'betas') else self.train_dataset.betas[0]
        except:
            betas = None
        
        if betas is not None:
            static_smpl_params = get_smpl_static_params(
                betas=betas,
                pose_type=self.cfg.obj.canon_pose_type if pose_type is None else pose_type,
                device=device
            )
            # static_smpl_params['transl'] = -self.train_dataset.misc[f'root.{self.mode}'][-1].to(device)
        else:
            static_smpl_params = {
                'global_orient' : None, # torch.zeros(3).to(device),
                'body_pose' : None,
                'betas' : None,
                'transl' : torch.zeros(3).to(device), #-self.train_dataset.misc['root.object'][-1].to(device), # torch.zeros(3).to(device),
                'smpl_scale' : torch.ones(1).to(device),
            }
        
        if is_train_progress:
            progress_imgs = []
        
        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")
        
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(static_smpl_params, **cam_p)

            if self.gs_model:
                human_gs_out = self.gs_model.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                    misc=self.train_dataset.misc,
                    device=device,
                    render_canon=True,
                    is_flip=self.cfg.obj.num_refine > 1
                )            
                
            if not self.gs_model.use_deformer:
                T = compute_bounding_box_centers(human_gs_out['xyz'].unsqueeze(0).detach().cpu().numpy())[0]
                human_gs_out['xyz'] -= torch.from_numpy(T).unsqueeze(0).to(device)                
            else:
                assert human_gs_out['joint3d'].shape[0] == 1
                human_gs_out['xyz'] -= human_gs_out['joint3d'][0, :1, :].to(device)
                
            data = custom_to(data, device, cast_type=True)
                
            if is_train_progress:
                scale_mod = 0.5
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    scaling_modifier=scale_mod,
                    device=device
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    device=device
                )
                
                image = render_pkg["render"]
                
                progress_imgs.append(image)
                
            else:
                render_pkg = render_human_scene(
                    data=data, 
                    human_gs_out=human_gs_out, 
                    scene_gs_out=scene_gs_out, 
                    bg_color=self.bg_color,
                    render_mode='human',
                    device=device
                )
                
                image = render_pkg["render"]
                
                torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        if is_train_progress:
            os.makedirs(f'{self.cfg.logdir}/train_progress/', exist_ok=True)
            log_img = torchvision.utils.make_grid(progress_imgs, nrow=4, pad_value=0)
            save_image(log_img, f'{self.cfg.logdir}/train_progress/{iter:06d}.png', 
                       text_labels=f"{iter:06d}, n_gs={self.gs_model.n_gs}")
            return
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/canon/')
        
    def render_poses(self, camera_params, smpl_params, pose_type='a_pose', bg_color='white'):
    
        if self.gs_model:
            self.gs_model.eval()
        
        betas = self.gs_model.betas.detach() if hasattr(self.gs_model, 'betas') else self.val_dataset.betas[0]
        
        nframes = len(camera_params)
        
        canon_forward_out = None
        if hasattr(self.gs_model, 'canon_forward'):
            canon_forward_out = self.gs_model.canon_forward()
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        if bg_color == 'white':
            bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        elif bg_color == 'black':
            bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
            
            
        imgs = []
        for idx in pbar:
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            data = dict(smpl_params, **cam_p)

            if self.gs_model:
                if canon_forward_out is not None:
                    human_gs_out = self.gs_model.forward_test(
                        canon_forward_out,
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                        is_flip=self.cfg.obj.num_refine > 1
                    )
                else:
                    human_gs_out = self.gs_model.forward(
                        global_orient=data['global_orient'],
                        body_pose=data['body_pose'],
                        betas=data['betas'],
                        transl=data['transl'],
                        smpl_scale=data['smpl_scale'],
                        dataset_idx=-1,
                        is_train=False,
                        ext_tfs=None,
                        is_flip=self.cfg.obj.num_refine > 1
                    )

            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_out, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            imgs.append(image)
        return imgs
    
    @torch.no_grad()
    def stored_gs_novel_view_animate(self, iter=None, rot_idx=None, device=None):
        iter_s = 'HOI' if iter is None else f'{iter:06d}'
        if self.gs_model:
            self.gs_model.eval()
            
        gaussians = self.gs_model
        try:
            data = gaussians._get_dataset('left')[0]
        except:
            data = gaussians._get_dataset('right')[0]
        
        os.makedirs(f'{self.cfg.logdir}/anim/', exist_ok=True)
        
        rot_idx = 20 if rot_idx is None else rot_idx
        
        rot_cam = get_rotating_camera(dist=3.5, nframes=60)
        rot_cam_list = [rot_cam[rot_idx]]
        # rot_cam_list = [rot_cam[20], rot_cam[40]]
        
        for idx, cam in enumerate(rot_cam_list):
            for k,v in cam.items():
                if k not in ['image_height', 'image_width', 'camera_center']:
                    data[k] = v
            data['view'] = idx
            self.stored_gs_animate(iter=iter, custom_data=data, device=device, keep_images=True)
    
    @torch.no_grad()
    def stored_gs_mesh_animate(self, iter=None, keep_images=False, custom_data=None, device=None, export=True, mesh_save_dir=None):
        from bigs.utils.general import make_p3d_mesh_viewer
        from pytorch3d.structures import Meshes, Pointclouds
        from pytorch3d.renderer.mesh import Textures
        from pytorch3d.structures.meshes import join_meshes_as_scene
        from pytorch3d.renderer.blending import BlendParams        
        # from submodules.hold.code.visualize_ckpt import DataViewer
        # viewer = DataViewer(interactive=True, size=(2024, 2024))
        
        iter_s = 'HOI' if iter is None else f'{iter:06d}'
        if self.gs_model:
            self.gs_model.eval()
            
        gaussians = self.gs_model
        dbs = [gaussians._get_dataset(m) for m in gaussians.mode]
        gs = [gaussians._get_gaussian(m) for m in gaussians.mode]
        
        os.makedirs(f'{self.cfg.logdir}/obj/', exist_ok=True)
        
        # assert self.cfg.components == ['right', 'left', 'object']
        n_hand_faces = len(gs[0].smpl_template.v_template)
        obj_mesh = trimesh.load_mesh(mesh_save_dir)
        hoi_faces = np.concatenate([
            gs[0].smpl_template.faces,
            gs[1].smpl_template.faces + n_hand_faces,
            obj_mesh.faces + n_hand_faces*2
        ], axis=0)
        hoi_faces_alt = np.concatenate([
            gs[0].smpl_template.faces,
            obj_mesh.faces + n_hand_faces
        ], axis=0)        
        
        mesh_renderer = make_p3d_mesh_viewer(
            intrinsic = dbs[0][0]['cam_intrinsics'],
            img_size = [dbs[0][0]['image_height'], dbs[0][0]['image_width']],
            device = device
        )
        
        pbar = tqdm(range(len(self.train_dataset)), desc="Animation")
        for idx in pbar:
            image = torch.zeros_like(dbs[0][0]['rgb'], dtype=dbs[0][0]['rgb'].dtype).cuda()
            
            human_gs_outs = None
            # if dbs[0][idx]['is_valid'] != 0:
            for db, g in zip(dbs, gs):
                self.train_dataset = db
                data = db[idx] if custom_data is None else custom_data
                human_gs = g
                
                data = dict((k,v[0] if isinstance(v, list) else v) for k,v in data.items())
                data = custom_to(data, 'cuda')            
                human_gs_out, scene_gs_out = None, None
                
                if g.use_deformer:
                    mode = 'right' if g.smpl_template.is_rhand else 'left'
                else:
                    mode = 'object'
                
                if data['mask'][mode].sum() == 0: #FIXME: FIX ME!!!!!!
                    continue                
                
                human_gs_out = human_gs.forward(
                    global_orient=None,
                    body_pose=None,
                    betas=None,
                    transl=None,
                    smpl_scale=None, #data['smpl_scale'][None],
                    dataset_idx=idx,
                    is_train=False,
                    misc = db.misc,
                    device=device,
                    is_flip=self.cfg.obj.num_refine > 1,
                    t_mesh= obj_mesh,
                    # ext_tfs=ext_tfs,
                )
                
                human_gs_out = {
                    'xyz': human_gs_out['xyz']
                }

                if human_gs_outs is None:
                    human_gs_outs = human_gs_out
                else:
                    for (ts_k, ts_v), (t_k, t_v) in zip(human_gs_outs.items(), human_gs_out.items()):
                        human_gs_outs[ts_k] = torch.cat([ts_v, t_v], dim=0)

            if export:
                try:        
                    mesh = trimesh.Trimesh(human_gs_outs['xyz'].cpu().detach().numpy(), hoi_faces)
                except:
                    mesh = trimesh.Trimesh(human_gs_outs['xyz'].cpu().detach().numpy(), hoi_faces_alt)    
                mesh.export(f'{self.cfg.logdir}/obj/{idx:05d}.obj')
            
            try:
                trimesh.Trimesh(human_gs_outs['xyz'].cpu().detach().numpy(), hoi_faces)
                mesh = Meshes(
                    verts=human_gs_outs['xyz'].unsqueeze(0).to('cuda:0'),
                    faces=torch.from_numpy(hoi_faces).unsqueeze(0).to('cuda:0'),
                    textures=Textures(verts_rgb=torch.ones_like(human_gs_outs['xyz']).to('cuda:0').unsqueeze(0)*255.)
                )
                mesh = join_meshes_as_scene(mesh, True)
                smpl_images = mesh_renderer(meshes_world=mesh)
                
                smpl_images[..., -1] *= 0
                smpl_images[..., -1] += 255.
                
                cv2.imwrite(f'{self.cfg.logdir}/obj/{idx:05d}.png', smpl_images[0].cpu().numpy())
            except:
                mesh = Meshes(
                    verts=human_gs_outs['xyz'].unsqueeze(0).to('cuda:0'),
                    faces=torch.from_numpy(hoi_faces_alt).unsqueeze(0).to('cuda:0'),
                    textures=Textures(verts_rgb=torch.ones_like(human_gs_outs['xyz']).to('cuda:0').unsqueeze(0)*255.)
                )
                mesh = join_meshes_as_scene(mesh, True)
                smpl_images = mesh_renderer(meshes_world=mesh)
                cv2.imwrite(f'{self.cfg.logdir}/obj/{idx:05d}.png', smpl_images[0].cpu().numpy())
        
        video_fname = f'{self.cfg.logdir}/mesh.mp4'
        create_video(f'{self.cfg.logdir}/obj/', video_fname, fps=20)
            
    @torch.no_grad()
    def stored_gs_animate(self, iter=None, keep_images=False, custom_data=None, vis_mesh=False, device=None):
        from bigs.utils.general import make_p3d_mesh_viewer
        from pytorch3d.structures import Meshes, Pointclouds
        from pytorch3d.renderer.mesh import Textures
        from pytorch3d.structures.meshes import join_meshes_as_scene
        from pytorch3d.renderer.blending import BlendParams        
        
        iter_s = 'HOI' if iter is None else f'{iter:06d}'
        if self.gs_model:
            self.gs_model.eval()
            
        gaussians = self.gs_model
        dbs = [gaussians._get_dataset(m) for m in gaussians.mode]
        gs = [gaussians._get_gaussian(m) for m in gaussians.mode]
        
        os.makedirs(f'{self.cfg.logdir}/anim_{self.cfg.dataset.gt_cam_num}/', exist_ok=True)
        os.makedirs(f'{self.cfg.logdir}/obj_{self.cfg.dataset.gt_cam_num}/', exist_ok=True)
        
        if vis_mesh:
            n_hand_faces = len(gs[0].smpl_template.v_template)
            obj_mesh = trimesh.load_mesh(f'{self.cfg.logdir}/{self.cfg.dataset.seq}.obj')
            hoi_faces = np.concatenate([
                gs[0].smpl_template.faces,
                gs[1].smpl_template.faces + n_hand_faces,
                obj_mesh.faces + n_hand_faces*2
            ], axis=0)
            hoi_faces_alt = np.concatenate([
                gs[0].smpl_template.faces,
                obj_mesh.faces + n_hand_faces
            ], axis=0)        
            
            mesh_renderer = make_p3d_mesh_viewer(
                intrinsic = dbs[0][0]['cam_intrinsics'],
                img_size = [dbs[0][0]['image_height'], dbs[0][0]['image_width']],
                device = device
        )        
        
        if 'gt' in self.train_dataset.misc:
            pbar = tqdm(range(len(self.train_dataset.misc['gt']['2d']['verts.right'])), desc="Animation")
        else:
            pbar = tqdm(range(len(self.train_dataset)), desc="Animation")
        
        time_list = []
        for idx in pbar:
            image = torch.zeros_like(dbs[0][0]['rgb'], dtype=dbs[0][0]['rgb'].dtype).cuda()
            
            if 'skip_frame' in dbs[0].misc and idx in dbs[0].misc['skip_frame']:
                continue
            
            human_gs_outs = None
            # if dbs[0][idx]['is_valid'] != 0:
            for db, g in zip(dbs, gs):
                self.train_dataset = db
                data = db[idx] if custom_data is None else custom_data
                human_gs = g
                
                data = dict((k,v[0] if isinstance(v, list) else v) for k,v in data.items())
                data = custom_to(data, 'cuda')            
                human_gs_out, scene_gs_out = None, None
                
                if g.use_deformer:
                    mode = 'right' if g.smpl_template.is_rhand else 'left'
                else:
                    mode = 'object'
                
                if data['mask'][mode].sum() == 0: #FIXME: FIX ME!!!!!!
                    continue                
                
                # stt = time.time()                
                with torch.no_grad():
                    human_gs_out = human_gs.forward(
                        global_orient=None,
                        body_pose=None,
                        betas=None,
                        transl=None,
                        smpl_scale=None, #data['smpl_scale'][None],
                        dataset_idx=idx,
                        is_train=False,
                        misc = db.misc,
                        device=device,
                        is_flip=self.cfg.obj.num_refine > 1
                        # ext_tfs=ext_tfs,
                    )
                
                if 'body_pose' in human_gs_out.keys():
                    human_gs_out.pop('body_pose')
                    
                if 'global_orient' in human_gs_out.keys():
                    human_gs_out.pop('global_orient')                    

                if human_gs_outs is None:
                    human_gs_outs = human_gs_out
                else:
                    for (ts_k, ts_v), (t_k, t_v) in zip(human_gs_outs.items(), human_gs_out.items()):
                        if t_v is None or ts_k == 'v_posed': continue
                        
                        if isinstance(ts_v, torch.Tensor):
                            human_gs_outs[ts_k] = torch.cat([ts_v, t_v], dim=0)
                        else:
                            human_gs_outs[ts_k] = t_v
            
            if custom_data is not None:
                T = compute_bounding_box_centers(human_gs_outs['xyz'][-gs[-1]._xyz.shape[0]: ,:].unsqueeze(0).detach().cpu().numpy())[0]
                T = torch.from_numpy(T).to(device)
                human_gs_outs['xyz'] -= T.unsqueeze(0).cuda()
            
            if 'gt' in self.train_dataset.misc:
                if self.cfg.dataset.gt_cam_num == 0:
                    world2cam = torch.from_numpy(self.train_dataset.misc['gt']['params']['world2ego'][idx]).unsqueeze(0).float().to('cuda:0')
                else:
                    world2cam = self.train_dataset.misc['gt']['params']['world2cam'][self.cfg.dataset.gt_cam_num - 1].unsqueeze(0)
                
                human_gs_outs['xyz'] = transform_points_batch(world2cam, human_gs_outs['xyz'].unsqueeze(0))[0]
            
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_outs, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode=self.cfg.mode,
            )
            # end = time.time()
            # time_list.append(end - stt)

            image = render_pkg["render"]
            # save_img(image*255)
            
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/anim_{self.cfg.dataset.gt_cam_num}/{idx:05d}.png')
            
            if vis_mesh:
                try:
                    trimesh.Trimesh(human_gs_outs['xyz'].cpu().detach().numpy(), hoi_faces)
                    mesh = Meshes(
                        verts=human_gs_outs['xyz'].unsqueeze(0).to('cuda:0'),
                        faces=torch.from_numpy(hoi_faces).unsqueeze(0).to('cuda:0'),
                        textures=Textures(verts_rgb=torch.ones_like(human_gs_outs['xyz']).to('cuda:0').unsqueeze(0)*255.)
                    )
                    mesh = join_meshes_as_scene(mesh, True)
                    smpl_images = mesh_renderer(meshes_world=mesh)
                    
                    smpl_images[..., -1] *= 0
                    smpl_images[..., -1] += 255.
                    
                    cv2.imwrite(f'{self.cfg.logdir}/obj_{self.cfg.dataset.gt_cam_num}/{idx:05d}.png', smpl_images[0].cpu().numpy())
                except:
                    mesh = Meshes(
                        verts=human_gs_outs['xyz'].unsqueeze(0).to('cuda:0'),
                        faces=torch.from_numpy(hoi_faces_alt).unsqueeze(0).to('cuda:0'),
                        textures=Textures(verts_rgb=torch.ones_like(human_gs_outs['xyz']).to('cuda:0').unsqueeze(0)*255.)
                    )
                    mesh = join_meshes_as_scene(mesh, True)
                    smpl_images = mesh_renderer(meshes_world=mesh)
                    cv2.imwrite(f'{self.cfg.logdir}/obj_{self.cfg.dataset.gt_cam_num}/{idx:05d}.png', smpl_images[0].cpu().numpy())

        prefix = 'anim' if custom_data is None else f"view_{custom_data['view']}"
        video_fname = f'{self.cfg.logdir}/{prefix}_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}_{self.cfg.dataset.gt_cam_num}.mp4'
        create_video(f'{self.cfg.logdir}/anim_{self.cfg.dataset.gt_cam_num}/', video_fname, fps=20)
        if not keep_images:
            try:
                shutil.rmtree(f'{self.cfg.logdir}/anim_{self.cfg.dataset.gt_cam_num}/')
            except:
                pass            
    
    @torch.no_grad()
    def stored_gs_render_canonical(self, iter=None, nframes=100, is_train_progress=False, pose_type=None, device=None):
        iter_s = 'HOI' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''        
        
        if self.gs_model:
            self.gs_model.eval()
        
        gaussians = self.gs_model
        gs = [gaussians._get_gaussian(m) for m in gaussians.mode]
        misc = gaussians._get_dataset(gaussians.mode[0]).misc
        
        os.makedirs(f'{self.cfg.logdir}/canon/', exist_ok=True)
        
        dist = 1.5
        
        camera_params = get_rotating_camera(
            dist=dist, img_size=256 if is_train_progress else 512, 
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2*torch.pi,
        )
        
        canon_scale = 1.5
        canon_gap = 0.75
        static_smpl_params_list = []
        for g in gs:
            try:
                betas = g.betas.detach()
            except:
                betas = None
                
            if betas is not None:
                mode = 'right' if g.smpl_template.is_rhand else 'left'
                static_smpl_params = get_smpl_static_params(
                        betas=betas,
                        pose_type=self.cfg.obj.canon_pose_type if pose_type is None else pose_type,
                )
                static_smpl_params['transl'] = -misc[f'root.{mode}'][-1].cuda() * canon_scale
                static_smpl_params_list.append(static_smpl_params)
            else:
                static_smpl_params_list.append({
                    'global_orient' : None, # torch.zeros(3).cuda(),
                    'body_pose' : None,
                    'betas' : None,
                    'transl' : torch.zeros(3).cuda(), # torch.tensor(misc['cached_data']['object']['transl'][-1]).cuda() * 2, # torch.zeros(3).cuda(),
                    'smpl_scale' : torch.ones(1).cuda(),
                })
        if 'right' in gaussians.mode:
            r_idx = gaussians.mode.index('right')
            static_smpl_params_list[r_idx]['transl'][0] += canon_gap
            static_smpl_params_list[r_idx]['smpl_scale'] *= canon_scale
        if 'left' in gaussians.mode:
            l_idx = gaussians.mode.index('left')
            static_smpl_params_list[l_idx]['transl'][0] -= canon_gap
            static_smpl_params_list[l_idx]['smpl_scale'] *= canon_scale
        
        pbar = tqdm(range(nframes), desc="Canonical:")
        for idx in pbar:
            human_gs_outs = None
            human_gs_out, scene_gs_out = None, None
            
            cam_p = camera_params[idx]
            
            for i, static_smpl_params in enumerate(static_smpl_params_list):
                human_gs = gs[i]
                data = dict(static_smpl_params, **cam_p)

                human_gs_out = human_gs.forward(
                    global_orient=data['global_orient'],
                    body_pose=data['body_pose'],
                    betas=data['betas'],
                    transl=data['transl'],
                    smpl_scale=data['smpl_scale'],
                    dataset_idx=-1,
                    is_train=False,
                    ext_tfs=None,
                    misc=misc,
                    device=device,
                    render_canon=True,
                    is_flip=self.cfg.obj.num_refine > 1
                )
                
                if human_gs_out['joint3d'] is None:
                    T = compute_bounding_box_centers(human_gs_out['xyz'].unsqueeze(0).detach().cpu().numpy())[0]
                    human_gs_out['xyz'] -= torch.from_numpy(T).unsqueeze(0).to(device)
                
                if 'body_pose' in human_gs_out.keys():
                    human_gs_out.pop('body_pose')
                    
                if 'global_orient' in human_gs_out.keys():
                    human_gs_out.pop('global_orient')                    

                #
                if human_gs_outs is None:
                    human_gs_outs = human_gs_out
                else:
                    for (ts_k, ts_v), (t_k, t_v) in zip(human_gs_outs.items(), human_gs_out.items()):
                        if t_v is None or ts_k == 'v_posed': continue
                        
                        if isinstance(ts_v, torch.Tensor):
                            human_gs_outs[ts_k] = torch.cat([ts_v, t_v], dim=0)
                        else:
                            human_gs_outs[ts_k] = t_v                
                
            render_pkg = render_human_scene(
                data=data, 
                human_gs_out=human_gs_outs, 
                scene_gs_out=scene_gs_out, 
                bg_color=self.bg_color,
                render_mode='human',
            )
            image = render_pkg["render"]
            torchvision.utils.save_image(image, f'{self.cfg.logdir}/canon/{idx:05d}.png')
        
        video_fname = f'{self.cfg.logdir}/canon_{self.cfg.dataset.name}_{self.cfg.dataset.seq}_{iter_s}.mp4'
        create_video(f'{self.cfg.logdir}/canon/', video_fname, fps=10)
        shutil.rmtree(f'{self.cfg.logdir}/canon/')    
        