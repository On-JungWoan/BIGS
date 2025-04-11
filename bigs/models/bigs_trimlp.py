#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
import trimesh
import numpy as np
from torch import nn
import os.path as op
from loguru import logger
import torch.nn.functional as F
from bigs.utils.general import make_and_export_mesh
from bigs.models import smpl_lbsweight_top_k

from bigs.utils.general import (
    inverse_sigmoid, 
    get_expon_lr_func, 
    strip_symmetric,
    build_scaling_rotation,
    apply_transformation,
    pcd_to_mesh,
    initialize_gaussian_scales,
)
from bigs.utils.rotations import (
    axis_angle_to_rotation_6d, 
    axis_angle_to_matrix,
    matrix_to_quaternion, 
    matrix_to_rotation_6d, 
    quaternion_multiply,
    quaternion_to_matrix, 
    rotation_6d_to_axis_angle, 
    rotation_6d_to_matrix,
    torch_rotation_matrix_from_vectors,
)
from bigs.cfg.constants import SMPL_PATH
from bigs.utils.subdivide_smpl import subdivide_smpl_model
from submodules.hold.common.rot import quaternion_apply
from pytorch3d.transforms import axis_angle_to_quaternion
from submodules.hold.code.src.utils.eval_modules import compute_bounding_box_centers, compute_bounding_box_centers_torch
import open3d as o3d

from .modules.lbs import lbs_extra
from .modules.smpl_layer import SMPL
from .modules.mano_layer import MANO
from .modules.triplane import TriPlane
from .modules.decoders import AppearanceDecoder, DeformationDecoder, GeometryDecoder

from bigs.utils.general import make_and_export_mesh, save_img, custom_to

from submodules.hold.common.xdict import xdict
from bigs.datasets.arctic.src.datasets.arctic_dataset import map_deform2eval_batch, map_deform2eval_batch_torch
from submodules.hold.common.transforms import transform_points_batch

SCALE_Z = 1e-5


class BIGS_TRIMLP:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(
        self, 
        sh_degree: int, 
        only_rgb: bool=False,
        n_subdivision: int=0,  
        use_surface=False,  
        init_2d=False,
        rotate_sh=False,
        isotropic=False,
        init_scale_multiplier=0.5,
        n_features=32,
        use_deformer=False,
        disable_posedirs=False,
        triplane_res=256,
        betas=None,
        is_rhand=True,
        device=None,
        misc=None,
        optim_rotations=False,
        optim_opacity=False,
        mode=None,
        refine_step=0,
    ):
        self.refine_step = refine_step
        self.only_rgb = only_rgb
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self.scaling_multiplier = torch.empty(0)
        # self.opacity_multiplier = torch.empty(0)

        self.max_wegiht = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.xyz_gradient_accum_abs = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.device = device
        self.use_surface = use_surface
        self.init_2d = init_2d
        self.rotate_sh = rotate_sh
        self.isotropic = isotropic
        self.init_scale_multiplier = init_scale_multiplier
        self.use_deformer = use_deformer
        self.disable_posedirs = disable_posedirs
        
        assert mode is not None
        self.mode = mode
        
        if betas is not None:
            self.create_betas(betas, requires_grad=False)
        
        self.triplane = TriPlane(n_features, resX=triplane_res, resY=triplane_res, resZ=triplane_res).to(device)
        self.appearance_dec = AppearanceDecoder(n_features=n_features*3, optim_opacity=optim_opacity).to(device)
        self.deformation_dec = DeformationDecoder(n_features=n_features*3, n_joints=16,
                                                  disable_posedirs=disable_posedirs).to(device)
        self.geometry_dec = GeometryDecoder(n_features=n_features*3, use_surface=use_surface, optim_rotations=optim_rotations).to(device)
        
        self.is_mesh = True
        if use_deformer:
            if n_subdivision > 0:
                logger.info(f"Subdividing SMPL model {n_subdivision} times")
                self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=n_subdivision, is_rhand=is_rhand).to(self.device)
            else:
                # SMPL_PATH = './misc/body_models/mano'
                self.smpl_template = MANO(SMPL_PATH, create_transl=False, use_pca=False, flat_hand_mean=False, is_rhand=is_rhand).to(self.device)
        else:
            from bigs.utils.subdivide_smpl import subdivide
            assert misc is not None
            
            if 'cano_mesh.object' in misc.keys():
                #* LOAD INIT MESH
                tgt = misc['cano_mesh.object']
                if isinstance(tgt, np.ndarray):
                    self.is_mesh = False
                    self.smpl_template = xdict()
                    self.smpl_template.vertices = tgt
                    self.smpl_template.faces = None
                    # self.smpl_template = pcd_to_mesh(tgt)
                else:
                    self.is_mesh = True
                    self.smpl_template = misc['cano_mesh.object']
                    self.smpl_template.vertices = torch.tensor(self.smpl_template.vertices)
                logger.info(f'Use {self.smpl_template.vertices.shape[0]} number of gaussians')
                
                
                #* UPSAMPLING
                min_obj_n_gaussians = 1e+4 # n_subdivision
                if self.is_mesh:
                    while (self.smpl_template.vertices.shape[0] < min_obj_n_gaussians):
                        sub_vertices, sub_faces, _ = subdivide(
                            vertices=self.smpl_template.vertices,
                            faces=self.smpl_template.faces,
                        )
                        self.smpl_template.vertices = sub_vertices
                        self.smpl_template.faces = sub_faces
                        logger.info(f'Use {self.smpl_template.vertices.shape[0]} number of gaussians')
                else:
                    from bigs.utils.general import upsample_point_cloud
                    
                    while (self.smpl_template.vertices.shape[0] < min_obj_n_gaussians):
                        self.smpl_template.vertices = upsample_point_cloud(self.smpl_template.vertices, noise_level=0.01)
                    logger.info(f'Use {self.smpl_template.vertices.shape[0]} number of gaussians')

                #* FOR HOLD INIT
                if 'servers' in misc.keys():
                    misc['servers']['object'].object_model.v3d_cano = torch.tensor(self.smpl_template.vertices, dtype=torch.float32).to(device)
                    misc['servers']['object'].verts_c = torch.tensor(self.smpl_template.vertices, dtype=torch.float32).to(device)            
            else:
                raise Exception('Not implemented yet!')
        
        #* SETUP (FOR SCALE INIT)
        if use_deformer:
            self.smpl = MANO(SMPL_PATH, create_transl=False, use_pca=False, flat_hand_mean=False, is_rhand=is_rhand).to(self.device)
            vertices=self.smpl_template.v_template.detach().cpu().numpy()
        else:
            self.smpl = None
            vertices=self.smpl_template.vertices
        
        if self.is_mesh:    
            edges = trimesh.Trimesh(
                vertices=vertices, 
                faces=self.smpl_template.faces, process=False
            ).edges_unique
            self.edges = torch.from_numpy(edges).to(self.device).long()
        else:
            self.edges = None
        
        self.init_values = {}
        self.get_vitruvian_verts()
        self.setup_functions()
        
        self.misc = misc
    
    def create_hand_pose(self, body_pose, requires_grad=False):
        body_pose = axis_angle_to_rotation_6d(body_pose.reshape(-1, 3)).reshape(-1, 15*6)
        self.body_pose = nn.Parameter(body_pose, requires_grad=requires_grad)
        logger.info(f"Created body pose with shape: {body_pose.shape}, requires_grad: {requires_grad}")
        
    def create_global_orient(self, global_orient, requires_grad=False):
        global_orient = axis_angle_to_rotation_6d(global_orient.reshape(-1, 3)).reshape(-1, 6)
        self.global_orient = nn.Parameter(global_orient, requires_grad=requires_grad)
        logger.info(f"Created global_orient with shape: {global_orient.shape}, requires_grad: {requires_grad}")
        
    def create_betas(self, betas, requires_grad=False):
        self.betas = nn.Parameter(betas, requires_grad=requires_grad)
        logger.info(f"Created betas with shape: {betas.shape}, requires_grad: {requires_grad}")
        
    def create_transl(self, transl, requires_grad=False):
        self.transl = nn.Parameter(transl, requires_grad=requires_grad)
        logger.info(f"Created transl with shape: {transl.shape}, requires_grad: {requires_grad}")
        
    def create_scale(self, scale, requires_grad=False):
        self.scale = nn.Parameter(scale, requires_grad=requires_grad)
        logger.info(f"Created scale with shape: {scale.shape}, requires_grad: {requires_grad}")        
        
    def create_eps_offsets(self, eps_offsets, requires_grad=False):
        logger.info(f"NOT CREATED eps_offsets with shape: {eps_offsets.shape}, requires_grad: {requires_grad}")
    
    @property
    def get_xyz(self):
        return self._xyz
    
    def state_dict(self):
        save_dict = {
            'active_sh_degree': self.active_sh_degree,
            'xyz': self._xyz,
            'triplane': self.triplane.state_dict(),
            'appearance_dec': self.appearance_dec.state_dict(),
            'geometry_dec': self.geometry_dec.state_dict(),
            'deformation_dec': self.deformation_dec.state_dict(),
            'scaling_multiplier': self.scaling_multiplier,
            # 'opacity_multiplier': self.opacity_multiplier,
            'max_radii2D': self.max_radii2D,
            'max_weight': self.max_weight,
            'xyz_gradient_accum': self.xyz_gradient_accum,
            'xyz_gradient_accum_abs': self.xyz_gradient_accum_abs,
            'denom': self.denom,
            'optimizer': self.optimizer.state_dict(),
            'spatial_lr_scale': self.spatial_lr_scale,
        }
        return save_dict
    
    def load_params(self, path, side, device, cfg=None):
        if side == 'object':
            return False
        
        oppo_side = 'left' if side == 'right' else 'right'
        oppo_prev_ckpt_path = f"{path.replace(side, oppo_side)}/human_final_{self.refine_step}.pth"
        
        if not op.isfile(oppo_prev_ckpt_path):
            oppo_prev_ckpt_path = f"{path.replace(side, oppo_side)}/human_final_{self.refine_step - 1}.pth"
            
        if op.isfile(oppo_prev_ckpt_path):
            state_dict = torch.load(oppo_prev_ckpt_path, map_location='cpu')
            
            # xyz
            state_dict['xyz'] = state_dict['xyz'].detach()
            state_dict['xyz'] = torch.nn.Parameter(state_dict['xyz'].to(device))
            
            self.active_sh_degree = state_dict['active_sh_degree']
            self._xyz = state_dict['xyz']
            self.max_radii2D = state_dict['max_radii2D'].to(device)
            self.max_weight = state_dict['max_weight'].to(device)
            xyz_gradient_accum = state_dict['xyz_gradient_accum'].to(device)
            xyz_gradient_accum_abs = state_dict['xyz_gradient_accum_abs'].to(device)
            denom = state_dict['denom'].to(device)
            self.spatial_lr_scale = state_dict['spatial_lr_scale']
            
            self.triplane.load_state_dict(custom_to(state_dict['triplane'], device))
            m_keys, u_keys = self.appearance_dec.load_state_dict(custom_to(state_dict['appearance_dec'], device), strict=False)
            self.geometry_dec.load_state_dict(custom_to(state_dict['geometry_dec'], device))
            # self.deformation_dec.load_state_dict(custom_to(state_dict['deformation_dec'], device))
            self.scaling_multiplier = state_dict['scaling_multiplier'].to(device)
            # self.opacity_multiplier = state_dict['opacity_multiplier'].to(device)
            
            if len(m_keys) > 0:
                logger.error(f'[Missing keys] {m_keys}')
            if len(u_keys) > 0:
                logger.error(f'[Unexpected keys] {u_keys}')

            self.xyz_gradient_accum = xyz_gradient_accum
            self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
            self.denom = denom      
            
            self.setup_optimizer(cfg.obj.lr)
            if cfg.obj.lr.use_optim_ckpt and op.isfile(f'{path}/human_final.pth'):
                logger.warning('Use optim ckpts!!')
                try:
                    self.optimizer.load_state_dict(
                        torch.load(f'{path}/human_final.pth')['optimizer']
                    )
                    params = self.optimizer.state_dict()['param_groups']
                    for param in params:
                        logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")                
                except ValueError as e:
                    logger.warning(f"Optimizer load failed: {e}")
                    logger.warning("Continue without a pretrained optimizer")            
            
            del state_dict
            torch.cuda.empty_cache()
            
            return True
        else:
            logger.warning("Previous checkpoints are not used!")
            return False
    
    def load_state_dict(self, state_dict, cfg=None, load_extra_param=True):
        self.active_sh_degree = state_dict['active_sh_degree']
        self._xyz = state_dict['xyz']
        self.max_radii2D = state_dict['max_radii2D']
        xyz_gradient_accum = state_dict['xyz_gradient_accum']
        denom = state_dict['denom']
        opt_dict = state_dict['optimizer']
        self.spatial_lr_scale = state_dict['spatial_lr_scale']
        try:
            self.max_weight = state_dict['max_weight']
            xyz_gradient_accum_abs = state_dict['xyz_gradient_accum_abs']
            # self.opacity_multiplier = state_dict['opacity_multiplier']
        except:
            self.max_weight = torch.zeros_like(self.max_radii2D).to(self.max_radii2D)
            xyz_gradient_accum_abs = torch.zeros_like(xyz_gradient_accum).to(xyz_gradient_accum)
            # self.opacity_multiplier = torch.zeros_like(state_dict['scaling_multiplier']).to(self.max_radii2D)
        
        self.setup_optimizer(cfg)
        self.triplane.load_state_dict(state_dict['triplane'])
        m_keys_appe, u_keys_appe = self.appearance_dec.load_state_dict(state_dict['appearance_dec'], strict=False)
        m_keys_geo, u_keys_geo = self.geometry_dec.load_state_dict(state_dict['geometry_dec'], strict=False)
        self.deformation_dec.load_state_dict(state_dict['deformation_dec'])
        self.scaling_multiplier = state_dict['scaling_multiplier']
        
        m_keys = m_keys_appe + m_keys_geo
        u_keys = u_keys_appe + u_keys_geo
        if len(m_keys) > 0:
            logger.error(f'[Missing keys] {m_keys}')
        if len(u_keys) > 0:
            logger.error(f'[Unexpected keys] {u_keys}')
        
        if load_extra_param:
            extra_params = ['body_pose', 'global_orient', 'betas', 'transl', 'scale']
            for p_key in extra_params:
                if p_key in state_dict.keys():
                    self.__setattr__(p_key, state_dict[p_key])
        
        if cfg is None:
            from bigs.cfg.config import cfg as default_cfg
            cfg = default_cfg.obj.lr
            
        self.setup_optimizer(cfg)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.xyz_gradient_accum_abs = xyz_gradient_accum_abs
        self.denom = denom
        if cfg.use_optim_ckpt:
            logger.warning('Use optim ckpts!!')
            try:
                self.optimizer.load_state_dict(opt_dict)
                params = self.optimizer.state_dict()['param_groups']
                for param in params:
                    logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")                
            except ValueError as e:
                logger.warning(f"Optimizer load failed: {e}")
                logger.warning("Continue without a pretrained optimizer")
            
    def __repr__(self):
        repr_str = "HUGS TRIMLP: \n"
        repr_str += "xyz: {} \n".format(self._xyz.shape)
        repr_str += "max_radii2D: {} \n".format(self.max_radii2D.shape)
        repr_str += "max_weight: {} \n".format(self.max_weight.shape)
        repr_str += "xyz_gradient_accum: {} \n".format(self.xyz_gradient_accum.shape)
        repr_str += "denom: {} \n".format(self.denom.shape)
        return repr_str

    def canon_forward(self):
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        if 'rotations' in geometry_out:
            gs_rot6d = geometry_out['rotations']
            xyz_offsets = geometry_out['xyz']
        else:
            gs_rot6d = self.rot6d
            xyz_offsets = self.xyz_offsets
        gs_scales = geometry_out['scales'] * self.scaling_multiplier
        
        if 'opacity' in appearance_out:
            gs_opacity = appearance_out['opacity']
        else:
            gs_opacity = torch.ones((xyz_offsets.shape[0], 1), dtype=torch.float, device="cuda")
        # gs_opacity = geometry_out['scales'] * self.opacity_multiplier
            
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)        
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
            
        return {
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'rot6d_canon': gs_rot6d,
            'shs': gs_shs,
            'opacity': gs_opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
        }

    def forward_test(
        self,
        canon_forward_out,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
    ):
        xyz_offsets = canon_forward_out['xyz_offsets']
        gs_rot6d = canon_forward_out['rot6d_canon']
        gs_scales = canon_forward_out['scales']
        
        gs_xyz = self.get_xyz + xyz_offsets
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        gs_opacity = canon_forward_out['opacity']
        gs_shs = canon_forward_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            lbs_weights = canon_forward_out['lbs_weights']
            posedirs = canon_forward_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
        
        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(15*3)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_output = self.smpl(
            hand_pose=body_pose.unsqueeze(0),
            betas=betas.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            transl=None,
            disable_posedirs=False,
            return_full_pose=True
        )
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, _, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
            T_t2pose = smpl_output.T[0]
            T_vitruvian2t = self.inv_T_t2vitruvian.clone()
            T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
            T_vitruvian2pose = T_t2pose @ T_vitruvian2t

            _, lbs_T = smpl_lbsmap_top_k(
                lbs_weights=self.smpl.lbs_weights,
                verts_transform=T_vitruvian2pose.unsqueeze(0),
                points=gs_xyz.unsqueeze(0),
                template_points=self.vitruvian_verts.unsqueeze(0),
                K=6,
            )
            lbs_T = lbs_T.squeeze(0)
        
            homogen_coord = torch.ones_like(gs_xyz[..., :1])
            gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
            deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
        
        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
        
        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales
            
            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
        
        self.normals = torch.zeros_like(gs_xyz)
        self.normals[:, 2] = 1.0
        
        canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
        
        deformed_gs_shs = gs_shs.clone()
        
        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': xyz_offsets,
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'normals': deformed_normals,
            'normals_canon': canon_normals,
            'active_sh_degree': self.active_sh_degree,
            'rot6d_canon': gs_rot6d,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'gt_lbs_weights': gt_lbs_weights,
        }
         
    def forward(
        self,
        global_orient=None, 
        body_pose=None, 
        betas=None, 
        transl=None, 
        smpl_scale=None,
        dataset_idx=-1,
        is_train=False,
        ext_tfs=None,
        misc=None,
        device=None,
        extract=False,
        render_canon=False,
        is_flip=True,
        t_mesh=None,
    ):
        human_gs_out = {}
        
        tri_feats = self.triplane(self.get_xyz)
        appearance_out = self.appearance_dec(tri_feats)
        geometry_out = self.geometry_dec(tri_feats)
        
        if 'rotations' in geometry_out and self.mode == 'object':
            gs_rot6d = geometry_out['rotations']
            xyz_offsets = geometry_out['xyz']
        else:
            gs_rot6d = self.rot6d
            xyz_offsets = self.xyz_offsets
        
        
        
        if self.mode == 'object':
            gs_scales = geometry_out['scales'] * self.scaling_multiplier
        else:
            gs_scales = self.scales
        
        
        gs_xyz = self.get_xyz + xyz_offsets
        if self.mode == 'left':
            gs_xyz[..., 0] *= -1
        
        gs_rotmat = rotation_6d_to_matrix(gs_rot6d)
        gs_rotq = matrix_to_quaternion(gs_rotmat)

        if 'opacity' in appearance_out and self.mode == 'object':
            gs_opacity = appearance_out['opacity']
        else:
            gs_opacity = torch.ones((xyz_offsets.shape[0], 1), device=device)
        # gs_opacity = geometry_out['scales'] * self.opacity_multiplier
            
        gs_shs = appearance_out['shs'].reshape(-1, 16, 3)
        
        if self.isotropic:
            gs_scales = torch.ones_like(gs_scales) * torch.mean(gs_scales, dim=-1, keepdim=True)
            
        gs_scales_canon = gs_scales.clone()
        
        if self.use_deformer:
            deformation_out = self.deformation_dec(tri_feats)
            lbs_weights = deformation_out['lbs_weights']
            lbs_weights = F.softmax(lbs_weights/0.1, dim=-1)
            posedirs = deformation_out['posedirs']
            if abs(lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                pass
            else:
                logger.warning(f"LBS weights should sum to 1, but it is: {lbs_weights.sum(-1).mean().item()}")
        else:
            lbs_weights = None
            posedirs = None
        
        if self.use_deformer:
            mode = self.mode # 'right' if self.smpl_template.is_rhand else 'left'
        else:
            mode = 'object'
        
        try:
            if hasattr(self, 'global_orient') and global_orient is None:
                global_orient_offset = rotation_6d_to_axis_angle(
                    self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)
                global_orient = torch.tensor(misc['cached_data'][mode]['global_orient'][dataset_idx], device=device) + global_orient_offset
                human_gs_out['global_orient'] = global_orient.clone()
            elif global_orient is None:
                global_orient = torch.zeros(3).to(device)
        except:
            global_orient = torch.zeros(3).to(device)
        
        try:
            if hasattr(self, 'body_pose') and body_pose is None:
                body_pose_offset = rotation_6d_to_axis_angle(
                    self.body_pose[dataset_idx].reshape(-1, 6)).reshape(15*3)
                body_pose = torch.tensor(misc['cached_data'][mode]['hand_pose'][dataset_idx], device=device) + body_pose_offset
                human_gs_out['body_pose'] = body_pose.clone()
            elif body_pose is None:
                body_pose = torch.zeros(45).to(device)
        except:
            body_pose = torch.zeros(45).to(device)
            
        if hasattr(self, 'betas') and betas is None:
            betas = self.betas
            
        try:
            if hasattr(self, 'transl') and transl is None:
                transl = self.transl[dataset_idx]
            elif transl is None:
                transl = torch.zeros(3).to(device)
        except:
            transl = torch.zeros(3).to(device)
            
        try:
            if hasattr(self, 'scale') and smpl_scale is None:
                smpl_scale = self.scale[dataset_idx]
            elif smpl_scale is None:
                smpl_scale = torch.ones(1).to(device)
        except:
            smpl_scale = torch.ones(1).to(device)
        
        # vitruvian -> t-pose -> posed
        # remove and reapply the blendshape
        smpl_joints = None
        v_posed = None
        if self.smpl is not None:
            # load scale & transl information
            scene_scale = misc['scale'].view(-1, 1, 1)
            
            if 'gt' in misc.keys():
                betas = torch.from_numpy(misc['gt']['params'][f'shape_{mode[0]}'][dataset_idx]).to(device)
                body_pose = torch.from_numpy(misc['gt']['params'][f'pose_{mode[0]}'][dataset_idx]).to(device)
                global_orient = torch.from_numpy(misc['gt']['params'][f'rot_{mode[0]}'][dataset_idx]).to(device)
                t_param = torch.from_numpy(misc['gt']['params'][f'trans_{mode[0]}'][dataset_idx]).to(device).view(-1, 1, 3)
                scene_scale= torch.ones_like(scene_scale).to(device)
                smpl_scale = torch.ones_like(smpl_scale).to(device)
                transl = torch.zeros_like(transl).to(device)
            else:
                t_param = torch.tensor(misc['cached_data'][mode]['transl'][dataset_idx]).to(device).view(-1, 1, 3)
            
            # ! Translation must be None
            smpl_output = self.smpl(
                betas=betas.unsqueeze(0),
                hand_pose=body_pose.unsqueeze(0),
                global_orient=global_orient.unsqueeze(0),
                transl=None, # misc[f'root.{mode}'][dataset_idx].unsqueeze(0).cuda(), #None,
                disable_posedirs=False,
                return_full_pose=True
            )
            
            # apply to verts
            verts = smpl_output.vertices
            verts = verts * scene_scale + t_param * scene_scale

            # np.allclose(
            #     verts.cpu().numpy(),
            #     misc['gt']['world_coord'][f'verts.{mode}'][dataset_idx]
            # )

            # apply to joints
            smpl_joints = smpl_output.joints
            smpl_joints = smpl_joints * scene_scale + t_param * scene_scale
            
            # apply to A matrix
            tf_mats = smpl_output.A
            tf_mats[:, :, :3, :] = tf_mats[:, :, :3, :] * scene_scale.view(-1, 1, 1, 1)
            tf_mats[:, :, :3, 3] = tf_mats[:, :, :3, 3] + t_param * scene_scale
            
            # save results
            smpl_output.vertices = verts
            smpl_output.joints = smpl_joints
            smpl_output.A = tf_mats
        
        gt_lbs_weights = None
        if self.use_deformer:
            A_t2pose = smpl_output.A[0]
            A_vitruvian2pose = A_t2pose @ self.inv_A_t2vitruvian
            deformed_xyz, _, lbs_T, v_posed, _ = lbs_extra(
                A_vitruvian2pose[None], gs_xyz[None], posedirs, lbs_weights, 
                smpl_output.full_pose, disable_posedirs=self.disable_posedirs, pose2rot=True
            )
            
            ext_verts = deformed_xyz.clone() if extract else None
            
            # 
            if 'gt' not in misc.keys():
                deformed_xyz = map_deform2eval_batch_torch(deformed_xyz, misc['inverse_scale'], torch.tensor(misc['normalize_shift']).to(device))
                smpl_joints = map_deform2eval_batch_torch(smpl_joints.float(), misc['inverse_scale'], torch.tensor(misc['normalize_shift']).to(device))
            
            deformed_xyz = deformed_xyz.squeeze(0)
            lbs_T = lbs_T.squeeze(0)

            with torch.no_grad():
                # gt lbs is needed for lbs regularization loss
                # predicted lbs should be close to gt lbs
                _, gt_lbs_weights = smpl_lbsweight_top_k(
                    lbs_weights=self.smpl.lbs_weights,
                    points=gs_xyz.unsqueeze(0),
                    template_points=self.vitruvian_verts.unsqueeze(0),
                )
                gt_lbs_weights = gt_lbs_weights.squeeze(0)
                if abs(gt_lbs_weights.sum(-1).mean().item() - 1) < 1e-7:
                    pass
                else:
                    logger.warning(f"GT LBS weights should sum to 1, but it is: {gt_lbs_weights.sum(-1).mean().item()}")
        else:
            if 'servers' in misc.keys():
                ext_verts, deformed_xyz = self.forward_obj_server(
                    gs_xyz, global_orient, dataset_idx, t_mesh,
                    misc, device, render_canon, mode, extract,
                )
            else:
                ext_verts, deformed_xyz = self.forward_obj(
                    gs_xyz, global_orient, dataset_idx,
                    misc, device, mode, extract,
                )
        
        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)
            if smpl_joints is not None:
                smpl_joints = smpl_joints * smpl_scale.unsqueeze(0)
        
        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)
            if smpl_joints is not None:
                smpl_joints = smpl_joints + transl.unsqueeze(0)
        
        if self.use_deformer:
            deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
            deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
            
            if ext_tfs is not None:
                tr, rotmat, sc = ext_tfs
                deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
                gs_scales = sc * gs_scales
                
                rotq = matrix_to_quaternion(rotmat)
                deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
                deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
            
            self.normals = torch.zeros_like(gs_xyz)
            self.normals[:, 2] = 1.0
            
            canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
            deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
            
            deformed_gs_shs = gs_shs.clone()
        else:
            deformed_gs_rotmat = gs_rotmat
            deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)
            
            if ext_tfs is not None:
                tr, rotmat, sc = ext_tfs
                deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
                gs_scales = sc * gs_scales
                
                rotq = matrix_to_quaternion(rotmat)
                deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
                deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)
            
            self.normals = torch.zeros_like(gs_xyz).float()
            self.normals[:, 2] = 1.0
            
            canon_normals = (gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
            deformed_normals = (deformed_gs_rotmat @ self.normals.unsqueeze(-1)).squeeze(-1)
            
            deformed_gs_shs = gs_shs.clone()
            
        human_gs_out.update(
            custom_to({
                'xyz': deformed_xyz,
                'xyz_canon': gs_xyz,
                'xyz_offsets': xyz_offsets,
                'scales': gs_scales,
                'scales_canon': gs_scales_canon,
                'rotq': deformed_gs_rotq,
                'rotq_canon': gs_rotq,
                'rotmat': deformed_gs_rotmat,
                'rotmat_canon': gs_rotmat,
                'shs': deformed_gs_shs,
                'opacity': gs_opacity,
                'normals': deformed_normals,
                'normals_canon': canon_normals,
                'active_sh_degree': self.active_sh_degree,
                'rot6d_canon': gs_rot6d,
                'lbs_weights': lbs_weights,
                'posedirs': posedirs,
                'gt_lbs_weights': gt_lbs_weights,
                'joint3d': smpl_joints,
                "v_posed": v_posed,
                "verts": ext_verts,
            }, device, cast_type=True)
        )            
        return human_gs_out
    
    def forward_obj_server(
            self, gs_xyz, global_orient, dataset_idx, t_mesh,
            misc, device, render_canon, mode, extract,
        ):
        
        misc['servers']['object'].object_model.v3d_cano = gs_xyz.clone().float().to(device)
        misc['servers']['object'].verts_c = gs_xyz.clone().float().to(device)
        
        if t_mesh is not None:
            misc['servers']['object'].object_model.v3d_cano = torch.from_numpy(t_mesh.vertices).float().to(device)
            misc['servers']['object'].verts_c = torch.from_numpy(t_mesh.vertices).float().to(device)
        
        if 'gt' in misc.keys():
            # new_gs_xyz = np.load('obj/espressomachine_obj.npy', allow_pickle=True).item()['v']
            
            if not hasattr(self, 'gt_verts'):
                obj_name = misc['full_seq_name'].split('_')[2]
                obj_trans_param = OBJ_SCALE[obj_name]
                
                gt_gs_xyz = np.load(f'preprocess/paper_output/obj_tplt/{obj_name}.npy', allow_pickle=True).item()['v']
                new_gs_xyz = gs_xyz.clone() * obj_trans_param['scale']
                new_T = compute_bounding_box_centers_torch(new_gs_xyz)
                new_gs_xyz = new_gs_xyz - new_T
                
                transform = np.eye(4)
                angle = obj_trans_param['angle']
                transform[:-1, :-1] = axis_angle_to_matrix(torch.tensor([a/180*np.pi for a in angle]))
                new_gs_xyz = apply_transformation(new_gs_xyz.cpu().numpy(), transform)
                new_gs_xyz = torch.from_numpy(new_gs_xyz).to(device).float() + compute_bounding_box_centers_torch(gt_gs_xyz)
                self.gt_verts = new_gs_xyz
            else:
                new_gs_xyz = self.gt_verts
            
            smpl_scale = torch.ones_like(smpl_scale).to(device)
            transl = torch.zeros_like(transl).to(device)
            
            gt_rot = torch.from_numpy(misc['gt']['params']['obj_rot'][dataset_idx]).to(device).unsqueeze(0)
            quat_global = axis_angle_to_quaternion(gt_rot.view(-1, 3))
            gt_trans = torch.from_numpy(misc['gt']['params']['obj_trans'][dataset_idx]).to(device)[None] / 1000
            
            val_rot = quaternion_apply(quat_global[:, None, :], new_gs_xyz)
            v3d = val_rot + gt_trans[:, None, :]
        else:
            if render_canon:
                t_param = torch.zeros(3).to(device).unsqueeze(0)
                s_param = torch.ones(1).to(device)
                norm_shift = np.zeros(3)
                # t_param = -torch.tensor(misc['cached_data'][mode]['transl'][dataset_idx]).to(device).unsqueeze(0)
            else:
                t_param = torch.tensor(misc['cached_data'][mode]['transl'][dataset_idx]).to(device).unsqueeze(0)
                s_param = misc['scale']
                norm_shift = misc['normalize_shift']
            
            server_param = xdict({
                "global_orient": global_orient.unsqueeze(0),
                "transl": t_param,
                "scene_scale": s_param,
            })                
            
            v3d = misc['servers']['object'].forward_param(server_param)['verts']
        
        ext_verts = v3d.clone() if extract else None
        if 'gt' not in misc.keys():
            deformed_xyz = map_deform2eval_batch_torch(v3d, float(1. / s_param[0]), torch.tensor(norm_shift).to(device))[0]
        else:
            deformed_xyz = v3d[0]
            
        return ext_verts, deformed_xyz
    
    def forward_obj(
            self, gs_xyz, global_orient, dataset_idx,
            misc, device, mode, extract,
        ):
        rot = global_orient.unsqueeze(0)
        trans = torch.tensor(misc['cached_data'][mode]['transl'][dataset_idx]).to(device).unsqueeze(0)
        obj_scale_v = misc['scale']
        norm_shift = misc['normalize_shift']
        denorm_mat = torch.inverse(misc['norm_mat']).to(device)
        
        batch_size = rot.shape[0]
        scene_scale = torch.ones(batch_size).to(device)
        rot_mat = axis_angle_to_matrix(rot).view(batch_size, 3, 3)

        # cano to camera
        batch_size = rot_mat.shape[0]
        tf_mats = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        tf_mats[:, :3, :3] = rot_mat
        tf_mats[:, :3, 3] = trans.view(batch_size, 3)

        scale_mat = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        scale_mat *= scene_scale[:, None, None]
        scale_mat[:, 3, 3] = 1
        v3d_cano_pad = torch.cat(
            [gs_xyz, torch.ones(gs_xyz.shape[0], 1, device=device)],
            dim=1,
        )
        v3d_cano_pad = v3d_cano_pad[None, :, :].repeat(batch_size, 1, 1)

        obj_scale = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
        obj_scale *= obj_scale_v
        obj_scale[:, 3, 3] = 1

        # deformalize
        tf_mats = torch.matmul(scale_mat, tf_mats)
        tf_mats = torch.matmul(tf_mats, obj_scale)
        tf_mats = torch.matmul(
            tf_mats, denorm_mat[None, :, :].repeat(batch_size, 1, 1).float()
        )

        vertices = torch.bmm(tf_mats.double(), v3d_cano_pad.permute(0, 2, 1)).permute(0, 2, 1)
        v3d = vertices[:, :, :3] / vertices[:, :, 3:4]
        ext_verts = v3d.clone() if extract else None
        deformed_xyz = map_deform2eval_batch_torch(v3d.float(), float(1. / scene_scale[0]), torch.tensor(norm_shift).to(device))[0]
        
        return ext_verts, deformed_xyz
        

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            logger.info(f"Going from SH degree {self.active_sh_degree} to {self.active_sh_degree + 1}")
            self.active_sh_degree += 1

    @torch.no_grad()
    def get_vitruvian_verts(self):
        if self.use_deformer:
            vitruvian_pose = torch.zeros(45, dtype=self.smpl.dtype, device=self.device)
            gt_rot_r = torch.zeros(3, dtype=self.smpl.dtype, device=self.device)

            smpl_output = self.smpl(hand_pose=vitruvian_pose[None], betas=self.betas[None], global_orient=gt_rot_r[None], transl=None, disable_posedirs=False)
            vitruvian_verts = smpl_output.vertices[0]
            self.A_t2vitruvian = smpl_output.A[0].detach()
            self.T_t2vitruvian = smpl_output.T[0].detach()
            self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
            self.inv_A_t2vitruvian = torch.inverse(self.A_t2vitruvian)
            self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
            self.canonical_offsets = self.canonical_offsets[0].detach()
            self.vitruvian_verts = vitruvian_verts.detach()
        else:
            vitruvian_verts = torch.tensor(self.smpl_template.vertices).cuda()
            self.vitruvian_verts = vitruvian_verts.detach()
        
        return vitruvian_verts.detach()
    
    @torch.no_grad()
    def get_vitruvian_verts_template(self):
        if self.use_deformer:
            vitruvian_pose = torch.zeros(45, dtype=self.smpl_template.dtype, device=self.device)
            gt_rot_r = torch.zeros(3, dtype=self.smpl_template.dtype, device=self.device)

            smpl_output = self.smpl_template(hand_pose=vitruvian_pose[None], betas=self.betas[None], global_orient=gt_rot_r[None], transl=None, disable_posedirs=False)
            vitruvian_verts = smpl_output.vertices[0]
        else:
            vitruvian_verts = self.vitruvian_verts
        
        return vitruvian_verts.detach()
    
    def train(self):
        pass
    
    def eval(self):
        pass
    
    def initialize(self):
        t_pose_verts = self.get_vitruvian_verts_template()
        
        self.scaling_multiplier = torch.ones((t_pose_verts.shape[0], 1), device=self.device)
        # self.opacity_multiplier = torch.ones((t_pose_verts.shape[0], 1), device=self.device)
        
        xyz_offsets = torch.zeros_like(t_pose_verts)
        colors = torch.ones_like(t_pose_verts) * 0.5
        
        shs = torch.zeros((colors.shape[0], 3, 16)).float().to(self.device)
        shs[:, :3, 0 ] = colors
        shs[:, 3:, 1:] = 0.0
        shs = shs.transpose(1, 2).contiguous()
        
        scales = torch.zeros_like(t_pose_verts)
        if self.edges is not None:
            # scale_factor = 1.75e-4
            # scales = torch.ones_like(t_pose_verts) * scale_factor            
            for v in range(t_pose_verts.shape[0]):
                selected_edges = torch.any(self.edges == v, dim=-1)
                selected_edges_len = torch.norm(
                    t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]], 
                    dim=-1
                )
                selected_edges_len *= self.init_scale_multiplier
                scales[v, 0] = torch.log(torch.max(selected_edges_len))
                scales[v, 1] = torch.log(torch.max(selected_edges_len))
                
                if not self.use_surface:
                    scales[v, 2] = torch.log(torch.max(selected_edges_len))
            
            if self.use_surface or self.init_2d:
                scales = scales[..., :2]
                
            scales = torch.exp(scales)
        else:
            s_factor = initialize_gaussian_scales(t_pose_verts)
            scales = scales * 0 + s_factor

        self.scales = scales
        
        if self.use_surface or self.init_2d:
            scale_z = torch.ones_like(scales[:, -1:]) * SCALE_Z
            scales = torch.cat([scales, scale_z], dim=-1)
        
        if self.is_mesh:
            mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)
            vert_normals = torch.tensor(mesh.vertex_normals).float().to(self.device)
        else:
            from pytorch3d.ops.points_normals import estimate_pointcloud_normals
            vert_normals = estimate_pointcloud_normals(t_pose_verts.unsqueeze(0))[0]
        
        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0
        
        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)
        rot6d = matrix_to_rotation_6d(norm_rotmat)
        self.rot6d = rot6d
        self.xyz_offsets = xyz_offsets
        if not self.geometry_dec.optim_rotations:
            rot6d = None
            xyz_offsets = None
                
        self.normals = gs_normals
        deformed_normals = (norm_rotmat @ gs_normals.unsqueeze(-1)).squeeze(-1)
        
        if self.appearance_dec.optim_opacity:
            opacity = 0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device=self.device)
        else:
            opacity = None
        
        if hasattr(self.smpl_template, 'posedirs'):
            posedirs = self.smpl_template.posedirs.detach().clone()
            lbs_weights = self.smpl_template.lbs_weights.detach().clone()
        else:
            posedirs = None
            lbs_weights = None

        self.n_gs = t_pose_verts.shape[0]
        self._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))
        
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        self.max_weight = torch.zeros((self.get_xyz.shape[0]), device=self.device)
        
        if self.smpl is not None:
            faces = self.smpl.faces_tensor
        else:
            faces = torch.tensor(self.smpl_template.faces) if self.smpl_template.faces is not None else None
        
        return {
            'xyz_offsets': xyz_offsets,
            'scales': scales,
            'rot6d_canon': rot6d,
            'shs': shs,
            'opacity': opacity,
            'lbs_weights': lbs_weights,
            'posedirs': posedirs,
            'deformed_normals': deformed_normals,
            'faces': faces,
            'edges': self.edges,
        }

    def setup_optimizer(self, cfg):
        self.percent_dense = cfg.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.spatial_lr_scale = cfg.smpl_spatial
        
        params = [
            {'params': [self._xyz], 'lr': cfg.position * cfg.smpl_spatial, "name": "xyz"},
            {'params': self.triplane.parameters(), 'lr': cfg.vembed, 'name': 'v_embed'},
            {'params': self.geometry_dec.parameters(), 'lr': cfg.geometry, 'name': 'geometry_dec'},
            {'params': self.appearance_dec.parameters(), 'lr': cfg.appearance, 'name': 'appearance_dec'},
            {'params': self.deformation_dec.parameters(), 'lr': cfg.deformation, 'name': 'deform_dec'},
        ]
        
        if hasattr(self, 'global_orient') and self.global_orient.requires_grad:
            params.append({'params': self.global_orient, 'lr': cfg.smpl_rot, 'name': 'global_orient'})
        
        if hasattr(self, 'body_pose') and self.body_pose.requires_grad:
            params.append({'params': self.body_pose, 'lr': cfg.smpl_pose, 'name': 'body_pose'})
            
        if hasattr(self, 'betas') and self.betas.requires_grad:
            params.append({'params': self.betas, 'lr': cfg.smpl_betas, 'name': 'betas'})
            
        if hasattr(self, 'transl') and self.transl.requires_grad:
            params.append({'params': self.transl, 'lr': cfg.smpl_trans, 'name': 'transl'})
            
        if hasattr(self, 'scale') and self.scale.requires_grad:
            params.append({'params': self.scale, 'lr': cfg.smpl_scale, 'name': 'scale'})            
        
        self.non_densify_params_keys = [
            'global_orient', 'body_pose', 'betas', 'transl', 'scale',
            'v_embed', 'geometry_dec', 'appearance_dec', 'deform_dec',
        ]
        
        for param in params:
            logger.info(f"Parameter: {param['name']}, lr: {param['lr']}")

        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=cfg.lr_init * cfg.smpl_spatial,
            lr_final=cfg.lr_final * cfg.smpl_spatial,
            lr_delay_mult=cfg.lr_delay_mult,
            max_steps=cfg.lr_max_steps,
        )

    def update_learning_rate(self, iteration, name='xyz'):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] in name:
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
        return lr


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = self.scaling_multiplier[valid_points_mask]
        # self.opacity_multiplier = self.opacity_multiplier[valid_points_mask]
        
        self.scales_tmp = self.scales_tmp[valid_points_mask]
        self.opacity_tmp = self.opacity_tmp[valid_points_mask]
        self.rotmat_tmp = self.rotmat_tmp[valid_points_mask]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.xyz_gradient_accum_abs = self.xyz_gradient_accum_abs[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        # self.max_weight = self.max_weight[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] in self.non_densify_params_keys:
                continue
            
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp):
        d = {
            "xyz": new_xyz,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self.scaling_multiplier = torch.cat((self.scaling_multiplier, new_scaling_multiplier), dim=0)
        ## self.opacity_multiplier = torch.cat((self.opacity_multiplier, new_opacity_multiplier), dim=0)
        self.opacity_tmp = torch.cat([self.opacity_tmp, new_opacity_tmp], dim=0)
        self.scales_tmp = torch.cat([self.scales_tmp, new_scales_tmp], dim=0)
        self.rotmat_tmp = torch.cat([self.rotmat_tmp, new_rotmat_tmp], dim=0)
        
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        # self.xyz_gradient_accum_abs = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        
        # max_weight = torch.zeros((new_xyz.shape[0]), device="cuda")
        # self.max_weight = torch.cat((self.max_weight,max_weight),dim=0)        

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        scales = self.scales_tmp
        rotation = self.rotmat_tmp
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values > self.percent_dense*scene_extent)
        # filter elongated gaussians
        med = scales.median(dim=1, keepdim=True).values 
        stdmed_mask = (((scales - med) / med).squeeze(-1) >= 1.0).any(dim=-1)
        selected_pts_mask = torch.logical_and(selected_pts_mask, stdmed_mask)
        
        stds = scales[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=torch.relu(stds))
        rots = rotation[selected_pts_mask].repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask].repeat(N,1) / (0.8*N)
        # new_opacity_multiplier = self.opacity_multiplier[selected_pts_mask].repeat(N,1) / (0.8*N)
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask].repeat(N,1)
        new_scales_tmp = self.scales_tmp[selected_pts_mask].repeat(N,1)
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask].repeat(N,1,1)
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        
        num_prune = sum(prune_filter)
        if (self.get_xyz.shape[0] - num_prune) < 1e4:
            tgt_num = (self.get_xyz.shape[0] - 1e4) - 1
            if tgt_num <= 0:
                logger.warning("There are no saturated points!")
                return
            new_mask_idx = torch.where(prune_filter == True)[0][torch.randperm(num_prune)[:int(num_prune - tgt_num)]]
            prune_filter[new_mask_idx] = False        
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        scales = self.scales_tmp
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(scales, dim=1).values <= self.percent_dense*scene_extent)        
        
        new_xyz = self._xyz[selected_pts_mask]
        new_scaling_multiplier = self.scaling_multiplier[selected_pts_mask]
        # # new_opacity_multiplier = self.opacity_multiplier[selected_pts_mask]
        new_opacity_tmp = self.opacity_tmp[selected_pts_mask]
        new_scales_tmp = self.scales_tmp[selected_pts_mask]
        new_rotmat_tmp = self.rotmat_tmp[selected_pts_mask]
        
        self.densification_postfix(new_xyz, new_scaling_multiplier, new_opacity_tmp, new_scales_tmp, new_rotmat_tmp)

    def reset_opacity(self):
        assert False, "Not implemented yet!"
        opacities_new = inverse_sigmoid(torch.min(self.opacity_multiplier, torch.ones_like(self.opacity_multiplier)*0.01))
        _ = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity_multiplier = opacities_new

    def reduce_opacity(self):
        assert False, "Not implemented yet!"
        opacities_new = inverse_sigmoid(torch.min(self.opacity_multiplier, torch.ones_like(self.opacity_multiplier)*0.8))
        _ = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self.opacity_multiplier = opacities_new
        
    def initial_prune(self, human_gs_out):
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']        
        
        pts_mask_1 = torch.max(self.scales_tmp, dim=1).values > torch.mean(self.scales_tmp)
        if len(self.scales_tmp) < 500_0000:
            pts_mask_2 = torch.max(self.scales_tmp, dim=1).values > torch.quantile(
                self.scales_tmp,
                0.999)
        else:
            pts_mask_2 = torch.max(self.scales_tmp, dim=1).values > torch.mean(
                self.scales_tmp) * 4

        selected_pts_mask = torch.logical_and(pts_mask_1, pts_mask_2)
        
        num_prune = sum(selected_pts_mask)
        logger.info(f"Initial pruning based on radius, GS num: {num_prune}")
        if (self.get_xyz.shape[0] - num_prune) < 1e4:
            tgt_num = (self.get_xyz.shape[0] - 1e4) - 1
            if tgt_num <= 0:
                logger.warning("There are no saturated points!")
                return
            new_mask_idx = torch.where(selected_pts_mask == True)[0][torch.randperm(num_prune)[:int(num_prune - tgt_num)]]
            selected_pts_mask[new_mask_idx] = False
            
        self.prune_points(selected_pts_mask)
        
    def prune_opacity(self, min_opacity):
        prune_mask = (self.opacity_tmp <= min_opacity).squeeze()
        num_prune = sum(prune_mask)
        if (self.get_xyz.shape[0] - num_prune) < 1e4:
            tgt_num = (self.get_xyz.shape[0] - 1e4) - 1
            if tgt_num <= 0:
                logger.warning("There are no saturated points!")
                return
            new_mask_idx = torch.where(prune_mask == True)[0][torch.randperm(num_prune)[:int(num_prune - tgt_num)]]
            prune_mask[new_mask_idx] = False
        if num_prune == 0:
            return
        
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    # def densify_and_prune(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
    def densify_and_prune(self, human_gs_out, max_grad, max_grad_abs, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        grads_abs = self.xyz_gradient_accum_abs / self.denom
        grads_abs[grads_abs.isnan()] = 0.0
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1        
        
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads_abs, max_grad_abs, extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        num_prune = sum(prune_mask)
        if (self.get_xyz.shape[0] - num_prune) < 1e4:
            tgt_num = (self.get_xyz.shape[0] - 1e4) - 1
            if tgt_num <= 0:
                logger.warning("There are no saturated points!")
                return
            new_mask_idx = torch.where(prune_mask == True)[0][torch.randperm(num_prune)[:int(num_prune - tgt_num)]]
            prune_mask[new_mask_idx] = False
        if num_prune == 0:
            return            
        
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    def densify_and_prune_origin(self, human_gs_out, max_grad, min_opacity, extent, max_screen_size, max_n_gs=None):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        self.opacity_tmp = human_gs_out['opacity']
        self.scales_tmp = human_gs_out['scales_canon']
        self.rotmat_tmp = human_gs_out['rotmat_canon']
        
        max_n_gs = max_n_gs if max_n_gs else self.get_xyz.shape[0] + 1
        
        if self.get_xyz.shape[0] <= max_n_gs:
            self.densify_and_clone(grads, max_grad, extent)
            self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.opacity_tmp < min_opacity).squeeze()
        # scale_prune_mask = self.scales_tmp.min(dim=1).values <= 1e-7 # 1e-7
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.scales_tmp.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            # prune_mask = torch.logical_or(prune_mask, scale_prune_mask)
        else:
            pass
            # prune_mask = torch.logical_or(prune_mask, scale_prune_mask)
            
        num_prune = sum(prune_mask)
        if (self.get_xyz.shape[0] - num_prune) < 1e4:
            tgt_num = (self.get_xyz.shape[0] - 1e4) - 1
            if tgt_num <= 0:
                logger.warning("There are no saturated points!")
                return
            new_mask_idx = torch.where(prune_mask == True)[0][torch.randperm(num_prune)[:int(num_prune - tgt_num)]]
            prune_mask[new_mask_idx] = False
        if num_prune == 0:
            self.n_gs = self.get_xyz.shape[0]
            torch.cuda.empty_cache()            
            return            
        
        self.prune_points(prune_mask)
        self.n_gs = self.get_xyz.shape[0]
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[:update_filter.shape[0]][update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        # self.xyz_gradient_accum_abs[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter, 2:], dim=-1,
        #                                                      keepdim=True)
        self.denom[update_filter] += 1        