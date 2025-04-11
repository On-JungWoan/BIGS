# Code based on 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/general_utils.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#


import os
import cv2
import torch
import random
import trimesh
import itertools
import subprocess
import numpy as np
import os.path as op
import open3d as o3d
from tqdm import tqdm
from loguru import logger
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from bigs.datasets.utils import get_rotating_camera
from bigs.renderer.gs_renderer import render_human_scene

def render_point_cloud(
        pcd, device, output_dir='canon_test',
        s_factor = 1e-2, c_factor = 0.5, r_factor = 0.5,
        dist=10.5, nframes=100,
    ):
    camera_params = get_rotating_camera(
        dist=dist, img_size=512, nframes=nframes,
        device=device, angle_limit=2*torch.pi,
    )

    x = torch.from_numpy(pcd).clone().float().to(device)
    s = torch.ones_like(x).float().to(device) * s_factor
    r = torch.ones(x.shape[0], 4).float().to(device) * r_factor
    c = torch.ones(x.shape[0], 16, 3).float().to(device) * c_factor
    o = torch.ones(x.shape[0], 1).float().to(device)
    human_gs_out = {
        'shs': c,
        'xyz': x,
        'opacity': o,
        'scales': s,
        'rotq': r,
        'active_sh_degree': 0,
    }

    os.makedirs(output_dir, exist_ok=True)
    pbar = enumerate(tqdm(camera_params, desc="[TEST PCD]"))
    for idx, data in pbar:
        image = render_human_scene(
            data, human_gs_out, None,
            torch.ones(3).float().to(device), 
            None, 1.0, '', None, t_iter=0, device=device
        )['render']
        save_img(image*255, f"{output_dir}/{idx:05}.png")    


def upsample_point_cloud(points, factor=2, noise_level=0.01):
    new_points = []
    
    for _ in range(factor - 1):
        jitter = np.random.normal(scale=noise_level, size=points.shape)
        new_points.append(points + jitter)
    
    return np.vstack([points] + new_points)


def compute_bounding_box_volume(pcd):
    """Compute the volume of the bounding box for a given point cloud."""
    min_corner = torch.min(pcd, axis=0)[0]
    max_corner = torch.max(pcd, axis=0)[0]
    volume = torch.prod(max_corner - min_corner)
    return volume


def initialize_gaussian_scales(pcd, alpha=10.0, beta=1/3):
    """Initialize Gaussian scales based on point cloud density."""
    N = len(pcd)
    V = compute_bounding_box_volume(pcd)
    
    if V == 0:
        raise ValueError("Bounding box volume is zero. Check the point cloud.")
    
    D = N / V  # Density
    scale = alpha * (D ** - beta)
    return scale


def pcd_to_mesh(points, out_path='test.obj'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    
    alpha = 0.1
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    obj_mesh = trimesh.smoothing.filter_mut_dif_laplacian(
        trimesh_mesh, 
        lamb=0.5,
        iterations=10,
        volume_constraint=True,
        laplacian_operator=None
    )
    obj_mesh.export(out_path)
    
    return obj_mesh


def make_p3d_mesh_viewer(intrinsic, img_size, device):
    from pytorch3d.renderer import (
        look_at_view_transform,
        OpenGLPerspectiveCameras,
        PerspectiveCameras,
        RasterizationSettings,
        MeshRenderer,
        MeshRasterizer,
        HardPhongShader,
        PointsRasterizationSettings,
        PointsRasterizer,
        PointsRenderer,
        PointLights,
        AlphaCompositor,
        HardPhongShader,
    )
    from pytorch3d.structures import Meshes, Pointclouds
    from pytorch3d.renderer.mesh import Textures
    from pytorch3d.structures.meshes import join_meshes_as_scene
    from pytorch3d.renderer.blending import BlendParams
    # from submodules.hold.code.visualize_ckpt import DataViewer
    # viewer = DataViewer(interactive=True, size=(2024, 2024))

    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    
    cx = intrinsic[0][-2]
    cy = intrinsic[1][-2]
    
    focal_length = torch.tensor([fx, fy]).unsqueeze(0).to(device)
    principal_point = torch.tensor([cx, cy]).unsqueeze(0).to(device)
    
    cam_R = torch.diag(torch.tensor([1, 1, 1]))[None].float()
    cam_T = torch.zeros(3)[None].float()
    cam_R[:, :2, :] *= -1.0
    cam_T[:, :1] *= -1.0
    cam_T[:, :2] *= -1.0
    
    raster_settings = RasterizationSettings(
        image_size=img_size,
        blur_radius=0.0,
        faces_per_pixel=10,
        max_faces_per_bin=1000000
    )
    torch3d_camera = PerspectiveCameras(
        focal_length=focal_length, principal_point=principal_point,
        R=cam_R, T=cam_T, device=device, in_ndc=False, image_size=[img_size]
    )
    mesh_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=torch3d_camera, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=torch3d_camera, blend_params=BlendParams(background_color= (255., 255., 255.)))  # Use the default shader
    )
    
    return mesh_renderer


def apply_transformation(points, transformation):
    """
    Apply a 4x4 transformation matrix to a set of 3D points.
    
    Parameters:
        points (np.ndarray): Source points as (N, 3).
        transformation (np.ndarray): 4x4 transformation matrix.
        
    Returns:
        transformed_points (np.ndarray): Transformed points as (N, 3).
    """
    # Convert points to homogeneous coordinates (N, 4)
    ones = np.ones((points.shape[0], 1))  # Column of 1s
    homogeneous_points = np.hstack([points, ones])  # Add homogeneous coordinate
    
    # Apply the transformation matrix (N, 4) * (4, 4).T
    transformed_homogeneous = homogeneous_points @ transformation.T
    
    # Convert back to 3D coordinates (drop the homogeneous part)
    transformed_points = transformed_homogeneous[:, :3]
    
    return transformed_points
                

def grad_off(human_gs, lrs, idx=0):
    tgt_layers = {
        'vembed':'triplane', 'appearance':'appearance_dec',
        'deformation':'deformation_dec', 'geometry':'geometry_dec'
    }
    tgt_values = {
        'position':'_xyz', 'smpl_pose':'body_pose', 'smpl_rot':'global_orient',
        'smpl_trans':'transl', 'smpl_scale':'scale'
    }
    optim_v = {
        '_xyz':'xyz', 'triplane':'v_embed', 'geometry_dec':'geometry_dec',
        'appearance_dec':'appearance_dec', 'deformation_dec':'deform_dec',
        'global_orient':'global_orient', 'body_pose':'body_pose',
        'transl':'transl', 'scale':'scale'
    }
    
    tgt_layers = [p_name for ag_name, p_name in tgt_layers.items() if lrs.get(ag_name) == 0]
    tgt_values = [p_name for ag_name, p_name in tgt_values.items() if lrs.get(ag_name) == 0]    
    
    if len(tgt_layers + tgt_values) > 0:
        for name in (tgt_layers + tgt_values):
            p_idx = [idx for idx, g in enumerate(human_gs.optimizer.param_groups) if optim_v[name] == g['name']]
            assert len(p_idx) == 1
            human_gs.optimizer.param_groups[p_idx[0]]['lr'] = 0.
        
        if len(tgt_layers) > 0:
            for layer_n in tgt_layers:
                logger.info(f'[{idx}_{layer_n}] grad off')
                for n, p in human_gs.__getattribute__(layer_n).named_parameters():
                    p.requires_grad = False
        
        if len(tgt_values) > 0:
            for value_n in tgt_values:
                if hasattr(human_gs, value_n):
                    human_gs.__getattribute__(value_n).requires_grad = False
                    logger.info(f'[{idx}_{value_n}] grad off')
                
    return human_gs


def debug_tensor(tensor, name):
    print(f'{name}: {tensor.shape} {tensor.dtype} {tensor.device}')
    print(f'{name}: min: {tensor.min().item():.5f} \
        max: {tensor.max().item():.5f} \
            mean: {tensor.mean().item():.5f} \
                std: {tensor.std().item():.5f}')


def load_human_ckpt(human_gs, ckpt_path):
    ckpt = torch.load(ckpt_path)
    persistent_buffers = {k: v for k, v in human_gs._buffers.items() if k not in human_gs._non_persistent_buffers_set}
    local_name_params = itertools.chain(human_gs._parameters.items(), persistent_buffers.items())
    local_state = {k: v for k, v in local_name_params if v is not None}
    for k, v in local_state.items():
        if v.shape != ckpt[k].shape:
            logger.warning(f'Warning: shape mismatch for {k}: {v.shape} vs {ckpt[k].shape}')
            if (isinstance(v, torch.nn.Parameter) and
                    not isinstance(ckpt[k], torch.nn.Parameter)):
                setattr(human_gs, k, torch.nn.Parameter(ckpt[k]))
            else:
                setattr(human_gs, k, ckpt[k])

    human_gs.load_state_dict(ckpt, strict=False)
    logger.info(f'Loaded human model from {ckpt_path}')
    return human_gs


class RandomIndexIterator:
    def __init__(self, max_index):
        self.max_index = max_index
        self.indices = list(range(max_index))
        random.shuffle(self.indices)
        self.current_index = 0
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current_index >= self.max_index:
            self.current_index = 0
            random.shuffle(self.indices)
        index = self.indices[self.current_index]
        self.current_index += 1
        return index


def find_cfg_diff(default_cfg, cfg, delimiter='_'):
    default_cfg_list = OmegaConf.to_yaml(default_cfg).split('\n')
    cfg_str_list = OmegaConf.to_yaml(cfg).split('\n')
    diff_str = ''
    nlines = len(default_cfg_list)
    for lnum in range(nlines):
        if default_cfg_list[lnum] != cfg_str_list[lnum]:
            diff_str += cfg_str_list[lnum].replace(': ', '-').replace(' ', '')
            diff_str += delimiter
    diff_str = diff_str[:-1]
    return diff_str
        

def create_video(img_folder, output_fname, fps=20):
    os.makedirs(os.path.dirname(output_fname), exist_ok=True)
    cmd = f"/usr/bin/ffmpeg -hide_banner -loglevel error -framerate {fps} -pattern_type glob -i '{img_folder}/*.png' \
        -vf \"pad=ceil(iw/2)*2:ceil(ih/2)*2\" \
            -c:v libx264 -pix_fmt yuv420p {output_fname} -y"
    logger.info(f"Video is saved under {output_fname}")
    subprocess.call(cmd, shell=True)


def save_images(img, img_fname, txt_label=None):
    if not os.path.isdir(os.path.dirname(img_fname)):
        os.makedirs(os.path.dirname(img_fname), exist_ok=True)
    im = Image.fromarray(img)
    if txt_label is not None:
        draw = ImageDraw.Draw(im)
        txt_font = ImageFont.load_default()
        draw.text((10, 10), txt_label, fill=(0, 0, 0), font=txt_font)
    im.save(img_fname)
        
        
def eps_denom(denom, eps=1e-17):
    """ Prepare denominator for division """
    denom_sign = denom.sign() + (denom == 0.0).type_as(denom)
    denom = denom_sign * torch.clamp(denom.abs(), eps)
    return denom


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L


def safe_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # torch.cuda.set_device(torch.device("cuda"))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def torch_rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector of shape N,3
    :param vec2: A 3d "destination" vector of shape N,3
    :return mat: A transform matrix (Nx3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
    b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
    
    v = torch.cross(a, b, dim=-1)
    c = torch.matmul(a.unsqueeze(1), b.unsqueeze(-1)).squeeze(-1)
    s = torch.norm(v, dim=-1, keepdim=True)
    kmat = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rot_mat = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
    rot_mat = rot_mat + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1)
    return rot_mat


class ClampWithGrad(torch.autograd.Function):
    @staticmethod
    def forward (ctx, input):
        ctx.save_for_backward(input)
        return torch.clamp(input, -1, 1)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        tanh = torch.tanh(input)
        grad_input[input <= -1] = (1.0 - tanh[input <= -1]**2.0) * grad_output[input <= -1]
        grad_input[input >= 1] = (1.0 - tanh[input >= 1]**2.0) * grad_output[input >= 1]
        max_norm = 1.0  # set the maximum gradient norm value
        torch.nn.utils.clip_grad_norm_(grad_input, max_norm)
        return grad_input
    

def make_and_export_mesh(vertices, faces, output_dirs='./test.obj'):
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().detach().numpy()
    
    mesh = trimesh.Trimesh(vertices.cpu().detach().numpy(), faces)
    mesh.export(output_dirs)
    
    return mesh

def draw_j2d(joint3d, K, rgb, path='test.png'):
    def to_xy_batch(x_homo):
        assert isinstance(x_homo, (torch.FloatTensor, torch.cuda.FloatTensor))
        assert x_homo.shape[2] == 3
        assert len(x_homo.shape) == 3
        batch_size = x_homo.shape[0]
        num_pts = x_homo.shape[1]
        x = torch.ones(batch_size, num_pts, 2, device=x_homo.device)
        x = x_homo[:, :, :2] / x_homo[:, :, 2:3]
        return x
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # 엄지
        (0, 5), (5, 6), (6, 7), (7, 8),        # 검지
        (0, 9), (9, 10), (10, 11), (11, 12),   # 중지
        (0, 13), (13, 14), (14, 15), (15, 16), # 약지
        (0, 17), (17, 18), (18, 19), (19, 20)  # 새끼
    ]
    
    pts2d_homo = torch.bmm(K, joint3d.permute(0, 2, 1)).permute(0, 2, 1)
    pts2d = to_xy_batch(pts2d_homo)[0]
    
    c_img = np.ascontiguousarray(rgb.cpu().permute(1,2,0))*255
    for j in range(21):
        joint = pts2d[j].type(torch.int).tolist()
        c_img = cv2.circle(c_img, (joint[0], joint[1]), 5, (255, 0, 0), -1)
        
    # for (start, end) in connections:
    #     x1, y1 = pts2d[start].type(torch.int).tolist()
    #     x2, y2 = pts2d[end].type(torch.int).tolist()
    #     cv2.line(c_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(path, c_img)
    
def save_img(img, save_path='test.png'):
    assert len(img.shape)==3
    assert isinstance(img, torch.Tensor)
    assert img.shape[0]==3 or img.shape[-1]==3
    
    if img.shape[0] == 3:
        img = img.permute(1,2,0)
    
    cv2.imwrite(save_path, img.cpu().detach().numpy())
    
    
def custom_to(tgt, device, cast_type=False):
    if isinstance(tgt, dict):
        for k, v in tgt.items():
            tgt[k] = custom_to(v, device, cast_type)
            
    elif isinstance(tgt, list):
        for idx, v in enumerate(tgt):
            tgt[idx] = custom_to(v, device, cast_type)
            
    elif isinstance(tgt, torch.Tensor):
        tgt = tgt.to(device)
        if cast_type:
            tgt = tgt.type(torch.float32)
    
    return tgt


def make_exp_dirs(logdir, exp_name):
    if '+' in exp_name:
        exp_name_list = exp_name.split('+')
        logdir = make_exp_dirs(logdir, exp_name_list[0])
        return make_exp_dirs(logdir, '+'.join(exp_name_list[1:]))
    else:
        exp_lists = os.listdir(logdir)
    
        if len(exp_lists) > 0:
            last_exp_num = max([int(l.split('_')[0]) for l in exp_lists])
            prev_exp_num = [int(exp.split('_')[0]) for exp in exp_lists if exp_name == '_'.join(exp.split('_')[1:])]
            
            if len(prev_exp_num) < 1:
                exp_num = last_exp_num+1
            else:
                assert len(prev_exp_num) == 1
                exp_num = prev_exp_num[0]
        else:
            exp_num = 0
        logdir = op.join(logdir, f'{exp_num:03}_{exp_name}')
    
    os.makedirs(logdir, exist_ok=True)
    return logdir


def find_outlier(tgt, weight=10.0):
    """_summary_

    Args:
        tgt (torch.FloatTensor): (b_idx, channel)
        weight (float, optional): iqr weight. Defaults to 3.5.

    Returns:
        is_valid (torch.BoolTensor): valid information
        outlier_idx (torch.IntTensor): outlier index
    """
    
    q = torch.tensor([0.25, 0.75])
    
    q_25, q_75 = torch.quantile(tgt, q, dim=0)
    iqr = (q_75 - q_25) * weight
    
    min = q_25 - iqr
    max = q_75 + iqr
    
    cond = (tgt < min.unsqueeze(0)) + (tgt > max.unsqueeze(0))
    
    cleaned_tgt = torch.where(cond, torch.nan, tgt)
    is_valid = torch.isnan(cleaned_tgt).sum(dim=1) == 0
    outlier_idx = (~is_valid).nonzero(as_tuple=True)[0]
    
    return is_valid, outlier_idx


def replace_outlier(tgt, is_valid):
    """_summary_

    Args:
        tgt (_type_): _description_
        is_valid (bool): _description_

    Returns:
        _type_: _description_
    """
    
    assert is_valid[0], 'Oops.....'
    for idx, valid in enumerate(is_valid):
        if not valid:
            tgt[idx] = tgt[idx-1]

    return tgt