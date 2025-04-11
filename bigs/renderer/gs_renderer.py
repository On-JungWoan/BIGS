# Code adapted from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py
# License from 3DGS: https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md

#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import cv2
import math
import torch
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings, 
    GaussianRasterizer
)

from bigs.utils.spherical_harmonics import SH2RGB
from bigs.utils.rotations import quaternion_to_matrix

def render_human_scene(
    data, 
    human_gs_out,
    scene_gs_out,
    bg_color, 
    human_bg_color=None,
    scaling_modifier=1.0, 
    render_mode='human_scene',
    render_human_separate=False,
    t_iter=0,
    device=None
):

    feats = human_gs_out['shs']
    means3D = human_gs_out['xyz']
    opacity = human_gs_out['opacity']
    scales = human_gs_out['scales']
    rotations = human_gs_out['rotq']
    active_sh_degree = human_gs_out['active_sh_degree']
    
    render_pkg = render(
        means3D=means3D,
        feats=feats,
        opacity=opacity,
        scales=scales,
        rotations=rotations,
        data=data,
        scaling_modifier=scaling_modifier,
        bg_color=bg_color,
        active_sh_degree=active_sh_degree,
        t_iter=t_iter,
        device=device
    )
        
    render_pkg['human_visibility_filter'] = render_pkg['visibility_filter']
    render_pkg['human_radii'] = render_pkg['radii']

    return render_pkg
    
    
def render(means3D, feats, opacity, scales, rotations, data, scaling_modifier=1.0, bg_color=None, active_sh_degree=0, t_iter=0, device=None):
    if bg_color is None:
        bg_color = torch.zeros(3, dtype=torch.float32, device=device)
        
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=device) + 0
    try:
        screenspace_points.retain_grad()
    except Exception as e:
        pass

    means2D = screenspace_points

    # Set up rasterization configuration
    tanfovx = math.tan(data['fovx'] * 0.5)
    tanfovy = math.tan(data['fovy'] * 0.5)

    shs, rgb = None, None
    if len(feats.shape) == 2:
        rgb = feats
    else:
        shs = feats

    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(data['image_height']),
        image_width=int(data['image_width']),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=data['world_view_transform'].to(device),
        projmatrix=data['full_proj_transform'].to(device),
        sh_degree=active_sh_degree,
        campos=data['camera_center'].to(device),
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    gs_w = None
    # rendered_image, radii, gs_w = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = rgb,
    #     opacities = opacity,
    #     scales = scales,
    #     rotations = rotations,
    #     cov3D_precomp = None)
            
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D.float(),
        means2D=means2D.float(),
        shs=shs.float(),
        opacities=opacity.float(),
        scales=scales.float(),
        rotations=rotations.float(),
        colors_precomp=rgb,
    )
    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    
    # cv2.imwrite(f'test/test_{t_iter}.png', rendered_image.permute(1,2,0).cpu().detach().numpy()*255)
    
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii,
    #         "gs_w": gs_w
    #         }    
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : radii > 0,
        "radii": radii,
    }
