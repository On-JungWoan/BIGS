#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import sys
import torch

from bigs.cfg.config import cfg as default_cfg
from copy import deepcopy


def optimize_init(model, save_path: str = 'test', lr: float = 1e-3, num_steps: int = 2000, cfg=None):
    model.train()

    if cfg is None:
        lr = 1e-3
        default_cfg.obj.lr.appearance = lr
        default_cfg.obj.lr.geometry = lr
        default_cfg.obj.lr.vembed = lr
        default_cfg.obj.lr.deformation = 5e-4
        model.setup_optimizer(default_cfg.obj.lr)
    else:
        # new_cfg = deepcopy(cfg)

        # new_cfg.obj.lr.position=0.0
        # new_cfg.obj.lr.deformation=0.0
        # new_cfg.obj.lr.geometry=0.0
        # model.setup_optimizer(new_cfg.obj.lr)
        model.setup_optimizer(cfg.obj.lr)
    optim = model.optimizer
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1000, verbose=True, factor=0.5)
    fn = torch.nn.MSELoss()
    
    body_pose = torch.zeros((45)).to("cuda").float()
    global_orient = torch.zeros((3)).to("cuda").float()
    betas = torch.zeros((10)).to("cuda").float()
    
    init_gt_vals = model.initialize()
    gt_vals = {}
    
    print("===== Ground truth values: =====")
    for k, v in init_gt_vals.items():
        if v is not None:
            print(k, v.shape)
            gt_vals[k] = v.detach().clone().to("cuda").float()
    print("================================")
    
    losses = []

    for i in range(num_steps):
        
        if hasattr(model, 'canon_forward'):
            model_out = model.canon_forward()
        else:
            assert False, "Not implemented yet!"
            model_out = model.forward(global_orient, body_pose, betas)
        
        if i % 1000 == 0:
            continue
            
        loss_dict = {}
        for k, v in gt_vals.items():
            if k in ['faces', 'deformed_normals', 'edges'] or v is None:
                continue
            if k in model_out:
                if model_out[k] is not None:
                    loss_dict['loss_' + k] = fn(model_out[k], v)
                    
        # for scale reg
        loss_dict['loss_' + k] = (model_out['scales']**2).mean(0).sum() * 100.0
        
        loss = sum(loss_dict.values())
        loss.backward()
        loss_str = ", ".join([f"{k}: {v.item():.7f}" for k, v in loss_dict.items()])
        print(f"Step {i:04d}: {loss.item():.7f} ({loss_str})", end='\r')
        
        optim.step()
        optim.zero_grad(set_to_none=True)
        lr_scheduler.step(loss.item())
            
        losses.append(loss.item())
    
    # save model's pth
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    return model
