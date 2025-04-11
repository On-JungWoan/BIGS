#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import os
import sys
import time
import torch
import argparse
import os.path as op
from loguru import logger
from omegaconf import OmegaConf

sys.path.append('.')

import warnings
warnings.filterwarnings('ignore')

import wandb
from bigs.utils.config import get_cfg_items
from bigs.cfg.config import cfg as default_cfg
from bigs.trainer import GaussianTrainer, Gaussians
from bigs.utils.general import safe_state, make_exp_dirs

import open3d
open3d.utility.set_verbosity_level(open3d.utility.VerbosityLevel.Error)

def get_logger(cfg):
    output_path = cfg.output_path
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    mode = 'eval' if cfg.eval else 'train'
    
    logdir = op.join(
        output_path, cfg.dataset.name, cfg.dataset.seq.replace('/', '_')
    )
    os.makedirs(logdir, exist_ok=True)
    logdir = make_exp_dirs(logdir, cfg.exp_name)
    cfg.logdir = logdir

    for side in ['right', 'left', 'object']:
        os.makedirs(op.join(logdir, side, 'val'), exist_ok=True)
        os.makedirs(op.join(logdir, side, 'train'), exist_ok=True)
        # os.makedirs(op.join(logdir, side, 'anim'), exist_ok=True)
        os.makedirs(op.join(logdir, side, 'meshes'), exist_ok=True)
    
    logger.remove()
    LOGURU_FORMAT="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level><cyan>{level}</cyan></level> | <level>{message}</level>"
    logger.add(sys.stdout, format=LOGURU_FORMAT)
    logger.add(op.join(logdir, f'{mode}.log'), format=LOGURU_FORMAT)
    
    logger.info(f'Logging to {logdir}')
    logger.info(OmegaConf.to_yaml(cfg))
    
    with open(op.join(logdir, f'config_{mode}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg)) 

from omegaconf.dictconfig import DictConfig
def make_simple_dict(tgt_dict):
    res = {}
    for k, v in tgt_dict.items():
        if isinstance(v, DictConfig):
            v = make_simple_dict(v)
            res[k] = dict(v)
        else:
            res[k] = v
    return res
    
def main(cfg):
    torch.cuda.set_device(cfg.gpus)
    cfg.device = f'cuda:{cfg.gpus}'
    if cfg.use_wandb:
        wandb.init(
            project='handgs',
            name=f'{cfg.exp_name}_{cfg.dataset.seq}',
            entity='jeongwanon'
        )
        wandb.config.update(make_simple_dict(dict(cfg)))
    
    safe_state(seed=cfg.seed)
    # create loggers
    get_logger(cfg)
    
    
    # ! Single Train ! #
    stored_gs = Gaussians(*cfg.components, cfg=None)
    cfg.obj.num_refine = 1 if cfg.train.joint_training else cfg.obj.num_refine
    
    for refine_step in range(cfg.obj.num_refine):
        for side in [*cfg.components]:
            cfg.logdir = op.join(cfg.logdir, side)
            cfg.logdir_ckpt = op.join(cfg.logdir, 'ckpt')
                
            os.makedirs(cfg.logdir_ckpt, exist_ok=True)
              
            cfg.dataset['side'] = side
            cfg.dataset['check_mode'] = cfg.check_mode
            
            trainer = GaussianTrainer(cfg, side, refine_step)
            
            if cfg.obj.ckpt is not None:
                if cfg.train.joint_training:
                    trainer.gs_model.load_state_dict(torch.load(cfg.obj.ckpt), cfg=cfg.obj.lr)
                    stored_gs._set_gaussian(trainer.gs_model, side)
                    stored_gs._set_dataset(trainer.train_dataset, side)

                    cfg.logdir = op.dirname(cfg.logdir); continue
            
            if not cfg.eval:
                trainer.train()
                trainer.save_ckpt()
                
                trainer.animate(device=cfg.device)
                stored_gs._set_gaussian(trainer.gs_model, side)
                trainer.render_canonical(pose_type='hand', device=cfg.device)
                stored_gs._set_dataset(trainer.train_dataset, side)            
            cfg.logdir = op.dirname(cfg.logdir)


    # ! Joint Train ! #
    stored_gs.cfg = cfg
    trainer.gs_model = stored_gs
    if cfg.train.joint_training or cfg.paper_vis_mode is not None:
        cfg.logdir = op.join(cfg.logdir, 'joint')
        cfg.logdir_ckpt = op.join(cfg.logdir, 'ckpt')
        
        os.makedirs(cfg.logdir_ckpt, exist_ok=True)
        trainer.cfg = cfg; trainer.gs_model.cfg = cfg

        trainer.train(joint_train=True)
        trainer.gs_model.save_ckpt()
    
    with torch.no_grad():
        trainer.stored_gs_animate(device=cfg.device, keep_images=cfg.check_mode)
        trainer.stored_gs_render_canonical(pose_type='hand', device=cfg.device)
        trainer.stored_gs_novel_view_animate(device=cfg.device)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="cfg_files/arctic.yaml", help="path to the yaml config file")
    parser.add_argument("--cfg_id", type=int, default=-1, help="id of the config to run")
    args, extras = parser.parse_known_args()
    
    cfg_file = OmegaConf.load(args.cfg_file)
    list_of_cfgs, hyperparam_search_keys = get_cfg_items(cfg_file)
    
    logger.info(f'Running {len(list_of_cfgs)} experiments')
    
    if args.cfg_id >= 0:
        cfg_item = list_of_cfgs[args.cfg_id]
        logger.info(f'Running experiment {args.cfg_id} -- {cfg_item.exp_name}')
        default_cfg.cfg_file = args.cfg_file
        cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
        main(cfg)
    else:
        for exp_id, cfg_item in enumerate(list_of_cfgs):
            logger.info(f'Running experiment {exp_id} -- {cfg_item.exp_name}')
            default_cfg.cfg_file = args.cfg_file
            cfg = OmegaConf.merge(default_cfg, cfg_item, OmegaConf.from_cli(extras))
            main(cfg)
            