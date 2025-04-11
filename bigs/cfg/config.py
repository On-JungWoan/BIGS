#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

from omegaconf import OmegaConf

# general configuration
cfg = OmegaConf.create()
cfg.seed = 0
cfg.mode = 'human' # 'human_scene' or 'scene'
cfg.output_path = 'output'
cfg.cfg_file = ''
cfg.exp_name = 'test'
cfg.dataset_path = ''
cfg.detect_anomaly = False
cfg.debug = False
cfg.wandb = False
cfg.logdir = ''
cfg.logdir_ckpt = ''
cfg.eval = False
cfg.bg_color = 'white'

# human dataset configuration
cfg.dataset = OmegaConf.create()
cfg.dataset.name = 'neuman' # 'zju', 'colmap', 'people_snapshot', 'itw'
cfg.dataset.seq = 'citron'

# training configuration
cfg.train = OmegaConf.create()
cfg.train.batch_size = 1
cfg.train.num_workers = 0
cfg.train.num_steps = 30_000
cfg.train.save_ckpt_interval = 4000
cfg.train.val_interval = 2000
cfg.train.anim_interval = 4000
cfg.train.optim_scene = True
cfg.train.save_progress_images = False
cfg.train.progress_save_interval = 10

# human model configuration
cfg.obj = OmegaConf.create()
cfg.obj.name = 'hugs'
cfg.obj.ckpt = None
cfg.obj.sh_degree = 3
cfg.obj.n_subdivision = 0
cfg.obj.only_rgb = False
cfg.obj.use_surface = False
cfg.obj.use_deformer = False
cfg.obj.init_2d = False
cfg.obj.disable_posedirs = False

cfg.obj.res_offset = False
cfg.obj.rotate_sh = False
cfg.obj.isotropic = False
cfg.obj.init_scale_multiplier = 1.0
cfg.obj.run_init = False
cfg.obj.estimate_delta = True
cfg.obj.triplane_res = 256

cfg.obj.optim_pose = False
cfg.obj.optim_betas = False
cfg.obj.optim_trans = False
cfg.obj.optim_eps_offsets = False
cfg.obj.activation = 'relu'

cfg.obj.canon_nframes = 60
cfg.obj.canon_pose_type = 'da_pose'
cfg.obj.knn_n_hops = 3

# human model learning rate configuration
cfg.obj.lr = OmegaConf.create()
cfg.obj.lr.wd = 0.0
cfg.obj.lr.position = 0.00016
cfg.obj.lr.lr_init = 0.00016
cfg.obj.lr.lr_final = 0.0000016
cfg.obj.lr.lr_delay_mult = 0.01
cfg.obj.lr.lr_max_steps = 30_000
cfg.obj.lr.opacity = 0.05
cfg.obj.lr.scaling = 0.005
cfg.obj.lr.rotation = 0.001
cfg.obj.lr.feature = 0.0025
cfg.obj.lr.smpl_spatial = 2.0
cfg.obj.lr.smpl_rot = 0.0001
cfg.obj.lr.smpl_pose = 0.0001
cfg.obj.lr.smpl_betas = 0.0001
cfg.obj.lr.smpl_trans = 0.0001
cfg.obj.lr.smpl_scale = 0.0001
cfg.obj.lr.smpl_eps_offset = 0.0001
cfg.obj.lr.lbs_weights = 0.0
cfg.obj.lr.posedirs = 0.0
cfg.obj.lr.percent_dense = 0.01

cfg.obj.lr.appearance = 1e-3
cfg.obj.lr.geometry = 1e-3
cfg.obj.lr.vembed = 1e-3
cfg.obj.lr.deformation = 1e-4
# scale
cfg.obj.lr.scale_lr_w_npoints = False

# human model loss coefficients
cfg.obj.loss = OmegaConf.create()
cfg.obj.loss.ssim_w = 0.2
cfg.obj.loss.l1_w = 0.8
cfg.obj.loss.lpips_w = 1.0
cfg.obj.loss.lbs_w = 0.0
cfg.obj.loss.humansep_w = 0.0
cfg.obj.loss.num_patches = 4
cfg.obj.loss.patch_size = 128
cfg.obj.loss.use_patches = 1

# human model densification configuration
cfg.obj.densification_interval = 100
cfg.obj.opacity_reset_interval = 3000
cfg.obj.densify_from_iter = 500
cfg.obj.densify_until_iter = 15_000
cfg.obj.densify_grad_threshold = 0.0002
cfg.obj.prune_min_opacity = 0.005
cfg.obj.densify_extent = 2.0
cfg.obj.max_n_gaussians = 2e5

# scene model configuration
cfg.scene = OmegaConf.create()
cfg.scene.name = 'scene_gs'
cfg.scene.ckpt = None
cfg.scene.sh_degree = 3
cfg.scene.add_bg_points = False
cfg.scene.num_bg_points = 204_800
cfg.scene.bg_sphere_dist = 5.0
cfg.scene.clean_pcd = False
cfg.scene.opt_start_iter = -1
cfg.scene.lr = OmegaConf.create()
cfg.scene.lr.percent_dense = 0.01
cfg.scene.lr.spatial_scale = 1.0
cfg.scene.lr.position_init = 0.00016
cfg.scene.lr.position_final = 0.0000016
cfg.scene.lr.position_delay_mult = 0.01
cfg.scene.lr.position_max_steps = 30_000
cfg.scene.lr.opacity = 0.05
cfg.scene.lr.scaling = 0.005
cfg.scene.lr.rotation = 0.001
cfg.scene.lr.feature = 0.0025

# scene model densification configuration
cfg.scene.percent_dense = 0.01
cfg.scene.densification_interval = 100
cfg.scene.opacity_reset_interval = 3000
cfg.scene.densify_from_iter = 500
cfg.scene.densify_until_iter = 15_000
cfg.scene.densify_grad_threshold = 0.0002
cfg.scene.prune_min_opacity = 0.005
cfg.scene.max_n_gaussians = 2e6

# scene model loss coefficients
cfg.scene.loss = OmegaConf.create()
cfg.scene.loss.ssim_w = 0.2
cfg.scene.loss.l1_w = 0.8
