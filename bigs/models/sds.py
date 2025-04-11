import os
from omegaconf import OmegaConf
from submodules.gtu.gtu.guidance import DiffusionGuidance

def setup_for_sds_loss(cfg):
    dg_log_dir = f'{cfg.logdir}/sds_loss'
    os.makedirs(dg_log_dir, exist_ok=True)
    dgm_opt = OmegaConf.load("submodules/gtu/gtu/guidance/configs/default.yaml")
    dgm_opt.density_start_iter = cfg.density_start_iter
    dgm_opt.density_end_iter = cfg.density_start_iter
    dgm_opt.densification_interval = cfg.density_start_iter
    dgm_opt.iter_smpl_densify = cfg.density_start_iter
    dgm_opt.iter_prune_smpl_until = cfg.density_start_iter
    dgm_opt.scene_extent = cfg.density_start_iter
    dgm_opt.lambda_rgb_loss = cfg.density_start_iter
    dgm_opt.lambda_density_reg = cfg.density_start_iter
    dgm_opt.dgm_loss_weight = cfg.density_start_iter
    dgm_opt.dgm_start_iter = cfg.density_start_iter
    
    textual_inversion_path = os.path.abspath(f'submodules/gtu/output_common/{cfg.db_name}')
    textual_inversion_expname = cfg.seq
    obj_name = cfg.obj_name if cfg.obj_name is not None else cfg.seq.split('_')[1]
    DGM = DiffusionGuidance(
        opt=dgm_opt, 
        log_dir=dg_log_dir, 
        textual_inversion_path=textual_inversion_path, 
        textual_inversion_expname=textual_inversion_expname,
        textual_inversion_in_controlnet=False,
        use_ti_free_prompt_on_controlnet = True,
        ti_load_epoch = -1,
        guidance_scale = cfg.cfg_weight,
        inpaint_guidance_scale = 7.5,
        controlnet_weight = cfg.controlnet_weight,
        lambda_percep=1.0,
        lambda_rgb=0.1,
        random_noise_step = True,
        noise_sched = 'time_annealing',
        camera_sched = 'default', #'defacto',
        do_guid_sched = False,
        sd_version="1.5",
        use_aux_prompt = True,
        use_view_prompt = True,
        cfg_sched = 'default',
        obj_name = obj_name,
    )
    DGM.prepare_train(
        pids=['0'], enable_controlnet = True, enable_zero123 = False,
        is_inpaint = False, do_cfg_rescale = cfg.do_cfg_rescale, do_multistep_sds = False,
        use_inpaint_unet = False, use_joint_diff = False
    )
    
    return DGM