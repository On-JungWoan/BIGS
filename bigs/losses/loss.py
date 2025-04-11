#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch
from lpips import LPIPS
import torch.nn as nn
import torch.nn.functional as F

from bigs.utils.sampler import PatchSampler

from bigs.utils.image import psnr
from .utils import l1_loss, l2_loss, ssim
import submodules.hold.common.transforms as tf
from bigs.utils.general import make_and_export_mesh, save_img
from submodules.hold.code.src.utils.eval_modules import compute_bounding_box_centers, compute_bounding_box_centers_torch
from bigs.utils.rotations import rotation_6d_to_axis_angle

class HandObjectLoss(nn.Module):
    def __init__(
        self,
        v_template_w=0.0,
        l1_bg_w=0.0,
        transl_w=1.0,
        j3d_w=0.0,
        rot_w=0.0,
        pose_w=0.0,
        l2_pre=0.0,
        
        l1_shs=0.0,
        l2_offset=0.0,
        l2_scale=0.0,
        l2_smt=0.0,
        
        sds_w=0.0,
        
        ssim_w=0.2,
        l1_w=0.8,
        lpips_w=0.0,
        lbs_w=0.0,
        humansep_w=0.0,
        num_patches=4,
        patch_size=32,
        use_patches=True,
        bg_color='white',
    ):
        super(HandObjectLoss, self).__init__()
        
        self.l_l2_pre = l2_pre
        self.l_j3d_w = j3d_w
        self.l_rot_w = rot_w
        self.l_pose_w = pose_w
        self.l_sds_w = sds_w
        
        # regularization
        self.l_l1_shs = l1_shs
        self.l_l2_offset = l2_offset
        self.l_l2_scale = l2_scale
        self.l_l2_smt = l2_smt
        
        self.l_transl_w = transl_w
        self.l_l1_bg_w = l1_bg_w
        self.l_v_tem_w = v_template_w
        
        self.l_ssim_w = ssim_w
        self.l_l1_w = l1_w
        self.l_lpips_w = lpips_w
        self.l_lbs_w = lbs_w
        self.l_humansep_w = humansep_w
        self.use_patches = use_patches
        
        self.bg_color = bg_color
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
    
        for param in self.lpips.parameters(): param.requires_grad=False
        
        if self.use_patches:
            self.patch_sampler = PatchSampler(num_patch=num_patches, patch_size=patch_size, ratio_mask=0.9, dilate=0)
        
    def forward(
        self, 
        data, 
        render_pkg,
        human_gs_out,
        render_mode, 
        human_gs_init_values=None,
        bg_color=None,
        human_bg_color=None,
        joint_train=False,
        
        sds_items=None,
        data_idx=None,
        misc=None,
        gs=None,
        mode='right',
        stored_j3d=None,
    ):
        loss_dict = {}
        extras_dict = {}
        device = human_gs_out['xyz'].device
        
        if bg_color is not None:
            self.bg_color = bg_color
            
        if human_bg_color is None:
            human_bg_color = self.bg_color
            
        joint_mask = sum(data['mask'].values()).unsqueeze(0).cuda()
        if joint_train:
            mask = joint_mask.clone()
        else:
            mask = data['mask'][mode].unsqueeze(0).cuda()
        
        gt_image = data['rgb']
        pred_img = render_pkg['render']
        
        tgt_idx = (data_idx - 1) if data_idx != 0 else (data_idx + 1)
        
        # cam alignment
        if mode=='object' and self.l_l2_pre != 0.:
            if not 'arctic' in misc['fnames'][0] or misc['is_valid'][data_idx]:
                l_pre = l2_loss(human_gs_out['xyz'], misc['v3d_c.object'][data_idx].to('cuda:0'))
                loss_dict['l_pre'] = l_pre * self.l_l2_pre
        
        # joint smt reg
        if stored_j3d is not None and self.l_j3d_w != 0.:
            if joint_train:
                    assert not gs[-1].use_deformer
                    assert gs[0].smpl_template.is_rhand
                    
                    pred_T = compute_bounding_box_centers_torch(human_gs_out['xyz'][-gs[-1]._xyz.shape[0]:, :])
                    gt_T = stored_j3d['object'][tgt_idx]
                    j3d_smt_loss_obj = torch.abs(pred_T - gt_T).mean()
                    j3d_smt_loss = j3d_smt_loss_obj
            else:
                if mode != 'object':
                    j3d_smt_loss = torch.abs(human_gs_out['joint3d'][0] - stored_j3d[tgt_idx]).mean(1).sum()
                else:
                    pred_T = compute_bounding_box_centers_torch(human_gs_out['xyz'])
                    gt_T = stored_j3d[tgt_idx]
                    j3d_smt_loss = torch.abs(pred_T - gt_T).mean()
            loss_dict['j3d_smt_reg'] = j3d_smt_loss * self.l_j3d_w
        
        # rot smt reg
        if 'global_orient' in human_gs_out and human_gs_out['global_orient'] is not None and self.l_rot_w != 0.:
            prev_global_orient_offset = rotation_6d_to_axis_angle(gs.global_orient[tgt_idx].reshape(-1, 6)).reshape(3)
            prev_global_orient = torch.tensor(misc['cached_data'][mode]['global_orient'][tgt_idx], device=device) + prev_global_orient_offset
            
            rot_smt_loss = torch.abs(human_gs_out['global_orient'] - prev_global_orient).sum()
            loss_dict['rot_smt_reg'] = rot_smt_loss * self.l_rot_w
        
        # pose smt reg
        if not joint_train and mode != 'object' and human_gs_out['body_pose'] is not None and self.l_pose_w != 0.:
            prev_body_pose_offset = rotation_6d_to_axis_angle(gs.body_pose[tgt_idx].reshape(-1, 6)).reshape(15*3)
            prev_body_pose = torch.tensor(misc['cached_data'][mode]['hand_pose'][tgt_idx], device=device) + prev_body_pose_offset
            
            pose_smt_loss = torch.abs(human_gs_out['body_pose'].reshape(15, 3) - prev_body_pose.reshape(15, 3)).mean(1).sum()
            loss_dict['pose_smt_reg'] = pose_smt_loss * self.l_pose_w
        
        if sds_items is not None:
            sds_pred, sds_gt, sds_mask = sds_items
            sds_mask = sds_mask.float()
            
            bg_img_sds = sds_gt * 0 + 1
            not_masked_pred_img_sds = sds_pred * sds_mask.unsqueeze(1) + (1. - sds_mask).unsqueeze(1)
            
            sds_gt = sds_gt * (1. - sds_mask).unsqueeze(1) + sds_mask.unsqueeze(1)
            sds_pred = sds_pred * (1. - sds_mask).unsqueeze(1) + sds_mask.unsqueeze(1)            
            
        
        if render_mode == "human":
            bg_img = gt_image * 0 + human_bg_color[:, None, None]
            not_masked_pred_img = pred_img * (1. - joint_mask) + human_bg_color[:, None, None] * joint_mask
            
            gt_image = gt_image * mask + human_bg_color[:, None, None] * (1. - mask)
            pred_img = pred_img * mask + human_bg_color[:, None, None] * (1. - mask)
            
            extras_dict['gt_img'] = gt_image
            extras_dict['pred_img'] = pred_img

        # * FOR REGULARIZATION * #
        if self.l_l2_smt > 0.0 and joint_train:
            for g in gs[:-1]:
                assert g.use_deformer
                postfix = 'right' if g.smpl_template.is_rhand else 'left'
                misc_transl = torch.from_numpy(misc['cached_data'][postfix]['transl']).to(device)
                approx_transl = misc_transl * misc['scale'].view(-1, 1).contiguous() + g.transl

                loss_dict[f'reg_smt_{postfix}'] = l2_loss(approx_transl[1:]*1000, approx_transl[:-1]*1000) * self.l_l2_smt

        if self.l_transl_w > 0.0 and joint_train:
            assert gs[0].smpl_template.is_rhand and not gs[-1].use_deformer
            assert len(human_gs_out['joint3d'].shape) == 3
            
            trans = human_gs_out['joint3d'].clone()
            
            x = human_gs_out['xyz'][-gs[-1]._xyz.shape[0]:, :]; assert len(x.shape) == 2
            bbox_min = torch.min(x, dim=0)[0]
            bbox_max = torch.max(x, dim=0)[0]
            T = (bbox_min + bbox_max) / 2

            for idx, t in enumerate(trans):
                loss_dict[f'reg_{idx}_t'] = l2_loss(t[0], T) * self.l_transl_w
        
        if self.l_l1_shs > 0.0:
            shs = human_gs_out['shs'].view(-1, 48).contiguous()
            shs_mean = shs.mean(0).unsqueeze(0)
            loss_dict['reg_shs'] = torch.abs(shs - shs_mean).mean(0).sum() * self.l_l1_shs
        
        if self.l_l2_offset > 0.0 and not joint_train:
            loss_dict['reg_offset'] = torch.abs(human_gs_out['xyz_offsets']).mean(0).sum() * self.l_l2_offset
            
        if self.l_l2_scale > 0.0:
            loss_dict['reg_scale'] = (human_gs_out['scales']**2).mean(0).sum() * self.l_l2_scale
        # * FOR REGULARIZATION * #
        
        if self.l_l1_w > 0.0:
            Ll1 = l1_loss(pred_img, gt_image, mask)
            Ll1_bg = l1_loss(not_masked_pred_img, bg_img, 1.-joint_mask)
            
            if sds_items is not None:
                Ll1_sds = l1_loss(sds_pred, sds_gt, 1.-sds_mask)
                Ll1_bg_sds = l1_loss(not_masked_pred_img_sds, bg_img_sds, sds_mask)
                loss_dict['l1_sds'] = self.l_l1_w * self.l_sds_w * Ll1_sds
                loss_dict['l1_bg_sds'] = self.l_l1_bg_w * self.l_sds_w * Ll1_bg_sds
            
            # save_img(not_masked_pred_img*255)
            loss_dict['l1'] = self.l_l1_w * Ll1
            loss_dict['l1_bg'] = self.l_l1_bg_w * Ll1_bg

        if self.l_ssim_w > 0.0:
            loss_ssim = 1.0 - ssim(pred_img, gt_image)
            loss_ssim = loss_ssim * (mask.sum() / (pred_img.shape[-1] * pred_img.shape[-2]))
            loss_dict['ssim'] = self.l_ssim_w * loss_ssim

            if sds_items is not None:
                loss_ssim_sds = 1.0 - ssim(sds_pred, sds_gt)
                loss_ssim_sds = loss_ssim_sds * ((1. - sds_mask).sum() / (sds_pred.shape[-1] * sds_pred.shape[-2]))
                loss_dict['ssim_sds'] = self.l_ssim_w * loss_ssim_sds * self.l_sds_w
        
        if self.l_lpips_w > 0.0 and not render_mode == "scene" and mask.sum() > 0:
            try:
                assert self.use_patches, "Not implemented yet!"
                
                if sds_items is not None:
                    loss_lpips_sds = 0.
                    for f_idx in range(len(sds_pred)):
                        sds_pred_single = sds_pred[f_idx]
                        sds_gt_single = sds_gt[f_idx]
                        sds_mask_single = sds_mask[f_idx].unsqueeze(0)
                    
                        bg_color_lpips_sds = torch.rand_like(sds_pred_single)
                        image_bg_sds = sds_pred_single * (1. - sds_mask_single) + bg_color_lpips_sds * sds_mask_single
                        gt_image_bg_sds = sds_gt_single * (1. - sds_mask_single) + bg_color_lpips_sds * sds_mask_single
                        _, pred_patches_sds, gt_patches_sds = self.patch_sampler.sample((1. - sds_mask_single), image_bg_sds, gt_image_bg_sds)
                        
                        loss_lpips_sds += self.lpips(pred_patches_sds.clip(max=1), gt_patches_sds).mean()
                    loss_dict['lpips_patch_sds'] = self.l_lpips_w * (loss_lpips_sds / len(sds_pred)) * self.l_sds_w
                
                bg_color_lpips = torch.rand_like(pred_img)
                image_bg = pred_img * mask + bg_color_lpips * (1. - mask)
                gt_image_bg = gt_image * mask + bg_color_lpips * (1. - mask)
                _, pred_patches, gt_patches = self.patch_sampler.sample(mask, image_bg, gt_image_bg)
                    
                loss_lpips = self.lpips(pred_patches.clip(max=1), gt_patches).mean()
                loss_dict['lpips_patch'] = self.l_lpips_w * loss_lpips
            except:
                pass

        if self.l_lbs_w > 0.0 and human_gs_out['lbs_weights'] is not None and not render_mode == "scene":
            if 'gt_lbs_weights' in human_gs_out.keys():
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_out['gt_lbs_weights'].detach()).mean()
            else:
                loss_lbs = F.mse_loss(
                    human_gs_out['lbs_weights'], 
                    human_gs_init_values['lbs_weights']).mean()
            loss_dict['lbs'] = self.l_lbs_w * loss_lbs
        
        loss = 0.0
        for k, v in loss_dict.items():
            loss = loss + v
        
        return loss, loss_dict, extras_dict
    