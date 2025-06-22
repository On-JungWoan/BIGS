#!/bin/bash

conda activate bigs

seq_name=$1

python main.py \
    exp_name=test+joint4 \
    dataset.seq=$seq_name dataset.name=arctic \
    human.object_ckpt=output/arctic/$seq_name/000_release/002_obj/object/ckpt/human_final.pth \
    human.left_ckpt=output/arctic/$seq_name/000_release/001_hand/left/ckpt/human_final.pth \
    human.right_ckpt=output/arctic/$seq_name/000_release/001_hand/right/ckpt/human_final.pth \
    train.num_steps=30000 human.lr.lr_max_steps=30000 train.save_ckpt_interval=15000 \
    train.anim_interval=5000 human.num_refine=1 \
    human.lr.name="['transl']" \
    human.lr.lr_init=1.0e-3 human.lr.lr_final=1.0e-5 human.lr.smpl_spatial=2.0 \
    human.lr.smpl_trans=1.0e-3 human.lr.smpl_rot=1.0e-4 human.lr.smpl_pose=0.0 human.lr.smpl_scale=0.0 \
    human.lr.position=0.0 human.lr.deformation=0.0 human.lr.geometry=0.0 human.lr.appearance=0.0 human.lr.vembed=0.0 \
    human.loss.l2_offset=0.0 human.loss.l2_scale=0.0 human.loss.l1_shs=0.0 \
    human.isotropic=true human.num_refine=1 \
    human.densify_until_iter=-1 human.lr.use_optim_ckpt=false \
    components="['right', 'left', 'object']" \
    human.loss.transl_w=1.0 \
    train.joint_training=true