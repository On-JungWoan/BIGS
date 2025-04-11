import torch
import common.camera as camera
import common.data_utils as data_utils
import common.transforms as tf
import src.callbacks.process.process_generic as generic
from common.xdict import xdict
from pytorch3d.transforms import matrix_to_axis_angle, rotation_6d_to_matrix

def process_data(
    models, inputs, targets, meta_info, mode, args, field_max=float("inf")
):
    # img_res = 224
    K = meta_info["intrinsics"]
    gt_pose_r = targets["mano.pose.r"]  # MANO pose parameters
    gt_betas_r = targets["mano.beta.r"]  # MANO beta parameters

    gt_pose_l = targets["mano.pose.l"]  # MANO pose parameters
    gt_betas_l = targets["mano.beta.l"]  # MANO beta parameters

    gt_kp2d_b = targets["object.kp2d.norm.b"]  # 2D keypoints for object base
    gt_object_rot = targets["object.rot"].view(-1, 3)

    # pose the object without translation (call it object cano space)
    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )
    diameters = out["diameter"]
    parts_idx = out["parts_ids"]
    meta_info["part_ids"] = parts_idx
    meta_info["diameter"] = diameters

    # targets keypoints of hand and objects are in camera coord (full resolution image) space
    # map all entities from camera coord to object cano space based on the rigid-transform
    # between the object base keypoints in camera coord and object cano space
    # since R, T is used, relative distance btw hand and object is preserved
    num_kps = out["kp3d"].shape[1] // 2
    kp3d_b_cano = out["kp3d"][:, num_kps:]
    R0, T0 = tf.batch_solve_rigid_tf(targets["object.kp3d.full.b"], kp3d_b_cano)
    joints3d_r0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.r"], R0, T0)
    joints3d_l0 = tf.rigid_tf_torch_batch(targets["mano.j3d.full.l"], R0, T0)

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r[:, 3:],
        global_orient=gt_pose_r[:, :3],
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l[:, 3:],
        global_orient=gt_pose_l[:, :3],
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints
    gt_vertices_l = gt_out_l.vertices
    gt_root_cano_l = gt_out_l.joints[:, 0]

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_model_joints_l = joints3d_l0
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]

    # now that everything is in the object canonical space
    # find camera translation for rendering relative to the object

    # unnorm 2d keypoints
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        meta_info["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]
    gt_model_joints_l = gt_model_joints_l + gt_transl[:, None, :]

    ####
    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_root_cam_patch_l = gt_model_joints_l[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    gt_cam_t_o = gt_transl

    targets["mano.cam_t.r"] = gt_cam_t_r
    targets["mano.cam_t.l"] = gt_cam_t_l
    targets["object.cam_t"] = gt_cam_t_o

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    gt_cam_t_wp_o = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    targets["mano.cam_t.wp.r"] = gt_cam_t_wp_r
    targets["mano.cam_t.wp.l"] = gt_cam_t_wp_l
    targets["object.cam_t.wp"] = gt_cam_t_wp_o

    # cam coord of patch
    targets["object.cam_t.kp3d.b"] = gt_transl

    targets["mano.v3d.cam.r"] = gt_vertices_r
    targets["mano.v3d.cam.l"] = gt_vertices_l
    targets["mano.j3d.cam.r"] = gt_model_joints_r
    targets["mano.j3d.cam.l"] = gt_model_joints_l
    targets["object.kp3d.cam"] = gt_kp3d_o
    targets["object.bbox3d.cam"] = gt_bbox3d_o

    out = models["arti_head"].object_tensors.forward(
        angles=targets["object.radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=meta_info["query_names"],
    )

    # GT vertices relative to right hand root
    targets["object.v.cam"] = out["v"] + gt_transl[:, None, :]
    targets["object.v_len"] = out["v_len"]

    targets["object.f"] = out["f"]
    targets["object.f_len"] = out["f_len"]

    targets = generic.prepare_interfield(targets, field_max)

    return inputs, targets, meta_info

def custom_process_data(
    models, targets, field_max=float("inf")
):
    img_res = targets['img'].shape[-1]
    K = targets["intrinsics"]
    gt_pose_r = targets['add']['pose']  # MANO pose parameters
    gt_rot_r = targets['add']['rot']  # MANO rot parameters
    gt_betas_r = targets['add']['shape']  # MANO beta parameters

    gt_pose_l = targets['anchor']['pose']  # MANO pose parameters
    gt_rot_l = targets['anchor']['rot']  # MANO rot parameters
    gt_betas_l = targets['anchor']['shape']  # MANO beta parameters

    gt_kp2d_b = targets["object"]["kp2d.norm.b"]  # 2D keypoints for object base
    gt_object_rot = targets["object"]["rot"].view(-1, 3)

    # pose the object without translation (call it object cano space)
    out = models["arti_head"].object_tensors.forward(
        angles=targets["object"]["radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=targets["query_names"],
    )
    diameters = out["diameter"]
    parts_idx = out["parts_ids"]
    targets["object"]["part_ids"] = parts_idx
    targets["object"]["diameter"] = diameters
    targets['object']['joint3d'] = out["kp3d"].clone()

    # targets keypoints of hand and objects are in camera coord (full resolution image) space
    # map all entities from camera coord to object cano space based on the rigid-transform
    # between the object base keypoints in camera coord and object cano space
    # since R, T is used, relative distance btw hand and object is preserved
    num_kps = out["kp3d"].shape[1] // 2
    kp3d_b_cano = out["kp3d"][:, num_kps:]
    R0, T0 = tf.batch_solve_rigid_tf(targets["object"]["kp3d.full.b"], kp3d_b_cano)
    joints3d_r0 = tf.rigid_tf_torch_batch(targets["add"]["j3d.full"], R0, T0)
    joints3d_l0 = tf.rigid_tf_torch_batch(targets["anchor"]["j3d.full"], R0, T0)

    # pose MANO in MANO canonical space
    gt_out_r = models["mano_r"](
        betas=gt_betas_r,
        hand_pose=gt_pose_r,
        global_orient=gt_rot_r,
        transl=None,
    )
    gt_model_joints_r = gt_out_r.joints
    gt_vertices_r = gt_out_r.vertices
    gt_root_cano_r = gt_out_r.joints[:, 0]
    targets['add']['joint3d'] = gt_model_joints_r.clone()

    gt_out_l = models["mano_l"](
        betas=gt_betas_l,
        hand_pose=gt_pose_l,
        global_orient=gt_rot_l,
        transl=None,
    )
    gt_model_joints_l = gt_out_l.joints
    gt_vertices_l = gt_out_l.vertices
    gt_root_cano_l = gt_out_l.joints[:, 0]
    targets['anchor']['joint3d'] = gt_model_joints_l.clone()

    # map MANO mesh to object canonical space
    Tr0 = (joints3d_r0 - gt_model_joints_r).mean(dim=1)
    Tl0 = (joints3d_l0 - gt_model_joints_l).mean(dim=1)
    gt_model_joints_r = joints3d_r0
    gt_model_joints_l = joints3d_l0
    gt_vertices_r += Tr0[:, None, :]
    gt_vertices_l += Tl0[:, None, :]

    # now that everything is in the object canonical space
    # find camera translation for rendering relative to the object

    # unnorm 2d keypoints
    gt_kp2d_b_cano = data_utils.unormalize_kp2d(gt_kp2d_b, img_res)

    # estimate camera translation by solving 2d to 3d correspondence
    gt_transl = camera.estimate_translation_k(
        kp3d_b_cano,
        gt_kp2d_b_cano,
        targets["intrinsics"].cpu().numpy(),
        use_all_joints=True,
        pad_2d=True,
    )

    # move to camera coord
    gt_vertices_r = gt_vertices_r + gt_transl[:, None, :]
    gt_vertices_l = gt_vertices_l + gt_transl[:, None, :]
    gt_model_joints_r = gt_model_joints_r + gt_transl[:, None, :]
    gt_model_joints_l = gt_model_joints_l + gt_transl[:, None, :]

    ####
    gt_kp3d_o = out["kp3d"] + gt_transl[:, None, :]
    gt_bbox3d_o = out["bbox3d"] + gt_transl[:, None, :]

    # roots
    gt_root_cam_patch_r = gt_model_joints_r[:, 0]
    gt_root_cam_patch_l = gt_model_joints_l[:, 0]
    gt_cam_t_r = gt_root_cam_patch_r - gt_root_cano_r
    gt_cam_t_l = gt_root_cam_patch_l - gt_root_cano_l
    gt_cam_t_o = gt_transl

    targets['add']['trans'] = gt_cam_t_r
    targets['anchor']['trans'] = gt_cam_t_l
    targets["object"]['trans'] = gt_cam_t_o

    avg_focal_length = (K[:, 0, 0] + K[:, 1, 1]) / 2.0
    gt_cam_t_wp_r = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_r, avg_focal_length, img_res
    )

    gt_cam_t_wp_l = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_l, avg_focal_length, img_res
    )

    gt_cam_t_wp_o = camera.perspective_to_weak_perspective_torch(
        gt_cam_t_o, avg_focal_length, img_res
    )

    targets['add']['trans_wp'] = gt_cam_t_wp_r
    targets['anchor']['trans_wp'] = gt_cam_t_wp_l
    targets["object"]['trans_wp'] = gt_cam_t_wp_o

    # cam coord of patch
    targets["object"]["cam_t.kp3d.b"] = gt_transl

    targets["add"]["v3d.cam"] = gt_vertices_r
    targets["anchor"]["v3d.cam"] = gt_vertices_l
    targets["add"]["j3d.cam"] = gt_model_joints_r
    targets["anchor"]["j3d.cam"] = gt_model_joints_l
    targets["object"]["kp3d.cam"] = gt_kp3d_o
    targets["object"]["bbox3d.cam"] = gt_bbox3d_o

    out = models["arti_head"].object_tensors.forward(
        angles=targets["object"]["radian"].view(-1, 1),
        global_orient=gt_object_rot,
        transl=None,
        query_names=targets["query_names"],
    )

    # GT vertices relative to right hand root
    targets["object"]["v.cam"] = out["v"] + gt_transl[:, None, :]
    targets["object"]["v_len"] = out["v_len"]

    targets["object"]["f"] = out["f"]
    targets["object"]["f_len"] = out["f_len"]

    targets = generic.prepare_interfield(targets, field_max)

    return targets


def get_arctic_item(outputs, cfg, device='cuda'):
    out_logits = outputs['pred_logits']
    hand_cam, obj_cam = outputs['pred_cams']
    mano_rot6d, mano_pose, mano_shape = outputs['pred_mano_params']
    out_obj_rad, out_obj_rot = outputs['pred_obj_params']
    prob = out_logits.sigmoid()
    bs, _, _ = prob.shape

    # query index select
    best_score = torch.zeros(bs).to(device).to(prob.dtype)
    # if dataset != 'AssemblyHands':
    obj_idx = torch.zeros(bs).to(device).to(torch.long)
    for i in range(1, cfg.hand_idx[0]):
        score, idx = torch.max(prob[:,:,i], dim=-1)
        obj_idx[best_score < score] = idx[best_score < score]
        best_score[best_score < score] = score[best_score < score]

    hand_idx = []
    for i in cfg.hand_idx:
        hand_idx.append(torch.argmax(prob[:,:,i], dim=-1)) 
    # hand_idx = torch.stack(hand_idx, dim=-1) 
    left_hand_idx, right_hand_idx = hand_idx

    # extract cam
    root_l = torch.gather(hand_cam, 1, left_hand_idx.view(-1,1,1).repeat(1,1,3))[:, 0, :].to(torch.float32)
    root_r = torch.gather(hand_cam, 1, right_hand_idx.view(-1,1,1).repeat(1,1,3))[:, 0, :].to(torch.float32)
    root_o = torch.gather(obj_cam, 1, obj_idx.view(-1,1,1).repeat(1,1,3))[:, 0, :].to(torch.float32)

    mano_rot6d_l = torch.gather(mano_rot6d, 1, left_hand_idx.view(-1,1,1).repeat(1,1,6))[:, 0, :].to(torch.float32)
    mano_rot6d_r = torch.gather(mano_rot6d, 1, right_hand_idx.view(-1,1,1).repeat(1,1,6))[:, 0, :].to(torch.float32)
    mano_pose_l = torch.gather(mano_pose, 1, left_hand_idx.view(-1,1,1).repeat(1,1,45))[:, 0, :].to(torch.float32)
    mano_pose_r = torch.gather(mano_pose, 1, right_hand_idx.view(-1,1,1).repeat(1,1,45))[:, 0, :].to(torch.float32)
    mano_shape_l = torch.gather(mano_shape, 1, left_hand_idx.view(-1,1,1).repeat(1,1,10))[:, 0, :].to(torch.float32)
    mano_shape_r = torch.gather(mano_shape, 1, right_hand_idx.view(-1,1,1).repeat(1,1,10))[:, 0, :].to(torch.float32)
    
    obj_rot = torch.gather(out_obj_rot, 1, obj_idx.view(-1,1,1).repeat(1,1,6))[:, 0, :].to(torch.float32)
    obj_rad = torch.gather(out_obj_rad, 1, obj_idx.view(-1,1,1).repeat(1,1,1))[:, 0, :].to(torch.float32)

    return [root_l, root_r, root_o], [mano_rot6d_l, mano_pose_l, mano_rot6d_r, mano_pose_r], [mano_shape_l, mano_shape_r], [obj_rot, obj_rad]


def post_process_detr_outputs(runner, outputs, data):
    # find querys
    root_lro, pose_lr, shape_lr, obj_rot_rad = get_arctic_item(outputs, runner.config.model, device=runner.device)
    root_l, root_r, root_o = root_lro
    rot_l, pose_l, rot_r, pose_r = pose_lr
    shape_l, shape_r = shape_lr
    obj_rot, obj_rad = obj_rot_rad
    
    # make outputs
    param_l = torch.cat([rot_l, pose_l, shape_l, root_l], dim=-1)
    param_r = torch.cat([rot_r, pose_r, shape_r, root_r], dim=-1)
    param_o = torch.zeros_like(param_r).float().to(param_r.device)
    param_o[:, :10] = torch.cat([obj_rad, obj_rot, root_o], dim=-1)
    
    return make_output(runner, torch.stack([param_l, param_r, param_o]), data)


def make_output(runner, params, data):
    mode_list = ['post_mano_r', 'post_mano_l', 'arti_head']
    
    out_param = xdict()
    for idx, param in enumerate(params):
        mode = mode_list[idx]
        out_param.merge(post_process(runner, param, data, mode))    
    return out_param

def post_process(runner, param, data, mode):
    # model settings
    model = runner.pre_process_models[mode]

    if mode == 'arti_head':
        rad = param[:, :1]
        rot = param[:, 1:7]
        trans = param[:, 7:10]
    else:
        rot = param[:, :6]
        pose = param[:, 6:51]
        shape = param[:, 51:61]
        trans = param[:, 61:64]
    # if param.shape[-1] == 61:
    #     if mode == 'arti_head':
    #         rad = param[:, :1]
    #         rot = param[:, 1:7]
    #         trans = param[:, 7:11]
    #     else:
    #         rot = param[:, :6]
    #         pose = param[:, 6:51]
    #         shape = param[:, 51:61]
    #         trans = param[:, 61:64]
    # elif param.shape[-1] == 64:
    #     if mode == 'arti_head':
    #         rad = param[:, :1]
    #     else:
    #         pose = param[:, :45]
    #         shape = param[:, 45:55]
    #     rot = matrix_to_axis_angle(rotation_6d_to_matrix(param[:, 55:61]))
    #     trans = param[:, 61:64]                
    # else:
    #     raise Exception("Something went wrong!")

    if mode == 'arti_head':
        output = model(
            rot=matrix_to_axis_angle(rotation_6d_to_matrix(rot)),
            angle=rad,
            query_names=data['query_names'],
            cam=trans,
            K=data['intrinsics'],
        )        
        kp3d = output["kp3d"]
        v = None
        joints3d_cam = output["kp3d.cam"]
        v3d_cam = output["v.cam"]
        joints2d = output["kp2d.norm"]
    else:
        output = model(
            rot=matrix_to_axis_angle(rotation_6d_to_matrix(rot)),
            pose=pose,
            shape=shape,
            K=data['intrinsics'],
            cam=trans,
        )
        kp3d = output["joints3d"]
        v = output["vertices"]
        joints3d_cam = output["j3d.cam"]
        v3d_cam = output["v3d.cam"]
        joints2d = output["j2d.norm"]
    
    # # translation
    # joints3d_cam = kp3d + trans[:, None, :]
    # v3d_cam = v + trans[:, None, :]
    # joints2d = tf.project2d_batch(data['intrinsics'], joints3d_cam)
    # joints2d = data_utils.normalize_kp2d(joints2d, data['img'].shape[-1])
    
    output = xdict()
    
    # origin param
    if mode == 'arti_head':
        mode = 'object'
        output[f"{mode}.rad"] = rad
    else:
        mode = mode.replace('post_', '')
        output[f"{mode}.pose"] = pose
        output[f"{mode}.shape"] = shape
    output[f"{mode}.trans"] = trans
    output[f"{mode}.rot"] = rot
    
    # output
    output[f"{mode}.joints3d"] = kp3d
    output[f"{mode}.vertices"] = v
    output[f"{mode}.j3d.cam"] = joints3d_cam
    output[f"{mode}.v3d.cam"] = v3d_cam
    output[f"{mode}.j2d.norm"] = joints2d

    return output