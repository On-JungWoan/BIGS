import os
import os.path as op
import re
from abc import abstractmethod

import matplotlib.cm as cm
import numpy as np
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.scene.material import Material
from aitviewer.utils.so3 import aa2rot_numpy
from aitviewer.viewer import Viewer
from easydict import EasyDict as edict
from loguru import logger
from PIL import Image
from tqdm import tqdm

import torch
from common.xdict import xdict
import common.viewer as viewer_utils
from common.body_models import build_layers, seal_mano_mesh

OBJ_ID = 100
SMPLX_ID = 150
LEFT_ID = 200
RIGHT_ID = 250
SEGM_IDS = {"object": OBJ_ID, "smplx": SMPLX_ID, "left": LEFT_ID, "right": RIGHT_ID}

cmap = cm.get_cmap("plasma")
materials = {
    "none": None,
    "white": Material(color=(1.0, 1.0, 1.0, 1.0), ambient=0.2),
    "red": Material(color=(0.969, 0.106, 0.059, 1.0), ambient=0.2),
    "blue": Material(color=(0.0, 0.0, 1.0, 1.0), ambient=0.2),
    "green": Material(color=(1.0, 0.0, 0.0, 1.0), ambient=0.2),
    "cyan": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "light-blue": Material(color=(0.588, 0.5647, 0.9725, 1.0), ambient=0.2),
    "cyan-light": Material(color=(0.051, 0.659, 0.051, 1.0), ambient=0.2),
    "dark-light": Material(color=(0.404, 0.278, 0.278, 1.0), ambient=0.2),
    "rice": Material(color=(0.922, 0.922, 0.102, 1.0), ambient=0.2),
}


class ViewerData(edict):
    """
    Interface to standardize viewer data.
    """

    def __init__(self, Rt, K, cols, rows, imgnames=None):
        self.imgnames = imgnames
        self.Rt = Rt
        self.K = K
        self.num_frames = Rt.shape[0]
        self.cols = cols
        self.rows = rows
        self.validate_format()

    def validate_format(self):
        assert len(self.Rt.shape) == 3
        assert self.Rt.shape[0] == self.num_frames
        assert self.Rt.shape[1] == 3
        assert self.Rt.shape[2] == 4

        assert len(self.K.shape) == 2
        assert self.K.shape[0] == 3
        assert self.K.shape[1] == 3
        if self.imgnames is not None:
            assert self.num_frames == len(self.imgnames)
            assert self.num_frames > 0
            im_p = self.imgnames[0]
            assert op.exists(im_p), f"Image path {im_p} does not exist"


class ARCTICViewer:
    def __init__(
        self,
        render_types=["rgb", "depth", "mask"],
        interactive=True,
        size=(2024, 2024),
    ):
        if not interactive:
            v = HeadlessRenderer()
        else:
            v = Viewer(size=size)

        self.v = v
        self.interactive = interactive
        # self.layers = layers
        self.render_types = render_types

    def view_interactive(self):
        self.v.run()

    def view_fn_headless(self, num_iter, out_folder, iter):
        v = self.v

        v._init_scene()

        logger.info("Rendering to video")
        if "video" in self.render_types:
            vid_p = op.join(out_folder, "video.mp4")
            v.save_video(video_dir=vid_p)

        pbar = tqdm(range(num_iter))
        for fidx in pbar:
            if iter is not None:
                rgb_name = f"rgb/{iter}_{fidx:04d}.png"
            else:
                rgb_name = f"rgb/{fidx:04d}.png"
            
            out_rgb = op.join(out_folder, "images", rgb_name)
            out_mask = op.join(out_folder, "images", f"mask/{fidx:04d}.png")
            out_depth = op.join(out_folder, "images", f"depth/{fidx:04d}.npy")

            # render RGB, depth, segmentation masks
            if "rgb" in self.render_types:
                v.export_frame(out_rgb)
            if "depth" in self.render_types:
                os.makedirs(op.dirname(out_depth), exist_ok=True)
                render_depth(v, out_depth)
            if "mask" in self.render_types:
                os.makedirs(op.dirname(out_mask), exist_ok=True)
                render_mask(v, out_mask)
            v.scene.next_frame()
        logger.info(f"Exported to {out_folder}")

    @abstractmethod
    def load_data(self):
        pass

    def check_format(self, batch):
        meshes_all, data = batch
        assert isinstance(meshes_all, dict)
        assert len(meshes_all) > 0
        for mesh in meshes_all.values():
            assert isinstance(mesh, Meshes)
        assert isinstance(data, ViewerData)

    def render_seq(self, batch, out_folder="./render_out", iter=None):
        meshes_all, data = batch
        self.setup_viewer(data)
        for mesh in meshes_all.values():
            self.v.scene.add(mesh)
        if self.interactive:
            self.view_interactive()
        else:
            num_iter = data["num_frames"]
            self.view_fn_headless(num_iter, out_folder, iter=iter)

    def setup_viewer(self, data):
        v = self.v
        fps = 30
        if "imgnames" in data:
            setup_billboard(data, v)

        # camera.show_path()
        v.run_animations = True  # autoplay
        v.run_animations = False  # autoplay
        v.playback_fps = fps
        v.scene.fps = fps
        v.scene.origin.enabled = False
        v.scene.floor.enabled = False
        v.auto_set_floor = False
        v.scene.floor.position[1] = -3
        # v.scene.camera.position = np.array((0.0, 0.0, 0))
        self.v = v


def dist2vc(dist_ro, dist_lo, dist_o, _cmap, tf_fn=None):
    if tf_fn is not None:
        exp_map = tf_fn
    else:
        exp_map = small_exp_map
    dist_ro = exp_map(dist_ro)
    dist_lo = exp_map(dist_lo)
    dist_o = exp_map(dist_o)

    vc_ro = _cmap(dist_ro)
    vc_lo = _cmap(dist_lo)
    vc_o = _cmap(dist_o)
    return vc_ro, vc_lo, vc_o


def small_exp_map(_dist):
    dist = np.copy(_dist)
    # dist = 1.0 - np.clip(dist, 0, 0.1) / 0.1
    dist = np.exp(-20.0 * dist)
    return dist


def construct_viewer_meshes(data, draw_edges=False, flat_shading=True):
    rotation_flip = aa2rot_numpy(np.array([1, 0, 0]) * np.pi)
    meshes = {}
    for key, val in data.items():
        if "object" in key:
            flat_shading = False
        else:
            flat_shading = flat_shading
        v3d = val["v3d"]
        meshes[key] = Meshes(
            v3d,
            val["f3d"],
            vertex_colors=val["vc"],
            name=val["name"],
            flat_shading=flat_shading,
            draw_edges=draw_edges,
            material=materials[val["color"]],
            rotation=rotation_flip,
        )
    return meshes


def setup_viewer(
    v, shared_folder_p, video, images_path, data, flag, seq_name, side_angle
):
    fps = 10

    # cols, rows = 224, 224
    focal = 1000.0

    # setup image paths
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    # setup billboard
    images_path = op.join(shared_folder_p, "images")
    images_paths = [
        os.path.join(images_path, f)
        for f in sorted(os.listdir(images_path), key=sort_key)
    ]
    assert len(images_paths) > 0

    cam_t = data[f"{flag}.object.cam_t"]
    num_frames = min(cam_t.shape[0], len(images_paths))
    cam_t = cam_t[:num_frames]
    # setup camera
    K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    Rt = np.zeros((num_frames, 3, 4))
    Rt[:, :, 3] = cam_t
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if side_angle is None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)
    v.run_animations = True  # autoplay
    v.playback_fps = fps
    v.scene.fps = fps
    v.scene.origin.enabled = False
    v.scene.floor.enabled = False
    v.auto_set_floor = False
    v.scene.floor.position[1] = -3
    v.set_temp_camera(camera)
    # v.scene.camera.position = np.array((0.0, 0.0, 0))
    return v


def render_depth(v, depth_p):
    depth = np.array(v.get_depth()).astype(np.float16)
    np.save(depth_p, depth)


def render_mask(v, mask_p):
    nodes_uid = {node.name: node.uid for node in v.scene.collect_nodes()}
    my_cmap = {
        uid: [SEGM_IDS[name], SEGM_IDS[name], SEGM_IDS[name]]
        for name, uid in nodes_uid.items()
        if name in SEGM_IDS.keys()
    }
    mask = np.array(v.get_mask(color_map=my_cmap)).astype(np.uint8)
    mask = Image.fromarray(mask)
    mask.save(mask_p)


def setup_billboard(data, v):
    images_paths = data.imgnames
    K = data.K
    Rt = data.Rt
    rows = data.rows
    cols = data.cols
    camera = OpenCVCamera(K, Rt, cols, rows, viewer=v)
    if images_paths is not None:
        billboard = Billboard.from_camera_and_distance(
            camera, 10.0, cols, rows, images_paths
        )
        v.scene.add(billboard)
    v.scene.add(camera)
    v.scene.camera.load_cam()
    v.set_temp_camera(camera)


def construct_meshes(data, out, device):
    # load object faces
    f3d_o = data['object']['f'][0]
    layers = build_layers(device)

    # center verts
    v3d_r = out['mano_r.v3d.cam']
    v3d_l = out['mano_l.v3d.cam']
    v3d_o = out['object.v3d.cam']
    cam_t = out['object.trans']
    # v3d_r = data['add']['v3d.cam']
    # v3d_l = data['anchor']['v3d.cam']
    # v3d_o = data['object']['v.cam']
    # cam_t = data['object']['trans']
    v3d_r -= cam_t[:, None, :] 
    v3d_l -= cam_t[:, None, :] 
    v3d_o -= cam_t[:, None, :] 

    # seal MANO mesh
    f3d_r = torch.LongTensor(layers["right"].faces.astype(np.int64))
    f3d_l = torch.LongTensor(layers["left"].faces.astype(np.int64))
    v3d_r, f3d_r = seal_mano_mesh(v3d_r, f3d_r, True)
    v3d_l, f3d_l = seal_mano_mesh(v3d_l, f3d_l, False)

    # AIT meshes
    right = {
        "v3d": v3d_r.cpu().numpy(),
        "f3d": f3d_r.cpu().numpy(),
        "vc": None,
        "name": "right",
        "color": "light-blue",
    }
    left = {
        "v3d": v3d_l.cpu().numpy(),
        "f3d": f3d_l.cpu().numpy(),
        "vc": None,
        "name": "left",
        "color": "red",
    }
    obj = {
        "v3d": v3d_o.cpu().numpy(),
        "f3d": f3d_o.cpu().numpy(),
        "vc": None,
        "name": "object",
        "color": "white",
    }

    meshes = viewer_utils.construct_viewer_meshes(
        {
            "right": right,
            "left": left,
            "object": obj,
        },
        draw_edges=False,
        flat_shading=True,
    )
    return meshes, data

def visualize_arctic_result(args, data, out, iter=None):
    # set viewer
    args.headless = True # aitviewer 작동 안해서 interactive 안됨
    viewer = ARCTICViewer(
        interactive=not args.headless,
        # size=(2048, 2048),
        size=(1000, 1000),
        render_types=["rgb", "video"],
    )
    imgnames = data['imgname']
    imgnames = [
                    args.data.dataset_path + \
                    name.replace('./', '/').replace('arctic_data/', 'data/arctic_data/data/') \
                        for name in imgnames
                ]
    num_frames = len(imgnames)

    # construct meshes
    meshes_all = xdict()
    meshes, data = construct_meshes(data, out, 'cuda')
    meshes_all.merge(meshes)
    

    ##################
    ## setup camera ##
    ##################
    
    ## intrinsic params ##
    # focal = 1000.0
    # rows = data['meta_info.center'][0][0] * 2
    # cols = data['meta_info.center'][0][1] * 2
    # K = np.array([[focal, 0, rows / 2.0], [0, focal, cols / 2.0], [0, 0, 1]])
    

    rows = data['img'].shape[-2]
    cols = data['img'].shape[-1]
    K = data['intrinsics'][0].cpu().numpy()
    
    ## extrinsic params ##
    # cam_t = data['extrinsics'][:, :-1, 3].cpu().numpy()
    # cam_t = out['object.trans']
    # cam_t = cam_t[:num_frames].cpu().numpy()
    Rt = np.zeros((num_frames, 3, 4))
    # Rt[:, :, 3] = cam_t
    Rt[:, -1, 3] = 1.
    Rt[:, :3, :3] = np.eye(3)
    Rt[:, 1:3, :3] *= -1.0

    data = ViewerData(Rt=Rt, K=K, cols=cols, rows=rows, imgnames=imgnames)
    batch = meshes_all, data

    exp_name = '_'.join(args.model_path.split('/')[-2:]).split('.')[0]
    save_foler = op.join(f'{args.log_path}/vis_{exp_name}')
    if not op.isdir(save_foler):
        os.makedirs(save_foler, exist_ok=True)

    viewer.check_format(batch)
    viewer.render_seq(batch, out_folder=save_foler, iter=iter)
    
    # destrory windows
    viewer.v.on_close()
    viewer.v.window.close()
    viewer.v.window.destroy()