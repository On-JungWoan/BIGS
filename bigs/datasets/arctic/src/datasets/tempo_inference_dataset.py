import os.path as op

import json
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

# from src.datasets.tempo_dataset import TempoDataset
import common.ld_utils as ld_utils
import src.datasets.dataset_utils as dataset_utils
from src.datasets.arctic_dataset import ArcticDataset

from environments import DATASET_ROOT

def create_windows(imgnames, window_size):
    def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        my_chunks = [lst[i : i + n] for i in range(0, len(lst), n)]
        if len(my_chunks[-1]) == n:
            return my_chunks
        last_chunk = my_chunks[-1]
        last_element = last_chunk[-1]
        last_chunk_pad = [last_element for _ in range(n)]
        for idx, element in enumerate(last_chunk):
            last_chunk_pad[idx] = element
        my_chunks[-1] = last_chunk_pad
        return my_chunks

    img_seq_dict = {}
    for imgname in imgnames:
        sid, seq_name, view_idx, _ = imgname.split("/")[-4:]
        seq_name = "/".join([sid, seq_name, view_idx])
        if seq_name not in img_seq_dict.keys():
            img_seq_dict[seq_name] = []
        img_seq_dict[seq_name].append(imgname)

    windows = []
    for seq_name in img_seq_dict.keys():
        windows.append(chunks(sorted(img_seq_dict[seq_name]), window_size))

    windows = sum(windows, [])
    return windows


class TempoInferenceDataset(ArcticDataset):
    def _load_data(self, args, split):
        self.aug_data = False
        # self.window_size = args.window_size
        self.window_size = args.batch_size

    def __init__(self, args, split, seq=None):        
        Dataset.__init__(self)
        
        super()._load_data(args, split, seq)
        self._load_data(args, split)
        super()._process_imgnames(seq, split)

        windows = create_windows(self.imgnames, self.window_size)

        self.windows = windows
        num_imgnames = len(sum(self.windows, []))
        logger.info(
            f"TempoInferDataset Loaded {self.split} split, num samples {num_imgnames}"
        )
        
        # for droid slam
        seqname = '/'.join(self.windows[0][0].replace(
            "/arctic_data/", "/data/arctic_data/data/"
        ).replace("/data/data/", "/data/").replace('./', '').split('/')[:-1])
        seqname = args.dataset_path + '/' + seqname.replace('images', 'samples').replace('/1', '/1_shape_of_motion')
        if op.isdir(seqname):
            
            tgt_seqname = op.join(seqname, 'images/mano_tgt/hold_fit.init.npy')
            self.tgt_data = np.load(tgt_seqname, allow_pickle=True).item()
            
            # intrinsics
            hamer_seqname = op.join(seqname, 'images/processed/v3d.npy')
            self.my_intris_mat = np.load(hamer_seqname, allow_pickle=True).item()['K']
            
            # extrinsics
            self.my_world2cam = np.eye(4)
            
            # masks
            self.mask_dir = op.join(seqname, 'masks')
        else:
            logger.warning(f'There is no dir : {seqname}')

    def __getitem__(self, index):
        imgnames = self.windows[index]
        targets_list = []

        for imgname in imgnames:
            targets = self.getitem(imgname, index)
            targets_list.append(targets)

        targets_list = ld_utils.stack_dl(
            ld_utils.ld2dl(targets_list), dim=0, verbose=False
        )
        targets_list["is_valid"] = torch.FloatTensor(np.array(targets_list["is_valid"]))
        targets_list["left_valid"] = torch.FloatTensor(
            np.array(targets_list["left_valid"])
        )
        targets_list["right_valid"] = torch.FloatTensor(
            np.array(targets_list["right_valid"])
        )
        return targets_list

    def __len__(self):
        return len(self.windows)
