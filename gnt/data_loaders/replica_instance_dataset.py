import random
import time
import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset
import glob
from PIL import Image
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from .utils.base_utils import downsample_gaussian_blur

def set_seed(index,is_train):
    if is_train:
        np.random.seed((index+int(time.time()))%(2**16))
        random.seed((index+int(time.time()))%(2**16)+1)
        torch.random.manual_seed((index+int(time.time()))%(2**16)+1)
    else:
        np.random.seed(index % (2 ** 16))
        random.seed(index % (2 ** 16) + 1)
        torch.random.manual_seed(index % (2 ** 16) + 1)

# only for validation
class ReplicaInsDataset(Dataset):
    def __init__(self, args, is_train, scenes=None, **kwargs):
        self.is_train = is_train
        self.num_source_views = args.num_source_views
        self.rectify_inplane_rotation = args.rectify_inplane_rotation

        image_size = 320
        self.ratio = image_size / 640
        self.h, self.w = int(self.ratio*480), int(image_size)

        scene_path = os.path.join(args.rootdir + 'data/Replica_DM', scenes)
        poses = np.loadtxt(f'{scene_path}/traj_w_c.txt',delimiter=' ').reshape(-1, 4, 4).astype(np.float32)

        self.img_num = len(poses)
        rgb_files = [os.path.join(scene_path, 'rgb', f'rgb_{idx}.png') for idx in range(self.img_num)]
                
        depth_files = [f.replace("rgb", "depth") for f in rgb_files]
        label_files = [f.replace("rgb", "semantic_instance") for f in rgb_files]

        index = np.arange(len(rgb_files))
        self.rgb_files = np.array(rgb_files, dtype=object)[index]
        self.depth_files = np.array(depth_files, dtype=object)[index]
        self.label_files = np.array(label_files, dtype=object)[index]
        self.poses = np.array(poses)[index]
        
        que_idxs = np.arange(len(self.rgb_files))
        self.train_que_idxs = que_idxs[:900:2]
        self.val_que_idxs = que_idxs[:900:20]

    def __len__(self):
        if self.is_train is True:
            return len(self.train_que_idxs)
        else:  
            return len(self.val_que_idxs)  
    
    def __getitem__(self, idx):
        set_seed(idx, is_train=self.is_train)
        if self.is_train is True:
            que_idx = self.train_que_idxs[idx]
        else:
            que_idx = self.val_que_idxs[idx]

        rgb_files = self.rgb_files
        depth_files = self.depth_files
        scene_poses = self.poses
        label_files = self.label_files

        render_pose = scene_poses[que_idx]

        subsample_factor = np.random.choice(np.arange(1, 6), p=[0.3, 0.25, 0.2, 0.2, 0.05])

        id_feat_pool = get_nearest_pose_ids(
            render_pose,
            scene_poses,
            self.num_source_views * subsample_factor,
            tar_id=que_idx,
            angular_dist_method="vector",
        )
        id_feat = np.random.choice(id_feat_pool, self.num_source_views, replace=False)

        if que_idx in id_feat:
            assert que_idx not in id_feat
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]):
            id_feat[np.random.choice(len(id_feat))] = que_idx

        img = Image.open(depth_files[que_idx])
        depth = np.asarray(img, dtype=np.float32) / 1000.0  # mm -> m
        depth = np.ascontiguousarray(depth, dtype=np.float32)
        depth = cv2.resize(depth, (self.w, self.h), interpolation=cv2.INTER_NEAREST)

        rgb = imageio.imread(rgb_files[que_idx]).astype(np.float32) / 255.0

        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            
        intrinsics = np.array([320.0, 0.0, 320.0, 0.0,
                                0.0, 319.5, 229.5, 0.0,
                                0.0, 0.0, 1.0, 0.0,
                                0.0, 0.0, 0.0, 1.0]).reshape(4,4)
        intrinsics[:2, :] *= self.ratio

        img_size = rgb.shape[:2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(
            np.float32
        )

        img = Image.open(label_files[que_idx])
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)

        all_poses = [render_pose]
        # get depth range
        # min_ratio = 0.1
        # origin_depth = np.linalg.inv(render_pose)[2, 3]
        # max_radius = 0.5 * np.sqrt(2) * 1.1
        # near_depth = max(origin_depth - max_radius, min_ratio * origin_depth)
        # far_depth = origin_depth + max_radius
        # depth_range = torch.tensor([near_depth, far_depth])
        depth_range = torch.tensor([0.1, 10.0])

        src_rgbs = []
        src_cameras = []
        for id in id_feat:
            src_rgb = imageio.imread(rgb_files[id]).astype(np.float32) / 255.0
            if self.w != 1296:
                src_rgb = cv2.resize(downsample_gaussian_blur(
                    src_rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
            pose = scene_poses[id]

            if self.rectify_inplane_rotation:
                pose, src_rgb = rectify_inplane_rotation(pose.reshape(4, 4), render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate((list(img_size), intrinsics.flatten(), pose.flatten())).astype(
                np.float32
            )
            src_cameras.append(src_camera)
            all_poses.append(pose)

        src_rgbs = np.stack(src_rgbs)
        src_cameras = np.stack(src_cameras)

        return {
            "rgb": torch.from_numpy(rgb),
            "true_depth": torch.from_numpy(depth),
            "labels": torch.from_numpy(label),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_files[que_idx],
            "src_rgbs": torch.from_numpy(src_rgbs),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }