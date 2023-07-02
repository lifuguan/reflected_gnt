import os
import numpy as np
import imageio
import cv2
import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image
from .utils.base_utils import downsample_gaussian_blur
from .asset import *
from .semantic_utils import PointSegClassMapping

scannet_set = scannet_train_scans_320

class OrderRendererDataset(Dataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        if self.is_train == 'train':
            self.scene_path_list = scannet_set
        else:
            self.scene_path_list = scannet_single

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        self.all_rgb_files = []
        self.all_label_files = []
        self.all_pose_files = []
        for i, scene_path in enumerate(self.scene_path_list):
            scene_path = os.path.join('data', scene_path[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            self.all_rgb_files.append(rgb_files)
            self.all_label_files.append(label_files)
            self.all_pose_files.append(pose_files)

        self.all_rgb_files = np.concatenate(self.all_rgb_files)
        self.all_label_files = np.concatenate(self.all_label_files)

        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )

    def __len__(self):
        return len(self.all_rgb_files)
    
    def __getitem__(self, idx):
        rgb_file = self.all_rgb_files[idx]
        label_file = self.all_label_files[idx]

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0

        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        img = Image.open(label_file)
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.scan2nyu[label]
        label = self.label_mapping(label)
    
        return {
            "image": torch.as_tensor(rgb.copy()).float().contiguous(),
            "mask": torch.as_tensor(label.copy()).long().contiguous()
        }


class RandomRendererDataset(Dataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        if self.is_train == True:
            self.scene_path_list = scannet_set
        else:
            self.scene_path_list = scannet_set

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        all_rgb_files = []
        all_label_files = []
        all_pose_files = []
        for i, scene_path in enumerate(self.scene_path_list):
            scene_path = os.path.join('data', scene_path[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            all_rgb_files.append(rgb_files)
            all_label_files.append(label_files)
            all_pose_files.append(pose_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_label_files = np.array(all_label_files)[index]

        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )

    def __len__(self):
        return 999999
    
    def __getitem__(self, idx):
        real_idx = idx % len(self.all_rgb_files)
        rgb_files = self.all_rgb_files[real_idx]
        label_files = self.all_label_files[real_idx]
        id_render = np.random.choice(np.arange(len(rgb_files)))
        rgb_file = rgb_files[id_render]
        label_file = label_files[id_render]

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0

        if self.w != 1296:
            rgb = cv2.resize(downsample_gaussian_blur(
                rgb, self.ratio), (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        img = Image.open(label_file)
        label = np.asarray(img, dtype=np.int32)
        label = np.ascontiguousarray(label)
        label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        label = label.astype(np.int32)
        label = self.scan2nyu[label]
        label = self.label_mapping(label)
    
        return {
            "image": torch.as_tensor(rgb.copy()).float().contiguous(),
            "mask": torch.as_tensor(label.copy()).long().contiguous()
        }



class StatisticRendererDataset(Dataset):
    def __init__(self, is_train, **kwargs):
        self.is_train = is_train
        if self.is_train == True:
            self.scene_path_list = scannet_train_scans_320
        else:
            self.scene_path_list = scannet_train_scans_320

        image_size = 320
        self.ratio = image_size / 1296
        self.h, self.w = int(self.ratio*972), int(image_size)

        all_rgb_files = []
        all_label_files = []
        all_pose_files = []
        for i, scene_path in enumerate(self.scene_path_list):
            scene_path = os.path.join('data', scene_path[:-10])
            pose_files = []
            for f in sorted(os.listdir(os.path.join(scene_path, "pose"))):
                path = os.path.join(scene_path, "pose", f)
                pose_files.append(path)
                    
            rgb_files = [f.replace("pose", "color").replace("txt", "jpg") for f in pose_files]
            label_files = [f.replace("pose", "label-filt").replace("txt", "png") for f in pose_files]

            all_rgb_files.append(rgb_files)
            all_label_files.append(label_files)
            all_pose_files.append(pose_files)

        index = np.arange(len(all_rgb_files))
        self.all_rgb_files = np.array(all_rgb_files)[index]
        self.all_label_files = np.array(all_label_files)[index]

        mapping_file = 'data/scannet/scannetv2-labels.combined.tsv'
        mapping_file = pd.read_csv(mapping_file, sep='\t', header=0)
        scan_ids = mapping_file['id'].values
        nyu40_ids = mapping_file['nyu40id'].values
        scan2nyu = np.zeros(max(scan_ids) + 1, dtype=np.int32)
        for i in range(len(scan_ids)):
            scan2nyu[scan_ids[i]] = nyu40_ids[i]
        self.scan2nyu = scan2nyu
        self.label_mapping = PointSegClassMapping(
            valid_cat_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                           11, 12, 14, 16, 24, 28, 33, 34, 36, 39],
            max_cat_id=40
        )

    def __len__(self):
        return len(self.all_rgb_files)
    
    def __getitem__(self, idx):
        label_files = self.all_label_files[idx]
        color_dicts = { i:0 for i in range(21)}
        for label_file in label_files:
            img = Image.open(label_file)
            label = np.asarray(img, dtype=np.int32)
            label = np.ascontiguousarray(label)
            label = cv2.resize(label, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            label = label.astype(np.int32)
            label = self.scan2nyu[label]
            label = self.label_mapping(label)
            unique_colors, color_counts = np.unique(label, return_counts=True)
            for (unique_color, color_count) in zip(unique_colors, color_counts):
                color_dicts[unique_color] += color_count
    
        return color_dicts

