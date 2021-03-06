# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com), Yang Zhao
# ------------------------------------------------------------------------------

import os
import random

import torch
import torch.utils.data as data
import pandas as pd
from PIL import Image, ImageFile
import numpy as np
import math
from ..utils.transforms import fliplr_joints, crop, generate_target, transform_pixel

ImageFile.LOAD_TRUNCATED_IMAGES = True


class RACE(data.Dataset):
    """race_data
    """
    def __init__(self, cfg, is_train=True, transform=None):
        # specify annotation file for dataset
        self.is_train = is_train

        self.is_trainval = cfg.DATASET.TRAIN_VAL

        if is_train or self.is_trainval:
            self.data_root = cfg.DATASET.TRAINSET
        else:
            self.data_root = cfg.DATASET.TESTSET
        
        self.transform = transform
        # self.data_root = cfg.DATASET.ROOT
        self.input_size = cfg.MODEL.IMAGE_SIZE
        self.output_size = cfg.MODEL.HEATMAP_SIZE
        self.sigma = cfg.MODEL.SIGMA
        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rot_factor = cfg.DATASET.ROT_FACTOR
        self.label_type = cfg.MODEL.TARGET_TYPE
        self.flip = cfg.DATASET.FLIP
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # load annotations
        # self.landmarks_frame = pd.read_csv(self.csv_file)
        
        if self.is_trainval:
            self.img_list = []
            if self.is_train:
                with open(os.path.join(self.data_root, 'train.txt')) as f:
                    for line in f.readlines():
                        self.img_list.append(line.strip()+'.jpg')
            else:
                with open(os.path.join(self.data_root, 'val.txt')) as f:
                    for line in f.readlines():
                        self.img_list.append(line.strip()+'.jpg')
        else:
            self.img_list = os.listdir(os.path.join(self.data_root,'picture'))

    def __len__(self):
        return len(self.img_list)

    # def read_txt(self, path):


    def __getitem__(self, idx):

        image_path = os.path.join(self.data_root,
                                  'picture',self.img_list[idx])
        bbox_path = os.path.join(self.data_root,
                                  'bbox',self.img_list[idx].split('.')[0]) + '.txt'
        if self.is_train or self.is_trainval:
            lmk_path = os.path.join(self.data_root,
                                    'landmark',self.img_list[idx].split('.')[0]) + '.txt'
        bbox = []
        with open(bbox_path, "r") as f:
            for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    bbox.extend(line)
        bbox = [float(i) for i in bbox]

        if self.is_train or self.is_trainval:
            pts = []
            with open(lmk_path, "r") as f:
                for line in f.readlines():
                    if line != 0:
                        line = line.strip('\n').split(' ')
                        pts.extend(line)
            pts = [float(i) for i in pts[1:]]
        
        center_w = (math.floor(bbox[0]) + math.ceil(bbox[2])) / 2.0
        center_h = (math.floor(bbox[1]) + math.ceil(bbox[3])) / 2.0

        scale = max(math.ceil(bbox[2]) - math.floor(bbox[0]), math.ceil(bbox[3]) - math.floor(bbox[1])) / 200.0

        center = torch.Tensor([center_w, center_h])

        if self.is_train or self.is_trainval:
            pts = np.array(pts).astype('float').reshape(-1, 2)

        scale *= 1.25
        if self.is_train or self.is_trainval:
            nparts = pts.shape[0]
        img = np.array(Image.open(image_path).convert('RGB'), dtype=np.float32)

        r = 0
        if self.is_train or self.is_trainval:
            scale = scale * (random.uniform(1 - self.scale_factor,
                                            1 + self.scale_factor))
            r = random.uniform(-self.rot_factor, self.rot_factor) \
                if random.random() <= 0.6 else 0
            if random.random() <= 0.5 and self.flip:
                img = np.fliplr(img)
                pts = fliplr_joints(pts, width=img.shape[1], dataset='RACE')
                center[0] = img.shape[1] - center[0]

            target = np.zeros((nparts, self.output_size[0], self.output_size[1]))
            tpts = pts.copy()

            for i in range(nparts):
                if tpts[i, 1] > 0:
                    tpts[i, 0:2] = transform_pixel(tpts[i, 0:2]+1, center,
                                                scale, self.output_size, rot=r)
                    target[i] = generate_target(target[i], tpts[i]-1, self.sigma,
                                                label_type=self.label_type)

        img = crop(img, center, scale, self.input_size, rot=r)
        img = img.astype(np.float32)
        img = (img/255.0 - self.mean) / self.std
        img = img.transpose([2, 0, 1])
        if self.is_train or self.is_trainval:
            target = torch.Tensor(target)
            tpts = torch.Tensor(tpts)
        center = torch.Tensor(center)
        bbox = torch.Tensor(bbox)

        if self.is_train or self.is_trainval:
            meta = {'index': idx, 'center': center, 'scale': scale, 'bbox': bbox,
                    'pts': torch.Tensor(pts), 'tpts': tpts}
            return img, target, meta
        else:
            meta = {'index': idx, 'center': center, 'scale': scale, 'bbox': bbox,
                    'img_name': self.img_list[idx]}
            return img, meta


if __name__ == '__main__':
    pass
