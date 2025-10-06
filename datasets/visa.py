import os
import cv2
import random
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from .utils import excluding_images

VISA_CLASS_NAMES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3',
    'pcb4', 'pipe_fryum'
]

class VisaDataset(Dataset):
    def __init__(self, c, is_train=True, excluded_images=None):
        assert c.class_name in VISA_CLASS_NAMES, 'class_name: {}, should be in {}'.format(c.class_name, VISA_CLASS_NAMES)
        self.dataset_path = c.data_path
        self.class_name = c.class_name
        self.is_train = is_train
        self.cropsize = c.crop_size
        # load dataset
        if excluded_images is not None:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
            self.x, self.y, self.mask, self.img_types = excluding_images(self.x, self.y, self.mask, self.img_types, excluded_images)
        else:
            self.x, self.y, self.mask, self.img_types = self.load_dataset_folder()
        # set transforms
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.ANTIALIAS),
                T.CenterCrop(c.crop_size),
                T.ToTensor()])
        # mask
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, img_type = self.x[idx], self.y[idx], self.mask[idx], self.img_types[idx]

        x = Image.open(img_path).convert('RGB')
        
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path[:-4]), img_type

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, types = [], [], [], []

        img_dir = os.path.join(self.dataset_path, self.class_name, phase)
        gt_dir = os.path.join(self.dataset_path, self.class_name, 'ground_truth')

        img_types = sorted(os.listdir(img_dir))
        for img_type in img_types:
            img_type_dir = os.path.join(img_dir, img_type)
            if not os.path.isdir(img_type_dir):
                continue
            img_fpath_list = sorted([os.path.join(img_type_dir, f)
                                     for f in os.listdir(img_type_dir)
                                     if f.endswith('.jpg')])
            x.extend(img_fpath_list)

            if img_type == 'good':
                y.extend([0] * len(img_fpath_list))
                mask.extend([None] * len(img_fpath_list))
                types.extend(['good'] * len(img_fpath_list))
            else:
                y.extend([1] * len(img_fpath_list))
                gt_type_dir = os.path.join(gt_dir, img_type)
                img_fname_list = [os.path.splitext(os.path.basename(f))[0] for f in img_fpath_list]
                gt_fpath_list = [os.path.join(gt_type_dir, img_fname + '.png')
                                 for img_fname in img_fname_list]
                mask.extend(gt_fpath_list)
                types.extend([img_type] * len(img_fpath_list))

        assert len(x) == len(y), 'number of x and y should be same'

        return list(x), list(y), list(mask), list(types)
