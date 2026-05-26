from copy import deepcopy
import math
import numpy as np
import os
import os.path as osp
import random

import kornia.augmentation as K

from util import utils
# from dataset.transform import *
import dataset.transform as trf

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

class SemiDataset(Dataset):
    def __init__(self, root, mode, valid_path=None, size=None, id_path=None, nsample=None):
        self.root = root
        self.mode = mode
        self.size = size
        
        self.ignore_label = 255
        self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
                              3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
                              7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
                              14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
                              18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                              28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}

        if mode == 'train_l' or mode == 'train_u': 
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines() 

            if mode == 'train_l' and nsample is not None: 
                self.ids *= math.ceil(nsample / len(self.ids)) 
                random.shuffle(self.ids)
                self.ids = self.ids[:nsample]
        else:
            with open(valid_path, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        image_path = self.ids[item]
        
        img = Image.open(osp.join(self.root, image_path.split(' ')[0])).convert('RGB')
        mask = Image.open(osp.join(self.root, image_path.split(' ')[1]))
        mask = np.array(mask)
        
        gt_copy = mask.copy()
        for key, value in self.id_to_trainid.items():
            gt_copy[mask == key] = value
        mask = Image.fromarray(gt_copy.astype(np.uint8))
        
        if self.mode == 'val' or self.mode == 'test':
            image_path = image_path.split(' ')[0]
            img, mask = trf.normalize(img, mask)
            
            return img, mask, image_path
        
        # -------------------- Weak Augmentation --------------------
        img_w, mask_w = self.apply_weak_augm(img, mask)
        img_w_norm, mask_w_norm = trf.normalize(img_w, mask_w)
        # ---------------------------------------------------------
        if self.mode == 'train_l':
            return img_w_norm, mask_w_norm, image_path
        
        # -------------------- Strong Augmentation --------------------
        img_s_norm, ignore_mask, cutmix_box = self.apply_strong_augm(img_w, mask_w)
        # ------------------------------------------------------------
        
        return img_w_norm, img_s_norm, ignore_mask, cutmix_box, image_path

    def __len__(self):
        return len(self.ids)
    
    
    def apply_weak_augm(self, img, mask):
        img, mask = trf.resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else self.ignore_label
        img, mask = trf.crop(img, mask, self.size, ignore_value)
        img, mask = trf.hflip(img, mask, p=0.5)
        
        return img, mask
        
    
    def apply_strong_augm(self, img, mask):
        if random.random() < 0.8:
            img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            
        img = transforms.RandomGrayscale(p=0.2)(img)
        img = trf.blur(img, p=0.5)
        cutmix_box = trf.obtain_cutmix_box(img.size[0], p=0.5)

        ignore_mask = np.zeros((self.size, self.size))
        img, ignore_mask = trf.normalize(img, ignore_mask) # img는 Tensor와 normalize, mask는 numpy to tensor형태로 변환

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = self.ignore_label # ignore_mask는 mask에서 254픽셀부분을 255로 변환
        
        # # --- any additional helper for augmentation ----------------------------------
        # # in dataset/transform.py
        # def random_scale(img, scale_range):
        #     scale = random.uniform(scale_range[0], scale_range[1])
        #     return F.interpolate(img.unsqueeze(0), scale_factor=scale, mode='bilinear',
        #                         align_corners=True)[0]

        # def random_gaussian_blur(img, prob):
        #     if random.random() < prob:
        #         return img.filter(ImageFilter.GaussianBlur(radius=random.random()*2))
        #     return img

        # def color_jitter(img, cfg):
        #     return torchvision.transforms.ColorJitter(cfg.brightness,
        #                                             cfg.contrast,
        #                                             cfg.saturation,
        #                                             cfg.hue)(img)
        return img, ignore_mask, cutmix_box
    
        

    
# class SemiDataset(Dataset):
#     def __init__(self, root, mode, valid_path=None, id_path=None, nsample=None):
#         self.root = root
#         self.mode = mode
        
#         path = id_path if 'train' in mode else valid_path
#         with open(path, 'r') as f:
#             self.ids = f.read().splitlines()
            
#         if mode == 'train_l' and nsample is not None: 
#             self.ids *= math.ceil(nsample / len(self.ids)) 
#             random.shuffle(self.ids)
#             self.ids = self.ids[:nsample]


#     def __getitem__(self, item):
#         image_mask_path = self.ids[item]
        
#         image_path, mask_path = image_mask_path.split(' ')[0], image_mask_path.split(' ')[1]
#         img = Image.open(osp.join(self.root, image_path)).convert('RGB')
#         mask = Image.open(osp.join(self.root, mask_path))
        
#         img = transforms.ToTensor()(img)
#         gt = torch.from_numpy(np.array(mask)).long()
        
#         return img, gt, image_path
            

#     def __len__(self):
#         return len(self.ids)


# class GPUAugmentation(nn.Module):
#     def __init__(self, size, ignore_label=255):
#         super().__init__()
#         self.size = size
#         self.ignore_label = ignore_label
        
#         self.id_map = torch.full((35, ), ignore_label, dtype=torch.long)
#         mapping = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 
#                    22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
#         for k, v in mapping.items():
#             self.id_map[k] = v
        
#         # Weak Augmentation (Resize + Crop + Flip)
#         self.weak_aug = K.AugmentationSequential(
#                     K.RandomResizedCrop(size=self.size, scale=(0.5, 2.0), p=1.0, resample='bilinear'),
#                     K.RandomHorizontalFlip(p=0.5),
#                     data_keys=["input", "mask"]
#                 )
        
#         # Strong Augmentation (Color + Blur)
#         self.strong_aug = K.AugmentationSequential(
#             K.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
#             K.RandomGrayscale(p=0.2),
#             K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
#             data_keys=["input"]
#         )
        
        
#     @torch.no_grad()
#     def map_id(self, mask):
#         mask = mask.long()
#         mask_mapped = self.id_map.to(mask.device)[mask]
#         return mask_mapped


#     @torch.no_grad()
#     def forward(self, img, mask, mode='train_l'):
#         mask = self.map_id(mask)
        
#         if mode == 'val':
#             mask = mask.squeeze(dim=1)
#             return img, mask
        
#         img, mask = self.weak_aug(img, mask.float())

#         if mode == 'train_l':
#             mask = mask.long()
#             return img, mask

#         img_s = self.strong_aug(img)

#         ignore_mask = torch.zeros_like(mask).long()
#         ignore_mask[mask == 254] = self.ignore_label
#         ignore_mask = ignore_mask.squeeze()
        
#         cutmix_box = dt.obtain_cutmix_box_gpu(batch_size=img_s.shape[0],
#                                               img_h=self.size[0],
#                                               img_w=self.size[1], 
#                                               device=img_s.device,
#                                               p=0.5)
        
#         return img, img_s, ignore_mask, cutmix_box