from copy import deepcopy
import math
import numpy as np
import os
import os.path as osp
import random

import kornia.augmentation as K

from util import utils
from dataset.transform import *

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn as nn

# class SemiDataset(Dataset):
#     def __init__(self, name, root, mode, vis_mask=False, valid_path=None, size=None, id_path=None, nsample=None):
#         self.name = name
#         self.root = root
#         self.mode = mode
#         self.size = size
        
#         self.vis_mask = vis_mask
        
#         self.ignore_label = 255
#         self.id_to_trainid = {-1: self.ignore_label, 0: self.ignore_label, 1: self.ignore_label, 2: self.ignore_label,
#                               3: self.ignore_label, 4: self.ignore_label, 5: self.ignore_label, 6: self.ignore_label,
#                               7: 0, 8: 1, 9: self.ignore_label, 10: self.ignore_label, 11: 2, 12: 3, 13: 4,
#                               14: self.ignore_label, 15: self.ignore_label, 16: self.ignore_label, 17: 5,
#                               18: self.ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
#                               28: 15, 29: self.ignore_label, 30: self.ignore_label, 31: 16, 32: 17, 33: 18}

#         if mode == 'train_l' or mode == 'train_u': # mode가 train_l이거나 train_u이면 self.mode로 써도 되지 않앗나?
#             with open(id_path, 'r') as f:
#                 self.ids = f.read().splitlines() # 텍스트파일 한 줄(enter) 씩 리스트로 반환
#             if mode == 'train_l' and nsample is not None: # train_l 모드이고 nsample이 있으면
#                 self.ids *= math.ceil(nsample / len(self.ids)) # nsample을 self.ids 길이로 나눈 값 반올림 한 값으로 반복
#                 random.shuffle(self.ids) # ids 랜덤 셔플링
#                 self.ids = self.ids[:nsample] # self.ids는 nsample 까지의 리스트로 재정의
#         else:
#             with open(valid_path, 'r') as f:
#                 self.ids = f.read().splitlines()

#     def __getitem__(self, item):
#         image_path = self.ids[item]
        
#         # ------------------- 데이터 읽기-------------------
#         img = Image.open(osp.join(self.root, image_path.split(' ')[0])).convert('RGB')
#         mask = Image.open(osp.join(self.root, image_path.split(' ')[1]))
#         mask = np.array(mask)
        
#         gt_copy = mask.copy()
#         for key, value in self.id_to_trainid.items():
#             gt_copy[mask == key] = value
#         mask = Image.fromarray(gt_copy.astype(np.uint8))
        
#         if self.vis_mask: 
#             vis_mask = utils.colorize_mask(mask)
            
#         if self.mode == 'val':
#             image_path = image_path.split(' ')[0]
            
#             img_ori = np.array(img) 
#             img, mask = normalize(img, mask)
            
#             return img, mask, image_path, img_ori
#         # ---------------------------------------------------------
        
#         # -------------------- Weak Augmentation --------------------
#         img, mask = resize(img, mask, (0.5, 2.0)) # 0.5와 2.0의 ratio를 적용한 랜덤 정수값으로 resize
#         ignore_value = 254 if self.mode == 'train_u' else self.ignore_label # unlabeld 데이터셋일때는 ignore_value=254로 대입, else ignore_v=255
#         img, mask = crop(img, mask, self.size, ignore_value) # 801-w, 801-h 를 하여 패딩설정후, 좌우를 0으로 채움
#         img, mask = hflip(img, mask, p=0.5)

#         if self.mode == 'train_l':
#             image, mask = normalize(img, mask) 
#             return image, mask, image_path
#         # ---------------------------------------------------------
        
#         img_w, img_s = deepcopy(img), deepcopy(img)
#         img_w = normalize(img_w)
        
#         # -------------------- Strong Augmentation --------------------
#         if random.random() < 0.8:
#             img_s = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s) # 색변환 80%확률 strong aug에 적용
#         img_s = transforms.RandomGrayscale(p=0.2)(img_s) # 회색조 20% 확률로 추가
#         img_s = blur(img_s, p=0.5) # 블러처리 50%확률
#         cutmix_box = obtain_cutmix_box(img_s.size[0], p=0.5)

#         ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0]))) # mask사이즈의 0으로 된 array를 Pil image로 재구성
#         img_s, ignore_mask = normalize(img_s, ignore_mask) # img는 Tensor와 normalize, mask는 numpy to tensor형태로 변환

#         mask = torch.from_numpy(np.array(mask)).long()
#         ignore_mask[mask == 254] = self.ignore_label # ignore_mask는 mask에서 254픽셀부분을 255로 변환
#         # ------------------------------------------------------------
        
#         # return img_w, img_s, np.array(img_w), ignore_mask, cutmix_box, image_path
#         return img_w, img_s, ignore_mask, cutmix_box, image_path

#     def __len__(self):
#         return len(self.ids)
    
    
class SemiDataset(Dataset):
    def __init__(self, root, mode, valid_path=None, id_path=None, nsample=None):
        self.root = root
        self.mode = mode
        
        path = id_path if 'train' in mode else valid_path
        with open(path, 'r') as f:
            self.ids = f.read().splitlines()
            
        if mode == 'train_l' and nsample is not None: 
            self.ids *= math.ceil(nsample / len(self.ids)) 
            random.shuffle(self.ids)
            self.ids = self.ids[:nsample]


    def __getitem__(self, item):
        image_mask_path = self.ids[item]
        
        image_path, mask_path = image_mask_path.split(' ')[0], image_mask_path.split(' ')[1]
        img = Image.open(osp.join(self.root, image_path)).convert('RGB')
        mask = Image.open(osp.join(self.root, mask_path))
        
        img = transforms.ToTensor()(img)
        gt = torch.from_numpy(np.array(mask)).long()
        
        return self.mode, img, gt, image_path
            

    def __len__(self):
        return len(self.ids)


class GPUAugmentation(nn.Module):
    def __init__(self, size, ignore_label=255):
        super().__init__()
        self.size = size
        self.ignore_label = ignore_label
        
        self.id_map = torch.full((35, ), ignore_label, dtype=torch.long)
        mapping = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 
                   22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18}
        for k, v in mapping.items():
            self.id_map[k] = v
        
        # Weak Augmentation (Resize + Crop + Flip)
        self.weak_aug = K.AugmentationSequential(
                    K.RandomResizedCrop(size=self.size, scale=(0.5, 2.0), p=1.0, resample='nearest'),
                    K.RandomHorizontalFlip(p=0.5),
                    data_keys=["input", "mask"]
                )
        
        # Strong Augmentation (Color + Blur)
        self.strong_aug = K.AugmentationSequential(
            K.ColorJitter(0.5, 0.5, 0.5, 0.25, p=0.8),
            K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            data_keys=["input"]
        )
        
        
    @torch.no_grad()
    def map_id(self, mask):
        mask = mask.long()
        mask_mapped = self.id_map.to(mask.device)[mask]
        return mask_mapped


    @torch.no_grad()
    def forward(self, img, mask, mode='train_l'):
        # 1. ID Mapping
        mask = self.map_id(mask)
        
        # 2. Weak Augmentation
        img, mask = self.weak_aug(img, mask.float())
        mask = mask.long()

        if mode == 'train_l':
            return img, mask

        # 3. Strong Augmentation (Unlabeled 데이터용)
        img_s = self.strong_aug(img)
        
        return img, img_s, mask