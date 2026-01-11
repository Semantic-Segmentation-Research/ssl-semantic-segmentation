from PIL import Image, ImageDraw
import numpy as np
import torch
import io

import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF


class SSLTensorBoard:
    def __init__(self, writer):
        self.writer = writer
        
    
    def draw_scalar(self, epoch, item):
        for key, value in item.items():
            self.writer.add_scalar(key, value, global_step=epoch)
            
            
    def draw_image(self, image, pred, image_path, epoch):
        rgb = np.clip(image, 0, 1)
        rgb = np.concatenate([cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)[np.newaxis, :] for sample in rgb], axis=0)

        pred = np.repeat(pred, repeats=3, axis=-1)
        
        batch_size, _, H, W = rgb.shape
        gap = 10 
        total_cols = 4
        row_width = W*total_cols + gap*3  # RGB | gap | GT | gap | PRED | gap | BINARY
        row_height = H+gap*3

        fig = plt.figure(figsize=(row_width/100, batch_size*row_height/100), dpi=100)

        for i in range(batch_size):
            image_name = image_path[i].split('/')[-1]
            
            row = np.concatenate([rgb[i],
                                np.ones((H, gap, 3), dtype=rgb.dtype),
                                pred[i]
                                ], axis=1)
                
            ax = fig.add_subplot(batch_size, 1, i+1)
            ax.imshow(row)
            ax.axis('off')
            ax.set_title(f"{image_name}")
            
        
        # BytesIO로 PNG 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05)
        plt.close(fig)
        
        buf.seek(0)
        img_pil = Image.open(buf)
        img_tensor = TF.to_tensor(img_pil) # (C, H, W) 0~1 float
        
        self.writer.add_image(f'train/unlabel image', img_tensor, global_step=epoch)