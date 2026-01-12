from PIL import Image
import numpy as np
import io

from util import utils
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms.functional as TF


class SSLTensorBoard:
    def __init__(self, writer):
        self.writer = writer
        self.gap = 5
        self.display_scale = 3 # 화면에 보일 크기 조절 계수
        
    
    def draw_scalar(self, epoch, item):
        for key, value in item.items():
            self.writer.add_scalar(key, value, global_step=epoch)
            
            
    def draw_image(self, tag, image, pred, conf, mask, image_path, epoch):
        rgb = np.clip(image, 0, 1)
        rgb = np.concatenate([cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)[np.newaxis, :] for sample in rgb], axis=0)
        rgb = (rgb * 255).astype(np.uint8)

        batch_size, H, W, _ = rgb.shape
        
        pred_rgb = np.repeat(pred, repeats=3, axis=-1)
        pred_rgb = np.stack(np.array([utils.colorize_mask(sample).convert('RGB') for sample in pred_rgb]))
        
        conf_rgb = np.repeat(conf, repeats=3, axis=-1)
        conf_rgb = (conf_rgb * 255).astype(np.uint8)
        if mask is not None:
            mask_rgb = np.repeat(mask, repeats=3, axis=-1)
            mask_rgb = np.stack(np.array([utils.colorize_mask(sample).convert('RGB') for sample in mask_rgb]))
        
        padding = np.ones((H, self.gap, 3), dtype=rgb.dtype) * 255
        
        fig_width = 12
        fig, axes = plt.subplots(batch_size, 1, figsize=(fig_width, 12))
        
        if batch_size == 1: axes = [axes]
        for i in range(batch_size):
            image_name = image_path[i].split('/')[-1].split('.')[0]
            
            if mask is not None:
                row = np.concatenate([
                    rgb[i],
                    padding,
                    mask_rgb[i],
                    padding,
                    conf_rgb[i],
                    padding,
                    pred_rgb[i]
                ], axis=1)
            else:
                row = np.concatenate([
                    rgb[i],
                    padding,
                    pred_rgb[i],
                    padding,
                    conf_rgb[i]
                ], axis=1)
                
            axes[i].imshow(row)
            axes[i].axis('off')
            axes[i].set_title(image_name, fontsize=12, pad=5)
            
        plt.subplots_adjust(hspace=0.2, top=0.95, bottom=0.05, left=0.05, right=0.95)
        
        # BytesIO로 PNG 변환
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.02)
        plt.close(fig)
        
        buf.seek(0)
        img_pil = Image.open(buf)
        img_tensor = TF.to_tensor(img_pil)
        
        self.writer.add_image(tag, img_tensor, global_step=epoch)