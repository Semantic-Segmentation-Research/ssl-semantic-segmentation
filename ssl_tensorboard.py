from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import torchvision.transforms.functional as F

class SSLTensorBoard:
    def __init__(self, writer):
        self.writer = writer
        
    
    def draw_text_on_tensor(self, tensor, text):
        # 1. 텐서(C, H, W)를 PIL 이미지로 변환
        # TensorBoard용으로 0~1 범위를 가정 (normalize=True 상태면 역정규화 필요)
        ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        
        # 2. 글자 그리기 준비
        draw = ImageDraw.Draw(img)
        # 폰트 설정 (기본 폰트 사용, 크기는 이미지 크기에 맞춰 조절 가능)
        # font = ImageFont.truetype("arial.ttf", 15) # 필요시 특정 폰트 경로 지정
        
        # 3. 텍스트 그리기 (왼쪽 상단 (5, 5) 위치에 흰색 글씨, 검은 테두리)
        draw.text((5, 5), text, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
        
        # 4. 다시 텐서로 변환
        return F.to_tensor(img)
    
    
    def draw_scalar(self, epoch, item):
        for key, value in item.items():
            self.writer.add_scalar(key, value, global_step=epoch)
            
    def draw_image(self, epoch, item):
        a=1