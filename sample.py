
# from transformers import pipeline
# from PIL import Image
# import numpy as np
# import cv2

# pipe = pipeline(
#     task="depth-estimation",
#     model="depth-anything/Depth-Anything-V2-Large-hf"
# )

# # 이미지 입력
# image = Image.open("/home/dev/ssl-semantic-segmentation/ac9be3fe-790d1f8e.jpg")

# # depth 추론
# depth = pipe(image)["depth"]

# output = np.array(depth, dtype=np.float32)
# output_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX)
# output_norm = output_norm.astype(np.uint8)
# # MAGMA 컬러맵 적용 (이미지와 가장 유사)
# output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

# 1. 필수 라이브러리 설치
# !pip install -q transformers PIL matplotlib

import torch
import segmentation_models_pytorch as smp

def colorize_mask(mask):
    palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

    zero_pad = 256 * 3 - len(palette)
    palette.extend([0] * zero_pad)

    if mask.ndim == 3:
        mask = mask.squeeze()
        
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

import torch
# from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

# # 2. 가장 정확도가 높은 B5 모델로 로드
# device = "cuda" if torch.cuda.is_available() else "cpu"
# # B5 모델은 메모리를 많이 차지하므로 사양이 낮은 환경에선 주의가 필요합니다.
# model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024" 

# processor = SegformerImageProcessor.from_pretrained(model_name)
# model = SegformerForSemanticSegmentation.from_pretrained(model_name)
# model.to(device)

model_name = "facebook/mask2former-swin-large-cityscapes-semantic"

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoImageProcessor.from_pretrained(model_name)
model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(device)

# 3. 이미지 불러오기
image = Image.open("/home/dev/ssl-semantic-segmentation/ac9be3fe-790d1f8e.jpg").convert("RGB")
# inputs = processor(images=image, return_tensors="pt").to(device)
w, h = image.size

# [ROI 설정] 상단 30%(하늘/건물 일부)와 하단 15%(보닛)를 제외
# 비율은 이미지의 특징에 따라 [0.3, 0.85] 등으로 조정 가능합니다.
top = int(h * 0.3)
bottom = int(h * 0.85)
left = 0
right = w

# 자르기 (Left, Top, Right, Bottom)
image = image.crop((left, top, right, bottom))
# 기존 방식보다 더 정밀하게 입력을 넣는 법
inputs = processor(images=image, return_tensors="pt", do_resize=True, size={"height": 1024, "width": 1024}).to(device)

# 4. 추론 (Inference)
with torch.no_grad():
    outputs = model(**inputs)
    # logits = outputs.logits

# # 5. 결과 복원 (Interpolation)
# upsampled_logits = nn.functional.interpolate(
#     logits, size=image.size[::-1], mode="bilinear", align_corners=False
# )
# pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
result = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
pred_seg = result.cpu().numpy()
# 원본 크기(h, w)의 빈 캔버스 생성

pred_seg = colorize_mask(pred_seg).convert('RGB')
pred_seg = pred_seg.resize((1280, 720))
a=1
# # 6. 결과 시각화 (원본과 오버레이 비교)
# plt.figure(figsize=(15, 10))

# plt.subplot(1, 2, 1)
# plt.title("Original Image")
# plt.imshow(image)
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("High-Accuracy Segmentation (B5)")
# plt.imshow(pred_seg, cmap='terrain') # 도로 상황을 보기 좋은 컬러맵
# plt.axis("off")

# plt.tight_layout()
# plt.show()