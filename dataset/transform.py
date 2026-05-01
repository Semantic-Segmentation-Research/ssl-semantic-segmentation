import random
import math

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torch
from torchvision import transforms
import torch.nn.functional as F

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)

def colorize_mask(mask):
    new_mask = mask.convert('P')
    new_mask.putpalette(palette)

    return new_mask


def crop(img, mask, size, ignore_value=255):
    w, h = img.size
    padw = size - w if w < size else 0
    padh = size - h if h < size else 0
    img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
    mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=ignore_value)
    
    w, h = img.size
    x = random.randint(0, w - size)
    y = random.randint(0, h - size)
    img = img.crop((x, y, x + size, y + size))
    mask = mask.crop((x, y, x + size, y + size))

    return img, mask


def hflip(img, mask, p=0.5):
    if random.random() < p:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
    return img, mask


def normalize(img, mask=None):
    img = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])(img)
    if mask is not None:
        mask = torch.from_numpy(np.array(mask)).long()
        return img, mask
    return img


def resize_certain(img, ratio_range):
    w, h = img.size
    ow = int(w * ratio_range + 0.5)
    oh = int(h * ratio_range + 0.5)
    img = img.resize((ow, oh), Image.BILINEAR)
    return img

def resize(img, mask, ratio_range):
    w, h = img.size
    long_side = random.randint(int(max(h, w) * ratio_range[0]), int(max(h, w) * ratio_range[1])) # ratio range[0]을 곱한 값과 ratio range[1]을 곱한 값 사이의 정수값 중 하나를 long_side로 정의
    if h > w: # 세로가 가로보다 길면
        oh = long_side # 높이는 long_side 랜덤 정수값
        ow = int(1.0 * w * long_side / h + 0.5)
    else:
        ow = long_side
        oh = int(1.0 * h * long_side / w + 0.5)

    img = img.resize((ow, oh), Image.BILINEAR)
    mask = mask.resize((ow, oh), Image.NEAREST)
    return img, mask


def blur(img, p=0.5):
    if random.random() < p:
        sigma = np.random.uniform(0.1, 2.0)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    return img


def gaussian_blur_feature(x, kernel_size=3, sigma=1.0):
    # 가우시안 커널 생성
    x_range = torch.arange(kernel_size, device=x.device) - (kernel_size - 1) / 2
    gauss = torch.exp(-x_range**2 / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    
    # 2D 커널로 확장 [C, 1, K, K]
    kernel = gauss[:, None] * gauss[None, :]
    kernel = kernel.expand(x.shape[1], 1, kernel_size, kernel_size).to(x.device)
    
    # Depthwise Convolution으로 블러 적용
    return F.conv2d(x, kernel, groups=x.shape[1], padding=kernel_size//2)


def obtain_cutmix_box(img_size, p=0.5, size_min=0.02, size_max=0.4, ratio_1=0.3, ratio_2=1/0.3):
    mask = torch.zeros(img_size, img_size) # img(h,w) 정사각형크기의 0으로 된 mask tensor
    if random.random() > p: # 확률에 따른 mask반환, 값이 p보다 크면 제로마스크 리턴
        return mask
    
    # 면적비율 크기
    # ?? 0.02~0.4 사이의 균등분포값을 랜덤으로 img size * img size값에 곱한 값 크기
    size = np.random.uniform(size_min, size_max) * img_size * img_size
    while True:
        ratio = np.random.uniform(ratio_1, ratio_2) # 0.3 ~ 3.33333... 값 사이의 균등분포값
        cutmix_w = int(np.sqrt(size / ratio)) # 자를 박스 width, int값으로 floor
        cutmix_h = int(np.sqrt(size * ratio)) # 자를 박스 height, int값으로 floor
        # 좌표 무작위 선택
        x = np.random.randint(0, img_size) # 0 ~ img_size-1 사이의 x축값
        y = np.random.randint(0, img_size) # 0 ~ img_size-1 사이의 y축값

        if x + cutmix_w <= img_size and y + cutmix_h <= img_size: # 자를박스의 width,height가 img_size를 넘지않으면 종료(자를영역찾으면 다음으로 넘어감)
            break

    mask[y:y + cutmix_h, x:x + cutmix_w] = 1 # mask의 y축값 ~ y+ bbox height까지, x축값 ~ x+ bbox width까지를 1로 채움

    return mask


def img_aug_autocontrast(img, scale=None):
    return ImageOps.autocontrast(img)


def img_aug_equalize(img, scale=None):
    return ImageOps.equalize(img)


def img_aug_invert(img, scale=None):
    return ImageOps.invert(img)


def img_aug_identity(img, scale=None):
    return img


def img_aug_blur(img, scale=[0.1, 2.0]):
    assert scale[0] < scale[1]
    sigma = np.random.uniform(scale[0], scale[1])
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def img_aug_contrast(img, scale=[0.05, 0.95], p=0.2):
    if random.random() < p:
        min_v, max_v = min(scale), max(scale)
        v = float(max_v - min_v) * random.random()
        v = max_v - v
        return ImageEnhance.Contrast(img).enhance(v)
    else:
        return img


def img_aug_brightness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Brightness(img).enhance(v)


def img_aug_color(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Color(img).enhance(v)


def img_aug_sharpness(img, scale=[0.05, 0.95]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v = max_v - v
    # print(f"final:{v}")
    return ImageEnhance.Sharpness(img).enhance(v)


def img_aug_hue(img, scale=[0, 0.5]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    v += min_v
    if np.random.random() < 0.5:
        hue_factor = -v
    else:
        hue_factor = v
    # print(f"Final-V:{hue_factor}")
    input_mode = img.mode
    if input_mode in {"L", "1", "I", "F"}:
        return img
    h, s, v = img.convert("HSV").split()
    np_h = np.array(h, dtype=np.uint8)
    # uint8 addition take cares of rotation across boundaries
    with np.errstate(over="ignore"):
        np_h += np.uint8(hue_factor * 255)
    h = Image.fromarray(np_h, "L")
    img = Image.merge("HSV", (h, s, v)).convert(input_mode)
    return img


def img_aug_posterize(img, scale=[4, 8]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.posterize(img, v)


def img_aug_solarize(img, scale=[1, 256]):
    min_v, max_v = min(scale), max(scale)
    v = float(max_v - min_v) * random.random()
    # print(min_v, max_v, v)
    v = int(np.ceil(v))
    v = max(1, v)
    v = max_v - v
    # print(f"final:{v}")
    return ImageOps.solarize(img, v)


def get_augment_list():
    l = [
        (img_aug_identity, None),
        (img_aug_autocontrast, None),
        (img_aug_equalize, None),
        (img_aug_blur, [0.1, 2.0]),
        (img_aug_contrast, [0.05, 0.95]),
        (img_aug_brightness, [0.05, 0.95]),
        (img_aug_color, [0.05, 0.95]),
        (img_aug_sharpness, [0.05, 0.95]),
        (img_aug_posterize, [4, 8]),
        (img_aug_solarize, [1, 256]),
        (img_aug_hue, [0, 0.5])
    ]
    return l


class strong_img_aug:
    def __init__(self, num_augs=4, flag_using_random_num=True):
        self.n = num_augs
        self.augment_list = get_augment_list()
        self.flag_using_random_num = flag_using_random_num

    def __call__(self, img):
        if self.flag_using_random_num:
            max_num = np.random.randint(1, high=self.n + 1)
        else:
            max_num = self.n
        ops = random.choices(self.augment_list, k=max_num)
        for op, scales in ops:
            img = op(img, scales)
        return img