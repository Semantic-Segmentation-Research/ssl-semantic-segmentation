import numpy as np
import logging
import os
import os.path as osp
import shutil
from PIL import Image
import torch
import torch.nn.functional as F
from collections import namedtuple


# region - Label Table
# ========================== Label Table ==========================
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]
# =================================================================



def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


# def color_map(dataset='pascal'):
#     cmap = np.zeros((256, 3), dtype='uint8')

#     if dataset == 'pascal' or dataset == 'coco':
#         def bitget(byteval, idx):
#             return (byteval & (1 << idx)) != 0

#         for i in range(256):
#             r = g = b = 0
#             c = i
#             for j in range(8):
#                 r = r | (bitget(c, 0) << 7-j)
#                 g = g | (bitget(c, 1) << 7-j)
#                 b = b | (bitget(c, 2) << 7-j)
#                 c = c >> 3

#             cmap[i] = np.array([r, g, b])

#     elif dataset == 'cityscapes':
#         cmap[0] = np.array([128, 64, 128])
#         cmap[1] = np.array([244, 35, 232])
#         cmap[2] = np.array([70, 70, 70])
#         cmap[3] = np.array([102, 102, 156])
#         cmap[4] = np.array([190, 153, 153])
#         cmap[5] = np.array([153, 153, 153])
#         cmap[6] = np.array([250, 170, 30])
#         cmap[7] = np.array([220, 220, 0])
#         cmap[8] = np.array([107, 142, 35])
#         cmap[9] = np.array([152, 251, 152])
#         cmap[10] = np.array([70, 130, 180])
#         cmap[11] = np.array([220, 20, 60])
#         cmap[12] = np.array([255,  0,  0])
#         cmap[13] = np.array([0,  0, 142])
#         cmap[14] = np.array([0,  0, 70])
#         cmap[15] = np.array([0, 60, 100])
#         cmap[16] = np.array([0, 80, 100])
#         cmap[17] = np.array([0,  0, 230])
#         cmap[18] = np.array([119, 11, 32])

#     return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnion_gpu(output, target, K, ignore_index=255):
    # output, target shape: [B, H, W] 또는 [N]
    assert output.shape == target.shape
    
    # 1차원으로 펼치기
    output = output.view(-1)
    target = target.view(-1)
    
    output = output.clone() # 원본 보존을 위해 클론
    output[target == ignore_index] = ignore_index

    # 3. Intersection 계산: output과 target이 같은 위치의 값들만 추출
    intersection = output[output == target]

    # 4. 빈도수 계산 (bincount)
    # 이 때 ignore_index(255)는 minlength=K에 의해 자동으로 제외됩니다.
    area_intersection = torch.bincount(intersection.long(), minlength=K)[:K]
    area_output = torch.bincount(output[output < K].long(), minlength=K)[:K]
    area_target = torch.bincount(target[target < K].long(), minlength=K)[:K]

    # 5. Union 계산
    area_union = area_output + area_target - area_intersection
    
    return area_intersection, area_union, area_target



logs = set()

def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name) # logger 객체 생성
    logger.setLevel(level) # 로그 레벨 설정, INFO, DEBUG 등
    ch = logging.StreamHandler() # 콘솔에 로그를 찍기위함
    ch.setLevel(level) # 핸들러도 동일한 로그 레벨
    if "SLURM_PROCID" in os.environ: # SLURM_PROCID 환경변수가 있으면 현재 분산 학습 환경에서 실행 중이라고 판단
        rank = int(os.environ["SLURM_PROCID"]) # 
        logger.addFilter(lambda record: rank == 0) # rank==0인 프로세스만 출력하도록 
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s" # 현재시간, 로그레벨, 메시지 내용
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def save_codes(tcfg, directory):
    for f in os.listdir(directory):
        if f.endswith('.py'):
            shutil.copy(osp.join(directory, f), osp.join(tcfg.exp_dir, "codes", tcfg.model_name))


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

# region - initialization
def init_non_backbone(m):
    import torch.nn as nn
    
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
