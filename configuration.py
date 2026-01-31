from dataclasses import dataclass
from dataclasses import field
import os.path as osp
import os

BASE_DIR = '/home/dev'

# region - Data
@dataclass
class DataConfig:
    labeled_id_path: str    = osp.join(osp.dirname(__file__), "partitions/cityscapes/1_4/labeled.txt")
    unlabeled_id_path: str  = osp.join(osp.dirname(__file__), "partitions/cityscapes/1_4/unlabeled.txt")
    val_id_path: str        = osp.join(osp.dirname(__file__), "partitions/cityscapes/val.txt")


# region - Model
@dataclass
class ModelConfig:
    backbone:    str    = "resnet50"
    num_classes: int    = 19
    multi_grid: bool    = False
    dilations: list     = field(default_factory=lambda: [12, 24, 36])
    replace_stride_with_dilation: list = field(default_factory=lambda: [False, True, True])
    

# region - Train
@dataclass
class TrainConfig:
    dataset: str        = "cityscapes"
    model_name: str     = "custom_resnet50_xca"

    exp_dir: str        = osp.join(BASE_DIR, "experiments")
    data_root: str      = osp.join(BASE_DIR, 'data')
    pretrained_path: str = osp.join(osp.dirname(__file__), 'pretrained')
    
    batch_size: int     = 8
    lr: float           = 0.005
    num_epochs: int     = 240
    num_workers: int    = 8
    
    crop_size: int      = 448
    
    @dataclass
    class LossConfig:
        name: str = "OHEM"
        
        
    @dataclass
    class AugConfig:
        resize_raio: list = field(default_factory=lambda: [0.5, 2.0])
        hflip_prob: float = 0.5
        brightness: float = 0.5
        contrast: float = 0.5
        saturation: float = 0.5
        hue: float = 0.5
        
        blur_prob: float = 0.5
        cutmix_prob: float = 0.5