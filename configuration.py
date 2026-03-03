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
    dilations: list     = field(default_factory=lambda: [12, 24, 36])
    replace_stride_with_dilation: list = field(default_factory=lambda: [False, False, True])
    norm_layer: str     = "BatchNorm2d"
    
    nf: int       = 64
    bttln_nf: int  = 128
    bttln_exp: int = 3
    groups: int     = 1
    width_per_group:int = 64
    num_blocks: list = field(default_factory=lambda: [3, 4, 6, 3])

    multi_grid: bool = False
    zero_init_residual: bool = False
    
    
    
# region - Train
@dataclass
class TrainConfig:
    dataset: str        = "cityscapes"
    model_name: str     = "v1.1.8_custom_resnet50_xca"

    exp_dir: str            = osp.join(BASE_DIR, "experiments")
    model_save_dir: str     = osp.join(exp_dir, "models", model_name)
    data_root: str          = osp.join(BASE_DIR, 'data')
    pretrained_path: str    = osp.join(osp.dirname(__file__), 'pretrained')
    
    batch_size: int     = 8
    lr: float           = 5e-4 # 5e-3
    lr_multi: float     = 1.0
    num_epochs: int     = 800
    num_workers: int    = 8
    
    crop_size: int      = 448
    local_rank: int     = 0
    port: int           = 0
    
    lr_period: int       = 400
    label_lr_decay: float = 0.9
    unlabel_lr_decay: float = 0.98
    resume: bool        = False
    
    eval_mode: str      = 'original'
    
    @dataclass
    class LossConfig:
        name: str = "OHEM" # OHEM, CELoss
        ignore_index: int = 255
        ohem_threshold: float = 0.9
        ohem_min_kept: int = 100000
        
        aux_loss_weight: float = 1.0 # 0.4
        
        
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
        
        # random_scale: list = field(default_factory=lambda: [0.5, 2.0])
        # gaussian_blur_prob: float = 0.5
        # color_jitter: bool = True