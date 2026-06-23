from dataclasses import dataclass
from dataclasses import field
import os.path as osp

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
    
    input_channel : int = 3
    nf: int       = 48
    bttln_nf: int  = 96
    bttln_exp: int = 3
    groups: int     = 1
    width_per_group:int = 64
    num_blocks: list = field(default_factory=lambda: [3, 4, 6, 3])
    init_value: float = 1e-6
    
    multi_grid: bool = False
    zero_init_residual: bool = False
    
    enc_c1_ratio: int   = 1
    enc_c2_ratio: int   = 2
    enc_c3_ratio: int   = 4
    enc_c4_ratio: int   = 6
    
    
# region - Train
@dataclass
class TrainConfig:
    dataset: str        = "cityscapes"
    model_name: str     = "v1.6.4_LTU" 
    # model_name: str     = "test" 

    exp_dir: str            = osp.join(BASE_DIR, "experiments")
    model_save_dir: str     = osp.join(exp_dir, "models", model_name)
    data_root: str          = osp.join(BASE_DIR, 'data')
    pretrained_path: str    = osp.join(osp.dirname(__file__), 'pretrained')
    valid_path: str         = osp.join(osp.dirname(__file__), 'partitions', 'cityscapes', 'val.txt')
    
    optimizer: str      = "Adam"
    scheduler: str      = "cosineDecay"
    
    batch_size: int     = 4
    accumulation_steps: int = 4
    # lr: float           = 2e-4
    lr: float           = 5e-4
    lr_period: int      = 31 # accumulation_step 사용 시 lr_period는 1/accm
    
    
    num_epochs: int     = 240
    num_workers: int    = 8
    
    crop_size: int      = 448
    
    decay_power: float  = 0.9
    lr_multi: float     = 1.0
    label_lr_decay: float   = 0.5
    unlabel_lr_decay: float = 0.9
    resume: bool            = False
    thresh_init: float      = 0.85
    
    eval_mode: str      = 'original'
    
    @dataclass
    class LossConfig:
        ignore_index: int = 255
        ohem_threshold: float = 0.9
        ohem_min_kept: int = 100000 # 지금 width, height, batch를 고려해 상위 84% 정도
        
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
        
        
        
@dataclass
class TestConfig:
    dataset: str        = "cityscapes"
    model_name: str     = "v1.6.3_LTU"

    valid_path: str         = osp.join(osp.dirname(__file__), 'partitions', 'cityscapes', 'val.txt')
    exp_dir: str            = osp.join(BASE_DIR, "experiments")

    model_save_dir: str     = osp.join(exp_dir, "models", model_name)
    data_root: str          = osp.join(BASE_DIR, 'data')
    pretrained_path: str    = osp.join(osp.dirname(__file__), 'pretrained')
    result_dir: str         = osp.join(exp_dir, "results")
    
    crop_size: int      = 448
    threshold: float    = 0.5
