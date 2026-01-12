from dataclasses import dataclass
from dataclasses import field
import os.path as osp
import os

@dataclass
class ModelConfig:
    backbone:    str    = "resnet101"
    num_classes: int    = 19



@dataclass
class TrainConfig:
    dataset: str        = "cityscapes"
    model_name: str     = "deeplabv3plus_resnet101"
    # model_name: str     = "test"

    exp_dir: str        = osp.join(os.getcwd(), "experiments")
    data_root: str      = osp.join(os.getcwd(), 'data')
    pretrained_path: str = osp.join(osp.dirname(__file__), 'pretrained')
    
    batch_size: int     = 4
    lr: float           = 0.005
    num_epochs: int     = 240
    num_workers: int    = 8
    
    crop_size: int     = 384
    
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