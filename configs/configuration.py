from dataclasses import dataclass
from dataclasses import field

@dataclass
class TrainConfig:
    batch_size: int     = 4
    lr: float           = 0.005
    
    @dataclass
    class AugConfig:
        resize_raio: list[float, float] = [0.5, 2.0]
        hflip_prob: float = 0.5
        brightness: float = 0.5
        contrast: float = 0.5
        saturation: float = 0.5
        hue: float = 0.5
        
        blur_prob: float = 0.5
        cutmix_prob: float = 0.5