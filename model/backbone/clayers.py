import torch.nn as nn
import torch



# region - Initial Convolution Layer
class InitConv(nn.Module):
    def __init__(self, input_ch, mid_ch, out_ch):
        super(InitConv, self).__init__()
        
        self.init_conv = nn.Sequential(
            nn.Conv2d(input_ch, mid_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch, bias=False),
            nn.Conv2d(mid_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=1, padding=1, groups=mid_ch, bias=False),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, input):
        return self.init_conv(input)
    
    
# region - Layer Scale
class LayerScale(nn.Module):
    def __init__(self, mcfg, input_channel, gamma_channel, init_value=1e-6):
        super(LayerScale, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channel, mcfg.nf*2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(mcfg.nf*2),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(init_value * torch.ones((1, gamma_channel, 1, 1)), requires_grad=True)
        
        
    def forward(self, x):
        return self.gamma * x