import torch
import torch.nn as nn
from model.backbone import clayers

__all__ = ['ResNet', 'resnet50', 'resnet101']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# region - Bottleneck
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, expansion, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * expansion)
        self.bn3 = norm_layer(planes * expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# region - ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, mcfg):
        super(ResNet, self).__init__()

        self.bttln_nf = mcfg.bttln_nf
        self.bttln_exp = mcfg.bttln_exp
        
        if mcfg.norm_layer == "BatchNorm2d": 
            self._norm_layer = nn.BatchNorm2d

        self.dilation = 1
        
        if mcfg.replace_stride_with_dilation is None:
            mcfg.replace_stride_with_dilation = [False, False, False]
        
        if len(mcfg.replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(mcfg.replace_stride_with_dilation))
            
        self.groups = mcfg.groups
        self.base_width = mcfg.width_per_group
        
        
        self.gamma_layer1 = clayers.LayerScale(input_ch=mcfg.input_channel, 
                                               out_ch=mcfg.nf, 
                                               gamma_channel=mcfg.nf)
        self.gamma_layer2 = clayers.LayerScale(input_ch=mcfg.bttln_nf*mcfg.bttln_exp, 
                                               out_ch=mcfg.bttln_nf*mcfg.bttln_exp, 
                                               gamma_channel=mcfg.bttln_nf*mcfg.bttln_exp)

        self.init_conv = clayers.InitConv(mcfg.input_channel, mcfg.nf)

        self.ds_conv = nn.Conv2d(mcfg.nf, mcfg.bttln_nf, kernel_size=3, stride=2, padding=1, bias=False)
        
        self.layer1 = self._make_layer(block, mcfg.bttln_nf, layers[0])
        self.layer2 = self._make_layer(block, mcfg.nf*mcfg.enc_c2_ratio, layers[1], stride=2,
                                       dilate=mcfg.replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, mcfg.nf*mcfg.enc_c3_ratio, layers[2], stride=2,
                                       dilate=mcfg.replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, mcfg.nf*mcfg.enc_c4_ratio, layers[3], stride=2,
                                       dilate=mcfg.replace_stride_with_dilation[2], multi_grid=mcfg.multi_grid)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if mcfg.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    # region make_layer
    def _make_layer(self, block, planes, num_block, stride=1, dilate=False, multi_grid=False):
        downsample = None
        norm_layer = self._norm_layer
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.bttln_nf != planes * self.bttln_exp:
            downsample = nn.Sequential(
                conv1x1(self.bttln_nf, planes * self.bttln_exp, stride),
                norm_layer(planes * self.bttln_exp),
            )

        grids = [1] * num_block
        if multi_grid:
            grids = [2, 2, 4]

        layers = list()
        layers.append(block(self.bttln_nf, planes, self.bttln_exp, stride, downsample, self.groups,
                            self.base_width, previous_dilation * grids[0], norm_layer))
        
        self.bttln_nf = planes * self.bttln_exp
        for i in range(1, num_block):
            layers.append(block(self.bttln_nf, planes, self.bttln_exp, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation * grids[i],
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def base_forward(self, x):
        x   = self.init_conv(x)
        x_g = self.gamma_layer1(x)
        x   = torch.add(x, x_g)

        x  = self.ds_conv(x)
        
        c1   = self.layer1(x)
        c1_g = self.gamma_layer2(c1)
        c1   = torch.add(c1, c1_g)

        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        return c1, c2, c3, c4


def _resnet(arch, block, layers, pretrained, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        pretrained_path = "pretrained/%s.pth" % arch
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
    return model


def resnet50(pretrained=False, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, **kwargs)


def resnet101(pretrained=False, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, **kwargs)
