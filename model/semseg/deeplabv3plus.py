import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import os.path as osp
from dataset import transform as dtf

# region - DeepLabV3+
class DeepLabV3Plus(nn.Module):
    def __init__(self, tcfg, mcfg, pretrained_path):
        super(DeepLabV3Plus, self).__init__()
        self.tcfg = tcfg
        self.is_corr = True
        # self.pretrained_path = pretrained_path
        if not osp.exists(pretrained_path): 
            pretrained_path = False
            
        if 'resnet' in mcfg.backbone:
            backbone = resnet.__dict__[mcfg.backbone]
            self.backbone = backbone(pretrained_path,
                                     nf=mcfg.nf,
                                     bottleneck_nf=mcfg.bottleneck_nf,
                                     bottleneck_exp=mcfg.bottleneck_exp,
                                     multi_grid=mcfg.multi_grid,
                                     replace_stride_with_dilation=mcfg.replace_stride_with_dilation
                                     )
        else:
            assert mcfg.backbone == 'xception'
            self.backbone = xception(True)

        high_channels = mcfg.nf*8 * mcfg.bottleneck_exp

        self.aspp = ASPPModule(high_channels, mcfg.dilations)

        self.reduce = nn.Sequential(nn.Conv2d(256, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))
        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, mcfg.num_classes, 1, bias=True)


        if self.is_corr:
            self.proj = nn.Sequential(
                # nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, bias=True),
                # nn.BatchNorm2d(128),
                nn.Conv2d(high_channels, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )
            self.corr = CrossCovarianceAtt(256, 128, nclass=mcfg.num_classes)
            # self.corr = Corr(nclass=mcfg.num_classes)

    # region forward
    def forward(self, x, mode='train'):
        result_dict = {}
        image_height, image_width = x.shape[2:]

        c1, _, c4 = self.backbone.base_forward(x)
        if mode =='train':
            _, c1_u_w = torch.split(c1, split_size_or_sections=self.tcfg.batch_size, dim=0)
            _, c4_u_w = torch.split(c4, split_size_or_sections=self.tcfg.batch_size, dim=0)
            
            sigma_sampler = torch.distributions.Uniform(low=0.1, high=2.0)
            sigma = sigma_sampler.sample().to(c4.device)
            
            c1_u_s1= dtf.gaussian_blur_feature(c1_u_w, kernel_size=5, sigma=sigma)
            c4_u_s1= dtf.gaussian_blur_feature(c4_u_w, kernel_size=5, sigma=sigma)
            
            c1_u_s2 = c1_u_w + torch.randn_like(c1_u_w) * (torch.std(c1_u_w) * 0.1)
            c4_u_s2 = c4_u_w + torch.randn_like(c4_u_w) * (torch.std(c4_u_w) * 0.1)
            
            c1_u_s = c1_u_s1 * c1_u_s2
            c4_u_s = c4_u_s1 * c4_u_s2
            
            out_u_s = self._decode(c1_u_s, c4_u_s, size=(image_height, image_width))
            result_dict['out_u_s'] = out_u_s
            
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))), 
                                size=(image_height, image_width))
            out, out_fp = outs.chunk(2)
                
            binary_norm_corr_map, corr_out = self._feature_corr(enc_out=c4, dec_out=out, aug_type='weak')
            
            result_dict['binary_norm_corr_map'] = binary_norm_corr_map
            result_dict['corr_out'] = corr_out
            result_dict['out_fp'] = out_fp

            corr_out_u_s = self._feature_corr(enc_out=c4_u_s, dec_out=out_u_s, aug_type='strong')
            result_dict['corr_out_u_s'] = corr_out_u_s
            
        elif mode == 'test':
            out = self._decode(c1, c4, size=(image_height, image_width))

        result_dict['out'] = out
        
        return result_dict


    def _decode(self, c1, c4, size):
        image_height, image_width = size
        
        c4 = self.aspp(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)
        
        out = self.classifier(feature)
        out = F.interpolate(out, (image_height, image_width), mode="bilinear", align_corners=True)
        
        return out
    
    def _feature_corr(self, enc_out, dec_out, aug_type='weak'):
        proj_feats = self.proj(enc_out)
        corr_out_dict = self.corr(proj_feats, dec_out, aug_type)
        
        corr_out = corr_out_dict['corr_dec_out']
        corr_out = F.interpolate(corr_out, size=(self.tcfg.crop_size, self.tcfg.crop_size), mode="bilinear", align_corners=True)
        
        if aug_type == 'weak':
            return corr_out_dict['binary_norm_corr_map'], corr_out
        else:
            return corr_out
            

def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


# region - ASPPPooling
class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


# region - ASPPModule
class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)


# region - Corr
class Corr(nn.Module):
    def __init__(self, nclass=21):
        super(Corr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    # Encoder output & Decoder output
    def forward(self, enc_out, dec_out):
        result_dict = {}
        
        enc_height, enc_width = enc_out.shape[-2:]
        dec_height, dec_width = dec_out.shape[-2:]
        
        dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        # feature = F.interpolate(enc_out, (enc_height, enc_width), mode='bilinear', align_corners=True)
        
        f1 = rearrange(self.conv1(enc_out), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(enc_out), 'n c h w -> n c (h w)')
        dec_out_reshape = rearrange(dec_out, 'n c h w -> n c (h w)')
        
        # 수식 4번
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        # corr_map_sample: (8, 128, 2304)
        corr_map_sample = self.sample(corr_map.detach(), enc_height, enc_width)
        # 7번 수식에서 C_hat (8, 128, 384, 384)
        result_dict['binary_norm_corr_map'] = self.normalize_corr_map(corr_map_sample, enc_height, enc_width, dec_height, dec_width)
        
        # 5번 수식 (8, 19, 48, 48)
        result_dict['corr_dec_out'] = rearrange(torch.matmul(dec_out_reshape, corr_map), 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        
        return result_dict


    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample
    
    
    # region - region propagation
    def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m = corr_map.shape[:2]
        
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        # Min - Max scaling (normalization), 수식 7번
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        
        return norm_corr_map
    


# region - XCA
class CrossCovarianceAtt(nn.Module):
    def __init__(self, in_ch, out_ch, nclass=21):
        super(CrossCovarianceAtt, self).__init__()
    
        self.nclass = nclass
        # self.in_ch = 128
        # self.out_ch = 64
        self.in_ch = in_ch
        self.out_ch = out_ch
        
        self.qkv_conv = nn.Conv2d(self.in_ch, self.out_ch * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1, groups=self.out_ch)
        self.proj   = nn.Conv2d(self.out_ch, nclass, kernel_size=3, padding=1)
        
        
    # Encoder output & Decoder output
    def forward(self, enc_out, dec_out, aug_type='weak'):
        result_dict = {}
        
        _, _, enc_height, enc_width = enc_out.shape
        dec_height, dec_width = dec_out.shape[-2:]
        
        qkv = self.qkv_conv(enc_out).flatten(2)
        q, k, v= qkv.chunk(3, dim=1)
        
        # q = rearrange(self.conv1(enc_out), 'n c h w -> n c (h w)')
        # k = rearrange(self.conv2(enc_out), 'n c h w -> n c (h w)')
        # v = rearrange(self.conv3(enc_out), 'n c h w -> n c (h w)')
        
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        # attn = torch.matmul(q, k.transpose(1, 2))
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)
        # xca: [B*2, C, N]
        xca = torch.bmm(attn, v)
        xca_conf = F.softmax(xca, dim=-1)
        
        if aug_type =='weak':
            result_dict['binary_norm_corr_map'] = self.normalize_xca_map(xca_conf, enc_height, enc_width, dec_height, dec_width)
        
        xca_reshsape = rearrange(xca, 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        
        x_lp = xca_reshsape + self.dwconv(xca_reshsape)
        x_lp = self.proj(x_lp)
        
        # 7번 수식에서 C_hat (8, 128, 384, 384)
        dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        # 5번 수식
        # (8,19,48,48)
        result_dict['corr_dec_out'] = dec_out * x_lp
        
        return result_dict


    def normalize_xca_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m = corr_map.shape[:2]
        
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        # Min - Max scaling (normalization), 수식 7번
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        
        return norm_corr_map