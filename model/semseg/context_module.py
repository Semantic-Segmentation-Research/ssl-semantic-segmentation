import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from collections import OrderedDict

# region - ASPPConv
# def ASPPConv(in_channels, out_channels, atrous_rate):
#     block = nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, 3, 
#                   padding=atrous_rate,
#                   dilation=atrous_rate, 
#                   bias=False),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(True)
#         )
#     return block
def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(
        # depthwise conv: groups=in_channels keeps channels separate
        nn.Conv2d(in_channels,
                  in_channels,
                  kernel_size=3,
                  padding=atrous_rate,
                  dilation=atrous_rate,
                  groups=in_channels,
                  bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU6(True),
        # pointwise conv to mix channels and adjust to desired out_channels
        nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU6(True),
    )
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
    def __init__(self, in_ch, out_ch, atrous_rates):
        super(ASPPModule, self).__init__()
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False),
                                nn.BatchNorm2d(out_ch),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_ch, out_ch, rate1)
        self.b2 = ASPPConv(in_ch, out_ch, rate2)
        self.b3 = ASPPConv(in_ch, out_ch, rate3)
        self.b4 = ASPPPooling(in_ch, out_ch)

        self.project = nn.Sequential(nn.Conv2d(5 * out_ch, out_ch, 1, bias=False),
                                     nn.BatchNorm2d(out_ch),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)



# # region - Corr
# class Corr(nn.Module):
#     def __init__(self, nclass=21):
#         super(Corr, self).__init__()
#         self.nclass = nclass
#         self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
#         self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

#     # Encoder output & Decoder output
#     def forward(self, enc_out, dec_out):
#         result_dict = {}
        
#         enc_height, enc_width = enc_out.shape[-2:]
#         dec_height, dec_width = dec_out.shape[-2:]
        
#         dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
#         # feature = F.interpolate(enc_out, (enc_height, enc_width), mode='bilinear', align_corners=True)
        
#         f1 = rearrange(self.conv1(enc_out), 'n c h w -> n c (h w)')
#         f2 = rearrange(self.conv2(enc_out), 'n c h w -> n c (h w)')
#         dec_out_reshape = rearrange(dec_out, 'n c h w -> n c (h w)')
        
#         # 수식 4번
#         corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
#         corr_map = F.softmax(corr_map, dim=-1)
#         # corr_map_sample: (8, 128, 2304)
#         corr_map_sample = self.sample(corr_map.detach(), enc_height, enc_width)
#         # 7번 수식에서 C_hat (8, 128, 384, 384)
#         result_dict['binary_norm_corr_map'] = self.normalize_corr_map(corr_map_sample, enc_height, enc_width, dec_height, dec_width)
        
#         # 5번 수식 (8, 19, 48, 48)
#         result_dict['corr_dec_out'] = rearrange(torch.matmul(dec_out_reshape, corr_map), 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        
#         return result_dict


#     def sample(self, corr_map, h_in, w_in):
#         index = torch.randint(0, h_in * w_in - 1, [128])
#         corr_map_sample = corr_map[:, index.long(), :]
#         return corr_map_sample
    
    
#     # region - region propagation
#     def normalize_corr_map(self, corr_map, h_in, w_in, h_out, w_out):
#         n, m = corr_map.shape[:2]
        
#         corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
#         corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

#         corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
#         # Min - Max scaling (normalization), 수식 7번
#         range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
#         range_ = torch.clamp(range_, min=1e-6)
#         temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
#         corr_map = (temp_map > 0.5)
        
#         norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        
#         return norm_corr_map
    


# region - XCA
class CrossCovarianceAtt(nn.Module):
    def __init__(self, reduc_in_ch, reduc_out_ch, mid_ch, output_size, nclass=21):
        super(CrossCovarianceAtt, self).__init__()

        self.output_size = output_size
        self.reduction = nn.Sequential(
            nn.Conv2d(reduc_in_ch, reduc_out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(reduc_out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        self.kv_conv = nn.Conv2d(reduc_out_ch, mid_ch * 2, kernel_size=1, stride=1, padding=0, bias=True)
        self.q_conv = nn.Conv2d(reduc_out_ch, mid_ch, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch)
        # self.proj   = nn.Conv2d(mid_ch, nclass, kernel_size=3, padding=1)
        self.proj = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch*2, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch*2, nclass, kernel_size=3, padding=1, bias=True),
        )
        # self.temperature = nn.Parameter(torch.ones(1, mid_ch, 1))
        self.relu = nn.ReLU(inplace=True)
        
    # Encoder output & Decoder output
    def forward(self, enc_out, dec_out, aug_type='weak'):
        result_dict = {}
        
        proj_feats = self.reduction(enc_out)
        
        enc_height, enc_width = proj_feats.shape[-2:]
        dec_height, dec_width = dec_out.shape[-2:]
        
        # q    = self.q_conv(dec_out.detach())
        q    = self.q_conv(dec_out)
        q    = F.interpolate(q, (enc_height, enc_width), mode='bilinear', align_corners=True)
        q    = q.flatten(2)
        kv   = self.kv_conv(proj_feats).flatten(2)
        k, v = kv.chunk(2, dim=1)
        
        q = F.normalize(q, p=2, dim=1)
        k = F.normalize(k, p=2, dim=1)
        
        attn = torch.bmm(q, k.transpose(1, 2))
        # attn /= self.temperature
        # attn = F.softmax(attn, dim=-1)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        xca = torch.bmm(attn, v)

        # xca = F.softmax(xca, dim=1)

        if aug_type =='weak':
            xca_ = F.softmax(xca, dim=1)
            result_dict['binary_norm_corr_map'] = self.normalize_xca_map(xca_, enc_height, enc_width, dec_height, dec_width)
        
        xca_conf_reshape = rearrange(xca, 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        xca_conf_reshape = self.proj(xca_conf_reshape)
        
        # dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        dec_out = F.interpolate(dec_out, (enc_height, enc_width), mode='bilinear', align_corners=True)
        corr_dec_out = dec_out * xca_conf_reshape
        corr_dec_out = F.interpolate(corr_dec_out, size=(self.output_size, self.output_size), mode="bilinear", align_corners=True)
        # 5번 수식
        result_dict['corr_dec_out'] = corr_dec_out
        
        return result_dict


    def normalize_xca_map(self, corr_map, h_in, w_in, h_out, w_out):
        n, m = corr_map.shape[:2]
        
        corr_map = rearrange(corr_map, 'n m (h w) -> (n m) 1 h w', h=h_in, w=w_in)
        corr_map = F.interpolate(corr_map, (h_out, w_out), mode='bilinear', align_corners=True)

        corr_map = rearrange(corr_map, '(n m) 1 h w -> (n m) (h w)', n=n, m=m)
        # Min - Max scaling (normalization), 수식 7번
        range_ = torch.max(corr_map, dim=1, keepdim=True)[0] - torch.min(corr_map, dim=1, keepdim=True)[0]
        range_ = torch.clamp(range_, min=1e-6)
        temp_map = ((- torch.min(corr_map, dim=1, keepdim=True)[0]) + corr_map) / range_
        corr_map = (temp_map > 0.5)
        
        norm_corr_map = rearrange(corr_map, '(n m) (h w) -> n m h w', n=n, m=m, h=h_out, w=w_out)
        
        return norm_corr_map
    

# region - FlowAtt
class FlowAtt(nn.Module):
    """
    label의 정보를 unlabel에게 전달
    """
    def __init__(self, channel, reduc_ch, exp_ratio, drop_path=0.1):
        super(FlowAtt, self).__init__()

        # -------------------------- StarNet Module --------------------------
        self.star_layer = nn.Sequential(OrderedDict([
            ('reduction', nn.Sequential(
                nn.Conv2d(channel, reduc_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(reduc_ch),
                nn.Hardswish(inplace=True))
             ),
            # ('f1', nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=1, stride=1, padding=0, bias=True)),
            # ('f2', nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=1, stride=1, padding=0, bias=True)),
            ('asy_f1', nn.Sequential(
                nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
                nn.BatchNorm2d(reduc_ch*exp_ratio),
                nn.Hardswish(inplace=True),
            )),
            ('asy_f2', nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True)),
            ('g', nn.Conv2d(reduc_ch*exp_ratio, channel, kernel_size=1, stride=1, padding=0, bias=True)),
            # ('dwconv2', nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False)),
            ('dwconv2', nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(channel),
                nn.Hardswish(inplace=True)
            )),
            ('hswish', nn.Hardswish(inplace=True)),
            ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
        ]))
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # ------------------------------------------------------------------
        
        # -------------------------- XCA Module --------------------------
        self.xca_layer = nn.Sequential(OrderedDict([
            ('kv_conv', nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0, bias=True)),
            ('q_conv', nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)),
            ("proj", nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(channel),
                nn.Hardswish(inplace=True)
                ))
        ]))
        # self.temperature = nn.Parameter(1e-6 * torch.ones(1, channel, 1))
        # --------------------------------------------------------------------
        
        
    def star(self, feat):
        input = feat
        
        x = self.star_layer.reduction(feat)
        # x1, x2 = self.star_layer.f1(x), self.star_layer.f2(x)
        x1, x2 = self.star_layer.asy_f1(x), self.star_layer.asy_f2(x)
        x = self.star_layer.hswish(x1) * x2
        x = self.star_layer.dwconv2(self.star_layer.g(x))
        
        x = input + self.star_layer.drop_path(x)
                
        return x
    
    
    def xca(self, feat1, feat2):
        b, _, h, w = feat1.shape
        
        # q = self.xca_layer.q_conv(feat2.detach()).flatten(2)
        q = self.xca_layer.q_conv(feat2).flatten(2)
        kv = self.xca_layer.kv_conv(feat1).flatten(2)
        k, v = kv.chunk(2, dim=1)
        
        # q = F.normalize(q, p=2, dim=2)
        # k = F.normalize(k, p=2, dim=2)
        # 2. 치명적 버그 수정: dim=2(공간 차원)로 변경 + 역전파 NaN 방지용 1e-6 명시적 추가
        q_norm = torch.norm(q, p=2, dim=2, keepdim=True)
        k_norm = torch.norm(k, p=2, dim=2, keepdim=True)
        
        q = q / (q_norm + 1e-6)
        k = k / (k_norm + 1e-6)
        
        attn = torch.bmm(q, k.transpose(1, 2)).float()
        # 안정화를 위해 NaN/Inf를 0/대규모 값으로 치환
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1e6, neginf=-1e6)
        # attn /= self.temperature
        attn = F.softmax(attn, dim=-1)
        xca = torch.bmm(attn, v)

        xca = xca.view(b, -1, h, w)
        xca = self.xca_layer.proj(xca)
        
        return xca


    def forward(self, feat1, feat2=None):
        feat1 = self.star(feat1)
        xca = self.xca(feat1, feat2)
        
        return xca



# region - ASPP
class ASPP(nn.Module):
    def __init__(self, high_ch, low_ch, dilations, ratio=8):
        super().__init__()
        
        self.aspp = ASPPModule(in_ch=high_ch, 
                               out_ch=high_ch // ratio, 
                               atrous_rates=dilations)
        
        self.reduce = nn.Sequential(nn.Conv2d(high_ch // ratio, low_ch, 1, bias=False),
                                    nn.BatchNorm2d(low_ch),
                                    nn.ReLU(True))
        
    def forward(self, feat1, feat2):
        feat2 = self.aspp(feat2)
        feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=True)

        feat1 = self.reduce(feat1)

        return feat1, feat2


    
    
# region - SegHead
class SegHead(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(mid_ch),
                                  nn.ReLU(True),
                                  nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(mid_ch),
                                  nn.ReLU(True))
        
        self.classifier = nn.Conv2d(mid_ch, out_ch, 1, bias=True)
        
        
    def forward(self, feature, size):
        image_height, image_width = size

        feature = self.fuse(feature)
        
        out = self.classifier(feature)
        out = F.interpolate(out, (image_height, image_width), mode="bilinear", align_corners=True)

        return out