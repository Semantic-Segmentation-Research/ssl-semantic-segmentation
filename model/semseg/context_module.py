import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange




def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 
                                    padding=atrous_rate,
                                    dilation=atrous_rate, bias
                                    =False),
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
    def __init__(self, in_channels, out_channels, atrous_rates):
        super(ASPPModule, self).__init__()
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
    def __init__(self, high_ch, in_ch, out_ch, output_size, nclass=21):
        super(CrossCovarianceAtt, self).__init__()

        self.output_size = output_size
        self.reduction = nn.Sequential(
            nn.Conv2d(high_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        
        self.qkv_conv = nn.Conv2d(in_ch, out_ch * 3, kernel_size=1, stride=1, padding=0, bias=True)
        self.dwconv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, groups=out_ch)
        self.proj   = nn.Conv2d(out_ch, nclass, kernel_size=3, padding=1)
        
        
    # Encoder output & Decoder output
    def forward(self, enc_out, dec_out, aug_type='weak'):
        result_dict = {}
        
        proj_feats = self.reduction(enc_out)
        
        _, _, enc_height, enc_width = proj_feats.shape
        dec_height, dec_width = dec_out.shape[-2:]
        
        qkv = self.qkv_conv(proj_feats).flatten(2)
        q, k, v= qkv.chunk(3, dim=1)
        
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
        
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = F.softmax(attn, dim=-1)

        xca = torch.bmm(attn, v)
        xca = F.softmax(xca, dim=1)
        
        if aug_type =='weak':
            result_dict['binary_norm_corr_map'] = self.normalize_xca_map(xca, enc_height, enc_width, dec_height, dec_width)
        
        xca_conf_reshape = rearrange(xca, 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        xca_conf_reshape = self.proj(xca_conf_reshape)
        
        # 7번 수식에서 C_hat (8, 128, 384, 384)
        dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        corr_dec_out = dec_out * xca_conf_reshape
        corr_dec_out = F.interpolate(corr_dec_out, size=(self.output_size, self.output_size), mode="bilinear", align_corners=True)
        # 5번 수식
        # (8,19,48,48)
        result_dict['corr_dec_out'] = corr_dec_out
        
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
    

# region - SegHead
class SegHead(nn.Module):
    def __init__(self, mcfg, high_ch, mid_ch, low_ch, ratio=8):
        super().__init__()
        
        self.aspp = ASPPModule(in_channels=high_ch, 
                               out_channels=high_ch // ratio, 
                               atrous_rates=mcfg.dilations)
        
        self.reduce = nn.Sequential(nn.Conv2d(high_ch // ratio, low_ch, 1, bias=False),
                                    nn.BatchNorm2d(low_ch),
                                    nn.ReLU(True))
        
        self.fuse = nn.Sequential(nn.Conv2d(high_ch // ratio + low_ch, mid_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(mid_ch),
                                  nn.ReLU(True),
                                  nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(mid_ch),
                                  nn.ReLU(True))
        
        self.classifier = nn.Conv2d(mid_ch, mcfg.num_classes, 1, bias=True)
        
        
    def forward(self, feat1, feat2, size):
        image_height, image_width = size
        
        feat2 = self.aspp(feat2)
        feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=True)

        feat1 = self.reduce(feat1)

        feature = torch.cat([feat1, feat2], dim=1)
        feature = self.fuse(feature)
        
        out = self.classifier(feature)
        out = F.interpolate(out, (image_height, image_width), mode="bilinear", align_corners=True)
        
        return out
