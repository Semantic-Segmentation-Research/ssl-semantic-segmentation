import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange


class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg, mcfg, pretrained_path):
        super(DeepLabV3Plus, self).__init__()
        self.is_corr = True
        self.pretrained_path = pretrained_path

        if 'resnet' in mcfg.backbone:
            backbone = resnet.__dict__[mcfg.backbone]
            self.backbone = backbone(pretrained_path,
                                     multi_grid=cfg['multi_grid'],
                                     replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert mcfg.backbone == 'xception'
            self.backbone = xception(True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
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
            self.corr = CustomCorr(nclass=mcfg.num_classes)
            self.proj = nn.Sequential(
                nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=1, bias=True),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
            )

    def forward(self, x, need_fp=False, use_corr=False):
        result_dict = {}
        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if need_fp:
            feats_decode = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))), torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = self.classifier(feats_decode)
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            
            out, out_fp = outs.chunk(2)
            if use_corr:
                proj_feats = self.proj(c4) # 채널수 줄이는 연산
                corr_out_dict = self.corr(proj_feats, out)
                
                result_dict['binary_norm_corr_map'] = corr_out_dict['binary_norm_corr_map']
                
                corr_out = corr_out_dict['corr_dec_out']
                corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
                
                result_dict['corr_out'] = corr_out
            
            result_dict['out'] = out
            result_dict['out_fp'] = out_fp

            return result_dict

        feats_decode = self._decode(c1, c4)
        out = self.classifier(feats_decode)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if use_corr:
            proj_feats = self.proj(c4)
            corr_out_dict = self.corr(proj_feats, out)
            
            result_dict['binary_norm_corr_map'] = corr_out_dict['binary_norm_corr_map']
            
            corr_out = corr_out_dict['corr_dec_out']
            corr_out = F.interpolate(corr_out, size=(h, w), mode="bilinear", align_corners=True)
            result_dict['corr_out'] = corr_out
        result_dict['out'] = out
        return result_dict

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


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
        
        # gram matrix
        
        f1 = rearrange(self.conv1(enc_out), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(enc_out), 'n c h w -> n c (h w)')
        dec_out_reshape = rearrange(dec_out, 'n c h w -> n c (h w)')
        
        # 수식 4번
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        
        corr_map_sample = self.sample(corr_map.detach(), enc_height, enc_width)
        # 7번 수식에서 C_hat
        result_dict['binary_norm_corr_map'] = self.normalize_corr_map(corr_map_sample, enc_height, enc_width, dec_height, dec_width)
        
        # 5번 수식
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
    
    
class CustomCorr(nn.Module):
    def __init__(self, nclass=21):
        super(CustomCorr, self).__init__()
        self.nclass = nclass
        self.conv1 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(256, self.nclass, kernel_size=1, stride=1, padding=0, bias=True)

    # Encoder output & Decoder output
    def forward(self, enc_out, dec_out):
        result_dict = {}
        
        enc_batch, enc_channel, enc_height, enc_width = enc_out.shape
        dec_height, dec_width = dec_out.shape[-2:]
        
        dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        # feature = F.interpolate(enc_out, (enc_height, enc_width), mode='bilinear', align_corners=True)
        
        # gram matrix
        features = self.conv1(enc_out)
        feat_batch, feat_chaneel, feat_height, feat_width = features.shape
        features = features.view(feat_batch, feat_chaneel, feat_height*feat_width)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (enc_channel * enc_height * enc_width)
        gram_attn = F.softmax(gram, dim=-1)
        
        result_dict['binary_norm_corr_map'] = self.normalize_gram_map(gram_attn)
        
        dec_out_reshape = rearrange(dec_out, 'n c h w -> n c (h w)')
        refined_out = torch.matmul(gram_attn, dec_out_reshape) # (B, C, HW)
        result_dict['corr_dec_out'] = refined_out.view(enc_batch, enc_channel, dec_height, dec_width)
        
        
        f1 = rearrange(self.conv1(enc_out), 'n c h w -> n c (h w)')
        f2 = rearrange(self.conv2(enc_out), 'n c h w -> n c (h w)')
        dec_out_reshape = rearrange(dec_out, 'n c h w -> n c (h w)')
        
        # 수식 4번
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        
        corr_map_sample = self.sample(corr_map.detach(), enc_height, enc_width)
        # 7번 수식에서 C_hat
        # (8,128,48,48)
        result_dict['binary_norm_corr_map'] = self.normalize_corr_map(corr_map_sample, enc_height, enc_width, dec_height, dec_width)
        
        # 5번 수식
        # (8,19,48,48)
        result_dict['corr_dec_out'] = rearrange(torch.matmul(dec_out_reshape, corr_map), 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        
        return result_dict


    def sample(self, corr_map, h_in, w_in):
        index = torch.randint(0, h_in * w_in - 1, [128])
        corr_map_sample = corr_map[:, index.long(), :]
        return corr_map_sample
    
    
    # region - region propagation
    def normalize_gram_map(self, gram_map):
        """
        gram_map: (Batch, C, C) - 채널 간 상관관계 행렬
        """
        n, c, _ = gram_map.shape
        
        # 1. 공간 보간(F.interpolate)은 더 이상 필요 없음 (C x C 이므로)
        # 대신 flatten 하여 정규화 준비
        gram_flat = rearrange(gram_map, 'n c1 c2 -> n (c1 c2)')
        
        # 2. Min-Max Scaling (수식 7번 논리 그대로 적용)
        min_val = torch.min(gram_flat, dim=1, keepdim=True)[0]
        max_val = torch.max(gram_flat, dim=1, keepdim=True)[0]
        
        range_ = max_val - min_val + 1e-8 # 0 나누기 방지
        temp_map = (gram_flat - min_val) / range_
        
        # 3. 임계값(Thresholding) 적용
        # 0.5보다 큰 상관관계를 가진 채널 쌍만 1로 활성화
        gram_mask = (temp_map > 0.5)
        
        # 4. 다시 (Batch, C, C) 형태로 복원
        norm_gram_map = rearrange(gram_mask, 'n (c1 c2) -> n c1 c2', c1=c, c2=c)
        
        return norm_gram_map