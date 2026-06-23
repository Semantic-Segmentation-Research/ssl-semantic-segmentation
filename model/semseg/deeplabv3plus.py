import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import os.path as osp
from model.semseg import context_module as context
from collections import OrderedDict

# region - DeepLabV3+
class DeepLabV3Plus(nn.Module):
    def __init__(self, tcfg, mcfg, pretrained_path=''):
        super(DeepLabV3Plus, self).__init__()
        self.tcfg = tcfg
        self.mcfg = mcfg

        if not osp.exists(pretrained_path): pretrained_path = False
        
        backbone = resnet.__dict__[mcfg.backbone]
        self.backbone = backbone(pretrained_path, mcfg=mcfg)

        self.decoder_layer = nn.Sequential(OrderedDict([
            ('strong', context.SegHead(in_ch= mcfg.nf*mcfg.bttln_exp*(mcfg.enc_c1_ratio+mcfg.enc_c2_ratio+mcfg.enc_c3_ratio+mcfg.enc_c4_ratio),
                                       mid_ch=256,
                                       out_ch=mcfg.num_classes)),
            ('weak', context.SegHead(in_ch= mcfg.nf * mcfg.bttln_exp *4,
                                     mid_ch=256,
                                     out_ch=mcfg.num_classes))
        ]))
        
        self.xca_layer = nn.Sequential(OrderedDict([
            ("c4", context.CrossCovarianceAtt(reduc_in_ch=mcfg.nf * mcfg.enc_c4_ratio * mcfg.bttln_exp,
                                                reduc_out_ch=mcfg.num_classes,
                                                mid_ch=128,
                                                output_size=self.tcfg.crop_size,
                                                nclass=mcfg.num_classes))
        ]))

        
        self.flow_layer = nn.Sequential(OrderedDict([
            ("c1", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp,
                                #   reduc_ch=mcfg.bttln_exp,
                                  reduc_ch=mcfg.nf,
                                  exp_ratio=4,
                                  method='sum')),
            ("c2", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c2_ratio,
                                #   reduc_ch=mcfg.bttln_exp*mcfg.enc_c2_ratio,
                                  reduc_ch=mcfg.nf*mcfg.enc_c2_ratio,
                                  exp_ratio=4,
                                  method='sum')),
            ("c3", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c3_ratio,
                                #   reduc_ch=mcfg.bttln_exp*mcfg.enc_c3_ratio,
                                  reduc_ch=mcfg.nf*mcfg.enc_c3_ratio,
                                  exp_ratio=4,
                                  method='mul')),
            ("c4", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c4_ratio,
                                #   reduc_ch=mcfg.bttln_exp*mcfg.enc_c4_ratio,
                                  reduc_ch=mcfg.nf*mcfg.enc_c4_ratio,
                                  exp_ratio=4,
                                  method='mul')
             )]))
        
        # region aspp
        self.aspp_layer = nn.Sequential(OrderedDict([
            ("c14", context.ASPP(high_ch= mcfg.nf * mcfg.enc_c4_ratio * mcfg.bttln_exp,
                                 low_ch=mcfg.nf * mcfg.enc_c2_ratio * mcfg.bttln_exp,
                                 dilations=mcfg.dilations,
                                 ratio=mcfg.bttln_exp))
            ]))
        
        # region fuse
        self.fuse = nn.Sequential(OrderedDict([
            # (112, 112, 114)
            ("c1", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c1_ratio, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (56, 56, 288)
            ("c2", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c2_ratio, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(mcfg.nf*mcfg.bttln_exp, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (28, 28, 576)
            ("c3", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c3_ratio, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(mcfg.nf*mcfg.bttln_exp, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (28, 28, 864)
            ("c4", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c4_ratio, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(mcfg.nf*mcfg.bttln_exp, mcfg.nf*mcfg.bttln_exp, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(mcfg.nf*mcfg.bttln_exp),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 ))
        ]))
        
    
    # region update proto
    # 정답지(Label)를 보고 메모리 뱅크 업데이트
    @torch.no_grad()
    def update_prototypes(self, c1, c2, c3, c4, label, momentum=0.999):
        """
        Labeled 배치의 backbone 피처(c1~c4)와 GT label을 이용해
        각 FlowAtt 인스턴스의 class_prototypes를 EMA 방식으로 갱신.

        갱신 절차:
          1. 각 스케일의 피처를 xca.reduction 통과 → [B, reduc_ch, H', W']
          2. label을 해당 해상도로 nearest 리사이즈
          3. 클래스별 픽셀 피처 평균 계산
          4. class_prototypes[cls] = momentum * old + (1-momentum) * new_mean

        Args:
            c1~c4  : backbone 출력 피처 (Labeled 이미지 기준)
            label  : GT 세그멘테이션 레이블 [B, H, W]
            momentum: EMA 갱신 계수 (기본 0.999) #TODO: epoch에 따른 값 변화(0.999 ->0.5)
        """
        scales = [
            (c1, self.flow_layer.c1),
            (c2, self.flow_layer.c2),
            (c3, self.flow_layer.c3),
            (c4, self.flow_layer.c4),
        ]

        for feat, layer in scales:
            b, c, h, w = feat.shape

            # 레이블을 피처 해상도에 맞게 리사이즈
            label_resized = F.interpolate(
                label.unsqueeze(1).float(), size=(h, w), mode='nearest'
            ).squeeze(1).long()   # [B, H', W']

            # protoAttn.reduction 통과 → [B, reduc_ch, H', W']
            feat_reduced = layer.protoAttn.reduction(feat)
            reduc_ch = feat_reduced.shape[1]

            # 픽셀 방향으로 펼치기: [reduc_ch, B*H'*W']
            feat_flat  = feat_reduced.transpose(1, 0).contiguous().view(reduc_ch, -1)
            label_flat = label_resized.view(-1)  # [B*H'*W']

            for cls_idx in range(self.mcfg.num_classes):
                mask = (label_flat == cls_idx)
                if mask.sum() == 0: continue
                
                # 해당 클래스 픽셀 피처 평균 [reduc_ch]
                cls_mean = feat_flat[:, mask].mean(dim=1)
                
                # EMA 갱신
                layer.class_prototypes[cls_idx] = (momentum * layer.class_prototypes[cls_idx] + (1 - momentum) * cls_mean)
                
                
    # region forward
    def forward(self, x, mode='train'):
        result_dict = {}
        image_height, image_width = x.shape[2:]

        c1, c2, c3, c4 = self.backbone.base_forward(x)
        if mode =='train':
            c1_lw_uw, c1_us = torch.split(c1, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            c2_lw_uw, c2_us = torch.split(c2, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            c3_lw_uw, c3_us = torch.split(c3, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            c4_lw_uw, c4_us = torch.split(c4, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            
            c1_lw, c1_uw = c1_lw_uw[:self.tcfg.batch_size], c1_lw_uw[self.tcfg.batch_size:]
            c2_lw, c2_uw = c2_lw_uw[:self.tcfg.batch_size], c2_lw_uw[self.tcfg.batch_size:]
            c3_lw, c3_uw = c3_lw_uw[:self.tcfg.batch_size], c3_lw_uw[self.tcfg.batch_size:]
            c4_lw, c4_uw = c4_lw_uw[:self.tcfg.batch_size], c4_lw_uw[self.tcfg.batch_size:]
            
            # ---------------- Unlabel Strong Part ----------------
            # c1_us = self.flow_layer.c1(c1_lw, c1_us)
            # c2_us = self.flow_layer.c2(c2_lw, c2_us)
            # c3_us = self.flow_layer.c3(c3_lw, c3_us)
            # c4_us = self.flow_layer.c4(c4_lw, c4_us)
            c1_u = self.flow_layer.c1(torch.cat([c1_us, c1_uw], dim=0))
            c2_u = self.flow_layer.c2(torch.cat([c2_us, c2_uw], dim=0))
            c3_u = self.flow_layer.c3(torch.cat([c3_us, c3_uw], dim=0))
            c4_u = self.flow_layer.c4(torch.cat([c4_us, c4_uw], dim=0))
            
            c1_us, c1_uw = c1_u.chunk(2, dim=0)
            c2_us, c2_uw = c2_u.chunk(2, dim=0)
            c3_us, c3_uw = c3_u.chunk(2, dim=0)
            c4_us, c4_uw = c4_u.chunk(2, dim=0)
            
            c2_us = F.interpolate(c2_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c3_us = F.interpolate(c3_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c4_us = F.interpolate(c4_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            
            feature = torch.cat([c1_us, c2_us, c3_us, c4_us], dim=1)
            flow_logit_us = self.decoder_layer.strong(feature, size=(image_height, image_width))
            result_dict['flow_logit_us'] = flow_logit_us
            
            result_corr = self.xca_layer.c4(enc_out=c4_us, dec_out=flow_logit_us, aug_type='strong')
            result_dict['corr_logit_us'] = result_corr["corr_dec_out"]
            
            # ---------------- Unlabel Weak Part ----------------
            c1_uw = self.flow_layer.c1(c1_uw)
            c2_uw = self.flow_layer.c2(c2_uw)
            c3_uw = self.flow_layer.c3(c3_uw)
            c4_uw = self.flow_layer.c4(c4_uw)
            
            c2_uw = F.interpolate(c2_uw, size=c1_uw.shape[-2:], mode='bilinear', align_corners=True)
            c3_uw = F.interpolate(c3_uw, size=c1_uw.shape[-2:], mode='bilinear', align_corners=True)
            c4_uw = F.interpolate(c4_uw, size=c1_uw.shape[-2:], mode='bilinear', align_corners=True)

            feature_uw = torch.cat([c1_uw, c2_uw, c3_uw, c4_uw], dim=1)
            flow_logit_uw = self.decoder_layer.strong(feature_uw, size=(image_height, image_width))
            result_dict['flow_logit_uw'] = flow_logit_uw
            # ---------------------------------------------------------
            
            result_dict['flow_logit_uws'] = flow_logit_uw + flow_logit_us
            
            # ---------------- label+unlabel Weak Part ----------------
            c1_lw_uw_fp, c4_lw_uw_fp = self.aspp_layer.c14(
                torch.cat((c1_lw_uw, nn.Dropout2d(0.5)(c1_lw_uw))),
                torch.cat((c4_lw_uw, nn.Dropout2d(0.5)(c4_lw_uw)))
                )

            feature     = torch.cat([c1_lw_uw_fp, c4_lw_uw_fp], dim=1)
            logits_fp   = self.decoder_layer.weak(feature, size=(image_height, image_width))

            logit_uw_lw, logit_uw_lw_fp = logits_fp.chunk(2)
            result_corr = self.xca_layer.c4(enc_out=c4_lw_uw, dec_out=logit_uw_lw, aug_type='weak')
            
            result_dict['binary_norm_corr_map'] = result_corr["binary_norm_corr_map"]
            result_dict['corr_logit_lw_uw']     = result_corr["corr_dec_out"]
            result_dict['logit_lw_uw_fp']       = logit_uw_lw_fp
            result_dict['logit_lw_uw']          = logit_uw_lw
            # ---------------------------------------------------------
            
            # ----------------------- label Part -----------------------
            c1_lw = self.fuse.c1(c1_lw)
            c2_lw = self.fuse.c2(c2_lw)
            c3_lw = self.fuse.c3(c3_lw)
            c4_lw = self.fuse.c4(c4_lw)
            
            c2_lw = F.interpolate(c2_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            c3_lw = F.interpolate(c3_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            c4_lw = F.interpolate(c4_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            
            feature_lw = torch.cat([c1_lw, c2_lw, c3_lw, c4_lw], dim=1)
            # logit = self.decoder_layer.strong(feature_lw, size=(image_height, image_width))
            logit = self.decoder_layer.weak(feature_lw, size=(image_height, image_width))

            result_dict['flow_logit_lw'] = logit
            # ---------------------------------------------------------
            
        elif mode == 'val':
            c1_, c4_  = self.aspp_layer.c14(c1, c4)
            feature_ = torch.cat([c1_, c4_], dim=1)
            # out      = self.decoder_layer.weak(feature, size=(image_height, image_width))

            # result_dict['out'] = out
        
            # c1_val = self.flow_layer.c1(c1)
            # c2_val = self.flow_layer.c2(c2)
            # c3_val = self.flow_layer.c3(c3)
            # c4_val = self.flow_layer.c4(c4)
            
            c1_val = self.fuse.c1(c1)
            c2_val = self.fuse.c2(c2)
            c3_val = self.fuse.c3(c3)
            c4_val = self.fuse.c4(c4)
            
            c2_val = F.interpolate(c2_val, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c3_val = F.interpolate(c3_val, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c4_val = F.interpolate(c4_val, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            
            feature = torch.cat([c1_val, c2_val, c3_val, c4_val], dim=1)
            # out = self.decoder_layer.strong(feature, size=(image_height, image_width))
            feautre_val = feature_ + feature
            out = self.decoder_layer.weak(feautre_val, size=(image_height, image_width))

            result_dict['out'] = out
        
        return result_dict