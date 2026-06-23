import model.backbone.resnet as resnet
from model.backbone.xception import xception

import torch
from torch import nn
import torch.nn.functional as F
import math
from einops import rearrange
import os.path as osp
from dataset import transform as dtf
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
            ('weak', context.SegHead(in_ch= 36 + mcfg.nf * mcfg.bttln_exp,
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
                                  reduc_ch=mcfg.bttln_exp,
                                  exp_ratio=4)),
            ("c2", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c2_ratio,
                                  reduc_ch=mcfg.bttln_exp*mcfg.enc_c2_ratio,
                                  exp_ratio=4)),
            ("c3", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c3_ratio,
                                  reduc_ch=mcfg.bttln_exp*mcfg.enc_c3_ratio,
                                  exp_ratio=4)),
            ("c4", context.FlowAtt(channel=mcfg.nf*mcfg.bttln_exp*mcfg.enc_c4_ratio,
                                  reduc_ch=mcfg.bttln_exp*mcfg.enc_c4_ratio,
                                  exp_ratio=4)
             )]))
        
        self.aspp_layer = nn.Sequential(OrderedDict([
            ("c14", context.ASPP(high_ch= mcfg.nf * mcfg.enc_c4_ratio * mcfg.bttln_exp,
                                 low_ch=36,
                                 dilations=mcfg.dilations,
                                 ratio=6))
            ]))
        
        self.fuse = nn.Sequential(OrderedDict([
            # (112, 112, 114)
            ("c1", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c1_ratio, 64, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (56, 56, 288)
            ("c2", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c2_ratio, 114, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(114),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(114, 64, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (28, 28, 576)
            ("c3", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c3_ratio, 288, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(288),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(288, 64, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 )),
            # (28, 28, 864)
            ("c4", nn.Sequential(nn.Conv2d(mcfg.nf*mcfg.bttln_exp*mcfg.enc_c4_ratio, 288, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(288),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 nn.Conv2d(288, 64, 3, padding=1, bias=True),
                                 nn.BatchNorm2d(64),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout2d(0.1),
                                 ))
        ]))
        
        self.cls = nn.Conv2d(64*4, mcfg.num_classes, 1, padding=0, bias=True)
        
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
            
            # ---------------- Unlabel Strong Part ----------------
            c1_us = self.flow_layer.c1(c1_lw_uw[:self.tcfg.batch_size], c1_us)
            c2_us = self.flow_layer.c2(c2_lw_uw[:self.tcfg.batch_size], c2_us)
            c3_us = self.flow_layer.c3(c3_lw_uw[:self.tcfg.batch_size], c3_us)
            c4_us = self.flow_layer.c4(c4_lw_uw[:self.tcfg.batch_size], c4_us)
            
            c1_us = F.interpolate(c1_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c2_us = F.interpolate(c2_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c3_us = F.interpolate(c3_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            c4_us = F.interpolate(c4_us, size=c1.shape[-2:], mode='bilinear', align_corners=True)
            
            feature = torch.cat([c1_us, c2_us, c3_us, c4_us], dim=1)
            pred_mask = self.decoder_layer.strong(feature, size=(image_height, image_width))
            result_dict['flow_mask_us'] = pred_mask
            
            result_corr = self.xca_layer.c4(enc_out=c4_us, dec_out=pred_mask, aug_type='strong')
            result_dict['corr_mask_us'] = result_corr["corr_dec_out"]
            # ---------------------------------------------------------
            
            # ---------------- label+unlabel Weak Part ----------------
            c1_lw_uw_fp, c4_lw_uw_fp = self.aspp_layer.c14(
                torch.cat((c1_lw_uw, nn.Dropout2d(0.5)(c1_lw_uw))),
                torch.cat((c4_lw_uw, nn.Dropout2d(0.5)(c4_lw_uw)))
                )
            
            feature         = torch.cat([c1_lw_uw_fp, c4_lw_uw_fp], dim=1)
            pred_masks      = self.decoder_layer.weak(feature, size=(image_height, image_width))

            pred_mask, pred_mask_fp = pred_masks.chunk(2)
            result_corr = self.xca_layer.c4(enc_out=c4_lw_uw, dec_out=pred_mask, aug_type='weak')
            
            result_dict['binary_norm_corr_map'] = result_corr["binary_norm_corr_map"]
            result_dict['corr_mask_lw'] = result_corr["corr_dec_out"]
            result_dict['mask_lw_uw_fp'] = pred_mask_fp
            result_dict['mask_lw_uw'] = pred_mask
            # ---------------------------------------------------------
            
            # ----------------------- label Part -----------------------
            c1_lw = self.flow_layer.c1(c1_lw_uw[:self.tcfg.batch_size], c1_lw_uw[:self.tcfg.batch_size])
            c2_lw = self.flow_layer.c2(c2_lw_uw[:self.tcfg.batch_size], c2_lw_uw[:self.tcfg.batch_size])
            c3_lw = self.flow_layer.c3(c3_lw_uw[:self.tcfg.batch_size], c3_lw_uw[:self.tcfg.batch_size])
            c4_lw = self.flow_layer.c4(c4_lw_uw[:self.tcfg.batch_size], c4_lw_uw[:self.tcfg.batch_size])
            
            c1_lw = self.fuse.c1(c1_lw)
            c2_lw = self.fuse.c2(c2_lw)
            c3_lw = self.fuse.c3(c3_lw)
            c4_lw = self.fuse.c4(c4_lw)
            
            c2_lw = F.interpolate(c2_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            c3_lw = F.interpolate(c3_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            c4_lw = F.interpolate(c4_lw, size=c1_lw.shape[-2:], mode='bilinear', align_corners=True)
            
            feature = torch.cat([c1_lw, c2_lw, c3_lw, c4_lw], dim=1)
            feature = self.cls(feature)
            pred_mask = F.interpolate(feature, size=(image_height, image_width), mode='bilinear', align_corners=True)
            
            result_dict['flow_mask_lw'] = pred_mask
            # ---------------------------------------------------------
            
        elif mode == 'val':
            c1_us, c4_us  = self.aspp_layer.c14(c1, c4)
            feature         = torch.cat([c1_us, c4_us], dim=1)
            out             = self.decoder_layer.weak(feature, size=(image_height, image_width))

            result_dict['out'] = out
        
        return result_dict