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

# region - DeepLabV3+
class DeepLabV3Plus(nn.Module):
    def __init__(self, tcfg, mcfg, pretrained_path):
        super(DeepLabV3Plus, self).__init__()
        self.tcfg = tcfg
        self.mcfg = mcfg

        if not osp.exists(pretrained_path): pretrained_path = False
            
        if 'resnet' in self.mcfg.backbone:
            backbone = resnet.__dict__[self.mcfg.backbone]
            self.backbone = backbone(pretrained_path,
                                     nf=self.mcfg.nf,
                                     bottleneck_nf=self.mcfg.bottleneck_nf,
                                     bottleneck_exp=self.mcfg.bottleneck_exp,
                                     multi_grid=self.mcfg.multi_grid,
                                     replace_stride_with_dilation=self.mcfg.replace_stride_with_dilation
                                     )
        else:
            assert self.mcfg.backbone == 'xception'
            self.backbone = xception(True)

        self.c14_aspp_module = context.ASPP(mcfg, 
                                 high_ch= self.mcfg.nf * 8 * self.mcfg.bottleneck_exp,
                                 low_ch=36,
                                 ratio=8)
        self.c12_aspp_module = context.ASPP(mcfg, 
                                 high_ch= self.mcfg.nf * 2 * self.mcfg.bottleneck_exp,
                                 low_ch=36,
                                 ratio=2)
        self.decoder = context.SegHead(in_ch= 36 + self.mcfg.nf * self.mcfg.bottleneck_exp,
                                           mid_ch=256,
                                           out_ch=self.mcfg.num_classes)
        self.us_decoder = context.SegHead(in_ch= 36 + 2*(self.mcfg.nf * self.mcfg.bottleneck_exp),
                                           mid_ch=256,
                                           out_ch=self.mcfg.num_classes)

        self.c4_corr = context.CrossCovarianceAtt(high_ch=self.mcfg.nf * 8 * self.mcfg.bottleneck_exp,
                                                in_ch=256,
                                                out_ch=128,
                                                output_size=self.tcfg.crop_size,
                                                nclass=self.mcfg.num_classes)
        self.c2_corr = context.CrossCovarianceAtt(high_ch=self.mcfg.nf * 2 * self.mcfg.bottleneck_exp,
                                                in_ch=256,
                                                out_ch=128,
                                                output_size=self.tcfg.crop_size,
                                                nclass=self.mcfg.num_classes)

    # region forward
    def forward(self, x, mode='train'):
        result_dict = {}
        image_height, image_width = x.shape[2:]

        c1, c2, c3, c4 = self.backbone.base_forward(x)
        if mode =='train':
            c1_lw_uw, c1_u_s = torch.split(c1, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            _, c2_u_s = torch.split(c2, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            c4_lw_uw, c4_u_s = torch.split(c4, [self.tcfg.batch_size*2, self.tcfg.batch_size], dim=0)
            
            # ---------------- Unlabel Strong Part ----------------
            c1_u_s1, c4_u_s1 = self.c14_aspp_module(c1_u_s, c4_u_s)
            c1_u_s2, c2_u_s1 = self.c12_aspp_module(c1_u_s, c2_u_s)
            c1_u_s = c1_u_s1 + c1_u_s2
            
            feature = torch.cat([c1_u_s, c2_u_s1, c4_u_s1], dim=1)
            out_u_s = self.us_decoder(feature, size=(image_height, image_width))
            result_dict['out_u_s'] = out_u_s
            
            result_corr = self.c4_corr(enc_out=c4_u_s, dec_out=out_u_s, aug_type='strong')
            result_dict['corr_out_u_s'] = result_corr["corr_dec_out"]
            # ---------------------------------------------------------
            
            # ---------------- label+unlabel Weak Part ----------------
            c1_lw_uw_fp, c4_lw_uw_fp = self.c14_aspp_module(
                torch.cat((c1_lw_uw, nn.Dropout2d(0.5)(c1_lw_uw))),
                torch.cat((c4_lw_uw, nn.Dropout2d(0.5)(c4_lw_uw)))
                )
            
            feature         = torch.cat([c1_lw_uw_fp, c4_lw_uw_fp], dim=1)
            outs            = self.decoder(feature, size=(image_height, image_width))

            out, out_fp = outs.chunk(2)
            result_c4corr = self.c4_corr(enc_out=c4_lw_uw, dec_out=out, aug_type='weak')
            # result_c2corr = self.c2_corr(enc_out=c2_lw_uw, dec_out=out, aug_type='weak')
            
            result_dict['binary_norm_corr_map'] = result_c4corr["binary_norm_corr_map"]
            result_dict['corr_out'] = result_c4corr["corr_dec_out"]
            result_dict['out_fp'] = out_fp
            # ---------------------------------------------------------
            
        elif mode == 'test':
            c1_u_s, c4_u_s  = self.c14_aspp_module(c1, c4)
            feature         = torch.cat([c1_u_s, c4_u_s], dim=1)
            out             = self.decoder(feature, size=(image_height, image_width))
            
        result_dict['out'] = out
        
        return result_dict