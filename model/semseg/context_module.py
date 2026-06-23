import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath
from collections import OrderedDict
import math


# region - ASPPConv
def ASPPConv(in_channels, out_channels, atrous_rate):
    # block = nn.Sequential(
    #     # depthwise conv: groups=in_channels keeps channels separate
    #     nn.Conv2d(in_channels,
    #               in_channels,
    #               kernel_size=3,
    #               padding=atrous_rate,
    #               dilation=atrous_rate,
    #               groups=in_channels,
    #               bias=False),
    #     nn.BatchNorm2d(in_channels),
    #     nn.ReLU6(True),
    #     # pointwise conv to mix channels and adjust to desired out_channels
    #     nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
    #     nn.BatchNorm2d(out_channels),
    #     nn.ReLU6(True),
    # )
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
        
        # q = q / F.normalize(q, p=2, dim=1).clamp_min(1e-6)
        # k = k / F.normalize(k, p=2, dim=1).clamp_min(1e-6)
        
        attn = torch.bmm(q, k.transpose(1, 2))
        # attn /= self.temperature
        # attn = F.softmax(attn, dim=-1)
        attn = attn / math.sqrt(q.shape[-1])
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1e6, neginf=-1e6)
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        xca = torch.bmm(attn, v)

        corr_prob = F.softmax(xca, dim=1)

        if aug_type =='weak':
            # corr_prob = F.softmax(xca, dim=1)
            result_dict['binary_norm_corr_map'] = self.normalize_xca_map(corr_prob, enc_height, enc_width, dec_height, dec_width)
        
        corr_prob_reshaped = rearrange(corr_prob, 'n c (h w) -> n c h w', h=enc_height, w=enc_width)
        corr_prob_reshaped = self.proj(corr_prob_reshaped)
        
        dec_out = F.interpolate(dec_out, (enc_height, enc_width), mode='bilinear', align_corners=True)
        corr_dec_out = dec_out * corr_prob_reshaped
        # dec_out = F.interpolate(dec_out.detach(), (enc_height, enc_width), mode='bilinear', align_corners=True)
        # corr_dec_out = dec_out + corr_prob_reshaped
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
    
    
# region - PrototypeAttention
class PrototypeAttention(nn.Module):
    """
    픽셀 피처(feat)가 클래스 프로토타입 메모리 뱅크를 참조하여 자신을 보정.
 
    Args:
        in_ch      : 입력/출력 채널 수 (residual 연결을 위해 동일하게 유지)
        out_ch     : reduction 후 채널 수 (= FlowAtt의 reduc_ch)
        num_classes: 클래스 프로토타입 개수 (기본 19)
 
    Flow:
        feat [B, C, H, W]
          → reduction → x [B, C', H, W]
          → q_conv    → Q [B, H*W, C']
          K, V ← k_proj / v_proj (prototypes [B, 19, C'])
          → attention [B, H*W, 19] × V → out [B, C', H, W]
          → proj (C' → C)
          → feat + out   (residual)
    """
    def __init__(self, in_ch, out_ch, num_classes=19):
        super(PrototypeAttention, self).__init__()
        self.num_classes = num_classes
 
        self.reduction = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.q_conv = nn.Conv2d(out_ch, out_ch, kernel_size=1, bias=True)
        self.k_proj = nn.Linear(out_ch, out_ch, bias=True)
        self.v_proj = nn.Linear(out_ch, out_ch, bias=True)
        self.proj = nn.Sequential(
            nn.Conv2d(out_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Hardswish(inplace=True)
        )
        self.temperature = nn.Parameter(torch.tensor(0.05))
        self.gamma = nn.Parameter(1e-6 * torch.ones((1, in_ch, 1, 1)), requires_grad=True)
        
        

    def forward(self, feat, prototypes):
        """
        feat       : [B, C,  H, W], Unlabel
        prototypes : [num_classes, C']  (FlowAtt.class_prototypes 버퍼)
        returns    : [B, C,  H, W]  (feat + attention-refined residual)
        """
        b, _, h, w = feat.shape

        x = self.reduction(feat)                          # [B, C', H, W]

        # Q: 픽셀별 피처  [B, H*W, C']
        q = self.q_conv(x).flatten(2).transpose(1, 2)

        # K, V: 프로토타입을 배치 크기만큼 복사  [B, 19, C']
        proto = prototypes.unsqueeze(0).expand(b, -1, -1)
        k = self.k_proj(proto)
        v = self.v_proj(proto)

        # q_norm = q / F.normalize(q, p=2, dim=1).clamp_min(1e-6)
        # k_norm = k / F.normalize(k, p=2, dim=1).clamp_min(1e-6)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        # [B, H*W, C'] × [B, C', 19] → [B, H*W, 19]
        # temperature = self.temperature.clamp_min(1e-3)
        # attn = torch.bmm(q_norm, k_norm.transpose(1, 2)) / temperature
        attn = torch.bmm(q_norm, k_norm.transpose(1, 2))
        attn = attn / math.sqrt(q_norm.shape[-1])
        attn = torch.nan_to_num(attn, nan=0.0, posinf=1e6, neginf=-1e6) # 방어코드
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)

        # [B, H*W, 19] × [B, 19, C'] → [B, H*W, C'] → [B, C', H, W]
        out = torch.bmm(attn, v).transpose(1, 2).view(b, -1, h, w)
        out = self.proj(out)  # [B, C, H, W]

        return out + self.gamma * feat



# region - FlowAtt
class FlowAtt(nn.Module):
    """
    Memory Bank 기반 피처 정제 모듈.
 
    구성:
      1. PrototypeAttention (xca)
         - 입력 피처를 reduction 후 class_prototypes(메모리 뱅크)를 Key/Value로 삼아
           Cross-Attention → residual 보정
      2. StarNet (star_layer)
         - Asymmetric depthwise conv (H×1, 1×W 분리) × element-wise 곱으로 피처 정제
         - DropPath 적용
 
    메모리 뱅크 갱신:
      - class_prototypes [num_classes, reduc_ch] : 학습 파라미터가 아닌 버퍼
      - DeepLabV3Plus.update_prototypes() 에서 Labeled 피처 기준 EMA 갱신
      - 갱신 시 xca.reduction 을 통과한 [reduc_ch] 차원 벡터를 사용 (차원 일치)
 
    Args:
        channel    : 입력/출력 채널 (backbone feature 채널)
        reduc_ch   : reduction 후 채널 (= 메모리 뱅크 채널 크기)
        exp_ratio  : StarNet 내부 expansion ratio
        num_classes: 클래스 수 (기본 19)
        drop_path  : DropPath rate (기본 0.1)
        method     : StarNet 분기 결합 방식 'sum' | 'mul' (기본 'sum')
    """
    def __init__(self, channel, reduc_ch, exp_ratio,
                 num_classes=19, drop_path=0.1, method='sum'):
        super(FlowAtt, self).__init__()
        self.method = method

        # ── 메모리 뱅크: [num_classes, reduc_ch] ──────────────────────────────
        # 역전파로 갱신되지 않으며, update_prototypes()에서 EMA 방식으로만 갱신됨
        self.register_buffer("class_prototypes", torch.zeros(num_classes, reduc_ch))

        # ── PrototypeAttention (Cross-Attention w/ Memory Bank) ───────────────
        self.protoAttn = PrototypeAttention(in_ch=channel, out_ch=reduc_ch, num_classes=num_classes)

        # ── StarNet (Asymmetric Depthwise Star Operation) ─────────────────────
        self.star_layer = nn.Sequential(OrderedDict([
            ('reduction', nn.Sequential(
                nn.Conv2d(channel, reduc_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(reduc_ch),
                nn.ReLU(True)
            )),
            ('asy_f1_sum', nn.Sequential(
                nn.Conv2d(reduc_ch, reduc_ch * exp_ratio, (1, 3), 1, (0, 1), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.Conv2d(reduc_ch * exp_ratio, reduc_ch * exp_ratio, (3, 1), 1, (1, 0), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.ReLU(True)
            )),
            ('asy_f2_sum', nn.Sequential(
                nn.Conv2d(reduc_ch, reduc_ch * exp_ratio, (3, 1), 1, (1, 0), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.Conv2d(reduc_ch * exp_ratio, reduc_ch * exp_ratio, (1, 3), 1, (0, 1), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.ReLU(True)
            )),
            ('asy_f1', nn.Sequential(
                nn.Conv2d(reduc_ch, reduc_ch * exp_ratio, (1, 3), 1, (0, 1), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.Conv2d(reduc_ch * exp_ratio, reduc_ch * exp_ratio, (3, 1), 1, (1, 0), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
            )),
            ('asy_f2', nn.Sequential(
                nn.Conv2d(reduc_ch, reduc_ch * exp_ratio, (3, 1), 1, (1, 0), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio),
                nn.Conv2d(reduc_ch * exp_ratio, reduc_ch * exp_ratio, (1, 3), 1, (0, 1), bias=False),
                nn.BatchNorm2d(reduc_ch * exp_ratio)
            )),
            ('g',      nn.Conv2d(reduc_ch * exp_ratio, channel, 1, 1, 0, bias=True)),
            ('dwconv', nn.Sequential(
                nn.Conv2d(channel, channel, 1, 1, 0, bias=False),
                nn.BatchNorm2d(channel),
                nn.ReLU6(True)
            )),
            ('relu',      nn.ReLU(inplace=True)),
            ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
        ]))

    def forward(self, feat):
        """
        feat : [B, C, H, W]
        returns : [B, C, H, W]

        1) xca: 메모리 뱅크 참조 → 피처 보정 (feat_att)
        2) star: asymmetric star op → residual 추가
        """
        # 1. Memory Bank Cross-Attention
        feat_att = self.protoAttn(feat, self.class_prototypes)

        # 2. StarNet
        x = self.star_layer.reduction(feat_att)
        if self.method == 'sum':
            x1 = self.star_layer.asy_f1_sum(x)
            x2 = self.star_layer.asy_f2_sum(x)
            x  = x1 + x2
        elif self.method == 'mul':
            x1 = self.star_layer.asy_f1(x)
            x2 = self.star_layer.asy_f2(x)
            x  = x1 * self.star_layer.relu(x2)

        x  = self.star_layer.dwconv(self.star_layer.g(x))

        return feat_att + self.star_layer.drop_path(x)
    
    
# class _FlowAtt(nn.Module):
#     """
#     label의 정보를 unlabel에게 전달
#     """
#     def __init__(self, channel, reduc_ch, exp_ratio, num_classes=19, drop_path=0.1, method='sum'):
#         super(_FlowAtt, self).__init__()

#         self.method = method
#         # -------------------------- Class Prototype Memory Bank --------------------------
#         # 역전파로 업데이트되지 않고, Labeled 데이터의 피처 평균값으로 갱신되는 버퍼 메모리
#         # 크기: [19(클래스 수), 채널수]
#         self.register_buffer("class_prototypes", torch.zeros(num_classes, channel))
#         # ----------------------------------------------------------------------------------
        
#         # -------------------------- StarNet Module --------------------------
#         self.star_layer = nn.Sequential(OrderedDict([
#             ('reduction', nn.Sequential(
#                 nn.Conv2d(channel, reduc_ch, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(reduc_ch),
#                 nn.ReLU(inplace=True))
#             ),
            
#             ('asy_f1', nn.Sequential(
#                 nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=True),
#                 # nn.BatchNorm2d(reduc_ch*exp_ratio),
#                 nn.Conv2d(reduc_ch*exp_ratio, reduc_ch*exp_ratio, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=False),
#                 nn.BatchNorm2d(reduc_ch*exp_ratio))
#             ),
            
#             ('asy_f2', nn.Sequential(
#                 nn.Conv2d(reduc_ch, reduc_ch*exp_ratio, kernel_size=(3, 1), stride=1, padding=(1, 0), bias=True),
#                 # nn.BatchNorm2d(reduc_ch*exp_ratio),
#                 nn.Conv2d(reduc_ch*exp_ratio, reduc_ch*exp_ratio, kernel_size=(1, 3), stride=1, padding=(0, 1), bias=False),
#                 nn.BatchNorm2d(reduc_ch*exp_ratio))
#             ),
            
#             ('g', nn.Conv2d(reduc_ch*exp_ratio, channel, kernel_size=1, stride=1, padding=0, bias=True)),

#             ('dwconv', nn.Sequential(
#                 nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(channel),
#                 nn.ReLU6(inplace=True)
#             )),
#             # ('hswish', nn.Hardswish(inplace=True)),
#             ('relu', nn.ReLU(inplace=True)),
#             ('drop_path', DropPath(drop_path) if drop_path > 0. else nn.Identity())
#         ]))
        
#         for m in self.modules():
#             if isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#         # ------------------------------------------------------------------
        
#         # -------------------------- XCA Module --------------------------
#         self.xca_layer = nn.Sequential(OrderedDict([
#             # ('kv_conv', nn.Conv2d(channel, channel * 2, kernel_size=1, stride=1, padding=0, bias=True)),
#             ('q_conv', nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)),
#             # ('k_conv', nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)),
#             # ('v_conv', nn.Conv2d(channel, channel, kernel_size=1, stride=1, padding=0, bias=True)),
#             ('k_conv', nn.Linear(channel, channel, bias=True)),
#             ('v_conv', nn.Linear(channel, channel, bias=True)),
#             ("proj", nn.Sequential(
#                 nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, bias=False),
#                 nn.BatchNorm2d(channel),
#                 nn.Hardswish(inplace=True)
#                 ))
#         ]))
#         # self.temperature = nn.Parameter(0.05 * torch.ones(1, channel, 1))
#         self.temperature = nn.Parameter(torch.ones(1) * 0.05)
#         # --------------------------------------------------------------------
        
        
#     def star(self, feat):
#         input = feat
        
#         x = self.star_layer.reduction(feat)
        
#         x1 = self.star_layer.asy_f1(x)
#         x2 = self.star_layer.asy_f2(x)
            
#         if self.method == 'sum':
#             x = x1 + self.star_layer.relu(x2)
        
#         elif self.method == 'mul':
#             # x1, x2 = self.star_layer.asy_f1(x), self.star_layer.asy_f2(x)
#             x = x1 * self.star_layer.relu(x2)
            
#         x = self.star_layer.dwconv(self.star_layer.g(x))
        
#         x = input + self.star_layer.drop_path(x)
                
#         return x
    
    
#     # def xca(self, feat1, feat2):
#     #     b, _, h, w = feat1.shape
        
#     #     q = self.xca_layer.q_conv(feat1).flatten(2)
#     #     # kv = self.xca_layer.kv_conv(feat1).flatten(2)
#     #     k = self.xca_layer.k_conv(feat1).flatten(2)
#     #     v = self.xca_layer.v_conv(feat2).flatten(2)
#     #     # k, v = kv.chunk(2, dim=1)
        
#     #     q_norm = torch.norm(q, p=2, dim=2, keepdim=True)
#     #     k_norm = torch.norm(k, p=2, dim=2, keepdim=True)
        
#     #     q = q / (q_norm + 1e-6)
#     #     k = k / (k_norm + 1e-6)
        
#     #     attn = torch.bmm(q, k.transpose(1, 2)).float()
#     #     attn = attn / self.temperature.clamp(min=0.01)
#     #     attn = F.softmax(attn, dim=-1)
#     #     xca = torch.bmm(attn, v)

#     #     xca = xca.view(b, -1, h, w)
#     #     xca = self.xca_layer.proj(xca)
        
#     #     unlabel_feat = feat2 + xca
        
#     #     return unlabel_feat
    
#     def xca_w_memory(self, feat):
#         """
#         입력된 피처(feat)가 기학습된 정답 가이드라인(class_prototypes)을 참조하게 만듦
#         """
#         b, c, h, w = feat.shape
        
#         # 1. Query: 입력 피처 벡터들 [B, C, H*W] -> [B, H*W, C]
#         q = self.xca_layer.q_conv(feat).flatten(2).transpose(1, 2)
        
#         # 2. Key, Value: 메모리 뱅크에서 가져옴 [19, C] -> 배치 크기만큼 복사 [B, 19, C]
#         proto = self.class_prototypes.unsqueeze(0).expand(b, -1, -1)
#         k = self.xca_layer.k_conv(proto)
#         v = self.xca_layer.v_conv(proto)
        
#         # L2 정규화 (채널 방향 안정성 확보)
#         q_norm = F.normalize(q, p=2, dim=-1)
#         k_norm = F.normalize(k, p=2, dim=-1)
        
#         # 3. Attention 계산: 각 픽셀이 19개 클래스 중 어느 것과 가장 유사한지 공통점을 찾음
#         # [B, H*W, C] x [B, C, 19] -> [B, H*W, 19]
#         attn = torch.bmm(q_norm, k_norm.transpose(1, 2)) / self.temperature
#         attn = F.softmax(attn, dim=-1)
        
#         # 4. 정보 전수: 유사도에 따라 클래스 메모리 피처(v)를 융합함
#         # [B, H*W, 19] x [B, 19, C] -> [B, H*W, C] -> [B, C, H, w]
#         out = torch.bmm(attn, v).transpose(1, 2).view(b, c, h, w)
#         out = self.xca_layer.proj(out)
        
#         return feat + out


#     def forward(self, feat1, feat2=None):
#         # feat1 = self.star(feat1)
#         # flow_feat2 = self.xca(feat1, feat2)
#         # unlabel_feat = self.xca(feat1, feat2)
#         unlabel_feat = self.xca_w_memory(feat1)
#         flow_feat = self.star(unlabel_feat)
        
#         return flow_feat



# region - ASPP
class ASPP(nn.Module):
    def __init__(self, high_ch, low_ch, dilations, ratio):
        super().__init__()
        
        self.aspp = ASPPModule(in_ch=high_ch, 
                               out_ch=high_ch // ratio, 
                               atrous_rates=dilations)
        
        # self.reduce = nn.Sequential(nn.Conv2d(high_ch // ratio, low_ch, 1, bias=False),
        self.expand = nn.Sequential(nn.Conv2d(144, low_ch, 1, bias=False),
                                    nn.BatchNorm2d(low_ch),
                                    nn.ReLU(True))
        
    def forward(self, feat1, feat2):
        feat2 = self.aspp(feat2)
        feat2 = F.interpolate(feat2, size=feat1.shape[-2:], mode="bilinear", align_corners=True)

        # feat1 = self.reduce(feat1)
        feat1 = self.expand(feat1)

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