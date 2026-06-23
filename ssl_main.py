import logging
import os
import os.path as osp
import random
import datetime

import numpy as np
import torch
from torch import nn
from torchmetrics import MeanMetric
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('agg')

from dataset.semi import SemiDataset
from ssl_tensorboard import SSLTensorBoard
from model.semseg.deeplabv3plus import DeepLabV3Plus
from evaluate import evaluate
import util.utils as utils
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.thresh_helper import ThreshController
from einops import rearrange

from configuration import DataConfig, TrainConfig, ModelConfig
from losses import LossFactory
from torch.optim.lr_scheduler import LambdaLR
from thop import profile
from tqdm import tqdm



def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


# region - test
def get_evaluate_train(model, img_uw, img_us, ignore_mask, cutmix_box):
    indices = torch.randperm(img_uw.size(0), device=device)
    ignore_mask = ignore_mask[indices]
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        model.eval()
        res_u_w_pred = model(img_uw[indices], mode='val')
        
        logit_u_w = res_u_w_pred['out'].detach()
        prob_u_w = logit_u_w.softmax(dim=1) # logit은 모델의 확신 점수이다.
        conf_u_w, mask_u_w = prob_u_w.max(dim=1) # pseudo label
        
        img_us[cutmix_box.unsqueeze(1).expand(img_us.shape) == 1] = \
            img_us[indices][cutmix_box.unsqueeze(1).expand(img_us.shape) == 1]
        
        return conf_u_w, mask_u_w, img_us, ignore_mask


# region - main
def main():
    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=tcfg.port)
    init_seeds(0, False)

    # model = DeepLabV3Plus(tcfg, mcfg, pretrained_path=osp.join(tcfg.pretrained_path, mcfg.backbone+'.pth'))
    model = DeepLabV3Plus(tcfg, mcfg)
    for name, module in model.named_modules():
        if name.startswith('backbone') or name == '': continue  
        utils.init_non_backbone(module)
    
    logger.info(f'Total params: {count_params(model):.1f}M\n')
    
    # ===================================================================
    # [FLOPs 계산 코드 추가 부분]
    # 모델의 forward가 다중 인자(need_fp, use_corr 등)를 요구하므로,
    # 더미 입력 텐서만 받는 단순한 Wrapper 클래스를 정의하여 계산합니다.
    # ===================================================================
    class WrapperModel(nn.Module):
        def __init__(self, basemodel):
            super().__init__()
            self.model = basemodel

        def forward(self, x):
            # 추론(eval) 기준 연산량을 구하기 위해 False로 설정
            return self.model(x, mode='val')

    # 모델을 GPU로 올리고 평가 모드로 전환
    wrapper_model = WrapperModel(model).cuda()
    wrapper_model.eval()

    # 설정 파일에서 이미지 크기(crop_size) 가져오기
    if isinstance(tcfg.crop_size, int):
        h, w = tcfg.crop_size, tcfg.crop_size
    else:
        h, w = tcfg.crop_size

    # Batch Size 1, Channel 3(RGB), H, W 크기의 더미(Dummy) 데이터 생성
    dummy_input = torch.randn(1, 3, h, w).cuda()

    try:
        # profile 함수로 MACs와 파라미터 계산 (verbose=False로 불필요한 출력 숨김)
        # macs, params = profile(wrapper_model, inputs=(dummy_input, ), verbose=False)
        flops, params = profile(model, inputs=(dummy_input, "val"))
        
        # 일반적으로 1 MAC(Multiply-Accumulate) = 2 FLOPs
        # flops = macs * 2
        
        logger.info('================ Model Complexity ================')
        logger.info(f"Input Shape : {dummy_input.shape}")
        logger.info(f"FLOPs       : {flops / 1e9:.3f} GFLOPs")
        # logger.info(f"MACs        : {macs / 1e9:.3f} GMACs")
        logger.info(f"Parameters  : {params / 1e6:.3f} M")
        logger.info('==================================================\n')
    except Exception as e:
        logger.error(f"FLOPs 계산 중 오류 발생: {e}")
        
    # 다시 학습을 위해 원래 모델을 훈련 모드로 복구
    model.train()
    # ===================================================================


    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # global하게 모든 mini-batch 통합하여 평균 분산 계산
    model.to(device)
    
    # if world_size > 1:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    losses = LossFactory()
    # loss_ce     = losses.label(mode='ce', device=device)
    loss_ohem   = losses.label(mode='ohem', device=device)
    loss_dice   = losses.label(mode='dice', device=device)
    
    loss_us_cr = losses.unlabel(mode='us_cr', device=device)
    loss_uw_cr = losses.unlabel(mode='uw_cr', device=device)
    loss_kl    = losses.unlabel(mode='kl', device=device)

    unlabel_train_set = SemiDataset(root=tcfg.data_root, 
                                    mode='train_u',
                                    size=tcfg.crop_size,
                                    id_path=dcfg.unlabeled_id_path)
    
    label_train_set = SemiDataset(root=tcfg.data_root, 
                                  mode='train_l',
                                  id_path=dcfg.labeled_id_path,
                                  size=tcfg.crop_size, 
                                  nsample=len(unlabel_train_set.ids))
    
    validation_set = SemiDataset(root=tcfg.data_root,
                                 mode='val',
                                 size=tcfg.crop_size,
                                 valid_path=dcfg.val_id_path) 

    use_ddp = torch.distributed.is_available() and torch.distributed.is_initialized()
    trainsampler_l = torch.utils.data.distributed.DistributedSampler(label_train_set) if use_ddp else None
    label_train_loader = DataLoader(label_train_set, 
                               batch_size=tcfg.batch_size,
                               pin_memory=True, 
                               num_workers=tcfg.num_workers, 
                               drop_last=True, 
                               shuffle=(trainsampler_l is None),
                               sampler=trainsampler_l)
    
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(unlabel_train_set) if use_ddp else None
    unlabel_train_loader = DataLoader(unlabel_train_set, 
                                      batch_size=tcfg.batch_size,
                                      pin_memory=True, 
                                      num_workers=tcfg.num_workers, 
                                      drop_last=True, 
                                      shuffle=(trainsampler_u is None),
                                      sampler=trainsampler_u)
    
    valsampler = torch.utils.data.distributed.DistributedSampler(validation_set) if use_ddp else None
    validation_loader = DataLoader(validation_set, 
                                   batch_size=1, 
                                   pin_memory=True, 
                                   num_workers=tcfg.num_workers,
                                   drop_last=False, 
                                   shuffle=(valsampler is None),
                                   sampler=valsampler)

    writer = SummaryWriter(osp.join(tcfg.exp_dir, "logs", tcfg.model_name))
    tb = SSLTensorBoard(writer)
    
    steps_per_epoch = len(unlabel_train_loader)
    num_total_steps = steps_per_epoch * tcfg.num_epochs
    
    # region optimizer
    if tcfg.optimizer == "SGD":
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': tcfg.lr},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': tcfg.lr * tcfg.lr_multi}], lr=tcfg.lr, momentum=0.9, weight_decay=1e-4)
    elif tcfg.optimizer == 'Adam':
        optimizer = Adam(model.parameters(), lr=tcfg.lr)

    if tcfg.scheduler == "cosineDecay":
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=24, T_mult=2)
        lr_cd = utils.get_tf_cosine_decay_restarts_lambda(first_decay_steps=steps_per_epoch * tcfg.lr_period,
                                                        t_mul=1.,
                                                        m_mul=0.5)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_cd)
    
    thresh_controller = ThreshController(nclass=mcfg.num_classes, momentum=0.999, thresh_init=tcfg.thresh_init)

    previous_best = 0.0
    full_metrics            = MeanMetric().to(device=device)
    # ------------------------------------------
    # label
    # ------------------------------------------
    label_metrics           = MeanMetric().to(device=device)
    label_fp_metrics        = MeanMetric().to(device=device)
    label_corr_metrics      = MeanMetric().to(device=device)
    label_flow_metrics      = MeanMetric().to(device=device)
    label_dice_metrics      = MeanMetric().to(device=device)
    label_corr_dice_metrics = MeanMetric().to(device=device)
    label_flow_dice_metrics = MeanMetric().to(device=device)
    total_label_metrics     = MeanMetric().to(device=device)
    # ------------------------------------------
    # unlabel
    # ------------------------------------------
    us_flow_metrics       = MeanMetric().to(device=device)
    us_corr_metrics       = MeanMetric().to(device=device)
    uw_corr_metrics       = MeanMetric().to(device=device)
    u_flow_metrics        = MeanMetric().to(device=device)
    uw_flow_metrics       = MeanMetric().to(device=device)
    uw_fp_metrics         = MeanMetric().to(device=device)
    total_unlabel_metrics = MeanMetric().to(device=device)
    
    start_epoch = 0
    if tcfg.resume:
        latest_model = os.listdir(tcfg.model_save_dir)[-1]
        checkpoint = torch.load(osp.join(tcfg.model_save_dir, latest_model), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if tcfg.scheduler == "cosineDecay":
            scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
            
        start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"Resuming training from epoch {start_epoch} with model {latest_model}")
        


    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    accum_counter = 0
    step_in_epoch = 0
    
    # region - Train
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    for epoch in range(start_epoch, tcfg.num_epochs):
        # ------------------------------------------
        # label
        # ------------------------------------------
        label_metrics.reset()
        label_fp_metrics.reset()
        label_corr_metrics.reset()
        label_flow_metrics.reset()
        label_dice_metrics.reset()
        label_corr_dice_metrics.reset()
        label_flow_dice_metrics.reset()
        total_label_metrics.reset()
        # ------------------------------------------
        # unlabel
        # ------------------------------------------
        us_flow_metrics.reset()
        us_corr_metrics.reset()
        uw_corr_metrics.reset()
        u_flow_metrics.reset()
        uw_flow_metrics.reset()
        uw_fp_metrics.reset()
        total_unlabel_metrics.reset()
        
        full_metrics.reset()
        total_mask_ratio = 0.0
        
        dataloader = zip(label_train_loader, unlabel_train_loader)
        pbar = tqdm(dataloader, total=steps_per_epoch, desc='🚀 Training', position=0, leave=True)
        
        line_sep    = tqdm(total=0, position=1, bar_format='{desc}')
        line_hyper  = tqdm(total=0, position=2, bar_format='{desc}')
        line_label  = tqdm(total=0, position=3, bar_format='{desc}')
        line_unlabel = tqdm(total=0, position=4, bar_format='{desc}')
        
        
        for step, ((img_lw, gt_lw, l_image_path), (img_uw, img_us, ignore_mask, cutmix_box, u_image_path)) in enumerate(pbar):
            start_event.record()

            img_lw, gt_lw       = img_lw.cuda(non_blocking=True), gt_lw.cuda(non_blocking=True)
            img_uw              = img_uw.cuda(non_blocking=True)
            img_us, ignore_mask = img_us.cuda(non_blocking=True), ignore_mask.cuda(non_blocking=True)
            cutmix_box          = cutmix_box.cuda(non_blocking=True)
            
            test_conf_uw, test_mask_uw, img_us, test_ignore_mask = get_evaluate_train(model, img_uw, img_us, ignore_mask, cutmix_box)
            
            model.train()
            label_batch, unlabel_batch = img_lw.shape[0], img_uw.shape[0]

            
            if epoch > 1:
                with torch.no_grad(), torch.amp.autocast('cuda'):
                    m_core = model.module if hasattr(model, 'module') else model
                    
                    # 1. Labeled 이미지(img_lw)만 백본에 통과시켜 깨끗한 피처 추출
                    c1_l, c2_l, c3_l, c4_l = m_core.backbone.base_forward(img_lw)
                    
                    # 2. 추출된 피처와 정답(gt_lw)을 이용해 19개 클래스 메모리 뱅크 갱신 (선생님의 지식 축적)
                    m_core.update_prototypes(c1_l, c2_l, c3_l, c4_l, gt_lw)
            
            
            with torch.amp.autocast('cuda'):
                results = model(torch.cat((img_lw, img_uw, img_us)))
            
                flow_mask_lw                = results['flow_mask_lw']
                # corr_mask_lw                = results['corr_mask_lw']
                
                pred_lw, pred_uw            = results['mask_lw_uw'].split([label_batch, unlabel_batch])
                pred_corr_lw, pred_corr_uw  = results['corr_mask_lw_uw'].split([label_batch, unlabel_batch]) # 6번 수식의 z값이 pred_uw_corr
                # pred_uw_fp                  = results['mask_lw_uw_fp'][label_batch:]
                pred_lw_fp, pred_uw_fp      = results['mask_lw_uw_fp'].split([label_batch, unlabel_batch])
                
                flow_mask_uw                = results['flow_mask_uw']
                flow_mask_us                = results['flow_mask_us']
                corr_mask_us                = results['corr_mask_us']

                # 2번 수식의 max F_hat
                pred_uw_prob = pred_uw.detach().softmax(dim=1)
                pred_conf_uw, pred_mask_uw = pred_uw_prob.max(dim=1)

                pred_mask_uw_cutmixed, pred_conf_uw_cutmixed, ignore_mask_cutmixed = pred_mask_uw.clone(), pred_conf_uw.clone(), ignore_mask.clone()
                # labeled + unlabeled weak간의 유사도가 높은부분에서의 unlabeled part
                binary_pred_corr_map_uw_cutm = results['binary_norm_corr_map'][label_batch:].detach().clone()

                # -------------------------- Test 결과를 모델 예측에 cutmix로 넣기 --------------------------
                cutmix_box = (cutmix_box == 1).squeeze(dim=1)
                pred_mask_uw_cutmixed[cutmix_box] = test_mask_uw[cutmix_box]
                pred_conf_uw_cutmixed[cutmix_box] = test_conf_uw[cutmix_box]
                ignore_mask_cutmixed[cutmix_box] = test_ignore_mask[cutmix_box]
                # ------------------------------------------------------------------------------------------
                
                # ------------------------ uw의 corr에 cutmix 부분은 제거 ------------------------
                cutmix_box = rearrange(cutmix_box, 'n h w -> n 1 h w')
                ignore_mask_cutmixed_arrange = rearrange((ignore_mask_cutmixed != 255), 'n h w -> n 1 h w')
                # cutmix된 부분은 모델의 예측이 아닌 test에서 얻은 예측을 사용하기 때문에 cutmix된 부분의 모델 예측과 유사도는 무의미하다. 
                # 따라서 cutmix된 부분의 유사도는 0으로 만들어준다.
                pred_corr_map_uw_wo_cutmixed: bool = (binary_pred_corr_map_uw_cutm * ~cutmix_box * ignore_mask_cutmixed_arrange).bool()
                # --------------------------------------------------------------------------------
                
                # ---------------------------- 모델이 예측한 신뢰도에서 threshold 걺 ----------------------------
                thresh_controller.thresh_update(pred_uw.detach(), ignore_mask_cutmixed, update_g=True)
                thresh_global = thresh_controller.get_thresh_global()
                # 2번 수식 (M_i)
                # conf_filter_uw : dynamic threshold를 통한 학습된 예측값 + 테스트 예측값 신뢰도에서 더 정확한 신뢰도만을 가져온것. 
                conf_filter_uw: bool = ((pred_conf_uw_cutmixed >= thresh_global) & (ignore_mask_cutmixed != 255))
                conf_filter_uw_wo_cutmix: bool = conf_filter_uw.clone()
                conf_filter_uw_wo_cutmix_arrange: bool = rearrange(conf_filter_uw_wo_cutmix, 'n h w -> n 1 h w')
                # ---------------------------------------------------------------------------------------------
                
                # ---------------- weak unlabel 중에 label과 공간적으로 가장 비슷하면서 모델 신뢰도가 높은 부분 ----------------
                # 9번 수식에서 M_i * c_hat
                # region_propagation - 더 정확한 경계를 얻기위함.
                # conf_filter_uw_wo_cutmix_arrange: M_i
                segments: bool = (pred_corr_map_uw_wo_cutmixed * conf_filter_uw_wo_cutmix_arrange).bool() # region-propa 재료
                # -----------------------------------------------------------------------------------------------------------
                
                # ------------------ region propagation: 신뢰도가 낮은 예측 영역 (pred_mask_uw_cutmixed)을 주변의 지표를 활용해 refinement ------------------
                """ label과 unlabel이 같은 클래스를 공간적으로 유사한 부분에서 공유하고 있으면 unique_cls에 그 클래스가 나타남."""
                segment = segments.view(tcfg.batch_size, -1, tcfg.crop_size*tcfg.crop_size)
                segment_ori = pred_corr_map_uw_wo_cutmixed.view(tcfg.batch_size, -1, tcfg.crop_size*tcfg.crop_size)
                high_conf_ratio = torch.sum(segment, dim=2) / (torch.sum(segment_ori, dim=2) + 1e-7)
                
                valid_mask = (torch.sum(segment, dim=2) > 0) & (high_conf_ratio >= thresh_global)
                valid_img_idx, valid_segment_idx = torch.where(valid_mask)
                for img_idx, segment_idx in zip(valid_img_idx, valid_segment_idx):
                    segment: bool = segments[img_idx, segment_idx]
                    segment_ori: bool = pred_corr_map_uw_wo_cutmixed[img_idx, segment_idx]
                    
                    unique_cls, count = torch.unique(pred_mask_uw_cutmixed[img_idx][segment==1], return_counts=True)
                    mask = torch.max(count) / torch.sum(count) > thresh_global
                    if mask:
                        top_class = unique_cls[count.argmax()] # 8번 수식 k*
                        pred_mask_uw_cutmixed[img_idx][segment_ori==1] = top_class # 10번 수식, top class를 찾아 수정
                        conf_filter_uw_wo_cutmix[img_idx] = conf_filter_uw_wo_cutmix[img_idx] | segment_ori # 수정
                        
                conf_filter_uw_wo_cutmix: bool = conf_filter_uw_wo_cutmix | conf_filter_uw
                # -----------------------------------------------------------------------------------------------------------------------------------------
                
                
                # region loss 계산
                # --------------------------------------------------------------------
                # label part
                # --------------------------------------------------------------------
                label_loss      = loss_ohem(pred_lw, gt_lw, 
                                            ignore_index=tcfg.LossConfig.ignore_index,
                                            threshold=tcfg.LossConfig.ohem_threshold,
                                            min_kept=tcfg.LossConfig.ohem_min_kept)
                label_fp_loss   = loss_ohem(pred_lw_fp, gt_lw, 
                                            ignore_index=tcfg.LossConfig.ignore_index,
                                            threshold=tcfg.LossConfig.ohem_threshold,
                                            min_kept=tcfg.LossConfig.ohem_min_kept)
                label_loss_corr = loss_ohem(pred_corr_lw, gt_lw, 
                                            ignore_index=tcfg.LossConfig.ignore_index,
                                            threshold=tcfg.LossConfig.ohem_threshold,
                                            min_kept=tcfg.LossConfig.ohem_min_kept)
                label_flow_loss  = loss_ohem(flow_mask_lw, gt_lw, 
                                            ignore_index=tcfg.LossConfig.ignore_index,
                                            threshold=tcfg.LossConfig.ohem_threshold,
                                            min_kept=tcfg.LossConfig.ohem_min_kept)
                
                label_dice_loss      = loss_dice(pred_lw, gt_lw)
                lw_corr_dice_loss    = loss_dice(pred_corr_lw, gt_lw)
                label_flow_dice_loss = loss_dice(flow_mask_lw, gt_lw)
                
                
                ohem_loss = label_loss + label_fp_loss + label_loss_corr + 1.5 * label_flow_loss
                dice_loss = (label_dice_loss + lw_corr_dice_loss)
                total_label_loss = ohem_loss + dice_loss
                
                # --------------------------------------------------------------------
                # unlabel part
                # --------------------------------------------------------------------
                us_flow_loss = loss_us_cr(pred=flow_mask_us,
                                    true=pred_mask_uw_cutmixed, 
                                    confidence=conf_filter_uw_wo_cutmix, 
                                    ignore_mask=ignore_mask_cutmixed)
                
                us_corr_loss = loss_us_cr(pred=corr_mask_us, 
                                        true=pred_mask_uw_cutmixed, 
                                        confidence=conf_filter_uw_wo_cutmix, 
                                        ignore_mask=ignore_mask_cutmixed)
                # 6번 수식
                uw_corr_loss = loss_uw_cr(pred=pred_corr_uw, 
                                        true=pred_mask_uw, 
                                        confidence=pred_conf_uw,
                                        threshold=thresh_global,
                                        ignore_mask=ignore_mask)
                # 3번 수식
                u_flow_kl   = loss_kl(flow_mask_us, pred_uw, confidence=conf_filter_uw, ignore_mask=ignore_mask_cutmixed)
                uw_flow_kl  = loss_kl(flow_mask_uw, pred_uw, confidence=conf_filter_uw, ignore_mask=ignore_mask_cutmixed)
                uw_fp_cr    = loss_uw_cr(pred=pred_uw_fp, 
                                        true=pred_mask_uw, 
                                        confidence=pred_conf_uw, 
                                        threshold=thresh_global, 
                                        ignore_mask=ignore_mask)
                
                # loss_uw_fp: UniMatch에서 가져온 loss인 것 같음.
                # loss = ( 0.5 * label_loss + 0.5 * label_loss_corr + loss_us * 0.25 + loss_u_kl * 0.25 + loss_uw_fp * 0.25 + 0.25 * loss_u_corr) / 2.0
                # total_unlabel_loss = 0.5*loss_us + 0.25 * loss_u_kl + 0.25 * loss_u_corr + 0.25 * loss_uw_fp
                # total_unlabel_loss = 0.5 * loss_us + 0.25 * (loss_us_kl + loss_uw_kl) + 0.25 * loss_u_corr + 0.25 * loss_uw_fp
                # total_unlabel_loss = 0.5 * loss_us + 0.25 * loss_us_kl + 0.25 * loss_u_corr + 0.25 * loss_uw_fp
                # ohem_loss = label_loss + label_fp_loss + label_loss_corr + 0.5 * label_loss_corr2 + tcfg.LossConfig.aux_loss_weight * label_flow_loss + 5 * label_dice_loss
                
                loss_u_corr = 0.5 * (us_corr_loss + uw_corr_loss)
                total_unlabel_loss = us_flow_loss + loss_u_corr + 0.25 * (u_flow_kl + uw_flow_kl) + 0.25 * uw_fp_cr
                
                full_loss = total_label_loss + total_unlabel_loss


            scaler.scale(full_loss / tcfg.accumulation_steps).backward()
            accum_counter += 1
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            iters = epoch * len(unlabel_train_loader) + step
            if accum_counter == tcfg.accumulation_steps:
                scale_before = scaler.get_scale()
                
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()
                scale_after = scaler.get_scale()
                if scale_before <= scale_after:
                    scheduler.step()
                
                iters = epoch * len(unlabel_train_loader) + step_in_epoch
                # # power = tcfg.unlabel_lr_decay if epoch >= tcfg.lr_period else tcfg.label_lr_decay
                # # current_cycle_epoch = epoch % tcfg.lr_period
                # # iters = current_cycle_epoch * len(unlabel_train_loader) + step
                # # num_cycle_steps = tcfg.lr_period * len(unlabel_train_loader)
                
                # lr = tcfg.lr * (1 - iters / num_total_steps) ** tcfg.decay_power
                # optimizer.param_groups[0]["lr"] = lr
                # optimizer.param_groups[1]["lr"] = lr * tcfg.lr_multi
                lr = scheduler.get_last_lr()[0]
                step_in_epoch += 1
                accum_counter = 0

            # --------------------------------------------------------------------
            # label part
            # --------------------------------------------------------------------
            label_metrics.update(label_loss.detach())
            label_fp_metrics.update(label_fp_loss.detach())
            label_corr_metrics.update(label_loss_corr.detach())
            label_flow_metrics.update(label_flow_loss.detach())
            label_dice_metrics.update(label_dice_loss.detach())
            label_corr_dice_metrics.update(lw_corr_dice_loss.detach())
            label_flow_dice_metrics.update(label_flow_dice_loss.detach())
            total_label_metrics.update(total_label_loss.detach())
            
            # --------------------------------------------------------------------
            # unlabel part
            # --------------------------------------------------------------------
            us_flow_metrics.update(us_flow_loss.detach())
            us_corr_metrics.update(us_corr_loss.detach())
            uw_corr_metrics.update(uw_corr_loss.detach())
            u_flow_metrics.update(u_flow_kl.detach())
            uw_flow_metrics.update(uw_flow_kl.detach())
            uw_fp_metrics.update(uw_fp_cr.detach())
            total_unlabel_metrics.update(total_unlabel_loss.detach())
            
            full_metrics.update(full_loss.detach())
            
            total_mask_ratio += ((pred_conf_uw >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()
            
            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            time_left = (num_total_steps - iters) * elapsed_time
            time_left = str(datetime.timedelta(seconds=int(time_left)))
            
            if step % 10 == 0:
                hyperparam = f"Model: [{tcfg.model_name:>5}] | Time Left: [{time_left:>5}] | Epoch: [{epoch:>3}/{tcfg.num_epochs:>5}] | Step: [{step}/{len(unlabel_train_loader):>5}] | Elapsed time: {elapsed_time:.2f}s"
                label_loss_info = f"Full: {full_metrics.compute():.3f}, 🏷️ Total Label: {total_label_metrics.compute():.3f}, Label fp: {label_fp_metrics.compute():.3f}, " \
                                  f"🏷️ Flow Label: {label_flow_metrics.compute():.3f}, Dice Label: {label_dice_metrics.compute():.3f}, 🏷️ Corr Dice Label: {label_corr_dice_metrics.compute():.3f}, " \
                                  f"🏷️ Flow Dice Label:{label_flow_dice_metrics.compute():.3f}"
                
                unlabel_loss_info = f"Total UnLabel: {total_unlabel_metrics.compute():.3f}, 🏷️ Flow US: {us_flow_metrics.compute():.3f}, Corr US: {us_corr_metrics.compute():.3f}, " \
                                    f"Corr UW: {uw_corr_metrics.compute():.3f}, 🏷️ Flow U: {u_flow_metrics.compute():.3f}, 🏷️ Flow UW: {uw_flow_metrics.compute():.3f}, " \
                                    f"UW FP: {uw_fp_metrics.compute():.3f}, " \
                                    f"Mask: {total_mask_ratio/(step+1):.3f}"
                
                line_sep.set_description_str("=" * 100)
                line_hyper.set_description_str(hyperparam)
                line_label.set_description_str(label_loss_info)
                line_unlabel.set_description_str(unlabel_loss_info)
                
            del results
            
            
        if accum_counter > 0:
            scale_before = scaler.get_scale()
            
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            scale_after = scaler.get_scale()
            if scale_before <= scale_after:
                scheduler.step()
            
            accum_counter = 0
            
        
        pbar.close()
        line_sep.close()
        line_hyper.close()
        line_label.close()
        line_unlabel.close()
        
        # region step 끝
        torch.cuda.empty_cache()
        res_val = evaluate(tcfg, mcfg, model, validation_loader, mode=tcfg.eval_mode)
        
        # region  tensorboard
        tb.draw_scalar(epoch=epoch, item={"Optimization/Full Loss": full_metrics.compute(), 
                                          "Label/Total Loss": total_label_metrics.compute(),
                                          "Label/FP Loss": label_fp_metrics.compute(), 
                                          "Label/Flow Loss": label_flow_metrics.compute(), 
                                          "Label/Dice Loss": label_dice_metrics.compute(), 
                                          "Label/Corr Dice Loss": label_corr_dice_metrics.compute(), 
                                          "Label/Flow Dice Loss": label_flow_dice_metrics.compute(), 
                                          
                                          "UnLabel/Total Loss": total_unlabel_metrics.compute(), 
                                          "UnLabel/Flow US": us_flow_metrics.compute(),
                                          "UnLabel/Corr US": us_corr_metrics.compute(), 
                                          "UnLabel/Corr UW": uw_corr_metrics.compute(),
                                          "UnLabel/Flow U": u_flow_metrics.compute(), 
                                          "UnLabel/Flow Uw": uw_flow_metrics.compute(),
                                          "UnLabel/FP UW": uw_fp_metrics.compute(),
                                          
                                          "Optimization/learning_rate": lr,
                                          "Accuracy/eval/mIOU": res_val['mIOU'],
                                          })
        
        img_us = img_us.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_mask_us = flow_mask_us.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_us = flow_mask_us.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/unlabel strong image", 
                      image=img_us, 
                      pred=pred_mask_us,
                      conf=conf_us,
                      mask=None,
                      image_path=l_image_path,
                      epoch=epoch)

        img_lw = img_lw.detach().cpu().permute(0, 2, 3, 1).numpy()
        flow_mask_lw = pred_lw.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_l = pred_lw.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        if gt_lw.dim() == 4:
            gt = gt_lw.detach().cpu().permute(0, 2, 3, 1).numpy()
        elif gt_lw.dim() == 3:
            gt = gt_lw.detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/label weak image", 
                      image=img_lw, 
                      pred=flow_mask_lw,
                      conf=conf_l,
                      mask=gt,
                      image_path=u_image_path,
                      epoch=epoch)
            
        val_image       = res_val['img'].detach().cpu().permute(0, 2, 3, 1).numpy()
        val_pred_mask   = res_val['pred'].detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        val_conf        = res_val['conf'].detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        val_gt          = res_val['mask'].detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="valid/label image", 
                      image=val_image, 
                      pred=val_pred_mask,
                      conf=val_conf,
                      mask=val_gt,
                      image_path=res_val['image_path'][0],
                      epoch=epoch)

        if res_val['mIOU'] > previous_best:
            if previous_best != 0:
                os.remove(osp.join(tcfg.model_save_dir, f'{mcfg.backbone}_{previous_best:.3f}.pth'))
            previous_best = res_val['mIOU']
            
            if tcfg.scheduler == "cosineDecay":
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict()}, 
                           osp.join(tcfg.model_save_dir, f'{mcfg.backbone}_{res_val["mIOU"]:.3f}.pth'))
            
            elif tcfg.scheduler == "Polynomial":
                torch.save({"epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                        osp.join(tcfg.model_save_dir, f'{mcfg.backbone}_{res_val["mIOU"]:.3f}.pth'))
        
        torch.cuda.empty_cache()


if __name__ == '__main__':
    dcfg = DataConfig()
    tcfg = TrainConfig()
    mcfg = ModelConfig()
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", local_rank)

    os.makedirs(osp.join(tcfg.exp_dir, "logs", tcfg.model_name), exist_ok=True)
    os.makedirs(osp.join(tcfg.exp_dir, "models", tcfg.model_name), exist_ok=True)
    os.makedirs(osp.join(tcfg.exp_dir, "codes", tcfg.model_name), exist_ok=True)
    
    utils.save_codes(tcfg, osp.dirname(__file__))
    main()
