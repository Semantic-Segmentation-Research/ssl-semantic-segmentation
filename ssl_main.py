import argparse
import logging
import os
import os.path as osp
import pprint
import random
import time
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
import yaml

from dataset.semi import SemiDataset
from ssl_tensorboard import SSLTensorBoard
from model.semseg.deeplabv3plus import DeepLabV3Plus
from evaluate import evaluate
import util.utils as utils
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
from util.thresh_helper import ThreshController
from einops import rearrange

from configuration import DataConfig, TrainConfig, ModelConfig
from losses import LossFactory

# parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
# parser.add_argument('--config', type=str, default=osp.join(osp.dirname(__file__), 'configs/cityscapes.yaml'))


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
# def test(model, dataloader, img_us, cutmix_box):
    # img_uw, img_us2, ignore_mask, _, _ = next(iter(dataloader))
def test(model, img_uw, img_us, ignore_mask, cutmix_box):
    indices = torch.randperm(img_uw.size(0), device=device)
    ignore_mask = ignore_mask[indices]
    
    with torch.no_grad():
        model.eval()
        res_u_w_pred = model(img_uw[indices], mode='test')
        
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

    rank, world_size = setup_distributed(port=tcfg.port)
    init_seeds(0, False)

    model = DeepLabV3Plus(tcfg, mcfg, pretrained_path=osp.join(tcfg.pretrained_path, mcfg.backbone+'.pth'))
    for name, module in model.named_modules():
        if name.startswith('backbone') or name == '': continue  
        utils.init_non_backbone(module)
    
    if rank == 0:
        logger.info(f'Total params: {count_params(model):.1f}M\n')

    # optimizer = SGD([{'params': model.backbone.parameters(), 'lr': tcfg.lr},
    #                  {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
    #                   'lr': tcfg.lr * tcfg.lr_multi}], lr=tcfg.lr, momentum=0.9, weight_decay=1e-4)
    
    optimizer = Adam(model.parameters(), lr=tcfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=24, T_mult=2)
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # global하게 모든 mini-batch 통합하여 평균 분산 계산
    model.to(device)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    losses = LossFactory()
    loss_ce     = losses.label(mode='ce', device=device)
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

    # aug_layer = GPUAugmentation(size=tcfg.crop_size).to(device)
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
    
    num_total_steps = len(unlabel_train_loader) * tcfg.num_epochs
    thresh_controller = ThreshController(nclass=mcfg.num_classes, momentum=0.999, thresh_init=tcfg.thresh_init)

    previous_best = 0.0
    total_loss              = MeanMetric().to(device=device)
    total_aux_loss          = MeanMetric().to(device=device)
    total_label_loss        = MeanMetric().to(device=device)
    total_label_loss_corr   = MeanMetric().to(device=device)
    total_dice_loss         = MeanMetric().to(device=device)
    total_loss_s            = MeanMetric().to(device=device)
    total_loss_kl           = MeanMetric().to(device=device)
    total_loss_w_fp         = MeanMetric().to(device=device)
    total_loss_corr_u       = MeanMetric().to(device=device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_epoch = 0
    
    if tcfg.resume:
        latest_model = os.listdir(tcfg.model_save_dir)[-1]
        checkpoint = torch.load(osp.join(tcfg.model_save_dir, latest_model), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        logger.info(f"Resuming training from epoch {start_epoch} with model {latest_model}")
        
        
    # region - Train
    for epoch in range(start_epoch, tcfg.num_epochs):
        total_aux_loss.reset()
        total_loss.reset()
        total_label_loss.reset()
        total_label_loss_corr.reset()
        total_dice_loss.reset()
        total_loss_s.reset()
        total_loss_kl.reset()
        total_mask_ratio = 0.0
        
        if use_ddp:
            label_train_loader.sampler.set_epoch(epoch)
            unlabel_train_loader.sampler.set_epoch(epoch)
        
        dataloader = zip(label_train_loader, unlabel_train_loader)
        for step, ((img_lw, mask_lw, l_image_path), 
                   (img_uw, img_us, ignore_mask, cutmix_box, u_image_path)) in enumerate(dataloader):

            # if step == 1: break
            
            start_event.record()
            
            img_lw, mask_lw = img_lw.cuda(non_blocking=True), mask_lw.cuda(non_blocking=True)
            img_uw = img_uw.cuda(non_blocking=True)
            img_us, ignore_mask = img_us.cuda(non_blocking=True), ignore_mask.cuda(non_blocking=True)
            cutmix_box = cutmix_box.cuda(non_blocking=True)
            
            test_conf_uw, test_mask_uw, img_us, test_ignore_mask = test(model, img_uw, img_us, ignore_mask, cutmix_box)
            
            model.train()
            label_batch, unlabel_batch = img_lw.shape[0], img_uw.shape[0]
            
            with torch.amp.autocast('cuda'):
                results = model(torch.cat((img_lw, img_uw, img_us)))
                
                pred_mask_lw = results['mask_lw']
                
                pred_lw, pred_uw = results['out'].split([label_batch, unlabel_batch])
                pred_lw_corr, pred_uw_corr = results['corr_out'].split([label_batch, unlabel_batch]) # 6번 수식의 z값이 pred_uw_corr
                pred_uw_fp = results['out_fp'][label_batch:]
                # pred_uw_corr_map : labeled + unlabeled weak간의 유사도가 높은부분에서의 unlabeled part
                pred_uw_corr_map: bool = results['binary_norm_corr_map'][label_batch:].detach()
                
                pred_us = results['out_us']
                pred_us_corr = results['corr_out_us']

            # 2번 수식의 max F_hat
            pred_uw_prob = pred_uw.detach().softmax(dim=1)
            pred_conf_uw, pred_mask_uw = pred_uw_prob.max(dim=1)

            pred_mask_uw_cutmixed, pred_conf_uw_cutmixed, ignore_mask_cutmixed = pred_mask_uw.clone(), pred_conf_uw.clone(), ignore_mask.clone()
            pred_corr_map_uw_cutmixed: bool = pred_uw_corr_map.clone()

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
            pred_corr_map_uw_wo_cutmixed: bool = (pred_corr_map_uw_cutmixed * ~cutmix_box * ignore_mask_cutmixed_arrange).bool()
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
            high_conf_ratio = torch.sum(segment, dim=2) / torch.sum(segment_ori, dim=2)
            
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
            # ---------------------- label part ----------------------
            label_loss      = loss_ohem(pred_lw, mask_lw, 
                                        ignore_index=tcfg.LossConfig.ignore_index,
                                        threshold=tcfg.LossConfig.ohem_threshold,
                                        min_kept=tcfg.LossConfig.ohem_min_kept)
            label_loss_corr = loss_ohem(pred_lw_corr, mask_lw, 
                                        ignore_index=tcfg.LossConfig.ignore_index,
                                        threshold=tcfg.LossConfig.ohem_threshold,
                                        min_kept=tcfg.LossConfig.ohem_min_kept)
            label_aux_loss  = loss_ohem(pred_mask_lw.float(), mask_lw, 
                                        ignore_index=tcfg.LossConfig.ignore_index,
                                        threshold=tcfg.LossConfig.ohem_threshold,
                                        min_kept=tcfg.LossConfig.ohem_min_kept)
            
            label_dice_loss = loss_dice(pred_lw, mask_lw)
            # ---------------------------------------------------------
            
            # ----------------------- unlabel part -----------------------
            loss_us = loss_us_cr(pred=pred_us,
                                 true=pred_mask_uw_cutmixed, 
                                 confidence=conf_filter_uw_wo_cutmix, 
                                 ignore_mask=ignore_mask_cutmixed)
            
            loss_us_corr = loss_us_cr(pred=pred_us_corr, 
                                      true=pred_mask_uw_cutmixed, 
                                      confidence=conf_filter_uw_wo_cutmix, 
                                      ignore_mask=ignore_mask_cutmixed)
            # 6번 수식
            loss_uw_corr = loss_uw_cr(pred=pred_uw_corr, 
                                      true=pred_mask_uw, 
                                      confidence=pred_conf_uw,
                                      threshold=thresh_global,
                                      ignore_mask=ignore_mask)

            loss_u_corr = 0.5 * (loss_us_corr + loss_uw_corr)
            
            # 3번 수식
            loss_u_kl = loss_kl(pred_us, pred_uw, confidence=conf_filter_uw, ignore_mask=ignore_mask_cutmixed)
            loss_uw_fp = loss_uw_cr(pred=pred_uw_fp, 
                                    true=pred_mask_uw, 
                                    confidence=pred_conf_uw, 
                                    threshold=thresh_global, 
                                    ignore_mask=ignore_mask)
            # ---------------------------------------------------------------------
            
            # loss_uw_fp: UniMatch에서 가져온 loss인 것 같음.
            # loss = ( 0.5 * label_loss + 0.5 * label_loss_corr + loss_us * 0.25 + loss_u_kl * 0.25 + loss_uw_fp * 0.25 + 0.25 * loss_u_corr) / 2.0
            label_loss = label_loss + label_loss_corr + tcfg.LossConfig.aux_loss_weight * label_aux_loss + 5 * label_dice_loss
            unlabel_loss = 0.5*loss_us + 0.25 * loss_u_kl + 0.25 * loss_u_corr + 0.25 * loss_uw_fp
            
            # weight_unlabel = torch.exp(torch.tensor(epoch - tcfg.lr_period, dtype=torch.float32))
            # weight_unlabel = torch.clip(weight_unlabel, 0., 1.)
            # weight_label = 2 - 0.5 * weight_unlabel

            # loss = weight_label * label_loss + weight_unlabel * unlabel_loss
            loss = label_loss + unlabel_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            total_aux_loss.update(label_aux_loss.detach())
            total_loss.update(loss.detach())
            total_label_loss.update(label_loss.detach())
            total_label_loss_corr.update(label_loss_corr.detach())
            total_dice_loss.update(label_dice_loss.detach())
            
            total_loss_s.update(loss_us.detach())
            total_loss_kl.update(loss_u_kl.detach())
            total_loss_w_fp.update(loss_uw_fp.detach())
            total_loss_corr_u.update(loss_u_corr.detach())
            total_mask_ratio += ((pred_conf_uw >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()
            
            
            # iters = epoch * len(unlabel_train_loader) + step
            # # power = tcfg.unlabel_lr_decay if epoch >= tcfg.lr_period else tcfg.label_lr_decay
            # # current_cycle_epoch = epoch % tcfg.lr_period
            # # iters = current_cycle_epoch * len(unlabel_train_loader) + step
            # # num_cycle_steps = tcfg.lr_period * len(unlabel_train_loader)
            
            # lr = tcfg.lr * (1 - iters / num_total_steps) ** tcfg.decay_power
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * tcfg.lr_multi

            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            time_left = (num_total_steps - iters) * elapsed_time
            time_left = str(datetime.timedelta(seconds=int(time_left)))
            
            
            if step % 10 == 0 and rank == 0:
                hyperparam = f"Model: [{tcfg.model_name:>5}] | Time Left: [{time_left:>5}] | Epoch: [{epoch:>3}/{tcfg.num_epochs:>5}] | Step: [{step}/{len(unlabel_train_loader):>5}] | Elapsed time: {elapsed_time*50:.2f}s | lr: {lr:5.4f}"
                loss_info = f"total loss: {total_loss.compute():.3f}, label loss: {total_label_loss.compute():.3f}, loss_corr_ce: {total_label_loss_corr.compute():.3f}, " \
                            f"loss s: {total_loss_s.compute():.3f}, loss w_fp: {total_loss_w_fp.compute():.3f}, loss_corr_u: {total_loss_corr_u.compute():.3f}, Mask: {total_mask_ratio/(step+1):.3f}"
                print(hyperparam + '\n' + loss_info)
                print('-'*100)
                
            del results
        # region step 끝
        
        # if tcfg.dataset == 'cityscapes':
        #     eval_mode = 'center_crop' if epoch < tcfg.num_epochs - 20 else 'slviding_window'
        # else:
            # eval_mode = 'original'
            
        torch.cuda.empty_cache()
        res_val = evaluate(tcfg, mcfg, rank, model, validation_loader, mode=tcfg.eval_mode)
        
        # region  tensorboard
        tb.draw_scalar(epoch=epoch, item={"Optimization/loss/total loss": total_loss.compute(), 
                                          "Optimization/loss/aux loss": total_aux_loss.compute(),
                                          "Optimization/loss/label loss": total_label_loss.compute(), 
                                          "Optimization/loss/label loss": total_label_loss.compute(), 
                                          "Optimization/loss/label loss corr": total_label_loss_corr.compute(), 
                                          "Optimization/loss/label dice loss": total_dice_loss.compute(), 
                                          "Optimization/loss/unlabel strong loss": total_loss_s.compute(), 
                                          "Optimization/loss/label weak fp loss": total_loss_w_fp.compute(), 
                                          "Optimization/loss/unlabel corr loss": total_loss_corr_u.compute(),
                                          "Optimization/learning_rate": lr,
                                          "Time/Elapsed time": elapsed_time,
                                          "Accuracy/eval/mIOU": res_val['mIOU'],
                                          })
        
        img_us = img_us.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_mask_us = pred_us.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_us = pred_us.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/unlabel strong image", 
                      image=img_us, 
                      pred=pred_mask_us,
                      conf=conf_us,
                      mask=None,
                      image_path=l_image_path,
                      epoch=epoch)

        img_lw = img_lw.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_mask_lw = pred_lw.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_l = pred_lw.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        if mask_lw.dim() == 4:
            gt = mask_lw.detach().cpu().permute(0, 2, 3, 1).numpy()
        elif mask_lw.dim() == 3:
            gt = mask_lw.detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/label weak image", 
                      image=img_lw, 
                      pred=pred_mask_lw,
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


        logger.info(f'***** Evaluation {tcfg.eval_mode} ***** >>>> meanIOU: {res_val["mIOU"]:.4f} \n')
        summary = " | ".join(f"[{k}:{v:.2f}%]" for k, v in res_val['iou_class'].items())
        print(f"[Class IoU]: {summary} \n")
                
        if res_val['mIOU'] > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(osp.join(tcfg.model_save_dir, f'{mcfg.backbone}_{previous_best:.3f}.pth'))
            previous_best = res_val['mIOU']
            torch.save({"epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, osp.join(tcfg.model_save_dir, f'{mcfg.backbone}_{res_val["mIOU"]:.3f}.pth'))
        
        if rank != 0:
            torch.distributed.barrier()
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
    os.makedirs(osp.join(tcfg.exp_dir, "backups", tcfg.model_name), exist_ok=True)
    
    utils.save_codes(tcfg, osp.dirname(__file__))
    main()
