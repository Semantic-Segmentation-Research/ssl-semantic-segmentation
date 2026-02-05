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
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('agg')
import yaml

# from dataset.semi import SemiDataset, GPUAugmentation
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
from util.vectorized_region_prop import vectorized_region_propagation

from configuration import DataConfig, TrainConfig, ModelConfig

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default=osp.join(osp.dirname(__file__), 'configs/cityscapes.yaml'))


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


# region - main
def main():
    args = parser.parse_args() # arg parser 정의

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader) # yaml 파일 로드 (config 옵션 쉽게 꺼내쓰기 위함)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=tcfg.port)
    init_seeds(0, False)

    model = DeepLabV3Plus(tcfg, mcfg, pretrained_path=osp.join(tcfg.pretrained_path, mcfg.backbone+'.pth'))

    if rank == 0:
        # logger.info(f'{pprint.pformat(cfg)}\n')
        logger.info(f'Total params: {count_params(model):.1f}M\n')

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': tcfg.lr}, # 옵티마이저 하이퍼파라미터 세팅
                     {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                      'lr': tcfg.lr * cfg['lr_multi']}], lr=tcfg.lr, momentum=0.9, weight_decay=1e-4)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) # global하게 모든 mini-batch 통합하여 평균 분산 계산
    model.to(device)
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    if tcfg.LossConfig().name == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device, non_blocking=True)
    elif tcfg.LossConfig().name == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device, non_blocking=True)
    else:
        raise NotImplementedError(f'{tcfg.LossConfig().name} criterion is not implemented')

    criterion_u = nn.CrossEntropyLoss(reduction='none').to(device, non_blocking=True)
    criterion_kl = nn.KLDivLoss(reduction='none').to(device, non_blocking=True)
    

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
    thresh_controller = ThreshController(nclass=mcfg.num_classes, momentum=0.999, thresh_init=cfg['thresh_init'])

    previous_best = 0.0
    total_loss              = MeanMetric().to(device=device)
    total_label_loss        = MeanMetric().to(device=device)
    total_label_loss_corr   = MeanMetric().to(device=device)
    total_loss_s            = MeanMetric().to(device=device)
    total_loss_w_fp         = MeanMetric().to(device=device)
    total_loss_corr_u       = MeanMetric().to(device=device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # region - Train
    for epoch in range(tcfg.num_epochs):
        total_loss.reset()
        total_label_loss.reset()
        total_label_loss_corr.reset()
        total_loss_s.reset()
        total_loss_kl = 0.0
        total_mask_ratio = 0.0
        
        if use_ddp:
            label_train_loader.sampler.set_epoch(epoch)
            unlabel_train_loader.sampler.set_epoch(epoch)
        
        dataloader = zip(label_train_loader, unlabel_train_loader)
        for step, ((img_l, mask_l, l_image_path), 
                   (img_u_w, img_u_s, ignore_mask, cutmix_box, u_image_path)) in enumerate(dataloader):

            # if step == 1: break
            start_event.record()
            
            img_l, mask_l = img_l.cuda(non_blocking=True), mask_l.cuda(non_blocking=True)
            img_u_w = img_u_w.cuda(non_blocking=True)
            img_u_s, ignore_mask = img_u_s.cuda(non_blocking=True), ignore_mask.cuda(non_blocking=True)
            cutmix_box = cutmix_box.cuda(non_blocking=True)
            
            idx = torch.randperm(img_u_w.size(0), device=device)
            img_u_w_mix = img_u_w[idx].cuda(non_blocking=True)
            img_u_s_mix = img_u_s[idx].cuda(non_blocking=True)
            ignore_mask_mix = ignore_mask[idx].cuda(non_blocking=True)
            
            with torch.no_grad():
                model.eval()
                res_u_w_mix = model(img_u_w_mix, mode='test')
                logit_u_w_mix = res_u_w_mix['out'].detach()
                # logit은 모델의 확신 점수이다. 
                prob_u_w_mix = logit_u_w_mix.softmax(dim=1)
                conf_u_w_mix, mask_u_w_mix = prob_u_w_mix.max(dim=1) # pseudo label
                img_u_s[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1] = \
                    img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]
                
                del logit_u_w_mix
                del res_u_w_mix
            
            model.train()
            label_batch, unlabel_batch = img_l.shape[0], img_u_w.shape[0]
            
            # res_w = model(torch.cat((img_l, img_u_w)), need_fp=True, use_corr=True)
            with torch.amp.autocast('cuda'):
                results = model(torch.cat((img_l, img_u_w, img_u_s)))
                
                pred_x, pred_u_w = results['out'].split([label_batch, unlabel_batch])
                # 6번 수식의 z값이 pred_u_w_corr
                pred_x_corr, pred_u_w_corr = results['corr_out'].split([label_batch, unlabel_batch])
                pred_u_w_fp = results['out_fp'][label_batch:]
                # pred_u_w_corr_map : labeled + unlabeled weak간의 유사도가 높은부분에서의 unlabeled part
                pred_u_w_corr_map = results['binary_norm_corr_map'][label_batch:].detach()
                
                # res_s = model(img_u_s, need_fp=False, use_corr=True)
                pred_u_s = results['out_u_s']
                pred_u_s_corr = results['corr_out_u_s']


            # 2번 수식의 max F_hat
            softmax_u_w = pred_u_w.detach().softmax(dim=1)
            conf_u_w, mask_u_w = softmax_u_w.max(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed1 = pred_u_w_corr_map.clone()

            cutmix_box_map = (cutmix_box == 1).squeeze(dim=1)
            mask_u_w_cutmixed1[cutmix_box_map] = mask_u_w_mix[cutmix_box_map]
            conf_u_w_cutmixed1[cutmix_box_map] = conf_u_w_mix[cutmix_box_map]
            ignore_mask_cutmixed1[cutmix_box_map] = ignore_mask_mix[cutmix_box_map]
            
            cutmix_box_sample = rearrange(cutmix_box_map, 'n h w -> n 1 h w')
            ignore_mask_cutmixed1_sample = rearrange((ignore_mask_cutmixed1 != 255), 'n h w -> n 1 h w')
            # corr_map_u_w_cutmixed1: unlabel 이미지에 대한 c_hat
            # 따라서, 9번 수식에서 쓰이는 c_hat
            corr_map_u_w_cutmixed1 = (corr_map_u_w_cutmixed1 * ~cutmix_box_sample * ignore_mask_cutmixed1_sample).bool()
            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, update_g=True)
            thresh_global = thresh_controller.get_thresh_global()

            # 2번 수식
            conf_filter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (ignore_mask_cutmixed1 != 255))
            # conf_filter_u_w : labeled + unlabeled 데이터에서 모델이 예측한 신뢰도에서 더 정확한 신뢰도만을 가져온것. dynamic threshold를 통한
            conf_filter_u_w_without_cutmix = conf_filter_u_w.clone()
            conf_filter_u_w_sample = rearrange(conf_filter_u_w_without_cutmix, 'n h w -> n 1 h w')

            # 9번 수식에서 M_i * c_hat
            # region_propagation - 더 정확한 경계를 얻기위함.
            # conf_filter_u_w_sample: M_i
            segments = (corr_map_u_w_cutmixed1 * conf_filter_u_w_sample).bool() # region-propa 재료
            # segments : encoder에서의 유사도와 모델 예측 confidence를 곱해서 더더욱 중요한 부분만 살리기 위함
            
            
            ########### 9번 수식 전체 ############### region propag
            # 신뢰도가 낮은 예측 영역 (mask_u_w_cutmixed1)을 주변의 지표를 활용해 refinement
            # with torch.no_grad():
            for img_idx in range(tcfg.batch_size):
                for segment_idx in range(corr_map_u_w_cutmixed1.shape[1]):

                    segment = segments[img_idx, segment_idx]
                    segment_ori = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    high_conf_ratio = torch.sum(segment)/torch.sum(segment_ori)
                    if torch.sum(segment) == 0 or high_conf_ratio < thresh_global:
                        continue
                    unique_cls, count = torch.unique(mask_u_w_cutmixed1[img_idx][segment==1], return_counts=True)

                    if torch.max(count) / torch.sum(count) > thresh_global:
                        # 신뢰할 수 있는 영역안에 있는 클래스들 중 가장 많이 나타나는 클래스를 찾음. 
                        top_class = unique_cls[torch.argmax(count)] # 8번수식 k*
                        mask_u_w_cutmixed1[img_idx][segment_ori==1] = top_class # 10번 수식
                        conf_filter_u_w_without_cutmix[img_idx] = conf_filter_u_w_without_cutmix[img_idx] | segment_ori
            
            conf_filter_u_w_without_cutmix = conf_filter_u_w_without_cutmix | conf_filter_u_w

            label_loss = criterion_l(pred_x, mask_l)
            label_loss_corr = criterion_l(pred_x_corr, mask_l)

            # 1번 수식: Consistency Regularization
            loss_u_s = criterion_u(pred_u_s, mask_u_w_cutmixed1)
            # 1번 수식에서의 M_i
            loss_u_s = loss_u_s * conf_filter_u_w_without_cutmix
            loss_u_s = torch.sum(loss_u_s) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s = criterion_u(pred_u_s_corr, mask_u_w_cutmixed1)
            loss_u_corr_s = loss_u_corr_s * conf_filter_u_w_without_cutmix
            loss_u_corr_s = torch.sum(loss_u_corr_s) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            # 6번 수식
            loss_u_corr_w = criterion_u(pred_u_w_corr, mask_u_w)
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)
            

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s = F.log_softmax(pred_u_s, dim=1)

            # 3번 수식
            loss_u_kl_sa2wa = criterion_kl(logsoftmax_pred_u_s, softmax_pred_u_w)
            # 3번 수식에서의 M_i 값
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa, dim=1) * conf_filter_u_w
            loss_u_kl_sa2wa = torch.sum(loss_u_kl_sa2wa) / torch.sum(ignore_mask_cutmixed1 != 255).item()
            loss_u_kl = loss_u_kl_sa2wa

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            # total loss = 0.5 * L_s + 0.5 * L_u
            # L_s = 0.5*loss_x + 0.5*loss_x_corr
            # L_u = 0.5*loss_u_s + 0.25 * loss_u_kl + 0.25 * loss_u_corr + 0.25 * loss_u_w_fp
            # loss_u_w_fp: UniMatch에서 가져온 loss인 것 같음.
            loss = ( 0.5 * label_loss + 0.5 * label_loss_corr + loss_u_s * 0.25 + loss_u_kl * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.detach())
            total_label_loss.update(label_loss.detach())
            total_label_loss_corr.update(label_loss_corr.detach())
            total_loss_s.update(loss_u_s.detach())
            total_loss_kl += loss_u_kl.item()
            total_loss_w_fp.update(loss_u_w_fp.detach())
            total_loss_corr_u.update(loss_u_corr.detach())
            total_mask_ratio += ((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                (ignore_mask != 255).sum().item()

            iters = epoch * len(unlabel_train_loader) + step
            lr = tcfg.lr * (1 - iters / num_total_steps) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            end_event.record()
            torch.cuda.synchronize()
            
            elapsed_time = start_event.elapsed_time(end_event) / 1000.0
            time_left = (num_total_steps - iters) * elapsed_time
            # time_left = time.strftime("%H:%M:%S", time.gmtime(time_left))
            time_left = str(datetime.timedelta(seconds=int(time_left)))
            
            
            if step % 10 == 0 and rank == 0:
                hyperparam = f"Model: [{tcfg.model_name:>5}] | Time Left: [{time_left:>5}] | Epoch: [{epoch:>3}/{tcfg.num_epochs:>5}] | Step: [{step}/{len(unlabel_train_loader):>5}] | Elapsed time: {elapsed_time*50:.2f}s | lr: {lr:5.4f}"
                loss_info = f"total loss: {total_loss.compute():.3f}, loss x: {total_label_loss.compute():.3f}, loss_corr_ce: {total_label_loss_corr.compute():.3f}, " \
                            f"loss s: {total_loss_s.compute():.3f}, loss w_fp: {total_loss_w_fp.compute():.3f}, loss_corr_u: {total_loss_corr_u.compute():.3f}, Mask: {total_mask_ratio/(step+1):.3f}"
                print(hyperparam + '\n' + loss_info)
                print('-'*100)
                
            del results
        # region step 끝
        
        if tcfg.dataset == 'cityscapes':
            eval_mode = 'center_crop' if epoch < tcfg.num_epochs - 20 else 'slviding_window'
        else:
            eval_mode = 'original'
            
        torch.cuda.empty_cache()
        # res_val = evaluate(tcfg, mcfg, rank, model, validation_loader, aug_layer, eval_mode="original")
        res_val = evaluate(tcfg, mcfg, rank, model, validation_loader, mode='original')
        class_IOU = res_val['iou_class']
        
        # region  tensorboard
        tb.draw_scalar(epoch=epoch, item={"Optimization/loss/total loss": total_loss.compute(), 
                                          "Optimization/loss/label loss": total_label_loss.compute(), 
                                          "Optimization/loss/label loss corr": total_label_loss_corr.compute(), 
                                          "Optimization/loss/unlabel strong loss": total_loss_s.compute(), 
                                          "Optimization/loss/label weak fp loss": total_loss_w_fp.compute(), 
                                          "Optimization/loss/unlabel corr loss": total_loss_corr_u.compute(),
                                          "Optimization/learning_rate": lr,
                                          "Time/Elapsed time": elapsed_time,
                                          "Accuracy/eval/mIOU": res_val['mIOU'],
                                          })
        
        img_us = img_u_s.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_mask_us = pred_u_s.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_us = pred_u_s.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/unlabel strong image", 
                      image=img_us, 
                      pred=pred_mask_us,
                      conf=conf_us,
                      mask=None,
                      image_path=l_image_path,
                      epoch=epoch)

        img_l = img_l.detach().cpu().permute(0, 2, 3, 1).numpy()
        pred_mask_l = pred_x.detach().argmax(dim=1).unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        conf_l = pred_x.detach().softmax(dim=1).max(dim=1).values.unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        if mask_l.dim() == 4:
            gt = mask_l.detach().cpu().permute(0, 2, 3, 1).numpy()
        elif mask_l.dim() == 3:
            gt = mask_l.detach().unsqueeze(1).cpu().permute(0, 2, 3, 1).numpy()
        tb.draw_image(tag="train/label weak image", 
                      image=img_l, 
                      pred=pred_mask_l,
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


        if rank == 0:
            logger.info(f'***** Evaluation {eval_mode} ***** >>>> meanIOU: {res_val["mIOU"]:.4f} \n')
            logger.info(f'***** ClassIOU ***** >>>> \n{class_IOU}\n')
        else:
            torch.distributed.barrier()

        if res_val['mIOU'] > previous_best and rank == 0:
            model_save_dir = osp.join(tcfg.exp_dir, "models", tcfg.model_name)
            
            if previous_best != 0:
                os.remove(osp.join(model_save_dir, f'{mcfg.backbone}_{previous_best:.3f}.pth'))
            previous_best = res_val['mIOU']
            torch.save(model.state_dict(), osp.join(model_save_dir, f'{mcfg.backbone}_{res_val["mIOU"]:.3f}.pth'))
        
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
