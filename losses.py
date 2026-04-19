import torch
import torch.nn as nn
import torch.nn.functional as F
from util.ohem import ProbOhemCrossEntropy2d


class LossFactory:
    @staticmethod
    def label(mode, device):
        def cross_entropy(pred, true, ignore_index):
            loss_ce = nn.CrossEntropyLoss(ignore_index=ignore_index).to(device, non_blocking=True)
            loss = loss_ce(pred, true)
            
            return loss

        
        def ohem(pred, true, ignore_index, threshold, min_kept):
            loss_ohem = ProbOhemCrossEntropy2d(ignore_index=ignore_index,
                                                thresh=threshold,
                                                min_kept=min_kept
                                                ).to(device, non_blocking=True)
            loss = loss_ohem(pred, true)
            
            return loss
        
        def dice(logits, targets, ignore_index=255, epsilon=1e-6):
            probs = F.softmax(logits, dim=1)
            # 1. ignore_index 위치를 찾음
            mask = (targets != ignore_index)
            mask = mask.unsqueeze(1).float()
            
            # 2. 에러 방지를 위해 ignore_index를 0으로 임시 치환 (나중에 마스킹할 것이므로 안전함)
            targets_temp = targets.clone()
            targets_temp[targets == ignore_index] = 0
            
            # 3. 이제 안전하게 one_hot 실행
            targets_one_hot = F.one_hot(targets_temp, num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()
            
            intersection = probs * targets_one_hot
            cardinality = probs + targets_one_hot
            
            intersection = torch.sum(intersection * mask, dim=(0, 2, 3))
            cardinality  = torch.sum(cardinality * mask, dim=(0, 2, 3))
            
            dice_coeff = (2. * intersection + epsilon) / (cardinality + epsilon)
            dice = 1. - dice_coeff
            
            dice_loss = torch.mean(dice)
            
            return dice_loss
            
        if mode == 'ce':
            return cross_entropy
        elif mode == 'ohem':
            return ohem
        elif mode == 'dice':
            return dice
    
    
    @staticmethod
    def unlabel(mode, device):
        def us_consistency_regularization(pred, true, confidence, ignore_mask):
            loss_ce = nn.CrossEntropyLoss(reduction='none').to(device, non_blocking=True)
            loss = loss_ce(pred, true)
            loss *= confidence
            denom = torch.sum(ignore_mask != 255)
            denom = denom.clamp_min(1)
            loss = torch.sum(loss) / denom

            return loss
        
        
        def uw_consistency_regularization(pred, true, confidence, threshold, ignore_mask):
            loss_ce = nn.CrossEntropyLoss(reduction='none').to(device, non_blocking=True)
            loss = loss_ce(pred, true)
            loss = loss * ((confidence >= threshold) & (ignore_mask != 255))
            denom = torch.sum(ignore_mask != 255)
            denom = denom.clamp_min(1)
            loss = torch.sum(loss) / denom
            
            return loss
        
        
        def kl_divergence(pred_us, pred_uw, confidence, ignore_mask):
            softmax_pred_u_w = F.softmax(pred_uw.detach(), dim=1)
            logsoftmax_pred_us = F.log_softmax(pred_us, dim=1)
            
            loss_kl = nn.KLDivLoss(reduction='none').to(device, non_blocking=True)
            loss = loss_kl(logsoftmax_pred_us, softmax_pred_u_w)
            loss = torch.sum(loss, dim=1) * confidence
            denom = torch.sum(ignore_mask != 255)
            denom = denom.clamp_min(1)
            loss = torch.sum(loss) / denom
            
            return loss
            
        if mode == 'us_cr':
            return us_consistency_regularization
        elif mode == 'uw_cr':
            return uw_consistency_regularization
        elif mode == 'kl':
            return kl_divergence