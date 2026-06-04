import os
import os.path as osp

import matplotlib
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
# from model.semseg.deeplabv3plus_vis import DeepLabV3Plus
from configuration import TestConfig, DataConfig, ModelConfig
from util.utils import intersectionAndUnion, AverageMeter
import pandas as pd
import cityscapesLabels as labels
from tqdm import tqdm




def color_map():
    cmap = np.zeros((256, 3), dtype='uint8')

    cmap[0] = np.array([128, 64, 128])
    cmap[1] = np.array([244, 35, 232])
    cmap[2] = np.array([70, 70, 70])
    cmap[3] = np.array([102, 102, 156])
    cmap[4] = np.array([190, 153, 153])
    cmap[5] = np.array([153, 153, 153])
    cmap[6] = np.array([250, 170, 30])
    cmap[7] = np.array([220, 220, 0])
    cmap[8] = np.array([107, 142, 35])
    cmap[9] = np.array([152, 251, 152])
    cmap[10] = np.array([70, 130, 180])
    cmap[11] = np.array([220, 20, 60])
    cmap[12] = np.array([255,  0,  0])
    cmap[13] = np.array([0,  0, 142])
    cmap[14] = np.array([0,  0, 70])
    cmap[15] = np.array([0, 60, 100])
    cmap[16] = np.array([0, 80, 100])
    cmap[17] = np.array([0,  0, 230])
    cmap[18] = np.array([119, 11, 32])

    return cmap


def main():
    model_name = os.listdir(tcfg.model_save_dir)[-1]
    model_path = osp.join(tcfg.model_save_dir, model_name)
    result_dir = osp.join(osp.dirname(__file__), 'results') 
    os.makedirs(osp.join(result_dir, 'rgb'), exist_ok=True)
    os.makedirs(osp.join(result_dir, 'gt'), exist_ok=True)
    os.makedirs(osp.join(result_dir, 'pred'), exist_ok=True)
    
    
    model = DeepLabV3Plus(tcfg, mcfg, pretrained_path='')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    valset = SemiDataset(root=tcfg.data_root, 
                         mode='test', 
                         valid_path=tcfg.valid_path,
                         size=tcfg.crop_size)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=False)

    
    results = []
    class_names = [l.name for l in labels.labels if l.trainId != 255]
    model.eval()
    with torch.no_grad():
        for img, mask, image_path in tqdm(valloader, desc='inference'):
            img = img.cuda(non_blocking=True)
            
            res = model(img, mode='test')
            pred = res['out']
            pred_mask = pred.argmax(dim=1)
            # pred_conf = pred.softmax(dim=1).max(dim=1)[0]
            
            # # take 0.95 as an example
            # pred_conf_fliter = (pred_conf <= tcfg.threshold)
            # mask_fliter = pred_mask.clone()
            # mask_fliter[pred_conf_fliter] = 255

            for i in range(pred_mask.shape[0]):
                file_name = osp.split(image_path[i])[-1][:-4]

                rgb             = img[i].cpu().numpy().transpose(1, 2, 0)
                rgb             = (rgb * 255).astype(np.uint8)
                mask_pred_i     = pred_mask[i]
                mask_i          = mask[i]
                
                # =================================================
                # Quantative Evaluation
                # =================================================
                intersection, union, target = intersectionAndUnion(mask_pred_i.cpu().numpy(), mask_i.cpu().numpy(), mcfg.num_classes, 255)
                
                iou_class_p_img = intersection / (union + 1e-10)

                class_p_img = np.unique(mask_i[mask_i != 255])
                nonzero_iou_p_img = iou_class_p_img[class_p_img]
                row_data = {'image_name': file_name, 'miou_p_img': np.mean(nonzero_iou_p_img) * 100}
                for class_name, iou in zip(class_names, nonzero_iou_p_img):
                    row_data[class_name] = iou
                results.append(row_data)
                
                
                # =================================================
                # Qualitative Evaluation
                # =================================================
                rgb             = Image.fromarray(rgb)
                mask_i          = Image.fromarray(mask_i.cpu().numpy().astype(np.uint8), mode='P')
                mask_pred_i     = Image.fromarray(mask_pred_i.cpu().numpy().astype(np.uint8), mode='P')
                
                platte = color_map()
                mask_i.putpalette(platte)
                mask_pred_i.putpalette(platte)
                
                rgb.save(osp.join(result_dir, 'rgb', f'{file_name}_rgb.png'))
                mask_i.save(osp.join(result_dir, 'gt', f'{file_name}_mask_gt.png'))
                mask_pred_i.save(osp.join(result_dir, 'pred', f'{file_name}_mask_pred.png'))
                
        df = pd.DataFrame(results)
        df.to_csv(osp.join(result_dir, 'evaluation_results.csv'), index=False)


if __name__ == "__main__":
    dcfg = DataConfig()
    tcfg = TestConfig()
    mcfg = ModelConfig()
    
    main()