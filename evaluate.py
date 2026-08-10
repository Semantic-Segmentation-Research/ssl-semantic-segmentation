import time
import argparse
import os
import numpy as np
import torch
import torch.distributed as dist
from util.dist_helper import setup_distributed
from model.semseg.deeplabv3plus import DeepLabV3Plus

from torch.utils.data import DataLoader
import yaml
from dataset.semi import SemiDataset
from util.utils import AverageMeter, intersectionAndUnion


def evaluate(model, loader, mode, cfg, rank=0):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for _ in range(10):
            sample = next(iter(loader))
            img = sample[0].cuda(non_blocking=True)
            _ = model(img)
            torch.cuda.synchronize()

        
        inference_time = 0
        num = 0
        for img, mask, _, _ in loader:
            # start_time = time.time()
            
            img = img.cuda()
            b, _, h, w = img.shape
            
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                final = torch.zeros(b, 19, h, w, device='cuda')
                stride = int(grid * 2 / 3)
                # row = 0
                # while row < h:
                #     col = 0
                #     while col < w:
                #         res = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                #         pred = res['out']
                #         final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                #         col += int(grid * 2 / 3)
                #     row += int(grid * 2 / 3)
                with torch.no_grad(), torch.autocast(device_type='cuda'):
                    row = 0
                    while row < h:
                        col = 0
                        while col < w:
                            r1 = min(h, row + grid)
                            c1 = min(w, col + grid)
                            
                            res = model(img[:, :, row:r1, col:c1])
                            # float32로 캐스팅 후 누적 (정확도 보존을 위해)
                            pred = res['out'].float().softmax(dim=1) 
                            
                            final[:, :, row:r1, col:c1] += pred
                            
                            col += stride
                        row += stride
                        
                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                torch.cuda.synchronize()
                start = time.perf_counter()

                res = model(img)

                torch.cuda.synchronize()
                end = time.perf_counter()

                pred = res['out'].argmax(dim=1)
                
                num += 1
                inference_time += (end - start)
            # elapsed_time = time.time() - start_time
            
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()
            if rank != 0:
                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())


    inference_time /= num
    inference_time *= 1000

    print(f"Inference Time: {inference_time:.3f} ms")

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    return_dict['iou_class'] = iou_class
    return_dict['mIOU'] = mIOU
    # return_dict['elapsed_time'] = elapsed_time

    return return_dict


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/home/dev/ssl-semantic-segmentation/configs/cityscapes.yaml')
    parser.add_argument('--checkpoint_path', type=str, default="/home/dev/ssl-semantic-segmentation/experiments/models/corrmatch_ver3_2/resnet50_53.906.pth")
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    args = parser.parse_args()
    # setup_distributed(port=args.port)
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    model = DeepLabV3Plus(cfg)

    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    state_dict = {
        k: v for k, v in state_dict.items()
        if not k.endswith('total_ops') and not k.endswith('total_params')
    }

    model.load_state_dict(state_dict)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # local_rank = int(os.environ["LOCAL_RANK"])
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=False)

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', size=cfg['crop_size'])
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
    #                        drop_last=False, sampler=valsampler)
    valloader = DataLoader(valset, 
                           batch_size=1, 
                           pin_memory=True, 
                           num_workers=4,
                           drop_last=False)

    model.eval()
    res_val = evaluate(model, valloader, 'original', cfg)

    mIOU = res_val['mIOU']
    iou_class = res_val['iou_class']

    print(mIOU)
    print(iou_class)


if __name__ == '__main__':
    main()
