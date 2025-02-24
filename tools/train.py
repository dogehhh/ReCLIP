import argparse
import torch
import clip
from torch import optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import time
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import torch.distributed as dist
import torch.multiprocessing as mp

import sys

import os


from config.configs import cfg_from_file
from model.model import RECLIPPP, ReCLIP
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_train_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='RECLIPPP', type=str)
    args = parser.parse_args()
    return args


class Train(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, _, self.train_images, self.train_labels, _, _, _, self.pseudo_classes = read_file_list(cfg)

    def __getitem__(self, idx):
        with open(self.train_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.train_labels[idx], 'rb') as f:
            label_buf = f.read()
        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=False)
        return img, label, img_metas, self.train_images[idx], self.pseudo_classes[idx]

    def __len__(self):
        return len(self.train_images)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    clip_model, clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(rank)

    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    log = open('experiments/log_voc_rectification.txt', mode='a')
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    cls_name_token, classes = prepare_dataset_cls_tokens(cfg)
    text_weight = torch.load(cfg.DATASET.TEXT_WEIGHT)

    train_data = Train(cfg)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    train_loader = DataLoader(dataset=train_data, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True, sampler=train_sampler)

    if args.model_name == 'RECLIPPP':
        model = RECLIPPP(cfg=cfg, clip_model=clip_model, rank=rank, zeroshot_weights=text_weight)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank], output_device=rank,
                                                          find_unused_parameters=True)
    else:
        model = ReCLIP(cfg=cfg, clip_model=clip_model, rank=rank, zeroshot_weights=text_weight)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[rank], output_device=rank,
                                                          find_unused_parameters=True)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9,
                          weight_decay=0.0005)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    if cfg.TRAIN.EPOCH >= 0:
        stop_epoch = cfg.TRAIN.EPOCH
    else:
        stop_epoch = max_epoch
    c_num = cfg.DATASET.NUM_CLASSES
    best_iou = 0.0
    for epoch in range(max_epoch):
        idx = 0
        model.train()
        running_loss = 0.0

        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, power=0.9)
        loop = tqdm(train_loader)

        for img, label, img_metas, filenames, pseudo_class in loop:
            time.sleep(0.08)
            gt_cls = []
            batch_size = img.shape[0]
            for i in range(batch_size):
                temp = [int(tensor.item()) for tensor in pseudo_class]
                gt_cls.append(temp)

                if len(temp) == 0:
                    continue
            if len(gt_cls[0]) == 0:
                continue
            output, loss = model(img.to(rank), gt_cls, text_weight, cls_name_token, training=True, img_metas=img_metas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            loop.set_postfix(epoch=epoch, img_loss=loss.item(), avg_loss=running_loss / (idx + 1))
            print('filenames:{}, img_idx:{}, img_loss:{:.5f}, avg_loss:{:.5f}'.format(filenames, idx, loss.item(), running_loss / (idx + 1)), file=log)
            idx += 1
        print('epoch {} finish, lr:{}'.format(epoch, lr), file=log)

        if rank == 0:
            model.eval()
            success_num = 0
            with torch.no_grad():
                for idx in range(len(val_images)):
                    with open(val_images[idx], 'rb') as f:
                        value_buf = f.read()
                    img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)
                    label = Image.open(val_labels[idx])
                    ori_shape = tuple((label.size[1], label.size[0]))
                    label = np.asarray(label)
                    gt_cls = []
                    label_cls = set(label.flatten().tolist()[1:])
                    for cls in label_cls:
                        if cls != 0 and cls != 255:
                            gt_cls.append(cls - 1)
                    shape = img.shape[2:]
                    output = model(img, gt_cls, text_weight, cls_name_token, training=False)

                    output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
                    output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])

                    output = F.softmax(output, dim=1)
                    output = torch.argmax(output, dim=1).squeeze(dim=0)
                    torch.save(output, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
                    success_num += 1

                    print('filenames:{}, img_idx:{}'.format(val_filenames[idx], idx))

                iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255, nan_to_num=0, reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
                print(iou['IoU'])
                avg = iou['IoU'].sum() / c_num
                print('avg:%.4f' % (avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg))
                print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, len(val_images), avg), file=log)
                log.write('miou:{}'.format(avg))
                if avg > best_iou:
                    best_iou = avg
                    torch.save(model.state_dict(), cfg.SAVE_DIR + 'best_weight.pth')
        if epoch == stop_epoch:
            break
    log.close()


if __name__ == '__main__':
    world_size = 1
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
