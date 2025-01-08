import torch
import clip
import argparse
import numpy as np
from numpy import random
from torch import optim
from tqdm import tqdm
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from PIL import Image
import cv2
import os.path as osp

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_preprocess = clip.load("ViT-B/16")
clip_model = clip_model.to(device)

from config.configs import cfg_from_file
from model.model import ReCLIP_DISTILL
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens
from utils.test_mIoU import mean_iou


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_distill_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='RECLIP', type=str)
    args = parser.parse_args()
    return args


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power, min_lr=1e-4):
    if epoch > num_epochs:
        return min_lr
    lr = base_lr * (1 - epoch / num_epochs) ** power
    if lr <= min_lr:
        lr = min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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

        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=True)
        return img, label, img_metas, self.train_images[idx], idx

    def __len__(self):
        return len(self.train_images)


class Test(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_filenames, self.val_filenames, self.train_images, self.train_labels, self.val_images, self.val_labels, _, _ = read_file_list(cfg)

    def __getitem__(self, idx):
        with open(self.val_images[idx], 'rb') as f:
            value_buf = f.read()
        with open(self.val_labels[idx], 'rb') as f:
            label_buf = f.read()

        img, label, img_metas = preprocess(self.cfg, value_buf, label_buf, return_meta=True, unlabeled=True)
        return img, label, img_metas, self.val_images[idx], idx


def train():
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    log = open('experiments/log_voc_distill.txt', mode='a')
    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    text_embeddings = torch.load(cfg.DATASET.TEXT_WEIGHT)
    model = ReCLIP_DISTILL(clip_model, cfg, cls_name_token, text_categories=cfg.DATASET.NUM_CLASSES, text_channels=512, text_embeddings=text_embeddings)
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(cfg)
    train_data = Train(cfg)
    train_loader = DataLoader(dataset=train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
                              num_workers=cfg.NUM_WORKERS, pin_memory=False)
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.TRAIN.LR, momentum=0.9,
                          weight_decay=0.0005)
    max_epoch = cfg.TRAIN.MAX_EPOCH
    best_iou = 0
    for epoch in range(max_epoch):
        print('epoch {} start'.format(epoch))
        print('epoch {} start'.format(epoch), file=log)
        idx = 0
        model.train()

        running_loss = 0.0
        loop = tqdm(train_loader)
        for img, label, img_metas, filenames, ids in loop:
            time.sleep(0.08)
            lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch, cfg.TRAIN.LR, power=0.9)
            gt_cls = []
            batch_size = img.shape[0]

            for i in range(batch_size):
                temp = pseudo_classes[ids[i]]
                gt_cls.append(temp)

            loss = model(img, label, train=True, filenames=filenames, text=text, cls=gt_cls)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(epoch=epoch, img_loss=loss.item())

            print('filenames:{}, img_idx:{}, img_loss:{:.5f}, avg_loss:{:.5f}'.format(filenames, idx, loss.item(), running_loss / (idx + 1)), file=log)

            idx += 1
        print('epoch {} finish, lr:{}'.format(epoch, lr))
        print('epoch {} finish, lr:{}'.format(epoch, lr), file=log)

        model.eval()
        success_num = 0
        c_num = cfg.DATASET.NUM_CLASSES
        with torch.no_grad():
            for idx in range(len(val_images)):
                with open(val_images[idx], 'rb') as f:
                    value_buf = f.read()
                img = val_preprocess(cfg, value_buf).unsqueeze(dim=0).to(device)
                label = Image.open(val_labels[idx])
                ori_shape = tuple((label.size[1], label.size[0]))
                label = np.asarray(label)
                shape = img.shape[2:]
                output = model(img, label, train=False, filenames=val_filenames[idx], text=text)

                output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
                output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0],
                                                                                           ori_shape[1])
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

            if avg > best_iou:
                best_iou = avg
                torch.save(model.state_dict(), cfg.SAVE_DIR + 'best_weight.pth')

    log.close()


if __name__ == '__main__':
    train()
