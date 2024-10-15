import torch

import clip
import argparse
from PIL import Image
from tqdm import tqdm
import time
import json

from utils.preprocess import read_file_list, prepare_dataset_cls_tokens, preprocess, val_preprocess
from config.configs import cfg_from_file

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

voc_classes = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table',
               'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor']
pascal_context_classes = ['airplane', 'bag', 'bed', 'bedclothes', 'bench', 'bicycle', 'bird', 'boat', 'book', 'bottle',
                          'building', 'bus', 'cabinet', 'car', 'cat', 'ceiling', 'chair', 'cloth', 'computer', 'cow',
                          'cup', 'curtain', 'dog', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground',
                          'horse', 'keyboard', 'light', 'motorbike', 'mountain', 'mouse', 'person', 'plate', 'platform',
                          'potted plant', 'road', 'rock', 'sheep', 'shelves', 'sidewalk', 'sign', 'sky', 'snow', 'sofa',
                          'table', 'track', 'train', 'tree', 'truck', 'tv monitor', 'wall', 'water', 'window', 'wood']
ade_classes = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet',
               'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water',
               'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence',
               'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
               'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator',
               'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway',
               'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench',
               'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar',
               'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning',
               'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land',
               'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
               'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket',
               'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name',
               'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood',
               'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor',
               'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']

coco_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                      'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                      'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
                      'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                      'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
                      'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
                      'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
                      'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                      'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                      'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'branch', 'bridge', 'building',
                      'bush', 'cabinet', 'cage', 'cardboard', 'carpet', 'ceiling', 'tile ceiling', 'cloth', 'clothes',
                      'clouds', 'counter', 'cupboard', 'curtain', 'desk', 'dirt', 'door', 'fence', 'marble floor',
                      'floor', 'stone floor', 'tile floor', 'wood floor', 'flower', 'fog', 'food', 'fruit', 'furniture',
                      'grass', 'gravel', 'ground', 'hill', 'house', 'leaves', 'light', 'mat', 'metal', 'mirror', 'moss',
                      'mountain', 'mud', 'napkin', 'net', 'paper', 'pavement', 'pillow', 'plant', 'plastic', 'platform',
                      'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof', 'rug', 'salad', 'sand',
                      'sea', 'shelf', 'sky', 'skyscraper', 'snow', 'solid', 'stairs', 'stone', 'straw', 'structural',
                      'table', 'tent', 'textile', 'towel', 'tree', 'vegetable', 'brick wall', 'concrete wall', 'wall',
                      'panel wall', 'stone wall', 'tile wall', 'wood wall', 'water', 'waterdrops', 'blind window',
                      'window', 'wood']

cityscapes_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                      'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                      'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_learnable_params_with_name(model, prefix=''):
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            print(f'{prefix}{name}, Shape: {param.shape}')
    for name, module in model.named_children():
        print_learnable_params_with_name(module, prefix=f'{name}.' if name else '')


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


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


def ReCLIP(images, labels, cfg):
    clip_model, new_clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    total_accuracy = 0
    total_recall = 0
    idx = 0
    loop = tqdm(zip(images, labels))
    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)
    print(len(images))
    for image, label in loop:
        time.sleep(0.08)
        with open(image, 'rb') as f:
            value_buf = f.read()
        with open(label, 'rb') as f:
            label_buf = f.read()
        _, label, img_metas = preprocess(cfg, value_buf, label_buf, return_meta=True, unlabeled=False)

        idx += 1
        img = Image.open(image)

        img = new_clip_preprocess(img).unsqueeze(dim=0).to(device)
        image_features = clip_model.encode_image(img)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = clip_model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()

        temp = []
        for j in range(cfg.DATASET.NUM_CLASSES):
            if logits_per_image[0][j] >= cfg.DATASET.THRESHOLD:
                temp.append(j)
        if len(temp) == 0:
            _, pred_label = torch.sort(logits_per_image, descending=True, dim=1)
            temp = pred_label[:, :cfg.DATASET.K].squeeze(dim=0).tolist()

        with open('text/voc_pseudo_label_ReCLIP.json', mode='a') as cls_json:
            cls_json.write(json.dumps(temp))
            cls_json.write('\n')

        label_cls = set(label.flatten().tolist()[1:])
        same_values = set(label_cls) & set(temp)
        acc = len(same_values) / len(temp)
        if 255 in label_cls and len(label_cls) > 1:
            recall = len(same_values) / (len(label_cls) - 1)
        else:
            recall = len(same_values) / (len(label_cls))
        total_accuracy += acc
        total_recall += recall
        loop.set_postfix(idx=idx, acc=acc, recall=recall, avg_acc=total_accuracy / idx, avg_recall=total_recall / idx)
    total_accuracy = total_accuracy / len(images)
    total_recall = total_recall / len(images)
    print('total acc = {}, total_recall = {}'.format(total_accuracy, total_recall))


def ReCLIPPP(images, labels, cfg, window_size, step_size):
    clip_model, new_clip_preprocess = clip.load("ViT-B/16")
    clip_model = clip_model.to(device)

    _, text = prepare_dataset_cls_tokens(cfg)

    total_accuracy = 0
    total_recall = 0
    text_features = torch.load(cfg.DATASET.TEXT_WEIGHT).to(device).to(torch.float16)

    loop = tqdm(zip(images, labels))
    idx = 0
    print(len(images))
    for image, label in loop:
        time.sleep(0.08)
        idx += 1
        with open(image, 'rb') as f:
            value_buf = f.read()
        with open(label, 'rb') as f:
            label_buf = f.read()
        _, label, img_metas = preprocess(cfg, value_buf, label_buf, return_meta=True, unlabeled=False)

        img = Image.open(image)
        width, height = img.size
        cls_dict = {}
        temp = []
        # 遍历窗口的所有位置
        for y in range(0, height - window_size[1], step_size):
            for x in range(0, width - window_size[0], step_size):
                # 使用crop方法裁剪图像
                box = (x, y, x + window_size[0], y + window_size[1])

                cropped_img = img.crop(box)
                # 在这里处理裁剪后的图像cropped_img
                cropped_img = new_clip_preprocess(cropped_img).unsqueeze(dim=0).to(device)
                image_features = clip_model.encode_image(cropped_img)
                # normalized features
                image_features = image_features / image_features.norm(dim=1, keepdim=True)

                logit_scale = clip_model.logit_scale.exp()
                logits_per_crop = logit_scale * image_features @ text_features.t()

                sort_logits, pred_label = torch.sort(logits_per_crop, descending=True, dim=1)
                cls = pred_label[0, 0].item()
                cls_dict[cls] = cls_dict.get(cls, 0) + 1
        for key in cls_dict.keys():
            if cls_dict[key] > 0:
                temp.append(key)
        if len(temp) == 0:
            sorted(cls_dict.items(), key=lambda x: x[1])
            keys = cls_dict.keys()
            temp = list(keys)[:cfg.DATASET.K]
        with open('text/voc_pseudo_label_ReCLIPPP.json', mode='a') as cls_json:
            cls_json.write(json.dumps(temp))
            cls_json.write('\n')

        label_cls = set(label.flatten().tolist()[1:])
        same_values = set(label_cls) & set(temp)
        if len(temp) != 0:
            acc = len(same_values) / len(temp)
        else:
            acc = 0
        if 255 in label_cls and len(label_cls) > 1:
            recall = len(same_values) / (len(label_cls) - 1)
        else:
            recall = len(same_values) / (len(label_cls))
        total_accuracy += acc
        total_recall += recall
        loop.set_postfix(idx=idx, acc=acc, recall=recall, avg_acc=total_accuracy / idx, avg_recall=total_recall / idx)
    total_accuracy = total_accuracy / len(images)
    total_recall = total_recall / len(images)
    print('total acc = {}, total recall = {}'.format(total_accuracy, total_recall))


if __name__ == '__main__':
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    crop_size = cfg.DATASET.CROP_SIZE
    w = crop_size[0] / 6
    s = w / 2

    _, _, train_images, train_labels, _, _, _, pseudo_classes = read_file_list(cfg)
    if args.model_name == 'RECLIPPP':
        ReCLIPPP(train_images, train_labels, cfg, window_size=(w, w), step_size=s)
    else:
        ReCLIP(train_images, train_labels, cfg)
