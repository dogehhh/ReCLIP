import argparse
import torch
import clip
import torch.nn.functional as F
import numpy as np
from PIL import Image


from config.configs import cfg_from_file
from model.model import RECLIPPP_DISTILL, ReCLIP_DISTILL
from utils.test_mIoU import mean_iou
from utils.preprocess import val_preprocess, preprocess, read_file_list, prepare_dataset_cls_tokens

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model, clip_preprocess = clip.load("ViT-B/16")
clip_model = clip_model.to(device)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='config/voc_distill_ori_cfg.yaml', type=str)
    parser.add_argument('--model', dest='model_name',
                        help='model name',
                        default='RECLIPPP', type=str)
    args = parser.parse_args()
    return args


def test():
    args = get_parser()
    cfg_file = args.cfg_file
    cfg = cfg_from_file(cfg_file)
    train_filenames, val_filenames, train_images, train_labels, val_images, val_labels, results_iou, pseudo_classes = read_file_list(
        cfg)
    cls_name_token, text = prepare_dataset_cls_tokens(cfg)
    text_embeddings = torch.load(cfg.DATASET.TEXT_WEIGHT)
    if args.model_name == 'RECLIPPP':
        model = RECLIPPP_DISTILL(clip_model, cfg, cls_name_token, text_categories=cfg.DATASET.NUM_CLASSES, text_channels=512, text_embeddings=text_embeddings)
    else:
        model = ReCLIP_DISTILL(clip_model, cfg, cls_name_token, text_categories=cfg.DATASET.NUM_CLASSES, text_channels=512, text_embeddings=text_embeddings)
    distill_weight = torch.load(cfg.LOAD_DISTILL_PATH)
    model.load_state_dict(distill_weight, strict=True)
    model = model.to(device)

    success_num = 0
    c_num = cfg.DATASET.NUM_CLASSES
    model.eval()
    with torch.no_grad():
        for idx in range(len(val_images)):
            with open(val_images[idx], 'rb') as f:
                value_buf = f.read()
            img = val_preprocess(cfg, value_buf).unsqueeze(dim=0)
            label = Image.open(val_labels[idx])
            ori_shape = tuple((label.size[1], label.size[0]))
            label = np.asarray(label).copy()
            label[label == 0] = 255

            shape = img.shape[2:]
            output = model(img, label, train=False, filenames=val_filenames[idx], text=text)

            # pd
            if args.model_name == 'RECLIPPP':
                N, C, H, W = output.shape
                _output = F.softmax(output*2, dim=1)
                max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
                selected_cls = (max_cls_conf < cfg.TEST.PD)[:, :, None, None].expand(N, C, H, W)
                output[selected_cls] = -100
            output = F.interpolate(output, shape, None, 'bilinear', False).reshape(1, c_num, shape[0], shape[1])
            output = F.interpolate(output, ori_shape, None, 'bilinear', False).reshape(1, c_num, ori_shape[0], ori_shape[1])
            output = F.softmax(output, dim=1)
            output = torch.argmax(output, dim=1).squeeze(dim=0)
            torch.save(output, cfg.SAVE_DIR + val_filenames[idx] + '.pt')
            success_num += 1
            print('filenames:{}, img_idx:{}'.format(val_filenames[idx], idx))

        iou = mean_iou(results_iou, val_labels, num_classes=c_num + 1, ignore_index=255, nan_to_num=0,
                       reduce_zero_label=cfg.DATASET.REDUCE_ZERO_LABEL)
        print(iou['IoU'])
        avg = iou['IoU'].sum() / c_num
        print('avg:%.4f' % (avg))
        print('\n\nfinish with %d/%d\nthe mIOU:%.4lf' % (success_num, 3616, avg))


if __name__ == '__main__':
    test()