# ReCLIP & ReCLIP++
This is the Pytorch Implementation for CVPR 2024  paper: [Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Learn_to_Rectify_the_Bias_of_CLIP_for_Unsupervised_Semantic_CVPR_2024_paper.pdf) and the extended version of the conference paper: [ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation](https://arxiv.org/abs/2408.06747).



#### Installation

##### Step 1 Install PyTorch and Torchvision

```python
pip install torch torchvision
# We're using CUDA 11.8 python==3.9 torch==2.1.0 
```

##### Step 2 Install CLIP

```
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```



#### Dataset

```
Maskclip
├── data
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   ├── ADEChallengeData2016
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   ├── Cityscapes
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```



#### Training

##### Step 1 Extract Text Embedding for datasets, e.g.,

```python
python utils/prompt_engineering.py --model ViT16 --class-set voc
# The Text Embeddings will be saved at 'text/voc_ViT16_clip_text.pth'
# Options for dataset: voc, context, ade, cityscapes, coco
```

##### Step 2 Extract Image-level Multi-label Hypothesis

```python
python tools/pseudo_class.py --cfg 'config/voc_train_ori_cfg.yaml' --model 'RECLIPPP'
# The Image-level Multi-label Hypothesis will be saved at 'text/voc_pseudo_label_ReCLIPPP.json'
# Options for dataset: voc, context, ade, cityscapes, coco
# Options for model: RECLIPPP(ReCLIP++), ReCLIP(ReCLIP)
```

##### Step 3 Rectification Stage, e.g.,

```python
python tools/train.py --cfg 'config/voc_train_ori_cfg.yaml' --model 'RECLIPPP'
# Options for dataset: voc, context, ade, cityscapes, coco
# Options for model: RECLIPPP(ReCLIP++), ReCLIP(ReCLIP)
```

##### Step 4 Distillation Stage, e.g.,

```python
python tools/distill.py --cfg 'config/voc_distill_ori_cfg.yaml'
# Options for dataset: voc, context, ade, cityscapes, coco
```



#### Test

##### Evaluation for Rectification Stage, e.g.,

```python
python tools/test.py --cfg 'config/voc_test_ori_cfg.yaml' --model 'RECLIPPP'
# Options for dataset: voc, context, ade, cityscapes, coco
# Options for model: RECLIPPP(ReCLIP++), ReCLIP(ReCLIP)
```

##### Evaluation for Distillation Stage, e.g.,

```python
python tools/distill_val.py --cfg 'config/voc_distill_ori_cfg.yaml'
# Options for dataset: voc, context, ade, cityscapes, coco
```



#### Weight

##### ReCLIP

|    Dataset     |                        Rectification                         |                         Distillation                         |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   PASCAL VOC   | [58.5](https://drive.google.com/file/d/1EtsaoIE6RrzckMNxJFUhBK0kmMFm3bID/view?usp=sharing) | [75.4](https://drive.google.com/file/d/1DHXRh6mb6SkmpWXbZYX1KkeGFUQq9t8Q/view?usp=sharing) |
| PASCAL Context | [25.8](https://drive.google.com/file/d/1Z-hWeLqXJ7niFafzXX3VX8t6Ou96zbgs/view?usp=sharing) | [33.8](https://drive.google.com/file/d/1jn8LhLc92IifGGyTK0hOZQ2Knj_IwJqs/view?usp=sharing) |
|     ADE20K     | [11.1](https://drive.google.com/file/d/1sfQLMYcIK0Q5JXu50fdBoF5X3zVmsS8K/view?usp=sharing) | [14.3](https://drive.google.com/file/d/1gAtmPW-oNPHMSBcI1eCH7tYIDhUWQEBP/view?usp=sharing) |



##### ReCLIP++

|    Dataset     |                        Rectification                         |
| :------------: | :----------------------------------------------------------: |
|   PASCAL VOC   | [85.4](https://drive.google.com/file/d/1FjTNj8PPGlce4xAQIXwjhi020glX3fSW/view?usp=drive_link) |
| PASCAL Context | [36.1](https://drive.google.com/file/d/1gYfh2fY5EyWCH8RXLgEq-bzcGToT40SM/view?usp=drive_link) |
|     ADE20K     | [16.4](https://drive.google.com/file/d/1p1b7WaaFHiNbq6ssvY1KZgqJ9G5XUKBr/view?usp=drive_link) |
|   Cityscapes   | [26.5](https://drive.google.com/file/d/18Hdorz5nbLS5DFbXFcXKvgknzyD0W4sQ/view?usp=drive_link) |
|   COCO Stuff   | [23.8](https://drive.google.com/file/d/1kqvNi_2J-IE0Tg9Rs2T2kkM5S_KRZsUz/view?usp=drive_link) |



#### Citing

Please cite our paper if you use our code in your research:

```
@inproceedings{wang2024learn,
  title={Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation},
  author={Wang, Jingyun and Kang, Guoliang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4102--4112},
  year={2024}
}

@article{wang2024reclip++,
  title={ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation},
  author={Wang, Jingyun and Kang, Guoliang},
  journal={arXiv preprint arXiv:2408.06747},
  year={2024}
}
```



#### Contact

For questions about our paper or code, please contact wangjingyun0730@gmail.com.
