# ReCLIP & ReCLIP++
This is the Pytorch Implementation for CVPR 2024  paper: [Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_Learn_to_Rectify_the_Bias_of_CLIP_for_Unsupervised_Semantic_CVPR_2024_paper.pdf) and the extended version of the conference paper: [ReCLIP++: Learn to Rectify the Bias of CLIP for Unsupervised Semantic Segmentation](https://arxiv.org/abs/2408.06747).



#### Installation

##### Step 1 Install PyTorch and Torchvision

```python
pip install torch torchvision
# We're using python==3.9 torch==1.11.0 and torchvision==0.12.0 
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
python tools/pseudo_label.py --cfg 'config/voc_train_ori_cfg.yaml' --model 'RECLIPPP'
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
python tools/distill.py --cfg 'config/voc_distill_ori_cfg.yaml' --model 'RECLIPPP'
# Options for dataset: voc, context, ade, cityscapes, coco
# Options for model: RECLIPPP(ReCLIP++), ReCLIP(ReCLIP)
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
python tools/distill_val.py --cfg 'config/voc_distill_ori_cfg.yaml' --model 'RECLIPPP'
# Options for dataset: voc, context, ade, cityscapes, coco
# Options for model: RECLIPPP(ReCLIP++), ReCLIP(ReCLIP)
```



#### Weight

##### ReCLIP

|    Dataset     |                        Rectification                         |                         Distillation                         |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   PASCAL VOC   | [58.5](https://drive.google.com/file/d/1EtsaoIE6RrzckMNxJFUhBK0kmMFm3bID/view?usp=sharing) | [75.4](https://drive.google.com/file/d/1DHXRh6mb6SkmpWXbZYX1KkeGFUQq9t8Q/view?usp=sharing) |
| PASCAL Context | [25.8](https://drive.google.com/file/d/1Z-hWeLqXJ7niFafzXX3VX8t6Ou96zbgs/view?usp=sharing) | [33.8](https://drive.google.com/file/d/1jn8LhLc92IifGGyTK0hOZQ2Knj_IwJqs/view?usp=sharing) |
|     ADE20K     | [11.1](https://drive.google.com/file/d/1sfQLMYcIK0Q5JXu50fdBoF5X3zVmsS8K/view?usp=sharing) | [14.3](https://drive.google.com/file/d/1gAtmPW-oNPHMSBcI1eCH7tYIDhUWQEBP/view?usp=sharing) |



##### ReCLIP++

|    Dataset     |                        Rectification                         |                         Distillation                         |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   PASCAL VOC   | [73.5](https://drive.google.com/file/d/1XvFJQ74xDkioQtMY9LMBYXxghhYZuLtO/view?usp=drive_link) | [79.2](https://drive.google.com/file/d/1rB17KIcAM0T7aBmS38IpgoebUw3Xj7Wo/view?usp=drive_link) |
| PASCAL Context | [34.0](https://drive.google.com/file/d/1r8b3U3i9z-_Qo7GGX8kre0WOzruVym-X/view?usp=sharing) | [34.5](https://drive.google.com/file/d/1AH5zIWRx7iLSct4z0uokK_jhGwC4wtpS/view?usp=sharing) |
|     ADE20K     | [13.5](https://drive.google.com/file/d/10GcLi-rh7wAgcP09TueWuBBuqoimEiG8/view?usp=drive_link) | [14.5](https://drive.google.com/file/d/1lSHivA9_MwMBK7qXgrzXg5TLkytIgRJ-/view?usp=sharing) |
|   Cityscapes   | [21.7](https://drive.google.com/file/d/1CcIGkPzRLK0rV2jV2CbJH3tEsbGBwUvG/view?usp=drive_link) | [26.9](https://drive.google.com/file/d/1exHb-cX2gIkaANATohrfg1CRzBbZ6xhm/view?usp=sharing) |
|   COCO Stuff   | [20.9](https://drive.google.com/file/d/1oYdTp998sUEy7dvWRfKeyo2iIbGvXEen/view?usp=drive_link) | [23.2](https://drive.google.com/file/d/1JyDOD-ugut6rbjDFAtbtfYMPvyqhnRwC/view?usp=sharing) |



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
