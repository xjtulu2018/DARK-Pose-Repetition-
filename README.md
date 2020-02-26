# Recurrence of DARK Pose
## News
- Our recurence of [*Recurrence of Distribution-Aware Coordinate Representation for Human Pose Estimation*](https://arxiv.org/abs/1910.06278) based on the codes of hrnet human-pose estimation(https://github.com/leoxiaobin/deep-high-resolution-net.pytorch) is released here.
- Our recurence of DARK Pose achieves 3.7 AP improvement on hrnet w32_128x96 and 1.1 AP on hrnet w32_256x128 on COCO.

## Introduction
This is an pytorch recurrence of [*Recurrence of Distribution-Aware Coordinate Representation for Human Pose Estimation*](https://arxiv.org/abs/1910.06278). 
The basic code is forked from hrnet human-pose estimation(https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).
In this work, we add 3 function to complete DARK_Pose:
1. Coordinate Encoding function: it is actually a unbiased heatmap generation function. (lib/dataset/JointsDataset.py).
2. Gaussian filter function: it is used to smooth the predict-heatmap. (lib/core/inference.py)
3. Coordinate Encoding function: it is used to predict keypoints from heatmap.  (lib/core/inference.py)
Besides, we add three parameter in config files to imply DARK pose in the code of hrnet：
1. HEATMAP_EN: true for applying coordinate encoding.
2. HEATMAP_DE: true for applying coordinate decoding.
3. HEATMAP_DE_DM: true for applying gaussian filter.</br>

## Main Results
### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | DARK  |   AP   |
|--------------------|------------|-------|--------|
| pose_resnet_50     |    256x192 | True  |   71.3 |
| pose_resnet_50     |    256x192 | False |   70.4 |
| **pose_hrnet_w32** |    256x192 | True  |   74.3 | 
| **pose_hrnet_w32** |    256x192 | False |   75.4 | 
| **pose_hrnet_w32** |    128x96  | True  |   65.8 |
| **pose_hrnet_w32** |    128x96  | False |   69.5 |
### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- pose_resnet_[50,101,152] is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).
- GFLOPs is for convolution and linear layers only.

## Environment
Same with hrnet.

## Quick start
Same with hrnet.

