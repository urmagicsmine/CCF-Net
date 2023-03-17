
# New Features for Lung General Detection.


## Introduction

The master branch works with **PyTorch 1.1** or higher.


### Major features

- **Training with post-combined CT-slices**

  Train 2D CT slice level detection models with post-combined slices, *e.g.* read in three consecutive slice and merge them into one RGB image for training.

- **Support of training with negative samples**

  Modify the data loading process and max-iou-assigner & smooth-l1-loss for training RCNN (w/o mask) based models.



## Contact

This repo is currently maintained by LGD Group.