# CCF-Net
Implementation of our ICIP 2021 paper : [CCF-Net: Composite Context Fusion Network with Inter-slice Correlative Fusion for Multi-disease Lesion Detection](https://ieeexplore.ieee.org/abstract/document/9506563).

## News!
To the best of our knowledge, CCF-Net demonstrates limitations when applied to datasets of moderately large scale.
In our recent study, we propose a practical and effective solution to pre-train 3D CNNs with 2D data, which results in highly robust and versatile 3D pre-trained models capable of handling a range of downstream tasks.
Further details on our approach can be found [here](https://github.com/urmagicsmine/CSPR).

## Installation
This code is based on [MMDetection](https://github.com/open-mmlab/mmdetection). Please see it for installation.


## Data preparation
We perform experiments on the [Tianchi Pulmonary Multi-disease datase](https://tianchi.aliyun.com/competition/entrance/231724/information), which is not publicly avalible now.
While, any datasets with coco-style annotations can be used to train a CCF model.

## Training
Please run this bash to train a model:
```
bash ccfnet.sh train
```
