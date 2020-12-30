# Center and Scale Prediction (CSP) for pedestrian detection
## Introduction
This is the unofficial pytorch implementation of [High-level Semantic Feature Detection: A New Perspective for Pedestrian Detection](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_High-Level_Semantic_Feature_Detection_A_New_Perspective_for_Pedestrian_Detection_CVPR_2019_paper.pdf). CSP is an effective and efficient method for pedestrian detector and achieves promising results on the CityPersons dataset. We implement CSP in pytorch based on previous works [offical code (keras)](https://github.com/liuwei16/CSP), [unofficial code (pytorch)](https://github.com/lwpyr/CSP-pedestrian-detection-in-pytorch). Compared with them, our codes have following features:
- **Support Apex Mix Precision** 
- **Support Distributed and Non-distributed training**
- **Support more backbones, such as ResNet, DLA-34, HRNet**

We obtain much faster training/inference speed (3 hours for 120 epoches using two gpus) and  comparable performance. We think CSP is a strong baseline for pedestrian detection, and it still has much room for improvement. 
We will **continuously update this repo**, and add some useful tricks (e.g. data augmentation) for better performance.

## models
| Model | Reasonable | Heavy Occlusion | All  | Training time | Link |
| ----- | :--------: | :-------------: | :----: | :----------: | :----: |
| ResNet-50 | 11.30 | 41.09 | 37.55 | ~5 hours |      |
| DLA-34 | 11.12 | 43.00 | 37.32 | ~3 hours |      |
| HRNet-18 | 10.24 | 37.72 | 36.15 | ~11 hours |      |
| HRNet-32| 9.69 | 36.48 | 35.47 | ~13 hours | |
|HRNet-32 + [SWA](https://arxiv.org/abs/2012.12645) | 9.66 | **34.61** | 34.86 | |

**Note**: Training time is evaluated in two 2080Ti GPUs for 120 epochs. We will further tune some hyperparameters (e.g. learning rate, batchsize) these days, then will release our models.

## Get Start

### Prerequisites
- Pytorch 1.2+
- Python 3.6+
- APEX (Install APEX following the [offical instruction](https://github.com/NVIDIA/apex))

### Installation
````bash
git clone git@github.com:ligang-cs/CSP-Pedestrain-detection.git
cd CSP-Pedestrian-detection/utils
make all
````

### Data preparation
You need to download the [CityPersons](https://github.com/cvgroup-njust/CityPersons) dataset.

Your directory tree should be look like this:
````bash
$root_path/
├── images
│   ├── train
│   └── val
├── annotations
│   ├── anno_train.mat
│   └── anno_val.mat
````

### Train and test

Please specify the configuration file.

#### Distributed training
````bash
CUDA_VISIBLE_DEVICES=<gpus_ids> python -m torch.distributed.launch --nproc_per_node <gpus_number> trainval_distributed.py --work-dir <save_path> 
````
#### Non-distributed training 
````bash
CUDA_VISIBLE_DEVICES=<gpus_ids> python trainval.py --work-dir <save_path>
````
#### Test
````bash
CUDA_VISIBLE_DEVICES=<gpus_ids> python test.py --val-path <checkpoint_path> --json-out <results_path>
````

## Contact

If you have any questions, please do not hesitate to contact Li Gang (gang.li@njust.edu.cn).  

We also appreciate all contributions to improve this repo. 

## Acknowledgement

- HRNet codes are folked from [HRNet offical codes](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1)
- DLA-34 codes are folked from [here](https://github.com/ucbdrive/dla)

Many thanks to them !







