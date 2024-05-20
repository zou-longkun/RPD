#Unsupervised Domain Adaptation on Point Cloud via Cross-Modal Knowledge Transfer

## Introduction


This repo is a PyTorch implementation for **Unsupervised Domain Adaptation on Point Cloud via Cross-Modal Knowledge Transfer(CMKT)**
[Paper]()
## Requirements
The code has been tested with

- Python >= 3.7
- PyTorch == 1.8.0+cu111
- torch-scatter == 2.0.7
- torchsampler == 0.1.2
- torchvision == 0.9.0+cu111

Some dependent packages:

- [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD)

```
cd PyTorchEMD
python setup.py install
```
## Dataset
Download the official [PointDA-10](https://drive.google.com/uc?id=1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J) dataset and put the folder under `[your_dataroot]/data/`.  
After download, the directory structure should be:

```
${ROOT}
|--PointDA_data
|  |--modelnet
|  |--scannet
|  |--shapenet
```

## Download Pre-trained Model

Download the MAE [pre-trained model](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put the folder under `pretrained/data/`. 

## Train
Run GAST on both source and target
```
python main.py --exp_name 'm2r' --src_dataset modelnet --trgt_dataset scannet --dataroot [your_dataroot] --batch_size 16
python main_spst.py --exp_name 'm2r' --trgt_dataset scannet --dataroot [your_dataroot] --batch_size 16 --lr 5e-5
```

If you want to test with pre-trained CMKT, download it from [here]() and place it at `experiments/'

## Acknowlegment
This repo benefits from [PointCLIP_V2](https://github.com/yangyangyang127/PointCLIP_V2), [MAE](https://github.com/facebookresearch/mae). Thanks for their wonderful works.
