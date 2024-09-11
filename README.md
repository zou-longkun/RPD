#Boosting Cross-Domain Point Classification via Distilling Relational Priors from 2D Transformers

## Introduction


This repo is a PyTorch implementation for **Boosting Cross-Domain Point Classification via Distilling Relational Priors from 2D Transformers**
[Paper](https://arxiv.org/abs/2407.18534)
## Requirements
The code has been tested with

- Python >= 3.7
- PyTorch == 1.8.0+cu111
- torch-scatter == 2.0.7
- torchsampler == 0.1.2
- torchvision == 0.9.0+cu111

Some dependent packages:

- [PyTorchEMD](https://github.com/daerduoCarey/PyTorchEMD)

Please refer to [issue #6](https://github.com/daerduoCarey/PyTorchEMD/issues/6) before installing.

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

## Download MAE Pre-trained Vit Model

Download the MAE [Pre-trained Vit Model](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and put the folder under `pretrained/`. 

## Train
Training on both source and target
```
python main.py --src_dataset modelnet --trgt_dataset scannet --dataroot [your_dataroot] --batch_size 16
python main_spst.py --exp_name 'spst' --trgt_dataset scannet --dataroot [your_dataroot] --batch_size 16 --lr 5e-5
```

If you want to test with pre-trained model, download it from [here](https://drive.google.com/file/d/1xV3XObyOOwHJ_dwiP4u-9ZSQMbiUY9lc/view?usp=sharing) and place it at `experiments/`

## Citation
Please cite this paper if you want to use it in your work,
```
@article{zou2024boosting,
  title={Boosting Cross-Domain Point Classification via Distilling Relational Priors from 2D Transformers},
  author={Zou, Longkun and Zhu, Wanru and Chen, Ke and Guo, Lihua and Guo, Kailing and Jia, Kui and Wang, Yaowei},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
```

## Acknowlegment
This repo benefits from [PointCLIP_V2](https://github.com/yangyangyang127/PointCLIP_V2), [MAE](https://github.com/facebookresearch/mae), [GAST](https://github.com/zou-longkun/GAST). Thanks for their wonderful works.
