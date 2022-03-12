# Regularizing Class-wise Predictions via Self-knowledge Distillation
### Unofficial implementation of "Regularizing Class-wise Predictions via Self-knowledge Distillation (CVPR 2020)" (PyTorch)
[Yechan Kim](https://github.com/unique-chan)

## This repository contains:
- Python3 / Pytorch code for multi-class image classification

## Prerequisites
- See `requirements.txt` for details.
~~~ME
torch == 1.7.1
torchvision == 0.8.2
torchmetrics == 0.7.2
tensorboard == 2.8.0
~~~


## How to run
- Example: `python train.py -dataroot='./dataroot' -saveroot='./saveroot' -dataset='cifar100' -epoch=200 --temp=4 --lamda=1 -ngpu=1 -model='CIFAR_ResNet34' -name=0`
- For details, please refer to `parse_args()` in `train.py`.


## Contribution
🐛 If you find any bugs or have opinions for further improvements, feel free to contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.


## Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/alinlab/cs-kd
