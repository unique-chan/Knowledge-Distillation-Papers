# Deep Mutual Learning
### Unofficial implementation of "Deep Mutual Learning (CVPR 2017)" (PyTorch)
[Yechan Kim](https://github.com/unique-chan)

## This repository contains:
- Python3 / Pytorch code for multi-class image classification

## Prerequisites
- See `requirements.txt` for details.
~~~ME
torch
torchvision
torchmetrics
tensorboard
~~~


## How to run
- Example: `python train.py -dataroot='./dataroot' -saveroot='./saveroot' -dataset='cifar100' -epoch=5 -ngpu=1 -model='CIFAR_ResNet18'`
- For details, please refer to `parse_args()` in `train.py`.

## Contribution
🐛 If you find any bugs or have opinions for further improvements, feel free to contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.


## Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/alinlab/cs-kd
