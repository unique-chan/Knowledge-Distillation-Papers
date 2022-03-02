# Deep Mutual Learning
### Unofficial implementation of "Deep Mutual Learning (CVPR 2017)" (PyTorch)
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
- Example: `python train.py -dataroot='./dataroot' -saveroot='./saveroot' -dataset='cifar100' -epoch=5 -ngpu=1 -cohort_size=2 -model='CIFAR_ResNet18'`
- For details, please refer to `parse_args()` in `train.py`.

## Experimental Results
- Here, `IND` and `DML` stand for 'individual training' and 'deep mutual learning', respectively.
- Experiments are performed on `cifar100` with 200 epochs. Other experimental setups are the same as in the code.
- Note that, for fair comparison, same model weights and PyTorch random states are used for both (A) and (B).

| Network | `IND` - Top1 Best Validation Acc (%) | `DML` - Top1 Best Validation Acc(%)|
|---------|:-----------:|:----------:|
|ResNet-34 (A) |   76.00%  | **76.13%** |
|ResNet-34 (B) |   75.37%  | **76.72%** |


## Contribution
🐛 If you find any bugs or have opinions for further improvements, feel free to contact me (yechankim@gm.gist.ac.kr). All contributions are welcome.


## Reference
1. https://github.com/weiaicunzai/pytorch-cifar100
2. https://github.com/alinlab/cs-kd
3. https://github.com/chxy95/Deep-Mutual-Learning