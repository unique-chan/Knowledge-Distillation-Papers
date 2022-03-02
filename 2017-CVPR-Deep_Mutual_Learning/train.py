from warnings import filterwarnings
import os, argparse, logging

import torch
import torchmetrics
from torch import optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


import models
from dataset import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Code (github.com/unique-chan)')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('-model', default='', type=str, help='model name')
    parser.add_argument('-name', default='0', type=str, help='name of experiment')
    parser.add_argument('-batch_size', default=128, type=int, help='batch size')
    parser.add_argument('-epoch', default=5, type=int, help='epochs to train')
    parser.add_argument('-decay', default=1e-4, type=float, help='weight decay (L2)')
    parser.add_argument('-ngpu', default=1, type=int, help='number of gpus (ex) 0: CPU, >= 1: # GPUs')
    parser.add_argument('-sgpu', default=0, type=int, help='gpu index (start) (ex) -ngpu=2 -sgpu=0 -> [0, 1] gpus used')
    parser.add_argument('-dataset', default='cifar100', type=str,
                        help='name of dataset (ex) cifar10 | cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67')
    parser.add_argument('-dataroot', default='./dataroot', type=str, help='directory for dataset')
    parser.add_argument('-saveroot', default='./saveroot', type=str, help='directory to store')
    parser.add_argument('-cohort_size', default=2, type=int, help='cohort_size (>= 2)')
    return parser.parse_args()


def set_logging_defaults():
    if os.path.isdir(log_dir):
        print(f'❌ Warning: {log_dir} already exists.')
        # raise Exception(f'❌  {log_dir} already exists.')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(format='[%(asctime)s] [%(name)s] %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.saveroot, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])
    logger = logging.getLogger(f'main-{log_dir}')
    logger.info(' '.join(os.sys.argv))
    logger.info(str(args))


def get_optimizers():
    return [optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay) for net in nets]


def adjust_lr():
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for i in range(len(optimizers)):
        optimizer = optimizers[i]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def train():
    cross_entropy = torch.nn.CrossEntropyLoss()
    kl_divergence = torch.nn.KLDivLoss(reduction='batchmean')
    acc_metrics = [torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
                   for _ in range(args.cohort_size)]
    loss_metrics = [torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
                    for _ in range(args.cohort_size)]
    kl_loss_metrics = [torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
                       for _ in range(args.cohort_size)]
    # switch to the 'train' mode.
    for i in range(args.cohort_size):
        nets[i].train()
    # train code
    for batch_idx, (x, y) in enumerate(loader_train):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        # for each peer net 'i',
        for i in range(args.cohort_size):
            # compute y_hat for all peer nets (per batch)
            y_hats = [nets[j](x) for j in range(args.cohort_size)]
            ce_loss = torch.mean(cross_entropy(y_hats[i], y))
            # kl loss calc,
            kl_loss = 0
            for j in range(args.cohort_size):
                if i != j:
                    kl_loss += kl_divergence(F.log_softmax(y_hats[i], dim=1),
                                             F.softmax(y_hats[j], dim=1))
            loss = ce_loss + kl_loss / (args.cohort_size - 1)
            # update the model 'i'
            optimizers[i].zero_grad()
            loss.backward(retain_graph=(True if i < args.cohort_size - 1 else False))  ## 🤔 retain_graph?? ->
                                                                                       ## to solve "RuntimeError:
                                                                                       ## Trying to backward
                                                                                       ## through the graph
                                                                                       ## a second time"
            optimizers[i].step()
            # logging on console
            print('\r' + f'⏩ model_{i} | epoch: {epoch} [{batch_idx + 1}/{len(loader_train)}] [train], '
                         f'best_acc_val: {best_acc_vals[i] * 100.: .3f}%, '
                         f'batch-acc: {acc_metrics[i](y_hats[i], y) * 100.: .3f}%, '
                         f'batch-loss: {loss_metrics[i](loss):.4f}', end='')
            kl_loss_metrics[i](kl_loss / (args.cohort_size - 1))
    # compute final losses / accs per each peer net
    total_losses, total_accs = [], []
    for i in range(args.cohort_size):
        total_loss = loss_metrics[i].compute()
        kl_loss_only = kl_loss_metrics[i].compute()
        total_acc = acc_metrics[i].compute()
        # logging on tensorboard
        tb_writer.add_scalar(f'model_{i}_train-loss', total_loss, epoch)
        tb_writer.add_scalar(f'model_{i}_train-kl_loss_only', kl_loss_only, epoch)
        tb_writer.add_scalar(f'model_{i}_train-acc', total_acc, epoch)
    return total_losses, total_accs


def val():
    cross_entropy = torch.nn.CrossEntropyLoss()
    acc_metrics = [torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
                   for _ in range(args.cohort_size)]
    loss_metrics = [torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
                    for _ in range(args.cohort_size)]
    with torch.no_grad():
        # switch to the 'eval' mode
        for i in range(args.cohort_size):
            nets[i].eval()
        # validation code
        for batch_idx, (x, y) in enumerate(loader_val):
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            y_hats, losses = [], []  ## per batch
            # compute y_hat, cross_entropy_loss, acc for all peer nets
            for i in range(args.cohort_size):
                y_hat = nets[i](x)
                ce_loss = torch.mean(cross_entropy(y_hat, y))
                y_hats.append(y_hat)
                losses.append(ce_loss)
                print('\r' + f'⏩ model_{i} | epoch: {epoch} [{batch_idx+1}/{len(loader_val)}] [valid], '
                             f'best_acc_val: {best_acc_vals[i] * 100.: .3f}%, '
                             f'batch-acc: {acc_metrics[i](y_hats[i], y) * 100.: .3f}%, '
                             f'batch-loss: {loss_metrics[i](losses[i]):.4f}', end='')
        # compute final losses / accs per each peer net
        total_losses, total_accs = [], []
        for i in range(args.cohort_size):
            total_loss = loss_metrics[i].compute()
            total_acc = acc_metrics[i].compute()
            # logging on tensorboard
            tb_writer.add_scalar(f'model_{i}_val-loss', total_loss, epoch)
            tb_writer.add_scalar(f'model_{i}_val-acc', total_acc, epoch)
            # best model update
            if total_acc > best_acc_vals[i]:
                best_acc_vals[i] = total_acc
                record_checkpoint(total_acc, epoch, i)
        return total_losses, total_accs


def record_checkpoint(total_acc, epoch, i, msg=''):
    print(end='\r' + '⭕')
    state = {
        'net': nets[i].state_dict(),
        'optimizer': optimizers[i].state_dict(),
        'acc': total_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(log_dir, f'model_{i}_ckpt{msg}.pt'))


########################################################################################################################
filterwarnings('ignore')

args = parse_args()
use_cuda = torch.cuda.is_available() if args.ngpu >= 1 else False

# experimental record
best_acc_vals = [0] * args.cohort_size
epoch_start, epoch_end = 0, args.epoch

# logger
log_dir = os.path.join(args.saveroot, args.dataset, args.model, f'cohort_size_{args.cohort_size}', f'run_{args.name}')
set_logging_defaults()
tb_writer = SummaryWriter(log_dir)  # (ex) tensorboard --logdir=[saveroot]

# data
print(f'⏺ Preparing dataset: {args.dataset}')
dataset_train, dataset_val = load_dataset(args.dataset, args.dataroot)
loader_train = data.DataLoader(dataset_train,
                               batch_size=args.batch_size, shuffle=True)
loader_val = data.DataLoader(dataset_val,
                             batch_size=args.batch_size, shuffle=False)
num_classes = len(dataset_train.classes)

# model
print(f'⏺ Building {args.cohort_size} peer models. Model name: {args.model}')
nets = [models.load_model(args.model, num_classes) for _ in range(args.cohort_size)]

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    for i in range(args.cohort_size):
        nets[i] = nets[i].cuda()
    if args.ngpu > 1:
        for i in range(args.cohort_size):
            nets[i] = torch.nn.DataParallel(nets[i], device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
optimizers = get_optimizers()
# lr_scheduler = pass

# resume
if args.resume:
    print(f'⏸ Resuming from checkpoint.')
    for i in range(args.cohort_size):
        checkpoint = torch.load(os.path.join(log_dir, f'model_{i}_ckpt.pt'))
        nets[i].load_state_dict(checkpoint['net'])
        optimizers[i].load_state_dict(checkpoint['optimizer'])
        best_acc_vals[i] = checkpoint['acc']
        epoch_start = checkpoint['epoch'] + 1
        torch.set_rng_state(checkpoint['rng_state'])
# save init states (for replication)
else:
    for i in range(args.cohort_size):
        record_checkpoint(total_acc=0, epoch=0, i=i, msg='_init')

# train/val
for epoch in range(epoch_start, epoch_end):
    train()
    val()
    adjust_lr()

# logging on console
print()
for i in range(args.cohort_size):
    print(f'⏹ model_{i}_best_acc_val: {best_acc_vals[i] * 100:.3f}%')

# logging on logs.txt
for i in range(args.cohort_size):
    logger = logging.getLogger(f'model_{i}_best_acc_val-{log_dir}')
    msg = f'model_{i}: {best_acc_vals[i] * 100:.3f}%'
    if i == args.cohort_size - 1:
        msg += '\n'
    logger.info(msg)
#######################################################################################################################
