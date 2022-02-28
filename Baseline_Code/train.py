from warnings import filterwarnings
import os, argparse, logging

import torch
import torchmetrics
from torch import optim
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
    return parser.parse_args()


def set_logging_defaults():
    global log_dir, args
    if os.path.isdir(log_dir):
        raise Exception(f'❌  {log_dir} already exists.')
    os.makedirs(log_dir)
    logging.basicConfig(format='[%(asctime)s] [%(name)s] %(message)s',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(args.saveroot, 'log.txt')),
                                  logging.StreamHandler(os.sys.stdout)])
    logger = logging.getLogger(f'main-{log_dir}')
    logger.info(' '.join(os.sys.argv))
    logger.info(str(args))


def get_optimizer():
    global net, args
    return optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


def adjust_lr():
    global optimizer, epoch, args
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_criterion():
    return torch.nn.CrossEntropyLoss()


def train():
    global net, loader_train, use_cuda, optimizer, epoch, tb_writer
    net.train()
    acc_metric = torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
    loss_metric = torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
    criterion = get_criterion()
    for batch_idx, (inputs, targets) in enumerate(loader_train):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = torch.mean(criterion(outputs, targets))
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logging on console
        print('\r' + f'⏩ epoch: {epoch} [{batch_idx+1}/{len(loader_train)}] [train], '
                     f'best_acc_val: {best_acc_val * 100.: .3f}%, '
                     f'batch-acc: {acc_metric(outputs, targets) * 100.: .3f}%, '
                     f'batch-loss: {loss_metric(loss):.4f}', end='')
    total_loss = loss_metric.compute()
    total_acc = acc_metric.compute()
    # logging on tensorboard
    tb_writer.add_scalar('train-loss', total_loss, epoch)
    tb_writer.add_scalar('train-acc', total_acc, epoch)
    return total_loss, total_acc


def val():
    global best_acc_val, net, loader_val, use_cuda, epoch, tb_writer
    net.eval()
    acc_metric = torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
    loss_metric = torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
    criterion = get_criterion()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader_train):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))
            # logging on console
            print('\r' + f'⏩ epoch: {epoch} [{batch_idx+1}/{len(loader_train)}] [valid], '
                         f'best_acc_val: {best_acc_val * 100.: .3f}%, '
                         f'batch-acc: {acc_metric(outputs, targets) * 100.: .3f}%, '
                         f'batch-loss: {loss_metric(loss):.4f}', end='')
        total_loss = loss_metric.compute()
        total_acc = acc_metric.compute()
        # logging on tensorboard
        tb_writer.add_scalar('val-loss', total_loss, epoch)
        tb_writer.add_scalar('val-acc', total_acc, epoch)
        # best model update
        if total_acc > best_acc_val:
            best_acc_val = total_acc
            checkpoint(total_acc, epoch)
    return total_loss, total_acc


def checkpoint(total_acc, epoch):
    global log_dir
    print(end='\r' + '⭕')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': total_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(log_dir, 'ckpt.pt'))


########################################################################################################################
filterwarnings('ignore')

args = parse_args()
use_cuda = torch.cuda.is_available() if args.ngpu >= 1 else False

# experimental record
best_acc_val = .0
epoch_start, epoch_end = 0, args.epoch

# logger
log_dir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
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
print(f'⏺ Building model: {args.model}')
net = models.load_model(args.model, num_classes)
if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))
optimizer = get_optimizer()
# lr_scheduler = pass

# resume
if args.resume:
    print(f'⏸ Resuming from checkpoint.')
    checkpoint = torch.load(os.path.join(log_dir, 'ckpt.pt'))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc_val = checkpoint['acc']
    epoch_start = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

# train/val
for epoch in range(epoch_start, epoch_end):
    loss_train, acc_train = train()
    loss_val, acc_val = val()
    adjust_lr()

print('\n' + f'⏹ best_acc_val: {best_acc_val * 100:.3f}%')
logger = logging.getLogger(f'best_acc_val-{log_dir}')
logger.info(f'{best_acc_val * 100:.3f}% \n')
#######################################################################################################################
