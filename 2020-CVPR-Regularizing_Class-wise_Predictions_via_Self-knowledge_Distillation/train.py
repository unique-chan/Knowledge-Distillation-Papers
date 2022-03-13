from warnings import filterwarnings
import os, argparse, logging

import torch
import torchmetrics
from torch import optim, nn
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter


import models
from dataset import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='2020-CVPR-'
                                                 'Regularizing Class-wise Predictions via Self-knowledge distillation '
                                                 '(github.com/unique-chan)')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('-ckpt', type=str, default='ckpt.pt',
                        help='checkpoint name (e.g. ckpt-epoch10.pt) (default: ckpt.pt)')
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
    parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
    parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio')
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


def get_optimizer():
    return optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)


def adjust_lr():
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class KDLoss(nn.Module):
    def __init__(self, temp):
        super(KDLoss, self).__init__()
        self.temp = temp
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input / self.temp, dim=1)
        q = torch.softmax(target / self.temp, dim=1)
        loss = self.kl_div(log_p, q) * (self.temp ** 2) / input.size(0)
        return loss


def train():
    net.train()
    acc_metric = torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
    loss_metric_1 = torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
    loss_metric_2 = torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
    for batch_idx, (x, y) in enumerate(loader_train):
        if use_cuda:
            x, y = x.cuda(), y.cuda()
        batch_size = x.size(0)
        # processing input batch [◼ | ⬜]
        y_front_half = y[:batch_size // 2]
        y_hat_front_half = net(x[:batch_size//2])
        train_loss = torch.mean(cross_entropy(y_hat_front_half, y_front_half))
        # processing input batch [⬜ | ◼]
        with torch.no_grad():
            y_hat_back_half = net(x[batch_size//2:]).detach()
            cls_loss = kd_loss(y_hat_front_half, y_hat_back_half)
        # loss calculation
        loss = train_loss + args.lamda * cls_loss
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # logging on console
        print('\r' + f'⏩ epoch: {epoch} [{batch_idx+1}/{len(loader_train)}] [train], '
                     f'best_acc_val: {best_acc_val * 100.: .3f}%, '
                     f'batch-acc: {acc_metric(y_hat_front_half, y_front_half) * 100.: .3f}%, '
                     f'batch-loss: {loss_metric_1(train_loss):.4f}',
                     f'batch-cls-loss: {loss_metric_2(cls_loss):.4f}', end='')
    total_train_loss, total_cls_loss = loss_metric_1.compute(), loss_metric_2.compute()
    total_acc = acc_metric.compute()
    # logging on tensorboard
    tb_writer.add_scalar('train-loss', total_train_loss, epoch)
    tb_writer.add_scalar('train-cls_loss', total_cls_loss, epoch)
    tb_writer.add_scalar('train-acc', total_acc, epoch)


def val():
    global best_acc_val
    net.eval()
    acc_metric = torchmetrics.Accuracy(top_k=1).cuda() if use_cuda else torchmetrics.Accuracy(top_k=1)
    loss_metric = torchmetrics.MeanMetric().cuda() if use_cuda else torchmetrics.MeanMetric()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader_val):
            if use_cuda:
                x, y = x.cuda(), y.cuda()
            y_hat = net(x)
            loss = torch.mean(cross_entropy(y_hat, y))
            # logging on console
            print('\r' + f'⏩ epoch: {epoch} [{batch_idx+1}/{len(loader_val)}] [valid], '
                         f'best_acc_val: {best_acc_val * 100.: .3f}%, '
                         f'batch-acc: {acc_metric(y_hat, y) * 100.: .3f}%, '
                         f'batch-loss: {loss_metric(loss):.4f}', end='')
        total_loss = loss_metric.compute()
        total_acc = acc_metric.compute()
        # logging on tensorboard
        tb_writer.add_scalar('val-loss', total_loss, epoch)
        tb_writer.add_scalar('val-acc', total_acc, epoch)
        # best model update
        if total_acc > best_acc_val:
            best_acc_val = total_acc
            record_checkpoint(total_acc, epoch)
    # return total_loss, total_acc


def record_checkpoint(total_acc, epoch, msg=''):
    print(end='\r' + '⭕')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': total_acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(log_dir, f'ckpt{msg}-epoch{epoch}.pt'))


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
loader_train, loader_val = load_dataset(args.dataset, args.dataroot, 'pair', batch_size=args.batch_size)
num_classes = loader_train.dataset.num_classes

# criterion
cross_entropy = torch.nn.CrossEntropyLoss()
kd_loss = KDLoss(args.temp)

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
    checkpoint = torch.load(os.path.join(log_dir, args.ckpt))
    net.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    best_acc_val = checkpoint['acc']
    epoch_start = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
# save init states (for replication)
else:
    record_checkpoint(total_acc=0, epoch=-1, msg='_init')

# train/val
for epoch in range(epoch_start, epoch_end):
    train()
    val()
    adjust_lr()

print('\n' + f'⏹ best_acc_val: {best_acc_val * 100:.3f}%')
logger = logging.getLogger(f'best_acc_val-{log_dir}')
logger.info(f'{best_acc_val * 100:.3f}% \n')
#######################################################################################################################
