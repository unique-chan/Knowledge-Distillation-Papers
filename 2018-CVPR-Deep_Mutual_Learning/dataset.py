import os

from torchvision import transforms, datasets


def load_dataset(name, root, **kwargs):
    if name in ['imagenet', 'tinyimagenet', 'CUB200', 'STANFORD120', 'MIT67']:
        if name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_valid = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        elif name == 'tinyimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_valid = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        else:  # CUB200 / STANFORD120 / MIT67
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_valid = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        dataset_train = datasets.ImageFolder(root=os.path.join(root, name, "train"), transform=transform_train)
        dataset_val = datasets.ImageFolder(root=os.path.join(root, name, "val"), transform=transform_valid)

    elif name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_valid = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        CIFAR = datasets.CIFAR10 if name == 'cifar10' else datasets.CIFAR100
        dataset_train = CIFAR(root, train=True,  download=True, transform=transform_train)
        dataset_val = CIFAR(root, train=False, download=True, transform=transform_valid)

    else:
        raise Exception('❌ Unknown dataset: {}'.format(name))

    return dataset_train, dataset_val