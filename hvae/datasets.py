# ---------------------------------------------------------------
# This file contains a modifed version of 
# code taken from https://github.com/NVlabs/NVAE
#-----------------------------------------------------------------


import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import os
import utils
from torch._utils import _accumulate
from lmdb_datasets import LMDBDataset, ImageArrayDataset
# from thirdparty.lsun import LSUN


class Binarize(object):
    """ This class introduces a binarization transformation
    """
    def __call__(self, pic):
        return torch.Tensor(pic.size()).bernoulli_(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class CropCelebA64(object):
    """ This class applies cropping for CelebA64. This is a simplified implementation of:
    https://github.com/andersbll/autoencoding_beyond_pixels/blob/master/dataset/celeba.py
    """
    def __call__(self, pic):
        new_pic = pic.crop((15, 40, 178 - 15, 218 - 30))
        return new_pic

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_loaders(args, eval=None):
    """Get data loaders for required dataset."""
    return get_loaders_eval(args.dataset, args)


def get_loaders_eval(dataset, args):
    """Get train and valid loaders for cifar10/tiny imagenet."""

    if dataset == 'cifar10':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_cifar10(args)
        # train_data = dset.CIFAR10(  
        #     root=args.data, train=True, download=True, transform=train_transform)
        # valid_data = dset.CIFAR10(
        #     root=args.data, train=False, download=True, transform=valid_transform)
        #! Using saved feats for now. Switch back to lines above if needed.
        train_data = ImageArrayDataset(args.data, args.dataset, train=True)
        valid_data = ImageArrayDataset(args.data, args.dataset, train=False)
        
    elif dataset == 'mnist':
        num_classes = 10
        train_transform, valid_transform = _data_transforms_mnist(args)
        train_data = dset.MNIST(
            root=args.data, train=True, download=True, transform=train_transform)
        valid_data = dset.MNIST(
            root=args.data, train=False, download=True, transform=valid_transform)
    elif dataset.startswith('celeba'):
        if dataset == 'celeba_64':
            resize = 64
            num_classes = 40
            train_transform, valid_transform = _data_transforms_celeba64(resize)
            #! Switch back to LMDBDataset if raw image data being used
            # train_data = LMDBDataset(root=args.data, name='celeba64', train=True, transform=train_transform, is_encoded=True)
            # valid_data = LMDBDataset(root=args.data, name='celeba64', train=False, transform=valid_transform, is_encoded=True)
            train_data = ImageArrayDataset(args.data, args.dataset, train=True)
            valid_data = ImageArrayDataset(args.data, args.dataset, train=False)
        elif dataset in {'celeba_256'}:
            num_classes = 1
            resize = int(dataset.split('_')[1])
            train_transform, valid_transform = _data_transforms_generic(resize)
            train_data = LMDBDataset(root=args.data, name='celeba', train=True, transform=train_transform)
            valid_data = LMDBDataset(root=args.data, name='celeba', train=False, transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('lsun'):
        if dataset.startswith('lsun_bedroom'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['bedroom_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['bedroom_val'], transform=valid_transform)
        elif dataset.startswith('lsun_church'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['church_outdoor_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['church_outdoor_val'], transform=valid_transform)
        elif dataset.startswith('lsun_tower'):
            resize = int(dataset.split('_')[-1])
            num_classes = 1
            train_transform, valid_transform = _data_transforms_lsun(resize)
            train_data = LSUN(root=args.data, classes=['tower_train'], transform=train_transform)
            valid_data = LSUN(root=args.data, classes=['tower_val'], transform=valid_transform)
        else:
            raise NotImplementedError
    elif dataset.startswith('imagenet'):
        num_classes = 1
        resize = int(dataset.split('_')[1])
        assert args.data.replace('/', '')[-3:] == dataset.replace('/', '')[-3:], 'the size should match'
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='imagenet-oord', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='imagenet-oord', train=False, transform=valid_transform)
    elif dataset.startswith('ffhq'):
        num_classes = 1
        resize = 256
        train_transform, valid_transform = _data_transforms_generic(resize)
        train_data = LMDBDataset(root=args.data, name='ffhq', train=True, transform=train_transform)
        valid_data = LMDBDataset(root=args.data, name='ffhq', train=False, transform=valid_transform)
    else:
        raise NotImplementedError

    train_sampler, valid_sampler = None, None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler, pin_memory=True, num_workers=4, drop_last=True) #!

    valid_queue = torch.utils.data.DataLoader(
        # valid_data, batch_size=args.batch_size,
        valid_data, batch_size=20,
        shuffle = False,

        # shuffle=(valid_sampler is None),
        sampler=valid_sampler, pin_memory=True, num_workers=1, drop_last=False)

    return train_queue, valid_queue, num_classes


def _data_transforms_cifar10(args):
    """Get data transforms for cifar10."""
    # mean = [0.49139968, 0.48215827, 0.44653124]
    # std = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

    return train_transform, valid_transform


def _data_transforms_mnist(args):
    """Get data transforms for cifar10."""
    train_transform = transforms.Compose([
        transforms.Pad(padding=2),
        # transforms.RandomCrop(32, padding=1),
        transforms.ToTensor(),
        Binarize(),
    ])

    valid_transform = transforms.Compose([
        transforms.Pad(padding=2),
        transforms.ToTensor(),
        Binarize(),
    ])

    return train_transform, valid_transform


def _data_transforms_generic(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_celeba64(size):
    train_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        CropCelebA64(),
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


def _data_transforms_lsun(size):
    train_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])

    return train_transform, valid_transform


if __name__ == '__main__':
    import argparse
    import matplotlib
    from time import time

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser('dataset examiner')

    parser.add_argument('--dataset', type=str, default='lsun_tower_128',
                        choices=['cifar10', 'mnist', 'celeba_32', 'celeba_64', 'celeba_256', 'imagenet_32', 'ffhq',
                                 'lsun_bedroom_128', 'lsun_church_128', 'lsun_tower_128'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/data1/datasets/LSUN/',
                        help='location of the data corpus')
    parser.add_argument('--train_portion', type=float, default=0.9,
                        help='portion of training data')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size')
    args = parser.parse_args()
    args.distributed = False
    train_queue, valid_queue, num_classes = get_loaders(args)

    if False:
        for i in range(1):
            s = time()
            count = 0
            for b in iter(train_queue):
                # print(b[0].size())
                count += 1
            print(count)
            e = time()
            print(i, e - s)

    batch = next(iter(train_queue))
    print(batch[0][0, :, :, :].squeeze().data.numpy())
    if args.dataset == 'mnist':
        plt.imshow(batch[0][0, 0, :, :].squeeze().data.numpy())
    else:
        plt.imshow(batch[0][0, :, :, :].squeeze().data.numpy().transpose([1, 2, 0]))
    plt.show()
