# ---------------------------------------------------------------
# This file contains a modifed version of 
# code taken from https://github.com/NVlabs/NVAE
#-----------------------------------------------------------------

import torch.utils.data as data
import numpy as np
import lmdb
import os
import io
from PIL import Image


def num_samples(dataset, train):
    if dataset == 'celeba':
        return 27000 if train else 3000
    elif dataset == 'celeba64':
        return 162770 if train else 19867
    elif dataset == 'imagenet-oord':
        return 1281147 if train else 50000
    elif dataset == 'ffhq':
        return 63000 if train else 7000
    else:
        raise NotImplementedError('dataset %s is unknown' % dataset)


class LMDBDataset(data.Dataset):
    def __init__(self, root, name='', train=True, transform=None, is_encoded=False):
        self.train = train
        self.name = name
        self.transform = transform
        if self.train:
            lmdb_path = os.path.join(root, 'train.lmdb')
        else:
            lmdb_path = os.path.join(root, 'validation.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded

    def __getitem__(self, index):
        target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return num_samples(self.name, self.train)


class ImageArrayDataset(data.Dataset):
    ''' Use this dataset for latent feats
    '''
    def __init__(self, root, dataset, train=True):
        self.train = train
        self.root = root
        if self.train:
            fname = f"{dataset}_qzx_samples_train_0.npy"
        else:
            fname = f"{dataset}_qzx_samples_valid_0.npy"
        self.data = np.load(os.path.join(self.root, fname))

    def __getitem__(self, idx):
        target = [0]
        x = self.data[idx]
        return x, target
    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
