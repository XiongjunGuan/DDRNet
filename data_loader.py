'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 15:53:08
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 16:33:59

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import logging
import os.path as osp
from glob import glob
from os import listdir
from os.path import splitext
from pathlib import Path
from random import randint

import cv2
import numpy as np
import scipy.io as scio
import torch
from PIL import Image
from scipy.ndimage import shift, zoom
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import transforms as T

from tools.fp_segmentation import segmentation_coherence


def transform_img(img):
    return (255 - img) / 255


def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes, )
    categorical = np.reshape(categorical, output_shape)

    return categorical


class load_dataset_predict_need_mask(Dataset):

    def __init__(self, img_dir: str, bimg_dir: str):
        self.img_dir = img_dir
        self.bimg_dir = bimg_dir

        fname_lst = glob(osp.join(img_dir, 'f_*.png'))
        self.fname_lst = [
            osp.basename(fname).replace('.png', '').replace('f_', '')
            for fname in fname_lst
        ]

        if not self.fname_lst:
            raise RuntimeError(
                f'No input file found in {img_dir}, make sure you put your data there'
            )
        logging.info(f'Creating dataset with {len(self.fname_lst)} examples')

    def __len__(self):
        return len(self.fname_lst)

    def __getitem__(self, idx):
        fname = self.fname_lst[idx]
        img = cv2.imread(osp.join(self.img_dir, fname + '.png'), 0)
        bimg = cv2.imread(osp.join(self.bimg_dir, 'b' + fname + '.png'), 0)
        mask = segmentation_coherence(img, win_size=16, stride=8)
        mask16 = zoom(mask, 1 / 16, order=0)

        img = np.float32(transform_img(img))[np.newaxis, :, :]
        bimg = np.float32(transform_img(bimg))[np.newaxis, :, :]
        mask16 = np.float32(mask16[np.newaxis, :, :])

        return {'img': img, 'bimg': bimg, 'mask16': mask16, 'fname': fname}


class load_dataset_test(Dataset):

    def __init__(self, data_dir: str, gt_dir: str, need_aug=True):
        self.data_dir = data_dir
        self.gt_dir = gt_dir

        self.need_aug = need_aug

        fname_lst = glob(osp.join(data_dir, 'f_*.png'))
        self.fname_lst = [
            osp.basename(fname).replace('.png', '').replace('f_', '')
            for fname in fname_lst
        ]

        if not self.fname_lst:
            raise RuntimeError(
                f'No input file found in {data_dir}, make sure you put your data there'
            )

    def __len__(self):
        return len(self.fname_lst)

    def __getitem__(self, idx):
        fname = self.fname_lst[idx]
        img = cv2.imread(osp.join(self.data_dir, 't_' + fname + '.png'), 0)
        mask = cv2.imread(osp.join(self.data_dir, 'm_' + fname + '.png'), 0)
        mask = (mask > 0)

        data = scio.loadmat(osp.join(self.gt_dir, fname + '.mat'))
        dx = data['dx_rect16']
        dy = data['dy_rect16']

        mask16 = zoom(mask, zoom=1 / 16, order=0)
        mask16 = mask16 > 0

        if self.need_aug is True:
            shiftx16 = randint(-2, 2)
            shifty16 = randint(-2, 2)

            img = shift(img,
                        shift=[shiftx16 * 16, shifty16 * 16],
                        mode='constant',
                        cval=255)
            dx = shift(dx, shift=[shiftx16, shifty16], mode='constant', cval=0)
            dy = shift(dy, shift=[shiftx16, shifty16], mode='constant', cval=0)
            mask16 = shift(mask16,
                           shift=[shiftx16, shifty16],
                           mode='constant',
                           cval=0)
            mask16 = mask16 > 0

        img = np.float32(transform_img(img))[np.newaxis, :, :]
        mask = mask16 * 1.0
        dx = dx[np.newaxis, :, :]
        dy = dy[np.newaxis, :, :]
        targets = np.float32(np.concatenate((dx, dy), axis=0))
        mask16 = np.float32(mask16[np.newaxis, :, :])

        return {'input': img, 'mask16': mask16, 'targets': targets}


class load_dataset_train(Dataset):

    def __init__(self,
                 data_dir: str,
                 feature_dir: str,
                 gt_dir: str,
                 need_aug=True):
        self.data_dir = data_dir
        self.feature_dir = feature_dir
        self.gt_dir = gt_dir

        self.need_aug = need_aug

        fname_lst = glob(osp.join(data_dir, 'f_*.png'))
        self.fname_lst = [
            osp.basename(fname).replace('.png', '').replace('f_', '')
            for fname in fname_lst
        ]

        if not self.fname_lst:
            raise RuntimeError(
                f'No input file found in {data_dir}, make sure you put your data there'
            )

    def __len__(self):
        return len(self.fname_lst)

    def __getitem__(self, idx):
        fname = self.fname_lst[idx]
        img = cv2.imread(osp.join(self.data_dir, 't_' + fname + '.png'), 0)
        mask = cv2.imread(osp.join(self.data_dir, 'm_' + fname + '.png'), 0)
        mask = (mask > 0)

        data = scio.loadmat(osp.join(self.gt_dir, fname + '.mat'))
        dx = data['dx_rect16']
        dy = data['dy_rect16']

        mask16 = zoom(mask, zoom=1 / 16, order=0)
        mask16 = mask16 > 0

        feature = np.load(osp.join(self.feature_dir, fname + '.npy'),
                          allow_pickle=True).item()
        DIR16 = zoom(feature['DIR'], zoom=1 / 2, mode='constant', cval=0)

        if self.need_aug is True:
            shiftx16 = randint(-4, 4)
            shifty16 = randint(-4, 4)
            img = shift(img,
                        shift=[shiftx16 * 16, shifty16 * 16],
                        mode='constant',
                        cval=255)
            dx = shift(dx, shift=[shiftx16, shifty16], mode='constant', cval=0)
            dy = shift(dy, shift=[shiftx16, shifty16], mode='constant', cval=0)
            mask16 = shift(mask16,
                           shift=[shiftx16, shifty16],
                           mode='constant',
                           cval=0)
            mask16 = mask16 > 0
            DIR16 = shift(DIR16,
                          shift=[shiftx16, shifty16],
                          mode='constant',
                          cval=0)

        img = np.float32(transform_img(img))[np.newaxis, :, :]
        dx = dx[np.newaxis, :, :]
        dy = dy[np.newaxis, :, :]
        targets = np.float32(np.concatenate((dx, dy), axis=0))
        mask16 = np.float32(mask16[np.newaxis, :, :])

        DIR16 = np.clip(np.round(DIR16 + 90), 0, 179)
        DIR16 = to_categorical(DIR16, num_classes=180).transpose(2, 0, 1)

        return {
            'input': img,
            'mask16': mask16,
            'DIR16': DIR16,
            'targets': targets
        }


def get_dataloader_test(img_dir, bimg_dir, batch_size):
    # Create dataset
    try:
        dataset = load_dataset_predict_need_mask(img_dir, bimg_dir)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    logging.info(f'n_test:{len(dataset)}')

    return test_loader


def get_dataloader_train(train_data_dir,
                         train_feature_dir,
                         train_gt_dir,
                         batch_size,
                         val_percent=0.3,
                         shuffle=False,
                         need_aug=True):
    # Create dataset
    try:
        dataset = load_dataset_train(train_data_dir,
                                     train_feature_dir,
                                     train_gt_dir,
                                     need_aug=need_aug)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    # Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, valid_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # Create data loaders
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    logging.info(f'n_train:{n_train}, n_val:{n_val}')

    return train_loader, valid_loader


def get_dataloader_valid(valid_data_dir, valid_gt_dir, batch_size):
    # Create dataset
    try:
        dataset = load_dataset_test(valid_data_dir,
                                    valid_gt_dir,
                                    need_aug=False)
    except Exception as e:
        logging.error("Error in DataLoader: ", repr(e))
        return

    n_val = int(len(dataset))

    # Create data loaders
    valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    logging.info(f'n_valid:{n_val}')

    return valid_loader
