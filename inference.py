'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 15:53:08
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 19:37:39

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import logging
import os
import os.path as osp
from glob import glob

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm

from data_loader import transform_img
from models.DDRNet_DIR import DDRNet_DIR
from tools.fp_distortion import apply_distortion
from tools.fp_draw import draw_img_with_orientation

if __name__ == '__main__':
    # set model path
    model_path = "./ckpts/DDRNet_DIR.pth"

    # set test dirs
    test_img_dir = "./examples/data_img/"
    test_timg_dir = "./examples/data_timg/"
    test_mask_dir = "./examples/data_mask/"
    save_img_dir = "./examples/result/"

    # specify the name of test files
    fname_lst = ['1', '2']

    # set sevice
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    if not osp.exists(save_img_dir):
        os.makedirs(save_img_dir)

    net = DDRNet_DIR()

    net.to(device=device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    logging.info(f'Using device {device}')

    with torch.no_grad():
        for fname in tqdm(fname_lst):
            # load data

            # rectification requires the original image
            img = cv2.imread(osp.join(test_img_dir, fname + '.png'), 0)

            # the model requires a skeleton image and a 1/16 mask as inputs
            timg = cv2.imread(osp.join(test_timg_dir, 't' + fname + '.png'), 0)
            mask = cv2.imread(osp.join(test_mask_dir, fname + '.png'), 0)

            mask = (mask > 0)
            mask16 = zoom(mask, 1 / 16, order=0)

            timg = torch.from_numpy(
                np.float32(transform_img(timg))[np.newaxis,
                                                np.newaxis, :, :]).to(device)
            mask16 = torch.from_numpy(
                np.float32(mask16[np.newaxis, np.newaxis, :, :])).to(device)

            # pred: (b, 2, h/16, w/16) for dx and dy
            # pred_DIR: (b, 180, h/16, w/16) represents the corresponding interval oritation field
            pred, pred_DIR = net(timg, mask16)

            pred = torch.squeeze(pred).cpu().numpy()
            mask16 = torch.squeeze(mask16).cpu().numpy()

            pred_DIR = torch.squeeze(pred_DIR).cpu().numpy()
            pred_DIR = np.argmax(pred_DIR, axis=0)
            pred_DIR[mask16 == 0] = 0
            pred_DIR = zoom(pred_DIR, zoom=2,
                            order=0) - 90  # [0,180] -> [-90, 90]

            # draw orientation field
            mask8 = zoom(mask, 1 / 8, order=0)
            mask8 = (mask8 > 0)
            draw_img_with_orientation(np.ones_like(img) * 255,
                                      -pred_DIR,
                                      osp.join(save_img_dir,
                                               fname + '_ori.png'),
                                      factor=8,
                                      stride=16,
                                      mask=mask8,
                                      vmin=0,
                                      vmax=255)
            # rectify fingerprint
            dx16 = pred[0, :, :]
            dy16 = pred[1, :, :]

            dx = zoom(dx16, 16, order=3)
            dy = zoom(dy16, 16, order=3)

            border_mask = np.ones_like(img)
            border_mask[2:-2, 2:-2] = 0
            img[border_mask == 1] = 255
            img_rect = apply_distortion(img, dx, dy)

            # draw original image
            cv2.imwrite(osp.join(save_img_dir, fname + '_raw.png'),
                        np.uint8(img))

            # draw rectified image
            cv2.imwrite(osp.join(save_img_dir, fname + '_rect.png'),
                        np.uint8(img_rect))
