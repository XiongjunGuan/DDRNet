'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 16:03:33
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 16:05:58

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import os

import numpy as np
import scipy.ndimage as ndi


def calc_orientation_graident(img, win_size=16, stride=8):
    # img = exposure.equalize_adapthist(img) * 255
    Gx, Gy = np.gradient(img.astype(np.float32))
    Gxx = ndi.gaussian_filter(Gx**2, win_size / 4)
    Gyy = ndi.gaussian_filter(Gy**2, win_size / 4)
    Gxy = ndi.gaussian_filter(-Gx * Gy, win_size / 4)
    coh = np.sqrt(
        (Gxx - Gyy)**2 + 4 * Gxy**2)  # / (Gxx + Gyy).clip(1e-6, None)
    if stride != 1:
        Gxx = ndi.uniform_filter(Gxx, stride)[::stride, ::stride]
        Gyy = ndi.uniform_filter(Gyy, stride)[::stride, ::stride]
        Gxy = ndi.uniform_filter(Gxy, stride)[::stride, ::stride]
        coh = ndi.uniform_filter(coh, stride)[::stride, ::stride]
    ori = np.arctan2(2 * Gxy, Gxx - Gyy) * 90 / np.pi
    return ori, coh
