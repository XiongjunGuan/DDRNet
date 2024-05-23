'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 16:03:05
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 16:19:24

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''

from glob import glob

import matplotlib.pylab as plt
import numpy as np


def draw_orientation(ax,
                     ori,
                     mask=None,
                     factor=8,
                     stride=32,
                     color="lime",
                     linewidth=1.5):
    """draw orientation filed

    Parameters:
        [None]
    Returns:
        [None]
    """
    ori = ori * np.pi / 180
    for ii in range(stride // factor // 2, ori.shape[0], stride // factor):
        for jj in range(stride // factor // 2, ori.shape[1], stride // factor):
            if mask is not None and mask[ii, jj] == 0:
                continue
            x, y, o, r = jj, ii, ori[ii, jj], stride * 0.8
            ax.plot(
                [
                    x * factor - 0.5 * r * np.cos(o),
                    x * factor + 0.5 * r * np.cos(o)
                ],
                [
                    y * factor - 0.5 * r * np.sin(o),
                    y * factor + 0.5 * r * np.sin(o)
                ],
                "-",
                color=color,
                linewidth=linewidth,
            )


def draw_img_with_orientation(img,
                              ori,
                              save_path,
                              factor=8,
                              stride=16,
                              cmap="gray",
                              vmin=None,
                              vmax=None,
                              mask=None,
                              color="lime",
                              dpi=100):
    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    draw_orientation(ax,
                     ori,
                     mask=mask,
                     factor=factor,
                     stride=stride,
                     color=color,
                     linewidth=dpi / 50)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    fig.tight_layout()
    fig.set_size_inches(img.shape[1] * 1.0 / dpi, img.shape[0] * 1.0 / dpi)
    fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    prefix = ""
