'''
Description: 
Author: Xiongjun Guan
Date: 2024-05-22 16:14:04
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2024-05-22 19:40:33

Copyright (C) 2024 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np
import scipy
import torch.nn.functional as F
from scipy.interpolate import griddata
from scipy.ndimage import zoom


def tps_module_numpy(src_cpts, tar_cpts, Lambda=0):
    assert tar_cpts.ndim == 2
    assert tar_cpts.shape[1] == 2
    N = src_cpts.shape[0]
    src_cpts = src_cpts.astype(np.float32)
    tar_cpts = tar_cpts.astype(np.float32)

    # create padded kernel matrix
    src_cc_partial_repr = compute_partial_repr(src_cpts,
                                               src_cpts) + Lambda * np.eye(N)
    forward_kernel = np.concatenate(
        (
            np.concatenate(
                (src_cc_partial_repr, np.ones([N, 1]), src_cpts), axis=1),
            np.concatenate((np.ones([1, N]), np.zeros([1, 3])), axis=1),
            np.concatenate((src_cpts.T, np.zeros([2, 3])), axis=1),
        ),
        axis=0,
    )
    # compute mapping matrix
    Y = np.concatenate([tar_cpts, np.zeros([3, 2])], axis=0)  # (M+3,2)
    mapping_matrix = scipy.linalg.solve(forward_kernel, Y)
    return mapping_matrix


def compute_partial_repr(input_points, control_points):
    pairwise_diff = input_points[:, None] - control_points[None]
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[..., 0] + pairwise_diff_square[..., 1]
    # fix numerical error for 0 * log(0), substitute all nan with 0
    repr_matrix = 0.5 * pairwise_dist * np.log(pairwise_dist.clip(1e-3, None))
    mask = (repr_matrix != repr_matrix) | np.isclose(pairwise_dist, 0)
    repr_matrix[mask] = 0
    return repr_matrix


def tps_apply_transform(src_pts, src_cpts, mapping_matrix):
    """
    Parameters:
        src_pts: points to be transformed
        src_cpts: control points
    Returns:
        [None]
    """
    assert src_pts.ndim == 2
    src_pc_partial_repr = compute_partial_repr(src_pts, src_cpts)
    N = src_pts.shape[0]
    src_pts_repr = np.concatenate(
        [src_pc_partial_repr, np.ones([N, 1]), src_pts], axis=1)
    tar_pts = np.matmul(src_pts_repr, mapping_matrix)
    return tar_pts


def apply_distortion_torch(img,
                           coordinate,
                           image_height,
                           image_width,
                           mode='bilinear'):
    """_summary_

    Args:
        img (_type_): _description_
        coordinate (_type_): dst coordinate. from -1 to 1 .
        image_height (_type_): _description_
        image_width (_type_): _description_
        mode (str, optional): _description_. Defaults to 'bilinear'.

    Returns:
        _type_: _description_
    """
    batch_size = img.size(0)
    coordinate = coordinate.view(batch_size, image_height, image_width, 2)
    img = F.grid_sample(img, coordinate, align_corners=False, mode=mode)
    return img


def apply_distortion(img, dx, dy, need_rect=False, method="linear"):
    """distorted image according to the displacement
    I'(x+dx, y+dy) = I(x,y)

    Args:
        img (_type_): M x N (gray style)
        dx (_type_): M x N 
        dy (_type_): M x N 
        need_rect (bool, optional): give the dual displacement.like DX'(x+dx, y+dy) = -DX(x,y)

    Returns:
        _type_: _description_
    """
    img_shape = img.shape
    x, y = np.meshgrid(np.arange(0, img_shape[0]), np.arange(0, img_shape[1]))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    img_dis = griddata(
        np.hstack((x + dx.reshape((-1, 1)), y + dy.reshape((-1, 1)))),
        img.reshape((-1, 1)),
        np.hstack((x, y)),
        method=method,
        fill_value=255,
    ).reshape(img_shape)

    img_dis = img_dis.clip(0, 255)
    if need_rect is True:
        dx_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)), y + dy.reshape((-1, 1)))),
            (-dx).reshape((-1, 1)),
            np.hstack((x, y)),
            method=method,
        ).reshape(img_shape)

        dy_rect = griddata(
            np.hstack((x + dx.reshape((-1, 1)), y + dy.reshape((-1, 1)))),
            (-dy).reshape((-1, 1)),
            np.hstack((x, y)),
            method=method,
        ).reshape(img_shape)

        return img_dis, dx_rect, dy_rect

    else:
        return img_dis


def transform_rect_to_dis(img_shape, dx_rect, dy_rect):
    """give the dual displacement. Noted that distortion should be the true size.

    Args:
        img_shape (_type_): (M, N, ...)
        dx_rect (_type_): M x N
        dy_rect (_type_): M x N

    Returns:
        _type_: _description_
    """
    x, y = np.meshgrid(np.arange(0, img_shape[0]), np.arange(0, img_shape[1]))
    x = x.reshape((-1, 1))
    y = y.reshape((-1, 1))

    dx = griddata(
        np.hstack((x + dx_rect.reshape((-1, 1)), y + dy_rect.reshape(
            (-1, 1)))),
        (-dx_rect).reshape((-1, 1)),
        np.hstack((x, y)),
        method="nearest",
    ).reshape(img_shape)

    dy = griddata(
        np.hstack((x + dx_rect.reshape((-1, 1)), y + dy_rect.reshape(
            (-1, 1)))),
        (-dy_rect).reshape((-1, 1)),
        np.hstack((x, y)),
        method="nearest",
    ).reshape(img_shape)

    return dx, dy


def generate_outside_distortion_by_common_area(dx, dy, common_mask):
    h, w = common_mask.shape
    x, y = np.meshgrid(np.arange(0, w), np.arange(0, h))

    zoom_param = 1 / 8
    dx_resize = zoom(dx, zoom=zoom_param, order=1)
    dy_resize = zoom(dy, zoom=zoom_param, order=1)
    mask_resize = zoom(common_mask, zoom=zoom_param, order=1)
    mask_resize = (mask_resize > 0.5)
    x_resize = zoom(x, zoom=zoom_param, order=1)
    y_resize = zoom(y, zoom=zoom_param, order=1)

    xs = x_resize[mask_resize > 0]
    ys = y_resize[mask_resize > 0]
    dxs = dx_resize[mask_resize > 0]
    dys = dy_resize[mask_resize > 0]

    src_cpts = np.float32(np.vstack((xs, ys)).T)
    src_pts = np.float32(
        np.vstack((x_resize.reshape((-1, )), y_resize.reshape((-1, )))).T)
    tar_cpts = np.float32(np.vstack(((xs + dxs, ys + dys))).T)

    mapping_matrix = tps_module_numpy(src_cpts, tar_cpts, 5)
    tar_pts = tps_apply_transform(src_pts, src_cpts, mapping_matrix)

    fx1, fx2 = x.shape[0] / x_resize.shape[0], x.shape[1] / x_resize.shape[1]
    fy1, fy2 = y.shape[0] / y_resize.shape[0], y.shape[1] / y_resize.shape[1]
    dx = zoom(tar_pts[:, 0].reshape(x_resize.shape), zoom=(fx1, fx2),
              order=1) - x
    dy = zoom(tar_pts[:, 1].reshape(y_resize.shape), zoom=(fy1, fy2),
              order=1) - y

    # dx = tar_pts[:,0].reshape(x.shape)-x
    # dy = tar_pts[:,1].reshape(y.shape)-y
    return dx, dy
