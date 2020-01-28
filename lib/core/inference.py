# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.ndimage import filters

import math

import numpy as np
import torch

from utils.transforms import transform_preds
'''code for DARK

'''


def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    return the point with max values and it's value
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def gaussian_modulation_torch(batch_heatmaps, sigma, eps=1e-8):
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    device = batch_heatmaps.device
    dtype = batch_heatmaps.dtype
    temp_size = sigma * 3
    size = 2 * temp_size + 1

    x = torch.arange(0, size, 1, dtype=dtype, requires_grad=False).to(device)
    y = x[:, None]
    x0 = size // 2
    y0 = size // 2

    heatmaps_reshaped = batch_heatmaps.view(batch_size, num_joints, -1)
    heatmaps_max = heatmaps_reshaped.max(2)[0][..., None, None]

    g = torch.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
    g = g.unsqueeze(0).unsqueeze(0).expand(num_joints, -1, -1, -1)
    assert (size + 1) % 2 == 0, 'only support odd kernel size now'
    padding = int((size - 1) / 2)

    with torch.no_grad():
        heatmaps_modulation = torch.nn.functional.conv2d(batch_heatmaps, g, padding=padding, groups=num_joints)
        heatmaps_modulation_reshaped = heatmaps_modulation.view(batch_size, num_joints, -1)
        heatmaps_modulation_max = heatmaps_modulation_reshaped.max(2)[0][..., None, None]
        heatmaps_modulation_min = heatmaps_modulation_reshaped.min(2)[0][..., None, None]

        heatmaps_modulation = (heatmaps_modulation-heatmaps_modulation_min) / (heatmaps_modulation_max-heatmaps_modulation_min+eps) * heatmaps_max
        heatmaps_modulation[heatmaps_modulation < 0] = 0

    return heatmaps_modulation


def get_final_preds(config, batch_heatmaps, center, scale,
                    batch_heatmaps_DM=None):
    '''return the pred point with and the value of the maximum values
    the pred point is calculated as：（x,y）=（x_max,y_max）+\
    0.25×（sign（value（x_max+1,y_max）-value（x_max-1,y_max）,
    value（x_max,y_max+1）-value（x_max-1,y_max-1）））'''
    sigma = config.MODEL.SIGMA
    DE = config.MODEL.HEATMAP_DE
    DM = config.MODEL.HEATMAP_DE_DM
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    # post-processing
    if DM:
        batch_heatmaps = batch_heatmaps_DM
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                # int（floor(+0.5)?）
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    if DE:
                        derivative_1 = np.array(
                            [
                                (2*np.log(hm[py][px+1])+np.log(hm[py+1][px+1])
                                 + np.log(hm[py-1][px+1]) -
                                 2*np.log(hm[py][px-1])-np.log(hm[py+1][px-1])
                                 - np.log(hm[py-1][px-1]))/4,
                                (2*np.log(hm[py+1][px])+np.log(hm[py+1][px+1])
                                 + np.log(hm[py+1][px-1]) -
                                 2*np.log(hm[py-1][px])-np.log(hm[py-1][px+1])
                                 - np.log(hm[py-1][px-1]))/4
                            ]
                        )
                        # derivative_2 laplace
                        derivative_2 = np.array(
                            [
                                -1/(sigma**2),
                                -1/(sigma**2)
                            ]
                        )
                        coords[n][p] -= derivative_1/derivative_2
                    else:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals


def demo_preds_function(config, batch_heatmaps, batch_heatmaps_DM=None):

    '''return the pred point with and the value of the maximum values
    the pred point is calculated as：（x,y）=（x_max,y_max）+\
    0.25×（sign（value（x_max+1,y_max）-value（x_max-1,y_max）,value（x_max,y_max+1）
    -value（x_max-1,y_max-1）））'''
    sigma = config.MODEL.SIGMA
    DE = config.MODEL.HEATMAP_DE
    DM = config.MODEL.HEATMAP_DE_DM
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    # post-processing
    if DM:
        batch_heatmaps = batch_heatmaps_DM
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                # int（floor(+0.5)?）
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    if DE:
                        derivative_1 = np.array(
                            [
                                (2*np.log(hm[py][px+1])+np.log(hm[py+1][px+1])
                                 + np.log(hm[py-1][px+1]) -
                                 2*np.log(hm[py][px-1])-np.log(hm[py+1][px-1])
                                 - np.log(hm[py-1][px-1]))/4,
                                (2*np.log(hm[py+1][px])+np.log(hm[py+1][px+1])
                                 + np.log(hm[py+1][px-1]) -
                                 2*np.log(hm[py-1][px])-np.log(hm[py-1][px+1])
                                 - np.log(hm[py-1][px-1]))/4
                            ]
                        )
                        # derivative_2 laplace
                        derivative_2 = np.array(
                            [
                                -1/(sigma**2),
                                -1/(sigma**2)
                            ]
                        )
                        coords[n][p] -= derivative_1/derivative_2
                    else:
                        diff = np.array(
                            [
                                hm[py][px+1] - hm[py][px-1],
                                hm[py+1][px]-hm[py-1][px]
                            ]
                        )
                        coords[n][p] += np.sign(diff) * .25

    preds = coords.copy()

    return preds, maxvals

# if __name__ == '__main__':
#     import sys
#     import matplotlib.pyplot as plt
#     sys.path.append('../')
#     batch_heatmaps = torch.randn(4, 19, 56, 56)
#     batch_heatmaps_show = batch_heatmaps.numpy()
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(batch_heatmaps_show[0, 0])
#     output = gaussian_modulation_torch(batch_heatmaps, 3)
#     output_show = output.numpy()
#     ax[1].imshow(output_show[0, 0])
#     plt.show()
