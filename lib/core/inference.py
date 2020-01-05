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


def gaussian_modulation_torch(batch_heatmaps, sug):
    pass


def gaussian_modulation(batch_heatmaps, sigma, gaussian_mode='nearest'):
    '''modulate pred_heatmap with gaussian filter:
    kernal size:sigma*6+1, default mode: ‘nearest’ '''
    temp_size = sigma*3
    #cal maxinum of heatmap
    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    height = batch_heatmaps.shape[2]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    maxvals = np.amax(heatmaps_reshaped, 2)
    maxvals = maxvals.reshape((batch_size, num_joints, 1, 1))*np.ones([1,1,height,width])
    #modulation
    heatmap_modulation=filters.gaussian_filter(batch_heatmaps, sigma, cval=0.0, mode=gaussian_mode, truncate=temp_size)
    #scale heatmap into 0-maximum value
    heatmaps_modulation_reshaped = heatmap_modulation.reshape((batch_size, num_joints, -1))
    maxvals_modulation = np.amax(heatmaps_modulation_reshaped, 2)
    minvals_modulation = np.amin(heatmaps_modulation_reshaped, 2)
    maxvals_modulation = maxvals_modulation.reshape((batch_size, num_joints, 1, 1))*np.ones([1,1,height,width])
    minvals_modulation = minvals_modulation.reshape((batch_size, num_joints, 1, 1))*np.ones([1,1,height,width])
    dis0 = ((maxvals_modulation-minvals_modulation)==0)
    dis1 = ((maxvals_modulation-minvals_modulation)!=0)
    heatmap_modulation = ((heatmap_modulation-minvals_modulation)/(maxvals_modulation-minvals_modulation+1*dis0)-1)\
    *dis1*maxvals+maxvals
    heatmap_modulation=heatmap_modulation*(heatmap_modulation>=0)
    return heatmap_modulation


def get_final_preds(config, batch_heatmaps, center, scale, gaussian_mode='constant'):
    
    '''return the pred point with and the value of the maximum values
    the pred point is calculated as：（x,y）=（x_max,y_max）+\
    0.25×（sign（value（x_max+1,y_max）-value（x_max-1,y_max）,value（x_max,y_max+1）-value（x_max-1,y_max-1）））'''
    sigma=config.MODEL.SIGMA
    DE=config.MODEL.HEATMAP_DE
    DM=config.MODEL.HEATMAP_DE_DM
    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    #DM
    if DM:
        batch_heatmaps=gaussian_modulation(batch_heatmaps,sigma, gaussian_mode)
    # post-processing
    if config.TEST.POST_PROCESS:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][p]
                px = int(math.floor(coords[n][p][0] + 0.5))
                py = int(math.floor(coords[n][p][1] + 0.5))
                #int（floor(+0.5)?）
                if 1 < px < heatmap_width-1 and 1 < py < heatmap_height-1:
                    if DE:
                        #derivative_1 sobel
                        derivative_1 = np.array(
                            [
                                (2*np.log(hm[py][px+1])+np.log(hm[py+1][px+1])+np.log(hm[py-1][px+1])- \
                                 2*np.log(hm[py][px-1])-np.log(hm[py+1][px-1])-np.log(hm[py-1][px-1]))/4,
                                (2*np.log(hm[py+1][px])+np.log(hm[py+1][px+1])+np.log(hm[py+1][px-1])- \
                                 2*np.log(hm[py-1][px])-np.log(hm[py-1][px+1])-np.log(hm[py-1][px-1]))/4
                            ]
                        )
                        #derivative_2 laplace
                        derivative_2=np.array(
                            [
                                -2*np.log(hm[py][px])+np.log(hm[py][px+1])+np.log(hm[py][px-1]),
                                -2*np.log(hm[py][px])+np.log(hm[py+1][px])+np.log(hm[py-1][px])
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


