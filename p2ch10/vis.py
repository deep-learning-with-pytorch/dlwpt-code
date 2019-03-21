import matplotlib
matplotlib.use('nbagg')

import numpy as np
import matplotlib.pyplot as plt

from p2ch11_old.dsets import Ct, LunaDataset

clim=(0.0, 1.3)

def findMalignantSamples(start_ndx=0, limit=10):
    ds = LunaDataset()

    malignantSample_list = []
    for sample_tup in ds.sample_list:
        if sample_tup[2]:
            print(len(malignantSample_list), sample_tup)
            malignantSample_list.append(sample_tup)

        if len(malignantSample_list) >= limit:
            break

    return malignantSample_list

def showNodule(series_uid, batch_ndx=None, **kwargs):
    ds = LunaDataset(series_uid=series_uid, **kwargs)
    malignant_list = [i for i, x in enumerate(ds.sample_list) if x[2]]

    if batch_ndx is None:
        if malignant_list:
            batch_ndx = malignant_list[0]
        else:
            print("Warning: no malignant samples found; using first non-malignant sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    # ct_tensor, malignant_tensor, series_uid, center_irc = ds[batch_ndx]
    malignant_tensor, diameter_mm, series_uid, center_irc, nodule_tensor = ds[batch_ndx]
    ct_ary = nodule_tensor[1].numpy()


    fig = plt.figure(figsize=(15, 25))

    group_list = [
        #[0,1,2],
        [3,4,5],
        [6,7,8],
        [9,10,11],
        #[12,13,14],
        #[15]
    ]

    subplot = fig.add_subplot(len(group_list) + 2, 3, 1)
    subplot.set_title('index {}'.format(int(center_irc.index)))
    plt.imshow(ct.ary[int(center_irc.index)], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 2)
    subplot.set_title('row {}'.format(int(center_irc.row)))
    plt.imshow(ct.ary[:,int(center_irc.row)], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 3)
    subplot.set_title('col {}'.format(int(center_irc.col)))
    plt.imshow(ct.ary[:,:,int(center_irc.col)], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 4)
    subplot.set_title('index {}'.format(int(center_irc.index)))
    plt.imshow(ct_ary[ct_ary.shape[0]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 5)
    subplot.set_title('row {}'.format(int(center_irc.row)))
    plt.imshow(ct_ary[:,ct_ary.shape[1]//2], clim=clim, cmap='gray')

    subplot = fig.add_subplot(len(group_list) + 2, 3, 6)
    subplot.set_title('col {}'.format(int(center_irc.col)))
    plt.imshow(ct_ary[:,:,ct_ary.shape[2]//2], clim=clim, cmap='gray')

    for row, index_list in enumerate(group_list):
        for col, index in enumerate(index_list):
            subplot = fig.add_subplot(len(group_list) + 2, 3, row * 3 + col + 7)
            subplot.set_title('slice {}'.format(index))
            plt.imshow(ct_ary[index*2], clim=clim, cmap='gray')


    print(series_uid, batch_ndx, bool(malignant_tensor[0]), malignant_list, ct.vxSize_xyz)

    return ct_ary
