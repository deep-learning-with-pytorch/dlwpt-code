import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from p2ch07.dsets import Ct, LunaDataset

clim=(0.0, 1.3)

def findMalignantSamples(start_ndx=0, limit=100):
    ds = LunaDataset()

    malignantSample_list = []
    for sample_tup in ds.noduleInfo_list:
        if sample_tup[0]:
            malignantSample_list.append(sample_tup)

        if len(malignantSample_list) >= limit:
            break

    return malignantSample_list

def showNodule(series_uid, batch_ndx=None):
    ds = LunaDataset(series_uid=series_uid)
    malignant_list = [i for i, x in enumerate(ds.noduleInfo_list) if x[0]]

    if batch_ndx is None:
        if malignant_list:
            batch_ndx = malignant_list[0]
        else:
            print("Warning: no malignant samples found; using first non-malignant sample.")
            batch_ndx = 0

    ct = Ct(series_uid)
    ct_tensor, malignant_tensor, series_uid, center_irc = ds[batch_ndx]
    ct_ary = ct_tensor[0].numpy()

    fig = plt.figure(figsize=(15, 25))

    group_list = [
        [9,11,13],
        [15, 16, 17],
        [19,21,23],
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
            plt.imshow(ct_ary[index], clim=clim, cmap='gray')


    print(series_uid, batch_ndx, bool(malignant_tensor[0]), malignant_list)


