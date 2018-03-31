import csv
import functools
import glob
import itertools
import math
import random
import time
import warnings

import scipy.ndimage
import SimpleITK as sitk

import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

cache = getCache('part2')

class Ct(object):
    def __init__(self, series_uid):
        mhd_path = glob.glob('data/luna/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_ary = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # This converts HU to g/cc.
        ct_ary += 1000
        ct_ary /= 1000

        # This gets rid of negative density stuff used to indicate out-of-FOV
        ct_ary[ct_ary < 0] = 0

        # This nukes any weird hotspots and clamps bone down
        ct_ary[ct_ary > 2] = 2

        self.series_uid = series_uid
        self.ary = ct_ary
        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_tup = tuple(int(round(x)) for x in ct_mhd.GetDirection())

    def getInputChunk(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_tup)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.ary.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.ary.shape, width_irc))
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.ary.shape[axis]:
                # log.warning("Crop outside of CT array: {} {}, center:{} shape:{} width:{}".format(
                #     self.series_uid, center_xyz, center_irc, self.ary.shape, width_irc))
                end_ndx = self.ary.shape[axis]
                start_ndx = int(self.ary.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.ary[slice_list]

        return ct_chunk, center_irc

    def getScaledInputChunk(self, center_xyz, width_mm, voxels_int):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_tup)

        ct_start = [int(round(i)) for i in xyz2irc(tuple(x - width_mm/2 for x in center_xyz), self.origin_xyz, self.vxSize_xyz, self.direction_tup)]
        ct_end = [int(round(i)) + 1 for i in xyz2irc(tuple(x + width_mm/2 for x in center_xyz), self.origin_xyz, self.vxSize_xyz, self.direction_tup)]

        for axis in range(3):
            if ct_start[axis] > ct_end[axis]:
                ct_start[axis], ct_end[axis] = ct_end[axis], ct_start[axis]

        pad_start = [0, 0, 0]
        pad_end = [ct_end[axis] - ct_start[axis] for axis in range(3)]
        # log.info([ct_end, ct_start, pad_end])
        pad_ary = np.zeros(pad_end, dtype=np.float32)

        for axis in range(3):
            if ct_start[axis] < 0:
                pad_start[axis] = -ct_start[axis]
                ct_start[axis] = 0

            if ct_end[axis] > self.ary.shape[axis]:
                pad_end[axis] -= ct_end[axis] - self.ary.shape[axis]
                ct_end[axis] = self.ary.shape[axis]

        pad_slices = tuple(slice(s,e) for s, e in zip(pad_start, pad_end))
        ct_slices = tuple(slice(s,e) for s, e in zip(ct_start, ct_end))
        pad_ary[pad_slices] = self.ary[ct_slices]

        try:
            zoom_seq = tuple(voxels_int/pad_ary.shape[axis] for axis in range(3))
        except:
            log.error([ct_end, ct_start, pad_end, center_irc, center_xyz, width_mm, self.vxSize_xyz])
            raise

        chunk_ary = scipy.ndimage.zoom(pad_ary, zoom_seq, order=1)

        # log.info("chunk_ary.shape {}".format([chunk_ary.shape, pad_ary.shape, zoom_seq, voxels_int]))

        return chunk_ary, center_irc


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@cache.memoize(typed=True)
def getCtInputChunk(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getInputChunk(center_xyz, width_irc)
    return ct_chunk, center_irc

@cache.memoize(typed=True)
def getScaledCtInputChunk(series_uid, center_xyz, width_mm, voxels_int):
    # log.info([series_uid, center_xyz, width_mm, voxels_int])
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getScaledInputChunk(center_xyz, width_mm, voxels_int)
    return ct_chunk, center_irc


def augmentChunk_shift(ct_chunk):

    for axis in range(1,3):
        new_chunk = np.zeros_like(ct_chunk)
        shift = random.randint(0, 2)

        slice_list = [slice(None)] * ct_chunk.ndim

        new_chunk


    return ct_chunk + np.random.normal(scale=0.1, size=ct_chunk.shape)

def augmentChunk_noise(ct_chunk):
    return ct_chunk + np.random.normal(scale=0.1, size=ct_chunk.shape)

def augmentChunk_mirror(ct_chunk):
    if random.random() > 0.5:
        ct_chunk = np.flip(ct_chunk, -1)
    return ct_chunk

def augmentChunk_rotate(ct_chunk):
    # Rotate the nodule around the head-foot axis
    angle = 360 * random.random()
    # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.rotate.html
    ct_chunk = scipy.ndimage.interpolation.rotate(
        ct_chunk,
        angle,
        axes=(-2, -1),
        reshape=False,
        order=1,
    )
    return ct_chunk

def augmentChunk_zoomAndCrop(ct_chunk):
    # log.info([ct_chunk.shape])
    zoom = 1.0 + 0.2 * random.random()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # https://docs.scipy.org/doc/scipy-0.16.1/reference/generated/scipy.ndimage.interpolation.zoom.html
        ct_chunk = scipy.ndimage.interpolation.zoom(
            ct_chunk,
            zoom,
            order=1
        )

    crop_list = [random.randint(0, ct_chunk.shape[axis]-16) for axis in range(1,4)]
    slice_list = [slice(None)] + [slice(start, start+16) for start in crop_list]

    ct_chunk = ct_chunk[slice_list]

    assert ct_chunk.shape[-3:] == (16, 16, 16), repr(ct_chunk.shape)

    return ct_chunk

def augmentCtInputChunk(ct_chunk):
    augment_list = [
        augmentChunk_mirror,
        augmentChunk_rotate,
        augmentChunk_noise,
        augmentChunk_zoomAndCrop,
    ]

    for augment_func in augment_list:
        ct_chunk = augment_func(ct_chunk)

    return ct_chunk


class LunaDataset(Dataset):
    def __init__(self, test_stride=0, isTestSet_bool=None, series_uid=None,
                 balanced_bool=False,
                 scaled_bool=False,
                 augmented_bool=False,
                 ):
        self.balanced_bool = balanced_bool
        self.scaled_bool = scaled_bool
        self.augmented_bool = augmented_bool

        # We construct a set with all series_uids that are present on disk.
        # This will let us use the data, even if we haven't downloaded all of
        # the subsets yet.
        mhd_list = glob.glob('data/luna/subset*/*.mhd')
        present_set = {p.rsplit('/', 1)[-1][:-4] for p in mhd_list}

        sample_list = []
        with open('data/luna/candidates.csv', "r") as f:
            csv_list = list(csv.reader(f))

        for row in csv_list[1:]:
            row_uid = row[0]

            if series_uid and series_uid != row_uid:
                continue

            # If a row_uid isn't present, that means it's in a subset that we
            # don't have on disk, so we should skip it.
            if row_uid not in present_set:
                continue

            center_xyz = tuple([float(x) for x in row[1:4]])
            isMalignant_bool = bool(int(row[4]))
            sample_list.append((row_uid, center_xyz, isMalignant_bool))

        sample_list.sort()
        if test_stride > 1:
            if isTestSet_bool:
                sample_list = sample_list[::test_stride]
            else:
                del sample_list[::test_stride]

        self.sample_list = sample_list
        self.benignIndex_list = [i for i, x in enumerate(sample_list) if not x[2]]
        self.malignantIndex_list = [i for i, x in enumerate(sample_list) if x[2]]

        self.shuffleSamples()

        log.info("{!r}: {} {} samples, {} ben, {} mal".format(
            self,
            len(sample_list),
            "testing" if isTestSet_bool else "training",
            len(self.benignIndex_list),
            len(self.malignantIndex_list),
        ))


    def shuffleSamples(self):
        if self.balanced_bool:
            log.warning("Shufflin'")
            random.shuffle(self.benignIndex_list)
            random.shuffle(self.malignantIndex_list)

    def __len__(self):
        if self.balanced_bool:
            return min(len(self.benignIndex_list), len(self.malignantIndex_list)) * 2 * 50
        else:
            return len(self.sample_list)

    def __getitem__(self, ndx):
        if self.balanced_bool:
            if ndx % 2:
                sample_ndx = self.benignIndex_list[(ndx // 2) % len(self.benignIndex_list)]
            else:
                sample_ndx = self.malignantIndex_list[(ndx // 2) % len(self.malignantIndex_list)]
        else:
            sample_ndx = ndx

        series_uid, center_xyz, isMalignant_bool = self.sample_list[sample_ndx]

        if self.scaled_bool:
            ct_chunk, center_irc = getScaledCtInputChunk(series_uid, center_xyz, 12, 20)
            # in:  dim=3, Index x Row x Col
            # out: dim=4, Channel x Index x Row x Col
            ct_chunk = np.expand_dims(ct_chunk, 0)

            if self.augmented_bool:
                ct_chunk = augmentCtInputChunk(ct_chunk)
            else:
                ct_chunk = ct_chunk[:, 2:-2, 2:-2, 2:-2]

        else:
            ct_chunk, center_irc = getCtInputChunk(series_uid, center_xyz, (16, 16, 16))
            ct_chunk = np.expand_dims(ct_chunk, 0)

        assert ct_chunk.shape[-3:] == (16, 16, 16), repr(ct_chunk.shape)


        ct_tensor = torch.from_numpy(np.array(ct_chunk, dtype=np.float32))
        # ct_tensor = ct_tensor.unsqueeze(0)

        # dim=1
        malignant_tensor = torch.from_numpy(np.array([isMalignant_bool], dtype=np.float32))
        malignant_tensor = malignant_tensor.unsqueeze(0)

        # Unpacked as: input_tensor, answer_int, series_uid, center_irc
        return ct_tensor, malignant_tensor, series_uid, center_irc



