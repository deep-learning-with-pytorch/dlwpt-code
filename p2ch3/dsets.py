import csv
import functools
import glob
import itertools
import math
import random
import time

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

# cache = getCache('p2ch3')
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


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@cache.memoize(typed=True)
def getCtInputChunk(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getInputChunk(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self, test_stride=0, isTestSet_bool=None, series_uid=None,
                 balanced_bool=False,
                 ):
        self.balanced_bool = balanced_bool

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
        ct_chunk, center_irc = getCtInputChunk(series_uid, center_xyz, (16, 16, 16))

        # dim=3, Index x Row x Col
        ct_tensor = torch.from_numpy(np.array(ct_chunk, dtype=np.float32))

        # dim=1
        malignant_tensor = torch.from_numpy(np.array([isMalignant_bool], dtype=np.float32))

        # dim=4, Channel x Index x Row x Col
        ct_tensor = ct_tensor.unsqueeze(0)
        malignant_tensor = malignant_tensor.unsqueeze(0)

        # Unpacked as: input_tensor, answer_int, series_uid, center_irc
        return ct_tensor, malignant_tensor, series_uid, center_irc



