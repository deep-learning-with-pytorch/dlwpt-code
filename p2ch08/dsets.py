import copy
import csv
import functools
import glob
import os
import random

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

raw_cache = getCache('part2ch09_raw')

@functools.lru_cache(1)
def getNoduleInfoList(requireDataOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob('data-unversioned/part2/luna/subset*/*.mhd')
    dataPresentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    diameter_dict = {}
    with open('data/part2/luna/annotations.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            diameter_dict.setdefault(series_uid, []).append((annotationCenter_xyz, annotationDiameter_mm))

    noduleInfo_list = []
    with open('data/part2/luna/candidates.csv', "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in dataPresentOnDisk_set and requireDataOnDisk_bool:
                continue

            isMalignant_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            candidateDiameter_mm = 0.0
            for annotationCenter_xyz, annotationDiameter_mm in diameter_dict.get(series_uid, []):
                for i in range(3):
                    delta_mm = abs(candidateCenter_xyz[i] - annotationCenter_xyz[i])
                    if delta_mm > annotationDiameter_mm / 4:
                        break
                else:
                    candidateDiameter_mm = annotationDiameter_mm
                    break

            noduleInfo_list.append((isMalignant_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz))

    noduleInfo_list.sort(reverse=True)
    return noduleInfo_list

class Ct(object):
    def __init__(self, series_uid):
        mhd_path = glob.glob('data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]

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

    def getRawNodule(self, center_xyz, width_irc):
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

@raw_cache.memoize(typed=True)
def getCtRawNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)
    return ct_chunk, center_irc

class LunaDataset(Dataset):
    def __init__(self,
                 test_stride=0,
                 isTestSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
            ):
        self.noduleInfo_list = copy.copy(getNoduleInfoList())

        if series_uid:
            self.noduleInfo_list = [x for x in self.noduleInfo_list if x[2] == series_uid]

        # __init__ continued...
        if test_stride > 1:
            if isTestSet_bool:
                self.noduleInfo_list = self.noduleInfo_list[::test_stride]
            else:
                del self.noduleInfo_list[::test_stride]

        if sortby_str == 'random':
            random.shuffle(self.noduleInfo_list)
        elif sortby_str == 'series_uid':
            self.noduleInfo_list.sort(key=lambda x: (x[2], x[3])) # sorting by series_uid, center_xyz)
        elif sortby_str == 'malignancy_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        log.info("{!r}: {} {} samples".format(
            self,
            len(self.noduleInfo_list),
            "testing" if isTestSet_bool else "training",
        ))


    def __len__(self):
        # if self.ratio_int:
        #     return min(len(self.benignIndex_list), len(self.malignantIndex_list)) * 4 * 90
        # else:
        return len(self.noduleInfo_list)

    def __getitem__(self, ndx):
        sample_ndx = ndx

        isMalignant_bool, _diameter_mm, series_uid, center_xyz = self.noduleInfo_list[sample_ndx]

        nodule_ary, center_irc = getCtRawNodule(series_uid, center_xyz, (32, 32, 32))

        nodule_tensor = torch.from_numpy(nodule_ary)
        nodule_tensor = nodule_tensor.unsqueeze(0)

        malignant_tensor = torch.tensor([isMalignant_bool], dtype=torch.float32)

        return nodule_tensor, malignant_tensor, series_uid, center_irc



