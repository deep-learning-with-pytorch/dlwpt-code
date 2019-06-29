import copy
import csv
import functools
import glob
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk

import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from util.disk import getCache
from util.util import XyzTuple, xyz2irc
from util.logconf import logging

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch12_raw')

NoduleInfoTuple = namedtuple('NoduleInfoTuple', 'isMalignant_bool, diameter_mm, series_uid, center_xyz')
MaskTuple = namedtuple('MaskTuple', 'air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask, ben_mask, mal_mask')

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

            noduleInfo_list.append(NoduleInfoTuple(isMalignant_bool, candidateDiameter_mm, series_uid, candidateCenter_xyz))

    noduleInfo_list.sort(reverse=True)
    return noduleInfo_list

class Ct(object):
    def __init__(self, series_uid, buildMasks_bool=True):
        mhd_path = glob.glob('data-unversioned/part2/luna/subset*/{}.mhd'.format(series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        ct_ary = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        # CTs are natively expressed in https://en.wikipedia.org/wiki/Hounsfield_scale
        # HU are scaled oddly, with 0 g/cc (air, approximately) being -1000 and 1 g/cc (water) being 0.
        # This gets rid of negative density stuff used to indicate out-of-FOV
        ct_ary[ct_ary < -1000] = -1000

        # This nukes any weird hotspots and clamps bone down
        ct_ary[ct_ary > 1000] = 1000

        self.series_uid = series_uid
        self.ary = ct_ary

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_tup = tuple(int(round(x)) for x in ct_mhd.GetDirection())

        noduleInfo_list = getNoduleInfoList()
        self.benignInfo_list = [ni_tup
                                for ni_tup in noduleInfo_list
                                    if not ni_tup.isMalignant_bool
                                        and ni_tup.series_uid == self.series_uid]
        self.benign_mask = self.buildAnnotationMask(self.benignInfo_list)[0]
        self.benign_indexes = sorted(set(self.benign_mask.nonzero()[0]))

        self.malignantInfo_list = [ni_tup
                                   for ni_tup in noduleInfo_list
                                        if ni_tup.isMalignant_bool
                                            and ni_tup.series_uid == self.series_uid]
        self.malignant_mask = self.buildAnnotationMask(self.malignantInfo_list)[0]
        self.malignant_indexes = sorted(set(self.malignant_mask.nonzero()[0]))

    def buildAnnotationMask(self, noduleInfo_list, threshold_gcc = -500):
        boundingBox_ary = np.zeros_like(self.ary, dtype=np.bool)

        for noduleInfo_tup in noduleInfo_list:
            center_irc = xyz2irc(
                noduleInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_tup,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.ary[ci + index_radius, cr, cc] > threshold_gcc and \
                        self.ary[ci - index_radius, cr, cc] > threshold_gcc:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.ary[ci, cr + row_radius, cc] > threshold_gcc and \
                        self.ary[ci, cr - row_radius, cc] > threshold_gcc:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.ary[ci, cr, cc + col_radius] > threshold_gcc and \
                        self.ary[ci, cr, cc - col_radius] > threshold_gcc:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([noduleInfo_tup.center_xyz, center_irc, self.ary[ci, cr, cc]])
            # assert row_radius > 0
            # assert col_radius > 0


            slice_tup = (
                slice(ci - index_radius, ci + index_radius + 1),
                slice(cr - row_radius, cr + row_radius + 1),
                slice(cc - col_radius, cc + row_radius + 1),
            )
            boundingBox_ary[slice_tup] = True

        thresholded_ary = boundingBox_ary & (self.ary > threshold_gcc)
        mask_ary = morph.binary_dilation(thresholded_ary, iterations=2)

        return mask_ary, thresholded_ary, boundingBox_ary

    def build2dLungMask(self, mask_ndx, threshold_gcc = -300):
        dense_mask = self.ary[mask_ndx] > threshold_gcc
        denoise_mask = morph.binary_closing(dense_mask, iterations=2)
        tissue_mask = morph.binary_opening(denoise_mask, iterations=10)
        body_mask = morph.binary_fill_holes(tissue_mask)
        air_mask = morph.binary_fill_holes(body_mask & ~tissue_mask)

        lung_mask = morph.binary_dilation(air_mask, iterations=2)

        ben_mask = denoise_mask & air_mask
        ben_mask = morph.binary_dilation(ben_mask, iterations=2)
        ben_mask &= ~self.malignant_mask[mask_ndx]

        mal_mask = self.malignant_mask[mask_ndx]

        return MaskTuple(
            air_mask,
            lung_mask,
            dense_mask,
            denoise_mask,
            tissue_mask,
            body_mask,
            ben_mask,
            mal_mask,
        )

    def build3dLungMask(self):
        air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask, ben_mask, mal_mask = mask_list = \
            [np.zeros_like(self.ary, dtype=np.bool) for _ in range(7)]

        for mask_ndx in range(self.ary.shape[0]):
            for i, mask_ary in enumerate(self.build2dLungMask(mask_ndx)):
                mask_list[i][mask_ndx] = mask_ary

        return MaskTuple(air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask, ben_mask, mal_mask)


    def getRawNodule(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_tup)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            try:
                start_ndx = int(round(center_val - width_irc[axis]/2))
            except:
                log.debug([center_val, width_irc, center_xyz, center_irc])
                raise
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

        ct_chunk = self.ary[tuple(slice_list)]

        return ct_chunk, center_irc

ctCache_depth = 5
@functools.lru_cache(ctCache_depth, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)
    return ct_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid, buildMasks_bool=False)
    return len(ct.benign_indexes)

def getCtAugmentedNodule(
        augmentation_dict,
        series_uid, center_xyz, width_irc,
        use_cache=True):
    if use_cache:
        ct_chunk, center_irc = getCtRawNodule(series_uid, center_xyz, width_irc)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)

    ct_tensor = torch.tensor(ct_chunk).unsqueeze(0).unsqueeze(0).to(torch.float32)

    transform_tensor = torch.eye(4).to(torch.float64)

    for i in range(3):
        if 'flip' in augmentation_dict:
            if random.random() > 0.5:
                transform_tensor[i,i] *= -1

        if 'offset' in augmentation_dict:
            offset_float = augmentation_dict['offset']
            random_float = (random.random() * 2 - 1)
            transform_tensor[3,i] = offset_float * random_float

        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            random_float = (random.random() * 2 - 1)
            transform_tensor[i,i] *= 1.0 + scale_float * random_float

    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)

        rotation_tensor = torch.tensor([
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float64)

        transform_tensor @= rotation_tensor

    affine_tensor = F.affine_grid(
            transform_tensor[:3].unsqueeze(0).to(torch.float32),
            ct_tensor.size(),
        )

    augmented_chunk = F.grid_sample(
            ct_tensor,
            affine_tensor,
            padding_mode='border'
        ).to('cpu')

    if 'noise' in augmentation_dict:
        noise_tensor = torch.randn_like(augmented_chunk)
        noise_tensor *= augmentation_dict['noise']

        augmented_chunk += noise_tensor

    return augmented_chunk[0], center_irc


class LunaDataset(Dataset):
    def __init__(self,
                 test_stride=0,
                 isTestSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 augmentation_dict=None,
                 noduleInfo_list=None,
            ):
        self.ratio_int = ratio_int
        self.augmentation_dict = augmentation_dict

        if noduleInfo_list:
            self.noduleInfo_list = copy.copy(noduleInfo_list)
            self.use_cache = False
        else:
            self.noduleInfo_list = copy.copy(getNoduleInfoList())
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(noduleInfo_tup.series_uid for noduleInfo_tup in getNoduleInfoList()))

        if isTestSet_bool:
            assert test_stride > 0, test_stride
            self.series_list = self.series_list[::test_stride]
            assert self.series_list
        elif test_stride > 0:
            del self.series_list[::test_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.noduleInfo_list = [x for x in self.noduleInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.noduleInfo_list)
        elif sortby_str == 'series_uid':
            self.noduleInfo_list.sort(key=lambda x: (x[2], x[3])) # sorting by series_uid, center_xyz)
        elif sortby_str == 'malignancy_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.benign_list = [nt for nt in self.noduleInfo_list if not nt.isMalignant_bool]
        self.malignant_list = [nt for nt in self.noduleInfo_list if nt.isMalignant_bool]

        log.info("{!r}: {} {} samples, {} ben, {} mal, {} ratio".format(
            self,
            len(self.noduleInfo_list),
            "testing" if isTestSet_bool else "training",
            len(self.benign_list),
            len(self.malignant_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.benign_list)
            random.shuffle(self.malignant_list)

    def __len__(self):
        if self.ratio_int:
            # return 20000
            return 200000
        else:
            return len(self.noduleInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            malignant_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                benign_ndx = ndx - 1 - malignant_ndx
                nodule_tup = self.benign_list[benign_ndx % len(self.benign_list)]
            else:
                nodule_tup = self.malignant_list[malignant_ndx % len(self.malignant_list)]
        else:
            nodule_tup = self.noduleInfo_list[ndx]

        width_irc = (24, 48, 48)

        if self.augmentation_dict:
            nodule_t, center_irc = getCtAugmentedNodule(
                self.augmentation_dict,
                nodule_tup.series_uid,
                nodule_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            nodule_ary, center_irc = getCtRawNodule(
                nodule_tup.series_uid,
                nodule_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_ary).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)
        else:
            ct = getCt(nodule_tup.series_uid)
            nodule_ary, center_irc = ct.getRawNodule(
                nodule_tup.center_xyz,
                width_irc,
            )
            nodule_t = torch.from_numpy(nodule_ary).to(torch.float32)
            nodule_t = nodule_t.unsqueeze(0)

        malignant_tensor = torch.tensor([
                not nodule_tup.isMalignant_bool,
                nodule_tup.isMalignant_bool
            ],
            dtype=torch.long,
        )

        # log.debug([type(center_irc), center_irc])

        return nodule_t, malignant_tensor, nodule_tup.series_uid, torch.tensor(center_irc)




class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 test_stride=0,
                 isTestSet_bool=None,
                 series_uid=None,
                 contextSlices_count=2,
                 augmentation_dict=None,
                 fullCt_bool=False,
            ):
        self.contextSlices_count = contextSlices_count
        self.augmentation_dict = augmentation_dict

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(noduleInfo_tup.series_uid for noduleInfo_tup in getNoduleInfoList()))

        if isTestSet_bool:
            assert test_stride > 0, test_stride
            self.series_list = self.series_list[::test_stride]
            assert self.series_list
        elif test_stride > 0:
            del self.series_list[::test_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in self.series_list:
            if fullCt_bool:
                self.sample_list.extend([(series_uid, ct_ndx) for ct_ndx in range(getCt(series_uid).ary.shape[0])])
            else:
                self.sample_list.extend([(series_uid, ct_ndx) for ct_ndx in range(getCtSampleSize(series_uid))])

        log.info("{!r}: {} {} series, {} slices".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'testing', False: 'training'}[isTestSet_bool],
            len(self.sample_list),
        ))

    def __len__(self):
        return len(self.sample_list) #// 100

    def __getitem__(self, ndx):
        if isinstance(ndx, int):
            series_uid, sample_ndx = self.sample_list[ndx % len(self.sample_list)]
            ct = getCt(series_uid)
            ct_ndx = self.sample_list[sample_ndx][1]
            useAugmentation_bool = False
        else:
            series_uid, ct_ndx, useAugmentation_bool = ndx
            ct = getCt(series_uid)

        ct_tensor = torch.zeros((self.contextSlices_count * 2 + 1 + 1, 512, 512))

        start_ndx = ct_ndx - self.contextSlices_count
        end_ndx = ct_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.ary.shape[0] - 1)

            ct_tensor[i] = torch.from_numpy(ct.ary[context_ndx].astype(np.float32))
        ct_tensor /= 1000

        mask_tup = ct.build2dLungMask(ct_ndx)

        ct_tensor[-1] = torch.from_numpy(mask_tup.body_mask.astype(np.float32))

        nodule_tensor = torch.from_numpy(
            (mask_tup.mal_mask | mask_tup.ben_mask).astype(np.float32)
        ).unsqueeze(0)
        ben_tensor = torch.from_numpy(mask_tup.ben_mask.astype(np.float32))
        mal_tensor = torch.from_numpy(mask_tup.mal_mask.astype(np.float32))
        label_int = mal_tensor.max() + ben_tensor.max() * 2

        if self.augmentation_dict and useAugmentation_bool:
            if 'rotate' in self.augmentation_dict:
                if random.random() > 0.5:
                    ct_tensor = ct_tensor.rot90(1, [1, 2])
                    nodule_tensor = nodule_tensor.rot90(1, [1, 2])

            if 'flip' in self.augmentation_dict:
                dims = [d+1 for d in range(2) if random.random() > 0.5]

                if dims:
                    ct_tensor = ct_tensor.flip(dims)
                    nodule_tensor = nodule_tensor.flip(dims)

            if 'noise' in self.augmentation_dict:
                noise_tensor = torch.randn_like(ct_tensor)
                noise_tensor *= self.augmentation_dict['noise']

                ct_tensor += noise_tensor
        return ct_tensor, nodule_tensor, label_int, ben_tensor, mal_tensor, ct.series_uid, ct_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, *args, batch_size=80, **kwargs):
        self.needsShuffle_bool = True
        self.batch_size = batch_size
        # self.rotate_frac = 0.5 * len(self.series_list) / len(self)
        super().__init__(*args, **kwargs)

    def __len__(self):
        return 50000

    def __getitem__(self, ndx):
        if self.needsShuffle_bool:
            random.shuffle(self.series_list)
            self.needsShuffle_bool = False

        if isinstance(ndx, int):
            if ndx % self.batch_size == 0:
                self.series_list.append(self.series_list.pop(0))

            series_uid = self.series_list[ndx % ctCache_depth]
            ct = getCt(series_uid)

            if ndx % 3 == 0:
                ct_ndx = random.choice(ct.malignant_indexes or ct.benign_indexes)
            elif ndx % 3 == 1:
                ct_ndx = random.choice(ct.benign_indexes)
            elif ndx % 3 == 2:
                ct_ndx = random.choice(list(range(ct.ary.shape[0])))

            useAugmentation_bool = True
        else:
            series_uid, ct_ndx, useAugmentation_bool = ndx

        return super().__getitem__((series_uid, ct_ndx, useAugmentation_bool))
