import copy
import csv
import functools
import glob
import itertools
import math
import os
import random

from collections import namedtuple

import SimpleITK as sitk

import scipy.ndimage.morphology

import numpy as np
import torch
import torch.cuda
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler

from util.disk import getCache
from util.util import XyzTuple, xyz2irc, IrcTuple
from util.logconf import logging
from util.affine import affine_grid_generator

log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2ch11_raw')
cubic_cache = getCache('part2ch11_cubic')

NoduleInfoTuple = namedtuple('NoduleInfoTuple', 'isMalignant_bool, diameter_mm, series_uid, center_xyz')

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

    def buildAnnotationMask(self, noduleInfo_list, threshold_gcc = 0.5):
        boundingBox_ary = np.zeros_like(self.ary, dtype=np.bool)

        for noduleInfo_tup in noduleInfo_list:
            center_irc = xyz2irc(noduleInfo_tup.center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_tup)
            center_index = int(center_irc.index)
            center_row = int(center_irc.row)
            center_col = int(center_irc.col)

            index_radius = 2
            try:
                while self.ary[center_index + index_radius, center_row, center_col] > threshold_gcc and \
                            self.ary[center_index - index_radius, center_row, center_col] > threshold_gcc:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.ary[center_index, center_row + row_radius, center_col] > threshold_gcc and \
                            self.ary[center_index, center_row - row_radius, center_col] > threshold_gcc:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.ary[center_index, center_row, center_col + col_radius] > threshold_gcc and \
                            self.ary[center_index, center_row, center_col - col_radius] > threshold_gcc:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            # assert index_radius > 0, repr([noduleInfo_tup.center_xyz, center_irc, self.ary[center_index, center_row, center_col]])
            # assert row_radius > 0
            # assert col_radius > 0


            slice_tup = (
                slice(
                    # max(0, center_index - index_radius),
                    center_index - index_radius,
                    center_index + index_radius + 1,
                ),
                slice(
                    # max(0, center_row - row_radius),
                    center_row - row_radius,
                    center_row + row_radius + 1,
                ),
                slice(
                    # max(0, center_col - col_radius),
                    center_col - col_radius,
                    center_col + row_radius + 1,
                ),
            )

            boundingBox_ary[slice_tup] = True

        thresholded_ary = boundingBox_ary & (self.ary > threshold_gcc)
        mask_ary = scipy.ndimage.morphology.binary_dilation(thresholded_ary, iterations=2)

        return mask_ary, thresholded_ary, boundingBox_ary

    def build2dLungMask(self, mask_ndx, threshold_gcc = 0.7):
        dense_mask = self.ary[mask_ndx] > threshold_gcc
        denoise_mask = scipy.ndimage.morphology.binary_closing(dense_mask, iterations=2)
        tissue_mask = scipy.ndimage.morphology.binary_opening(denoise_mask, iterations=10)
        body_mask = scipy.ndimage.morphology.binary_fill_holes(tissue_mask)
        air_mask = scipy.ndimage.morphology.binary_fill_holes(body_mask & ~tissue_mask)

        lung_mask = scipy.ndimage.morphology.binary_dilation(air_mask, iterations=2)

        return air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask

    def build3dLungMask(self):
        air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask = mask_list = \
            [np.zeros_like(self.ary, dtype=np.bool) for _ in range(6)]

        for mask_ndx in range(self.ary.shape[0]):
            for i, mask_ary in enumerate(self.build2dLungMask(mask_ndx)):
                mask_list[i][mask_ndx] = mask_ary

        return air_mask, lung_mask, dense_mask, denoise_mask, tissue_mask, body_mask




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

        ct_chunk = self.ary[tuple(slice_list)]

        return ct_chunk, center_irc

    def getCubicInputChunk(self, center_xyz, maxWidth_mm):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz, self.direction_tup)

        ct_start = [int(round(i)) for i in xyz2irc(tuple(x - maxWidth_mm / 2 for x in center_xyz), self.origin_xyz, self.vxSize_xyz, self.direction_tup)]
        ct_end = [int(round(i)) + 1 for i in xyz2irc(tuple(x + maxWidth_mm / 2 for x in center_xyz), self.origin_xyz, self.vxSize_xyz, self.direction_tup)]

        for axis in range(3):
            if ct_start[axis] > ct_end[axis]:
                ct_start[axis], ct_end[axis] = ct_end[axis], ct_start[axis]

        pad_start = [0, 0, 0]
        pad_end = [ct_end[axis] - ct_start[axis] for axis in range(3)]
        # log.info([ct_end, ct_start, pad_end])
        chunk_ary = np.zeros(pad_end, dtype=np.float32)

        for axis in range(3):
            if ct_start[axis] < 0:
                pad_start[axis] = -ct_start[axis]
                ct_start[axis] = 0

            if ct_end[axis] > self.ary.shape[axis]:
                pad_end[axis] -= ct_end[axis] - self.ary.shape[axis]
                ct_end[axis] = self.ary.shape[axis]

        pad_slices = tuple(slice(s,e) for s, e in zip(pad_start, pad_end))
        ct_slices = tuple(slice(s,e) for s, e in zip(ct_start, ct_end))
        chunk_ary[pad_slices] = self.ary[ct_slices]

        return chunk_ary, center_irc


ctCache_depth = 3
@functools.lru_cache(ctCache_depth, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtSize(series_uid):
    ct = Ct(series_uid, buildMasks_bool=False)
    return tuple(ct.ary.shape)

# @raw_cache.memoize(typed=True)
# def getCtLungExtents(series_uid):
#     ct = getCt(series_uid)
#     return (int(min(ct.lung_indexes)), int(max(ct.lung_indexes)))

@raw_cache.memoize(typed=True)
def getCtRawNodule(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getRawNodule(center_xyz, width_irc)

    return ct_chunk, center_irc

# clamp_value = 1.5
@functools.lru_cache(1, typed=True)
@cubic_cache.memoize(typed=True)
def getCtCubicChunk(series_uid, center_xyz, maxWidth_mm):
    ct = getCt(series_uid)
    ct_chunk, center_irc = ct.getCubicInputChunk(center_xyz, maxWidth_mm)

    # # ct_chunk has been clamped to [0, 2] at this point
    # # We are going to convert to uint8 to reduce size on disk and loading time
    # ct_chunk[ct_chunk > clamp_value] = clamp_value
    # ct_chunk *= 255/clamp_value
    # ct_chunk = np.array(ct_chunk, dtype=np.uint8)

    return ct_chunk, center_irc

def getCtAugmentedNodule(augmentation_dict, series_uid, center_xyz, width_mm, voxels_int, maxWidth_mm=32.0, use_cache=True):
    assert width_mm <= maxWidth_mm

    if use_cache:
        cubic_chunk, center_irc = getCtCubicChunk(series_uid, center_xyz, maxWidth_mm)
    else:
        ct = getCt(series_uid)
        ct_chunk, center_irc = ct.getCubicInputChunk(center_xyz, maxWidth_mm)

    slice_list = []
    for axis in range(3):
        crop_size = cubic_chunk.shape[axis] * width_mm / maxWidth_mm
        crop_size = int(math.ceil(crop_size))
        start_ndx = (cubic_chunk.shape[axis] - crop_size) // 2
        end_ndx = start_ndx + crop_size

        slice_list.append(slice(start_ndx, end_ndx))

    cropped_chunk = cubic_chunk[slice_list]

    # # inflate cropped_chunk back to float32
    # cropped_chunk = np.array(cropped_chunk, dtype=np.float32)
    # cropped_chunk *= clamp_value/255
    cropped_tensor = torch.tensor(cropped_chunk).unsqueeze(0).unsqueeze(0)

    transform_tensor = torch.eye(4).to(torch.float64)

    # Scale and Mirror
    for i in range(3):
        if 'scale' in augmentation_dict:
            scale_float = augmentation_dict['scale']
            transform_tensor[i,i] *= 1.0 - scale_float/2.0 + (random.random() * scale_float)

        if 'mirror' in augmentation_dict:
            if random.random() > 0.5:
                transform_tensor[i,i] *= -1

    # Rotate
    if 'rotate' in augmentation_dict:
        angle_rad = random.random() * math.pi * 2
        s = math.sin(angle_rad)
        c = math.cos(angle_rad)
        c1 = 1 - c

        axis_tensor = torch.rand([3], dtype=torch.float64)
        axis_tensor /= axis_tensor.pow(2).sum().pow(0.5)

        z, y, x = axis_tensor
        rotation_tensor = torch.tensor([
            [x*x*c1 +   c, y*x*c1 - z*s, z*x*c1 + y*s, 0],
            [x*y*c1 + z*s, y*y*c1 +   c, z*y*c1 - x*s, 0],
            [x*z*c1 - y*s, y*z*c1 + x*s, z*z*c1 +   c, 0],
            [0, 0, 0, 1],
        ], dtype=torch.float64)

        transform_tensor @= rotation_tensor

    # Transform into final desired shape
    affine_tensor = affine_grid_generator(
            transform_tensor[:3].unsqueeze(0).to(torch.float32),
            torch.Size([1, 1, voxels_int, voxels_int, voxels_int])
        )

    zoomed_chunk = torch.nn.functional.grid_sample(
            cropped_tensor,
            affine_tensor,
            padding_mode='border'
        ).to('cpu')

    # Noise
    if 'noise' in augmentation_dict:
        noise_tensor = torch.randn(
                zoomed_chunk.size(),
                dtype=zoomed_chunk.dtype,
            )
        noise_tensor *= augmentation_dict['noise']
        zoomed_chunk += noise_tensor

    return zoomed_chunk[0,0], center_irc


class LunaPrepcacheDataset(Dataset):
    def __init__(self):
        self.series_list = sorted(set(noduleInfo_tup.series_uid for noduleInfo_tup in getNoduleInfoList()))

    def __len__(self):
        return len(self.series_list)

    def __getitem__(self, ndx):
        getCtSize(self.series_list[ndx])
        # getCtLungExtents(self.series_list[ndx])

        return 0


class LunaClassificationDataset(Dataset):
    def __init__(self,
                 test_stride=0,
                 isTestSet_bool=None,
                 series_uid=None,
                 sortby_str='random',
                 ratio_int=0,
                 scaled_bool=False,
                 multiscaled_bool=False,
                 augmented_bool=False,
                 noduleInfo_list=None,
            ):
        self.ratio_int = ratio_int
        self.scaled_bool = scaled_bool
        self.multiscaled_bool = multiscaled_bool

        if augmented_bool:
            self.augmentation_dict = {
                'mirror': True,
                'rotate': True,
            }

            if isTestSet_bool:
                self.augmentation_dict['scale'] = 0.25
            else:
                self.augmentation_dict['scale'] = 0.5
                self.augmentation_dict['noise'] = 0.1
        else:
            self.augmentation_dict = {}

        if noduleInfo_list:
            self.noduleInfo_list = copy.copy(noduleInfo_list)
            self.use_cache = False
        else:
            self.noduleInfo_list = copy.copy(getNoduleInfoList())
            self.use_cache = True

        if series_uid:
            self.noduleInfo_list = [x for x in self.noduleInfo_list if x[2] == series_uid]

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

        self.benignIndex_list = [i for i, x in enumerate(self.noduleInfo_list) if not x[0]]
        self.malignantIndex_list = [i for i, x in enumerate(self.noduleInfo_list) if x[0]]

        log.info("{!r}: {} {} samples, {} ben, {} mal, {} ratio".format(
            self,
            len(self.noduleInfo_list),
            "testing" if isTestSet_bool else "training",
            len(self.benignIndex_list),
            len(self.malignantIndex_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffleSamples(self):
        if self.ratio_int:
            random.shuffle(self.benignIndex_list)
            random.shuffle(self.malignantIndex_list)

    def __len__(self):
        if self.ratio_int:
            # return 10000
            return 100000
        elif self.augmentation_dict:
            return len(self.noduleInfo_list) * 5
        else:
            return len(self.noduleInfo_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            malignant_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                benign_ndx = ndx - 1 - malignant_ndx
                nodule_ndx = self.benignIndex_list[benign_ndx % len(self.benignIndex_list)]
            else:
                nodule_ndx = self.malignantIndex_list[malignant_ndx % len(self.malignantIndex_list)]

            augmentation_dict = self.augmentation_dict
        else:
            nodule_ndx = ndx % len(self.noduleInfo_list)

            if ndx < len(self.noduleInfo_list):
                augmentation_dict = {}
            else:
                augmentation_dict = self.augmentation_dict

        isMalignant_bool, _diameter_mm, series_uid, center_xyz = self.noduleInfo_list[nodule_ndx]

        if self.scaled_bool:
            channel_list = []
            voxels_int = 32

            if self.multiscaled_bool:
                width_list = [8, 16, 32]
            else:
                width_list = [24]

            for width_mm in width_list:
                nodule_ary, center_irc = getCtAugmentedNodule(augmentation_dict, series_uid, center_xyz, width_mm, voxels_int)
                # in:  dim=3, Index x Row x Col
                # out: dim=4, Channel x Index x Row x Col
                nodule_ary = nodule_ary.unsqueeze(0)
                channel_list.append(nodule_ary)

            nodule_tensor = torch.cat(channel_list)

        else:
            nodule_ary, center_irc = getCtRawNodule(series_uid, center_xyz, (32, 32, 32))
            nodule_ary = np.expand_dims(nodule_ary, 0)
            nodule_tensor = torch.from_numpy(nodule_ary)

        # dim=1
        malignant_tensor = torch.tensor([isMalignant_bool], dtype=torch.float32)

        return nodule_tensor, malignant_tensor, series_uid, center_irc
        #
        # return malignant_tensor, diameter_mm, series_uid, center_irc, nodule_tensor

class Luna2dSegmentationDataset(Dataset):
    purpose_str = 'general'

    def __init__(self,
                 contextSlices_count=2,
                 series_uid=None,
                 test_stride=0,
            ):
        self.contextSlices_count = contextSlices_count
        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(noduleInfo_tup.series_uid for noduleInfo_tup in getNoduleInfoList()))
        self.cullTrainOrTestSeries(test_stride)


        self.sample_list = []
        for series_uid in self.series_list:
            self.sample_list.extend([(series_uid, i) for i in range(int(getCtSize(series_uid)[0]))])

        log.info("{!r}: {} {} series, {} slices".format(
            self,
            len(self.series_list),
            self.purpose_str,
            len(self.sample_list),
        ))

    def cullTrainOrTestSeries(self, test_stride):
        assert test_stride == 0

    def __len__(self):
        return len(self.sample_list) #// 100

    def __getitem__(self, ndx):
        if isinstance(ndx, int):
            series_uid, sample_ndx = self.sample_list[ndx % len(self.sample_list)]
        else:
            series_uid, sample_ndx = ndx
        ct = getCt(series_uid)

        ct_tensor = torch.zeros((self.contextSlices_count * 2 + 2, 512, 512))
        masks_tensor = torch.zeros((2, 512, 512))

        start_ndx = sample_ndx - self.contextSlices_count
        end_ndx = sample_ndx + self.contextSlices_count + 1
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.ary.shape[0] - 1)

            ct_tensor[i] = torch.from_numpy(ct.ary[context_ndx].astype(np.float32))

        air_mask, lung_mask = ct.build2dLungMask(sample_ndx)[:2]

        ct_tensor[-1] = torch.from_numpy(lung_mask.astype(np.float32))

        mal_mask = ct.malignant_mask[sample_ndx] & lung_mask
        ben_mask = ct.benign_mask[sample_ndx] & air_mask

        masks_tensor[0] = torch.from_numpy(mal_mask.astype(np.float32))
        masks_tensor[1] = torch.from_numpy((mal_mask | ben_mask).astype(np.float32))
        # masks_tensor[1] = torch.from_numpy(ben_mask.astype(np.float32))

        return ct_tensor.contiguous(), masks_tensor.contiguous(), ct.series_uid, sample_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    purpose_str = 'training'

    def __init__(self, *args, **kwargs):
        self.needsShuffle_bool = True
        super().__init__(*args, **kwargs)

    def cullTrainOrTestSeries(self, test_stride):
        assert test_stride > 0, test_stride
        del self.series_list[::test_stride]
        assert self.series_list

    def __len__(self):
        # return 100
        # return 1000
        # return 10000
        return 20000
        # return 40000

    def __getitem__(self, ndx):
        if self.needsShuffle_bool:
            random.shuffle(self.series_list)
            self.needsShuffle_bool = False

        if random.random() < 0.01:
            self.series_list.append(self.series_list.pop(0))

        if isinstance(ndx, int):
            series_uid = self.series_list[ndx % ctCache_depth]
            ct = getCt(series_uid)
            sample_ndx = random.choice(ct.malignant_indexes or ct.benign_indexes)
            # series_uid, sample_ndx = self.sample_list[ndx % len(self.sample_list)]
        else:
            series_uid, sample_ndx = ndx

        # if ndx % 2 == 0:
        #     sample_ndx = random.choice(ct.malignant_indexes or ct.benign_indexes)
        # else: #if ndx % 2 == 2:
        #     sample_ndx = random.choice(ct.benign_indexes)
        # else:
        #     sample_ndx = random.randint(*self.series2extents_dict[series_uid])

        return super().__getitem__((series_uid, sample_ndx))


class TestingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    purpose_str = 'testing'

    def cullTrainOrTestSeries(self, test_stride):
        assert test_stride > 0
        self.series_list = self.series_list[::test_stride]
        assert self.series_list

