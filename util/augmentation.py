import math
import random
import warnings

import numpy as np
import scipy.ndimage

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

def cropToShape(image, new_shape, center_list=None, fill=0.0):
    # log.debug([image.shape, new_shape, center_list])
    # assert len(image.shape) == 3, repr(image.shape)

    if center_list is None:
        center_list = [int(image.shape[i] / 2) for i in range(3)]

    crop_list = []
    for i in range(0, 3):
        crop_int = center_list[i]
        if image.shape[i] > new_shape[i] and crop_int is not None:

            # We can't just do crop_int +/- shape/2 since shape might be odd
            # and ints round down.
            start_int = crop_int - int(new_shape[i]/2)
            end_int = start_int + new_shape[i]
            crop_list.append(slice(max(0, start_int), end_int))
        else:
            crop_list.append(slice(0, image.shape[i]))

    # log.debug([image.shape, crop_list])
    image = image[crop_list]

    crop_list = []
    for i in range(0, 3):
        if image.shape[i] < new_shape[i]:
            crop_int = int((new_shape[i] - image.shape[i]) / 2)
            crop_list.append(slice(crop_int, crop_int + image.shape[i]))
        else:
            crop_list.append(slice(0, image.shape[i]))

    # log.debug([image.shape, crop_list])
    new_image = np.zeros(new_shape, dtype=image.dtype)
    new_image[:] = fill
    new_image[crop_list] = image

    return new_image


def zoomToShape(image, new_shape, square=True):
    # assert image.shape[-1] in {1, 3, 4}, repr(image.shape)

    if square and image.shape[0] != image.shape[1]:
        crop_int = min(image.shape[0], image.shape[1])
        new_shape = [crop_int, crop_int, image.shape[2]]
        image = cropToShape(image, new_shape)

    zoom_shape = [new_shape[i] / image.shape[i] for i in range(3)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = scipy.ndimage.interpolation.zoom(
            image, zoom_shape,
            output=None, order=0, mode='nearest', cval=0.0, prefilter=True)

    return image

def randomOffset(image_list, offset_rows=0.125, offset_cols=0.125):

    center_list = [int(image_list[0].shape[i] / 2) for i in range(3)]
    center_list[0] += int(offset_rows * (random.random() - 0.5) * 2)
    center_list[1] += int(offset_cols * (random.random() - 0.5) * 2)
    center_list[2] = None

    new_list = []
    for image in image_list:
        new_image = cropToShape(image, image.shape, center_list)
        new_list.append(new_image)

    return new_list


def randomZoom(image_list, scale=None, scale_min=0.8, scale_max=1.3):
    if scale is None:
        scale = scale_min + (scale_max - scale_min) * random.random()

    new_list = []
    for image in image_list:
        # assert image.shape[-1] in {1, 3, 4}, repr(image.shape)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # log.info([image.shape])
            zimage = scipy.ndimage.interpolation.zoom(
                image, [scale, scale, 1.0],
                output=None, order=0, mode='nearest', cval=0.0, prefilter=True)
        image = cropToShape(zimage, image.shape)

        new_list.append(image)

    return new_list


_randomFlip_transform_list = [
    # lambda a: np.rot90(a, axes=(0, 1)),
    # lambda a: np.flip(a, 0),
    lambda a: np.flip(a, 1),
]

def randomFlip(image_list, transform_bits=None):
    if transform_bits is None:
        transform_bits = random.randrange(0, 2 ** len(_randomFlip_transform_list))

    new_list = []
    for image in image_list:
        # assert image.shape[-1] in {1, 3, 4}, repr(image.shape)

        for n in range(len(_randomFlip_transform_list)):
            if transform_bits & 2**n:
                # prhist(image, 'before')
                image = _randomFlip_transform_list[n](image)
                # prhist(image, 'after ')

        new_list.append(image)

    return new_list


def randomSpin(image_list, angle=None, range_tup=None, axes=(0, 1)):
    if range_tup is None:
        range_tup = (0, 360)

    if angle is None:
        angle = range_tup[0] + (range_tup[1] - range_tup[0]) * random.random()

    new_list = []
    for image in image_list:
        # assert image.shape[-1] in {1, 3, 4}, repr(image.shape)

        image = scipy.ndimage.interpolation.rotate(
                image, angle, axes=axes, reshape=False,
                output=None, order=0, mode='nearest', cval=0.0, prefilter=True)

        new_list.append(image)

    return new_list


def randomNoise(image_list, noise_min=-0.1, noise_max=0.1):
    noise = np.zeros_like(image_list[0])
    noise += (noise_max - noise_min) * np.random.random_sample(image_list[0].shape) + noise_min
    noise *= 5
    noise = scipy.ndimage.filters.gaussian_filter(noise, 3)
    # noise += (noise_max - noise_min) * np.random.random_sample(image_hsv.shape) + noise_min

    new_list = []
    for image_hsv in image_list:
        image_hsv = image_hsv + noise

        new_list.append(image_hsv)

    return new_list


def randomHsvShift(image_list, h=None, s=None, v=None,
                   h_min=-0.1, h_max=0.1,
                   s_min=0.5, s_max=2.0,
                   v_min=0.5, v_max=2.0):
    if h is None:
        h = h_min + (h_max - h_min) * random.random()
    if s is None:
        s = s_min + (s_max - s_min) * random.random()
    if v is None:
        v = v_min + (v_max - v_min) * random.random()

    new_list = []
    for image_hsv in image_list:
        # assert image_hsv.shape[-1] == 3, repr(image_hsv.shape)

        image_hsv[:,:,0::3] += h
        image_hsv[:,:,1::3] = image_hsv[:,:,1::3] ** s
        image_hsv[:,:,2::3] = image_hsv[:,:,2::3] ** v

        new_list.append(image_hsv)

    return clampHsv(new_list)


def clampHsv(image_list):
    new_list = []
    for image_hsv in image_list:
        image_hsv = image_hsv.clone()

        # Hue wraps around
        image_hsv[:,:,0][image_hsv[:,:,0] > 1] -= 1
        image_hsv[:,:,0][image_hsv[:,:,0] < 0] += 1

        # Everything else clamps between 0 and 1
        image_hsv[image_hsv > 1] = 1
        image_hsv[image_hsv < 0] = 0

        new_list.append(image_hsv)

    return new_list


# def torch_augment(input):
#     theta = random.random() * math.pi * 2
#     s = math.sin(theta)
#     c = math.cos(theta)
#     c1 = 1 - c
#     axis_vector = torch.rand(3, device='cpu', dtype=torch.float64)
#     axis_vector -= 0.5
#     axis_vector /= axis_vector.abs().sum()
#     l, m, n = axis_vector
#
#     matrix = torch.tensor([
#         [l*l*c1 +   c, m*l*c1 - n*s, n*l*c1 + m*s, 0],
#         [l*m*c1 + n*s, m*m*c1 +   c, n*m*c1 - l*s, 0],
#         [l*n*c1 - m*s, m*n*c1 + l*s, n*n*c1 +   c, 0],
#         [0, 0, 0, 1],
#     ], device=input.device, dtype=torch.float32)
#
#     return th_affine3d(input, matrix)




# following from https://github.com/ncullen93/torchsample/blob/master/torchsample/utils.py
# MIT licensed

# def th_affine3d(input, matrix):
#     """
#     3D Affine image transform on torch.Tensor
#     """
#     A = matrix[:3,:3]
#     b = matrix[:3,3]
#
#     # make a meshgrid of normal coordinates
#     coords = th_iterproduct(input.size(-3), input.size(-2), input.size(-1), dtype=torch.float32)
#
#     # shift the coordinates so center is the origin
#     coords[:,0] = coords[:,0] - (input.size(-3) / 2. - 0.5)
#     coords[:,1] = coords[:,1] - (input.size(-2) / 2. - 0.5)
#     coords[:,2] = coords[:,2] - (input.size(-1) / 2. - 0.5)
#
#     # apply the coordinate transformation
#     new_coords = coords.mm(A.t().contiguous()) + b.expand_as(coords)
#
#     # shift the coordinates back so origin is origin
#     new_coords[:,0] = new_coords[:,0] + (input.size(-3) / 2. - 0.5)
#     new_coords[:,1] = new_coords[:,1] + (input.size(-2) / 2. - 0.5)
#     new_coords[:,2] = new_coords[:,2] + (input.size(-1) / 2. - 0.5)
#
#     # map new coordinates using bilinear interpolation
#     input_transformed = th_trilinear_interp3d(input, new_coords)
#
#     return input_transformed
#
#
# def th_trilinear_interp3d(input, coords):
#     """
#     trilinear interpolation of 3D torch.Tensor image
#     """
#     # take clamp then floor/ceil of x coords
#     x = torch.clamp(coords[:,0], 0, input.size(-3)-2)
#     x0 = x.floor()
#     x1 = x0 + 1
#     # take clamp then floor/ceil of y coords
#     y = torch.clamp(coords[:,1], 0, input.size(-2)-2)
#     y0 = y.floor()
#     y1 = y0 + 1
#     # take clamp then floor/ceil of z coords
#     z = torch.clamp(coords[:,2], 0, input.size(-1)-2)
#     z0 = z.floor()
#     z1 = z0 + 1
#
#     stride = torch.tensor(input.stride()[-3:], dtype=torch.int64, device=input.device)
#     x0_ix = x0.mul(stride[0]).long()
#     x1_ix = x1.mul(stride[0]).long()
#     y0_ix = y0.mul(stride[1]).long()
#     y1_ix = y1.mul(stride[1]).long()
#     z0_ix = z0.mul(stride[2]).long()
#     z1_ix = z1.mul(stride[2]).long()
#
#     # input_flat = th_flatten(input)
#     input_flat = x.contiguous().view(x[0], x[1], -1)
#
#     vals_000 = input_flat[:, :, x0_ix+y0_ix+z0_ix]
#     vals_001 = input_flat[:, :, x0_ix+y0_ix+z1_ix]
#     vals_010 = input_flat[:, :, x0_ix+y1_ix+z0_ix]
#     vals_011 = input_flat[:, :, x0_ix+y1_ix+z1_ix]
#     vals_100 = input_flat[:, :, x1_ix+y0_ix+z0_ix]
#     vals_101 = input_flat[:, :, x1_ix+y0_ix+z1_ix]
#     vals_110 = input_flat[:, :, x1_ix+y1_ix+z0_ix]
#     vals_111 = input_flat[:, :, x1_ix+y1_ix+z1_ix]
#
#     xd = x - x0
#     yd = y - y0
#     zd = z - z0
#     xm1 = 1 - xd
#     ym1 = 1 - yd
#     zm1 = 1 - zd
#
#     x_mapped = (
#             vals_000.mul(xm1).mul(ym1).mul(zm1) +
#             vals_001.mul(xm1).mul(ym1).mul(zd) +
#             vals_010.mul(xm1).mul(yd).mul(zm1) +
#             vals_011.mul(xm1).mul(yd).mul(zd) +
#             vals_100.mul(xd).mul(ym1).mul(zm1) +
#             vals_101.mul(xd).mul(ym1).mul(zd) +
#             vals_110.mul(xd).mul(yd).mul(zm1) +
#             vals_111.mul(xd).mul(yd).mul(zd)
#     )
#
#     return x_mapped.view_as(input)
#
# def th_iterproduct(*args, dtype=None):
#     return torch.from_numpy(np.indices(args).reshape((len(args),-1)).T)
#
# def th_flatten(x):
#     """Flatten tensor"""
#     return x.contiguous().view(x[0], x[1], -1)
