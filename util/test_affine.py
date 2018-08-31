import math
import random

import numpy as np
import scipy.ndimage

import torch

import pytest

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)

from .affine import affine_grid_generator


if torch.cuda.is_available():
    @pytest.fixture(params=['cpu', 'cuda'])
    def device(request):
        return request.param
else:
    @pytest.fixture(params=['cpu'])
    def device(request):
        return request.param

# @pytest.fixture(params=[0., 0.25])
@pytest.fixture(params=[0.0, 0.5, 0.25, 0.125, 'random'])
def angle_rad(request):
    if request.param == 'random':
        return random.random() * math.pi * 2
    return request.param * math.pi * 2

@pytest.fixture(params=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), 'random'])
def axis_vector(request):
    if request.param == 'random':
        t = (random.random(), random.random(), random.random())
        l = sum(x**2 for x in t)**0.5
        return tuple(x/l for x in t)
    return request.param

@pytest.fixture(params=[torch.nn.functional.affine_grid, affine_grid_generator])
def affine_func2d(request):
    return request.param

@pytest.fixture(params=[affine_grid_generator])
def affine_func3d(request):
    return request.param

# @pytest.fixture(params=[[1, 1, 3, 5], [1, 1, 3, 3]])
@pytest.fixture(params=[[1, 1, 3, 5], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 3, 4]])
def input_size2d(request):
    return request.param

# @pytest.fixture(params=[[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 5, 5]])
@pytest.fixture(params=[[1, 1, 5, 3], [1, 1, 3, 5], [1, 1, 4, 3], [1, 1, 5, 5], [1, 1, 6, 6]])
def output_size2d(request):
    return request.param

@pytest.fixture(params=[[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 6, 6], ])
def input_size2dsq(request):
    return request.param

@pytest.fixture(params=[[1, 1, 2, 2], [1, 1, 3, 3], [1, 1, 4, 4], [1, 1, 5, 5], [1, 1, 6, 6], ])
def output_size2dsq(request):
    return request.param


# @pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 2, 3, 4]])
@pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 2, 3, 4], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 3, 4, 5]])
def input_size3d(request):
    return request.param

@pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 6, 6, 6]])
def input_size3dsq(request):
    return request.param

@pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 4, 4, 4], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]])
def output_size3dsq(request):
    return request.param

# @pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5]])
@pytest.fixture(params=[[1, 1, 2, 2, 2], [1, 1, 3, 3, 3], [1, 1, 3, 4, 5], [1, 1, 4, 3, 2], [1, 1, 5, 5, 5], [1, 1, 6, 6, 6]])
def output_size3d(request):
    return request.param


def _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad):
    print("_buildEquivalentTransforms2d", device, input_size, output_size, angle_rad * 180 / math.pi)
    input_center = [(x-1)/2 for x in input_size]
    output_center = [(x-1)/2 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)

    intrans_ary = np.array([
        [1, 0, input_center[2]],
        [0, 1, input_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0],
        [0, input_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    rotation_ary = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1/output_center[2], 0, 0],
        [0, 1/output_center[3], 0],
        [0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, -output_center[2]],
        [0, 1, -output_center[3]],
        [0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ], dtype=np.float64)

    transform_ary = intrans_ary @ inscale_ary @ rotation_ary.T @ outscale_ary @ outtrans_ary
    grid_ary = reorder_ary @ rotation_ary.T @ outscale_ary @ outtrans_ary
    transform_tensor = torch.from_numpy((rotation_ary)).to(device, torch.float32)

    transform_tensor = transform_tensor[:2].unsqueeze(0)

    print('transform_tensor', transform_tensor.size(), transform_tensor.dtype, transform_tensor.device)
    print(transform_tensor)
    print('outtrans_ary', outtrans_ary.shape, outtrans_ary.dtype)
    print(outtrans_ary.round(3))
    print('outscale_ary', outscale_ary.shape, outscale_ary.dtype)
    print(outscale_ary.round(3))
    print('rotation_ary', rotation_ary.shape, rotation_ary.dtype)
    print(rotation_ary.round(3))
    print('inscale_ary', inscale_ary.shape, inscale_ary.dtype)
    print(inscale_ary.round(3))
    print('intrans_ary', intrans_ary.shape, intrans_ary.dtype)
    print(intrans_ary.round(3))
    print('transform_ary', transform_ary.shape, transform_ary.dtype)
    print(transform_ary.round(3))
    print('grid_ary', grid_ary.shape, grid_ary.dtype)
    print(grid_ary.round(3))

    def prtf(pt):
        print(pt, 'transformed', (transform_ary @ (pt + [1]))[:2].round(3))

    prtf([0, 0])
    prtf([1, 0])
    prtf([2, 0])

    print('')

    prtf([0, 0])
    prtf([0, 1])
    prtf([0, 2])
    prtf(output_center[2:])

    return transform_tensor, transform_ary, grid_ary

def _buildEquivalentTransforms3d(device, input_size, output_size, angle_rad, axis_vector):
    print("_buildEquivalentTransforms2d", device, input_size, output_size, angle_rad * 180 / math.pi, axis_vector)
    input_center = [(x-1)/2 for x in input_size]
    output_center = [(x-1)/2 for x in output_size]

    s = math.sin(angle_rad)
    c = math.cos(angle_rad)
    c1 = 1 - c

    intrans_ary = np.array([
        [1, 0, 0, input_center[2]],
        [0, 1, 0, input_center[3]],
        [0, 0, 1, input_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    inscale_ary = np.array([
        [input_center[2], 0, 0, 0],
        [0, input_center[3], 0, 0],
        [0, 0, input_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    l, m, n = axis_vector
    scipyRotation_ary = np.array([
        [l*l*c1 +   c, m*l*c1 - n*s, n*l*c1 + m*s, 0],
        [l*m*c1 + n*s, m*m*c1 +   c, n*m*c1 - l*s, 0],
        [l*n*c1 - m*s, m*n*c1 + l*s, n*n*c1 +   c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    z, y, x = axis_vector
    torchRotation_ary = np.array([
        [x*x*c1 +   c, y*x*c1 - z*s, z*x*c1 + y*s, 0],
        [x*y*c1 + z*s, y*y*c1 +   c, z*y*c1 - x*s, 0],
        [x*z*c1 - y*s, y*z*c1 + x*s, z*z*c1 +   c, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outscale_ary = np.array([
        [1/output_center[2], 0, 0, 0],
        [0, 1/output_center[3], 0, 0],
        [0, 0, 1/output_center[4], 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    outtrans_ary = np.array([
        [1, 0, 0, -output_center[2]],
        [0, 1, 0, -output_center[3]],
        [0, 0, 1, -output_center[4]],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    reorder_ary = np.array([
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
    ], dtype=np.float64)

    transform_ary = intrans_ary @ inscale_ary @ np.linalg.inv(scipyRotation_ary) @ outscale_ary @ outtrans_ary
    grid_ary = reorder_ary @ np.linalg.inv(scipyRotation_ary) @ outscale_ary @ outtrans_ary
    transform_tensor = torch.from_numpy((torchRotation_ary)).to(device, torch.float32)
    transform_tensor = transform_tensor[:3].unsqueeze(0)

    print('transform_tensor', transform_tensor.size(), transform_tensor.dtype, transform_tensor.device)
    print(transform_tensor)
    print('outtrans_ary', outtrans_ary.shape, outtrans_ary.dtype)
    print(outtrans_ary.round(3))
    print('outscale_ary', outscale_ary.shape, outscale_ary.dtype)
    print(outscale_ary.round(3))
    print('rotation_ary', scipyRotation_ary.shape, scipyRotation_ary.dtype, axis_vector, angle_rad)
    print(scipyRotation_ary.round(3))
    print('inscale_ary', inscale_ary.shape, inscale_ary.dtype)
    print(inscale_ary.round(3))
    print('intrans_ary', intrans_ary.shape, intrans_ary.dtype)
    print(intrans_ary.round(3))
    print('transform_ary', transform_ary.shape, transform_ary.dtype)
    print(transform_ary.round(3))
    print('grid_ary', grid_ary.shape, grid_ary.dtype)
    print(grid_ary.round(3))

    def prtf(pt):
        print(pt, 'transformed', (transform_ary @ (pt + [1]))[:3].round(3))

    prtf([0, 0, 0])
    prtf([1, 0, 0])
    prtf([2, 0, 0])

    print('')

    prtf([0, 0, 0])
    prtf([0, 1, 0])
    prtf([0, 2, 0])

    print('')

    prtf([0, 0, 0])
    prtf([0, 0, 1])
    prtf([0, 0, 2])

    prtf(output_center[2:])

    return transform_tensor, transform_ary, grid_ary


def test_affine_2d_rotate0(device, affine_func2d):
    input_size = [1, 1, 3, 3]
    input_ary = np.array(np.random.random(input_size), dtype=np.float32)
    output_size = [1, 1, 5, 5]
    angle_rad = 0.

    transform_tensor, transform_ary, offset = _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

    # reference
    # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    scipy_ary = scipy.ndimage.affine_transform(
        input_ary[0,0],
        transform_ary,
        offset=offset,
        output_shape=output_size[2:],
        # output=None,
        order=1,
        mode='nearest',
        # cval=0.0,
        prefilter=False)

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary)

    affine_tensor = affine_func2d(
            transform_tensor,
            torch.Size(output_size)
        )

    print('affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device)
    print(affine_tensor)

    gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border'
        ).to('cpu').numpy()

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary)
    print('gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype)
    print(gridsample_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary)

    assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
    assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
    # assert False

def test_affine_2d_rotate90(device, affine_func2d, input_size2dsq, output_size2dsq):
    input_size = input_size2dsq
    input_ary = np.array(np.random.random(input_size), dtype=np.float32)
    output_size = output_size2dsq
    angle_rad = 0.25 * math.pi * 2

    transform_tensor, transform_ary, offset = _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

    # reference
    # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    scipy_ary = scipy.ndimage.affine_transform(
        input_ary[0,0],
        transform_ary,
        offset=offset,
        output_shape=output_size[2:],
        # output=None,
        order=1,
        mode='nearest',
        # cval=0.0,
        prefilter=True)

    print('input_ary', input_ary.shape, input_ary.dtype, input_ary.mean())
    print(input_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype, scipy_ary.mean())
    print(scipy_ary)

    if input_size2dsq == output_size2dsq:
        assert np.abs(scipy_ary.mean() - input_ary.mean()) < 1e-6
    assert np.abs(scipy_ary[0,0] - input_ary[0,0,0,-1]).max() < 1e-6
    assert np.abs(scipy_ary[0,-1] - input_ary[0,0,-1,-1]).max() < 1e-6
    assert np.abs(scipy_ary[-1,-1] - input_ary[0,0,-1,0]).max() < 1e-6
    assert np.abs(scipy_ary[-1,0] - input_ary[0,0,0,0]).max() < 1e-6

    affine_tensor = affine_func2d(
            transform_tensor,
            torch.Size(output_size)
        )

    print('affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device)
    print(affine_tensor)

    gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border'
        ).to('cpu').numpy()

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary)
    print('gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype)
    print(gridsample_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary)

    assert np.abs(scipy_ary.mean() - gridsample_ary.mean()) < 1e-6
    assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
    # assert False

def test_affine_2d_rotate45(device, affine_func2d):
    input_size = [1, 1, 3, 3]
    input_ary = np.array(np.zeros(input_size), dtype=np.float32)
    input_ary[0,0,0,:] = 0.5
    input_ary[0,0,2,2] = 1.0
    output_size = [1, 1, 3, 3]
    angle_rad = 0.125 * math.pi * 2

    transform_tensor, transform_ary, offset = _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

    # reference
    # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    scipy_ary = scipy.ndimage.affine_transform(
        input_ary[0,0],
        transform_ary,
        offset=offset,
        output_shape=output_size[2:],
        # output=None,
        order=1,
        mode='nearest',
        # cval=0.0,
        prefilter=False)

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary)

    affine_tensor = affine_func2d(
            transform_tensor,
            torch.Size(output_size)
        )

    print('affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device)
    print(affine_tensor)

    gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border'
        ).to('cpu').numpy()

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary)
    print('gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype)
    print(gridsample_ary)
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary)

    assert np.abs(scipy_ary - gridsample_ary).max() < 1e-6
    # assert False

def test_affine_2d_rotateRandom(device, affine_func2d, angle_rad, input_size2d, output_size2d):
    input_size = input_size2d
    input_ary = np.array(np.random.random(input_size), dtype=np.float32).round(3)
    output_size = output_size2d

    input_ary[0,0,0,0] = 2
    input_ary[0,0,0,-1] = 4
    input_ary[0,0,-1,0] = 6
    input_ary[0,0,-1,-1] = 8

    transform_tensor, transform_ary, grid_ary = _buildEquivalentTransforms2d(device, input_size, output_size, angle_rad)

    # reference
    # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    scipy_ary = scipy.ndimage.affine_transform(
        input_ary[0,0],
        transform_ary,
        # offset=offset,
        output_shape=output_size[2:],
        # output=None,
        order=1,
        mode='nearest',
        # cval=0.0,
        prefilter=False)

    affine_tensor = affine_func2d(
            transform_tensor,
            torch.Size(output_size)
        )

    print('affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device)
    print(affine_tensor)

    for r in range(affine_tensor.size(1)):
        for c in range(affine_tensor.size(2)):
            grid_out = grid_ary @ [r, c, 1]
            print(r, c, 'affine:', affine_tensor[0,r,c], 'grid:', grid_out[:2])

    gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border'
        ).to('cpu').numpy()

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary.round(3))
    print('gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype)
    print(gridsample_ary.round(3))
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary.round(3))

    for r in range(affine_tensor.size(1)):
        for c in range(affine_tensor.size(2)):
            grid_out = grid_ary @ [r, c, 1]

            try:
                assert np.allclose(affine_tensor[0,r,c], grid_out[:2], atol=1e-5)
            except:
                print(r, c, 'affine:', affine_tensor[0,r,c], 'grid:', grid_out[:2])
                raise

    assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5
    # assert False

def test_affine_3d_rotateRandom(device, affine_func3d, angle_rad, axis_vector, input_size3d, output_size3d):
    input_size = input_size3d
    input_ary = np.array(np.random.random(input_size), dtype=np.float32)
    output_size = output_size3d

    input_ary[0,0,  0,  0,  0] = 2
    input_ary[0,0,  0,  0, -1] = 3
    input_ary[0,0,  0, -1,  0] = 4
    input_ary[0,0,  0, -1, -1] = 5
    input_ary[0,0, -1,  0,  0] = 6
    input_ary[0,0, -1,  0, -1] = 7
    input_ary[0,0, -1, -1,  0] = 8
    input_ary[0,0, -1, -1, -1] = 9

    transform_tensor, transform_ary, grid_ary = _buildEquivalentTransforms3d(device, input_size, output_size, angle_rad, axis_vector)

    # reference
    # https://stackoverflow.com/questions/20161175/how-can-i-use-scipy-ndimage-interpolation-affine-transform-to-rotate-an-image-ab
    scipy_ary = scipy.ndimage.affine_transform(
        input_ary[0,0],
        transform_ary,
        # offset=offset,
        output_shape=output_size[2:],
        # output=None,
        order=1,
        mode='nearest',
        # cval=0.0,
        prefilter=False)

    affine_tensor = affine_func3d(
            transform_tensor,
            torch.Size(output_size)
        )

    print('affine_tensor', affine_tensor.size(), affine_tensor.dtype, affine_tensor.device)
    print(affine_tensor)

    for i in range(affine_tensor.size(1)):
        for r in range(affine_tensor.size(2)):
            for c in range(affine_tensor.size(3)):
                grid_out = grid_ary @ [i, r, c, 1]
                print(i, r, c, 'affine:', affine_tensor[0,i,r,c], 'grid:', grid_out[:3].round(3))

    print('input_ary', input_ary.shape, input_ary.dtype)
    print(input_ary.round(3))

    gridsample_ary = torch.nn.functional.grid_sample(
            torch.tensor(input_ary, device=device).to(device),
            affine_tensor,
            padding_mode='border'
        ).to('cpu').numpy()

    print('gridsample_ary', gridsample_ary.shape, gridsample_ary.dtype)
    print(gridsample_ary.round(3))
    print('scipy_ary', scipy_ary.shape, scipy_ary.dtype)
    print(scipy_ary.round(3))

    for i in range(affine_tensor.size(1)):
        for r in range(affine_tensor.size(2)):
            for c in range(affine_tensor.size(3)):
                grid_out = grid_ary @ [i, r, c, 1]
                try:
                    assert np.allclose(affine_tensor[0,i,r,c], grid_out[:3], atol=1e-5)
                except:
                    print(i, r, c, 'affine:', affine_tensor[0,i,r,c], 'grid:', grid_out[:3].round(3))
                    raise

    assert np.abs(scipy_ary - gridsample_ary).max() < 1e-5
    # assert False
