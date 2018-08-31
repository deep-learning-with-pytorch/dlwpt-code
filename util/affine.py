import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.backends.cudnn as cudnn

from util.logconf import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
# log.setLevel(logging.INFO)
log.setLevel(logging.DEBUG)


def affine_grid_generator(theta, size):
    if theta.data.is_cuda and len(size) == 4:
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        if not cudnn.is_acceptable(theta.data):
            raise RuntimeError("AffineGridGenerator generator theta not acceptable for CuDNN")
        N, C, H, W = size
        return torch.cudnn_affine_grid_generator(theta, N, C, H, W)
    else:
        return AffineGridGenerator.apply(theta, size)

class AffineGridGenerator(Function):
    @staticmethod
    def _enforce_cudnn(input):
        if not cudnn.enabled:
            raise RuntimeError("AffineGridGenerator needs CuDNN for "
                               "processing CUDA inputs, but CuDNN is not enabled")
        assert cudnn.is_acceptable(input)

    @staticmethod
    def forward(ctx, theta, size):
        assert type(size) == torch.Size

        if len(size) == 5:
            N, C, D, H, W = size
            ctx.size = size
            ctx.is_cuda = theta.is_cuda
            base_grid = theta.new(N, D, H, W, 4)

            w_points = (torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1]))
            h_points = (torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])).unsqueeze(-1)
            d_points = (torch.linspace(-1, 1, D) if D > 1 else torch.Tensor([-1])).unsqueeze(-1).unsqueeze(-1)

            base_grid[:, :, :, :, 0] = w_points
            base_grid[:, :, :, :, 1] = h_points
            base_grid[:, :, :, :, 2] = d_points
            base_grid[:, :, :, :, 3] = 1
            ctx.base_grid = base_grid
            grid = torch.bmm(base_grid.view(N, D * H * W, 4), theta.transpose(1, 2))
            grid = grid.view(N, D, H, W, 3)

        elif len(size) == 4:
            N, C, H, W = size
            ctx.size = size
            if theta.is_cuda:
                AffineGridGenerator._enforce_cudnn(theta)
                assert False
            ctx.is_cuda = False
            base_grid = theta.new(N, H, W, 3)
            linear_points = torch.linspace(-1, 1, W) if W > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(base_grid[:, :, :, 0])
            linear_points = torch.linspace(-1, 1, H) if H > 1 else torch.Tensor([-1])
            base_grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(base_grid[:, :, :, 1])
            base_grid[:, :, :, 2] = 1
            ctx.base_grid = base_grid
            grid = torch.bmm(base_grid.view(N, H * W, 3), theta.transpose(1, 2))
            grid = grid.view(N, H, W, 2)
        else:
            raise RuntimeError("AffineGridGenerator needs 4d (spatial) or 5d (volumetric) inputs.")

        return grid

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_grid):
        if len(ctx.size) == 5:
            N, C, D, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, D, H, W, 3])
            assert ctx.is_cuda == grad_grid.is_cuda
            # if grad_grid.is_cuda:
            #     AffineGridGenerator._enforce_cudnn(grad_grid)
            #     assert False
            base_grid = ctx.base_grid
            grad_theta = torch.bmm(
                base_grid.view(N, D * H * W, 4).transpose(1, 2),
                grad_grid.view(N, D * H * W, 3))
            grad_theta = grad_theta.transpose(1, 2)
        elif len(ctx.size) == 4:
            N, C, H, W = ctx.size
            assert grad_grid.size() == torch.Size([N, H, W, 2])
            assert ctx.is_cuda == grad_grid.is_cuda
            if grad_grid.is_cuda:
                AffineGridGenerator._enforce_cudnn(grad_grid)
                assert False
            base_grid = ctx.base_grid
            grad_theta = torch.bmm(
                base_grid.view(N, H * W, 3).transpose(1, 2),
                grad_grid.view(N, H * W, 2))
            grad_theta = grad_theta.transpose(1, 2)
        else:
            assert False

        return grad_theta, None
