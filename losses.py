import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

# Normalized cross-correlation, adapted from the TorchIR repository by
# Bob de Vos (AmsterdamUMC): https://github.com/BDdeVos/TorchIR


class StableStd(torch.autograd.Function):
    """Standard deviation with a numerically stable gradient."""

    @staticmethod
    def forward(ctx, tensor):
        assert tensor.numel() > 1
        ctx.tensor = tensor.detach()
        ctx.result = torch.std(tensor).detach()
        return ctx.result

    @staticmethod
    def backward(ctx, grad_output):
        tensor, result = ctx.tensor, ctx.result
        e = 1e-6
        return ((2.0 / (tensor.numel() - 1.0))
                * (grad_output.detach() / (result * 2 + e))
                * (tensor - tensor.mean()))


class NCC(_Loss):
    """Negative global normalized cross-correlation."""

    def ncc(self, x1, x2, e=1e-10):
        assert x1.shape == x2.shape, "Inputs are not of similar shape"
        cc = ((x1 - x1.mean()) * (x2 - x2.mean())).mean()
        std = StableStd.apply(x1) * StableStd.apply(x2)
        return cc / (std + e)

    def forward(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return -self.ncc(fixed, warped).mean()


class MSE(_Loss):
    """Mean squared intensity difference."""

    def forward(self, fixed: Tensor, warped: Tensor) -> Tensor:
        return torch.nn.functional.mse_loss(warped, fixed)
