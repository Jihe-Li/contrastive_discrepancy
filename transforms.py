import torch
import pytorch3d.transforms as T

from utils import make_coords_3D


class Affine:
    """An element ``g`` of the affine group acting on the image domain.

    Applying ``g`` resamples the volume on a rotated and rescaled canonical grid,
    so that a structure sitting at position ``p`` of the input appears at
    ``R (s * (p - c)) + c`` in the output, where ``c`` is the volume centre.
    Landmarks are mapped with the same convention, which keeps the transformed
    moving image and its landmarks consistent.

    Args:
        arr_shape: volume shape in ``(z, y, x)`` order.
        scale: isotropic factor, or one factor per ``(x, y, z)`` axis.
        rotation: degrees, scalar or per ``(x, y, z)`` axis, read as an
            axis-angle vector. A scalar therefore rotates about ``(1, 1, 1)``.
        jitter: relative magnitude of additive Gaussian intensity noise.
    """

    def __init__(self, arr_shape, scale=None, rotation=None, jitter=None,
                 intensity_scale=1250.0, seed=2021):
        self.jitter = jitter
        self.intensity_scale = intensity_scale
        self.seed = seed
        self.img_size = torch.FloatTensor(list(arr_shape[::-1]))

        self.scale = None
        if scale is not None:
            self.scale = torch.as_tensor(scale, dtype=torch.float32).expand(3).clone()

        self.rot_mat = None
        if rotation is not None:
            axis_angle = torch.as_tensor(rotation, dtype=torch.float32).expand(3).clone()
            self.rot_mat = T.axis_angle_to_matrix(axis_angle * torch.pi / 180.0)

        self.resample = self.scale is not None or self.rot_mat is not None
        self.grid = self._make_grid(arr_shape) if self.resample else None

    def _make_grid(self, arr_shape):
        """Canonical grid pulled back through ``g``, ready for ``grid_sample``."""
        grid = make_coords_3D(arr_shape)
        if self.rot_mat is not None:
            grid = torch.einsum("ft,dhwf->dhwt", self.rot_mat, grid)  # R^T x
        if self.scale is not None:
            grid = grid / self.scale
        return grid

    def augment_landmarks(self, marks):
        """Map 1-indexed landmarks through the same transform."""
        center = (self.img_size.to(marks) + 1) / 2.0
        marks = marks - center
        if self.scale is not None:
            marks = marks * self.scale.to(marks)
        if self.rot_mat is not None:
            marks = marks @ self.rot_mat.to(marks).T
        return marks + center

    def __call__(self, tensor, mode="bilinear", marks=None):
        out = tensor
        if self.jitter:
            gen = torch.Generator(device=out.device).manual_seed(self.seed)
            noise = torch.randn(out.shape, generator=gen, device=out.device)
            out = out + noise * self.intensity_scale * self.jitter

        if self.resample:
            dtype = out.dtype
            out = torch.nn.functional.grid_sample(
                out.float()[None, None], self.grid.to(out.device)[None],
                mode=mode, align_corners=False,
            )[0, 0].to(dtype)

        if marks is not None:
            return out, self.augment_landmarks(marks)
        return out


class Identity(Affine):
    """The neutral element; useful as a sanity check (CD should vanish)."""

    def __init__(self, arr_shape, **kwargs):
        super().__init__(arr_shape)
