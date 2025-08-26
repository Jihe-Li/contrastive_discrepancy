from functools import partial

import torch
import pytorch3d.transforms as T
from omegaconf import OmegaConf


def create(module, cfg, *args, **kwargs):
    cls_ = module.__dict__[cfg.type]
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    return cls_(*args, **params)


def find(module, cfg):
    func = module.__dict__[cfg.type]
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    return partial(func, **params)

def make_coords(shape, mask=None):
    """Make a coordinate tensor."""

    coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in shape]
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords[::-1], dim=-1)
    coords = coords.view(-1, 3)

    if mask is not None:
        coords = coords[mask.flatten()]

    return coords

def make_coords_3D(shape):
    """Make a coordinate tensor."""

    coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in shape]
    coords = torch.meshgrid(*coords, indexing="ij")
    coords = torch.stack(coords[::-1], dim=-1)

    return coords

class Augment:
    def __init__(self, arr_shape, scale={'max': 1.05, 'min': 0.95}, rotation=None, jitter=None):
        self.scale = scale
        self.rotation = rotation
        self.jitter = jitter
        self.img_size = torch.FloatTensor(arr_shape[::-1])
        self.grid = self.make_coords(arr_shape)

        self.process_grid()

    def make_coords(self, arr_shape):
        coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in arr_shape]
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords[::-1], dim=-1)
        return coords

    def process_grid(self):
        if self.rotation is not None:
            rotation_xyz = (torch.ones(3) * 2. - 1.) * torch.tensor(self.rotation) / 180 * torch.pi
            self.apply_rotation(rotation_xyz)

        if self.scale is not None:
            self.scale_xyz = torch.ones(3) * (self.scale['max'] - self.scale['min']) + self.scale['min']
            self.grid = self.grid / self.scale_xyz

    def apply_rotation(self, rotation):
        self.rot_mat = T.axis_angle_to_matrix(rotation)
        self.grid = torch.einsum("ft,hdwf->hdwt", (self.rot_mat, self.grid))  # A.T dot x

    def augment_landmarks(self, marks):
        '''augment landmarks using scaling and rotation
        marks: [N, 3]
        operation: rotation -> scaling
        '''
        if self.rotation is not None:
            center = (self.img_size + 1) / 2.
            marks = (self.rot_mat @ (marks - center).T).T + center

        if self.scale is not None:
            center = (self.img_size + 1) / 2.
            marks = self.scale_xyz * (marks - center) + center
            
        return marks

    def __call__(self, tensor, mode='bilinear', marks=None):
        if self.jitter != None:
            if mode == 'bilinear':
                intensity_scale = 1250
                gen = torch.Generator(device=tensor.device)
                gen.manual_seed(2021)
                noise = torch.randn(tensor.shape, generator=gen, device=tensor.device) * intensity_scale * self.jitter
                tensor += noise

        if self.scale != None or self.rotation != None:
            dtype = tensor.dtype
            if dtype != torch.float:
                tensor = tensor.float()
            tensor = torch.nn.functional.grid_sample(tensor.unsqueeze(0).unsqueeze(0), 
                                                    self.grid.unsqueeze(0), 
                                                    mode, align_corners=False).squeeze(0).squeeze(0)
            if dtype != torch.float:
                tensor = tensor.to(dtype)

        if marks is not None:
            marks = self.augment_landmarks(marks)
            return tensor, marks
        
        return tensor
