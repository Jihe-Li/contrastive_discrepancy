import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

import utils
from loaders import load_DIRLab

OFFSET = -0.5


class RegDataset(IterableDataset):
    """This is a class for registrating implicitly represented images."""

    def __init__(self, case_idx):
        super().__init__()
        self.case_idx = case_idx
        self.batch_size = 20000
        self.neighs = 3
        self.device = torch.device("cpu")

        named_data = load_DIRLab(case_idx=self.case_idx)
        self.fix_arr = named_data["fix_arr"].unsqueeze(0).unsqueeze(0)
        self.mov_arr = named_data["mov_arr"].unsqueeze(0).unsqueeze(0)
        self.fix_mask = named_data["fix_mask"]
        self.mov_mask = named_data["mov_mask"]
        if "fix_marks" in named_data:
            self.fix_marks = named_data["fix_marks"]
        if "mov_marks" in named_data:
            self.mov_marks = named_data["mov_marks"]
        self.params = named_data["params"]

        self.voxel_size = torch.FloatTensor(self.params["spacing"])
        self.image_size = torch.FloatTensor(self.params["size"])

        self.batch_center_num = self.batch_size // (self.neighs + 1)
        self.voxel_units = torch.eye(3)[None] * (2 / self.image_size)

        self.masked_coords = utils.make_coords(self.fix_mask.shape, self.fix_mask)
        self.shuffle()

    @property
    def shape(self):
        return self.fix_arr.shape[-3:]

    def arr2nii(self, array, save_path="", params=None):
        if params is None:
            params = self.params
        direction = params.get("direction", [1, 0, 0, 0, 1, 0, 0, 0, 1])
        origin = params.get("origin", [0, 0, 0])
        spacing = params.get("spacing", self.voxel_size.tolist())

        image = sitk.GetImageFromArray(array)
        image.SetDirection(direction)
        image.SetOrigin(origin)
        image.SetSpacing(spacing)

        if save_path:
            sitk.WriteImage(image, save_path)
        return image

    def abs2rel(self, coords):
        return 2 * (coords + OFFSET) / self.image_size - 1.0

    def rel2abs(self, coords):
        return (coords + 1.0) * self.image_size / 2 - OFFSET

    def abs2phys(self, coords):
        return coords * self.voxel_size

    def shuffle(self):
        self.indices = torch.randperm(self.masked_coords.shape[0], device=self.device)
        self.iter_self = iter(range(0, len(self.indices), self.batch_center_num))

    def to(self, device):
        self.device = device
        self.fix_arr = self.fix_arr.to(device)
        self.mov_arr = self.mov_arr.to(device)
        self.fix_mask = self.fix_mask.to(device)
        self.mov_mask = self.mov_mask.to(device)
        if hasattr(self, "fix_marks"):
            self.fix_marks = self.fix_marks.to(device)
        if hasattr(self, "mov_marks"):
            self.mov_marks = self.mov_marks.to(device)
        if hasattr(self, "voxel_units"):
            self.voxel_units = self.voxel_units.to(device)

        self.indices = self.indices.to(device)
        self.voxel_size = self.voxel_size.to(device)
        self.image_size = self.image_size.to(device)
        self.masked_coords = self.masked_coords.to(device)

        return self

    def __iter__(self):
        while True:
            try:
                idx = next(self.iter_self)
                coords = self.masked_coords[self.indices[idx: idx + self.batch_center_num]]
                neigh_coords = coords[:, None] + self.voxel_units
                neigh_coords = neigh_coords.reshape(-1, 3)
                coords = torch.concat([coords, neigh_coords], dim=0)
                
                yield coords

            except StopIteration:
                self.shuffle()
                continue

    def __len__(self):
        return len(self.indices)

    def masked_gather(self, tensor, mask=None, is_flow=False):
        if is_flow:
            full_size = self.shape + (3,)
            full_tensor = torch.zeros(full_size, device=self.device)
        else:
            full_size = self.shape
            full_tensor = torch.ones(full_size, device=self.device) * -1000
        if mask is None:
            mask = self.fix_mask
        full_tensor[mask] = tensor
        return full_tensor

    def __getitem__(self, index):
        return self.masked_coords[index]

    def _sampling(self, coords, tensor, mode="bilinear"):
        coords = coords.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return (
            F.grid_sample(tensor, coords, mode=mode, align_corners=False)
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
            .squeeze(0)
        )

    def samp_fix(self, coords):
        return self._sampling(coords, self.fix_arr)

    def samp_mov(self, coords):
        return self._sampling(coords, self.mov_arr)


class CDDataset(RegDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        aug = utils.Augment(self.params["size"][::-1])
        self.aug_mov_arr = aug(self.mov_arr.squeeze(0).squeeze(0)).unsqueeze(0).unsqueeze(0) 
        if hasattr(self, "mov_mask"):
            self.aug_mov_mask = aug(self.mov_mask, mode="nearest")
        if hasattr(self, "mov_marks"):
            self.aug_mov_marks = aug.augment_landmarks(self.mov_marks)

    def to(self, device):
        super().to(device)
        self.aug_mov_arr = self.aug_mov_arr.to(device)
        if hasattr(self, "mov_mask"):
            self.aug_mov_mask = self.aug_mov_mask.to(device)
        if hasattr(self, "mov_marks"):
            self.aug_mov_marks = self.aug_mov_marks.to(device)

        return self

    def samp_aug_mov(self, coords):
        return self._sampling(coords, self.aug_mov_arr)
