import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset

import loaders
import utils

OFFSET = -0.5


class RegDataset(IterableDataset):
    """Base class for coordinate-sampled deformable image registration.

    The image pair is represented implicitly: each iteration yields a batch of
    normalised coordinates drawn from the ROI of the fixed image, each centre
    followed by ``neighs`` axis neighbours used by the TV regulariser.

    A subclass only has to implement :meth:`load`, returning the fixed and moving
    volumes, their ROI masks, the geometry of the fixed image and -- optionally --
    landmark pairs used to report a reference TRE.
    """

    _TENSORS = ("fix_arr", "mov_arr", "fix_mask", "mov_mask", "fix_marks",
                "mov_marks", "aug_mov_arr", "aug_mov_marks", "voxel_units",
                "voxel_size", "image_size", "masked_coords", "indices")

    def __init__(self, batch_size=20000, neighs=3, fill_value=-1000.0, seed=0):
        super().__init__()
        self.batch_size = batch_size
        self.neighs = neighs
        self.fill_value = fill_value
        self.device = torch.device("cpu")
        self.generator = torch.Generator().manual_seed(seed)

        data = self.load()
        self.fix_arr = data["fix_arr"][None, None]
        self.mov_arr = data["mov_arr"][None, None]
        self.fix_mask = data["fix_mask"]
        self.mov_mask = data["mov_mask"]
        self.fix_marks = data.get("fix_marks")
        self.mov_marks = data.get("mov_marks")
        self.params = data["params"]
        self.aug_mov_arr = None
        self.aug_mov_marks = None

        self.voxel_size = torch.FloatTensor(self.params["spacing"])
        self.image_size = torch.FloatTensor(self.params["size"])
        self.batch_center_num = max(self.batch_size // (self.neighs + 1), 1)
        self.voxel_units = torch.eye(3)[None] * (2 / self.image_size)
        self.masked_coords = utils.make_coords(self.fix_mask.shape, self.fix_mask)
        self.shuffle()

    # ------------------------------------------------------------------ hooks

    def load(self):
        """Return a dict with fix/mov volumes, masks, landmarks and geometry."""
        raise NotImplementedError

    @property
    def name(self):
        return type(self).__name__

    # ------------------------------------------------------------------ setup

    @property
    def shape(self):
        return self.fix_arr.shape[-3:]

    @property
    def has_landmarks(self):
        return self.fix_marks is not None and self.mov_marks is not None

    def build_orbit_sample(self, transform):
        """Create the second orbit observation ``M_b = g * M_a``."""
        self.aug_mov_arr = transform(self.mov_arr[0, 0])[None, None]
        if self.mov_marks is not None:
            self.aug_mov_marks = transform.augment_landmarks(self.mov_marks)
        return self

    def to(self, device):
        self.device = torch.device(device)
        for name in self._TENSORS:
            tensor = getattr(self, name, None)
            if torch.is_tensor(tensor):
                setattr(self, name, tensor.to(self.device))
        return self

    # ------------------------------------------------------------ coordinates

    def abs2rel(self, coords):
        return 2 * (coords + OFFSET) / self.image_size - 1.0

    def rel2abs(self, coords):
        return (coords + 1.0) * self.image_size / 2 - OFFSET

    def abs2phys(self, coords):
        return coords * self.voxel_size

    # --------------------------------------------------------------- sampling

    def reseed(self, seed):
        """Restart the sampling stream, so a registration always sees the same order.

        Called once per registration, which keeps the coordinate ordering a
        function of that branch's seed rather than of how many registrations
        happened to run before it.
        """
        self.generator.manual_seed(seed)
        self.shuffle()

    def shuffle(self):
        # Drawn from the dataset's own generator so that sampling order never
        # perturbs -- and is never perturbed by -- model initialisation.
        perm = torch.randperm(self.masked_coords.shape[0], generator=self.generator)
        self.indices = perm.to(self.device)
        self.iter_self = iter(range(0, len(self.indices), self.batch_center_num))

    def __iter__(self):
        while True:
            try:
                idx = next(self.iter_self)
            except StopIteration:
                self.shuffle()
                continue
            coords = self.masked_coords[self.indices[idx: idx + self.batch_center_num]]
            neigh_coords = (coords[:, None] + self.voxel_units).reshape(-1, 3)
            yield torch.concat([coords, neigh_coords], dim=0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.masked_coords[index]

    def scatter_to_volume(self, values, mask=None, is_flow=False):
        """Scatter ROI values back onto the full grid, padding with ``fill_value``."""
        mask = self.fix_mask if mask is None else mask
        if is_flow:
            volume = torch.zeros(self.shape + (3,), device=self.device)
        else:
            volume = torch.full(self.shape, self.fill_value, device=self.device)
        volume[mask] = values
        return volume

    def _sampling(self, coords, tensor, mode="bilinear"):
        coords = coords[None, None, None]
        return F.grid_sample(tensor, coords, mode=mode, align_corners=False).flatten()

    def samp_fix(self, coords):
        return self._sampling(coords, self.fix_arr)

    def samp_mov(self, coords):
        return self._sampling(coords, self.mov_arr)

    def samp_aug_mov(self, coords):
        return self._sampling(coords, self.aug_mov_arr)

    def sampler(self, volume):
        """Build a sampling callable for an arbitrary volume of shape ``shape``."""
        volume = volume[None, None]
        return lambda coords: self._sampling(coords, volume)


class DIRLab(RegDataset):
    """DIRLab 4DCT: 10 lung cases with 300 annotated landmarks each."""

    def __init__(self, root="data/DIRLab", case_idx=1, fix_phase=0, mov_phase=5,
                 mask_folder="Lungs", **kwargs):
        self.root = root
        self.case_idx = case_idx
        self.fix_phase = fix_phase
        self.mov_phase = mov_phase
        self.mask_folder = mask_folder
        super().__init__(**kwargs)

    @property
    def name(self):
        return f"DIRLab-Case{self.case_idx}"

    def load(self):
        return loaders.load_DIRLab(self.root, self.case_idx, self.fix_phase,
                                   self.mov_phase, self.mask_folder)


class NiftiPair(RegDataset):
    """A single fixed/moving pair given by explicit file paths."""

    def __init__(self, fix_image, mov_image, fix_mask=None, mov_mask=None,
                 fix_marks=None, mov_marks=None, name="pair", **kwargs):
        self.paths = dict(fix_image=fix_image, mov_image=mov_image,
                          fix_mask=fix_mask, mov_mask=mov_mask,
                          fix_marks=fix_marks, mov_marks=mov_marks)
        self._name = name
        super().__init__(**kwargs)

    @property
    def name(self):
        return self._name

    def load(self):
        return loaders.load_pair(**self.paths)
