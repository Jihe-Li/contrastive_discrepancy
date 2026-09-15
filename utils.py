import csv
from functools import partial
from pathlib import Path

import torch
from omegaconf import OmegaConf


def create(module, cfg, *args, **kwargs):
    """Instantiate ``module.<cfg.type>`` from a config node."""
    cls_ = getattr(module, cfg.type)
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    params.update(kwargs)
    return cls_(*args, **params)


def find(module, cfg):
    """Bind ``module.<cfg.type>`` to the remaining keys of a config node."""
    func = getattr(module, cfg.type)
    params = OmegaConf.to_container(cfg, resolve=True)
    del params["type"]
    return partial(func, **params)


def make_coords(shape, mask=None):
    """Voxel-centre coordinates in ``[-1, 1]``, flattened to ``[N, 3]`` as (x, y, z)."""
    coords = make_coords_3D(shape).view(-1, 3)
    if mask is not None:
        coords = coords[mask.flatten()]
    return coords


def make_coords_3D(shape):
    """Voxel-centre coordinates in ``[-1, 1]`` on the full grid, as (x, y, z)."""
    coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in shape]
    coords = torch.meshgrid(*coords, indexing="ij")
    return torch.stack(coords[::-1], dim=-1)


def append_csv(path, row):
    """Append one result row, writing a header if the file does not exist yet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def get_by_path(cfg, path):
    """Read a dotted config path, e.g. ``network.num_gaussians``."""
    node = cfg
    for key in path.split("."):
        node = node[key]
    return node
