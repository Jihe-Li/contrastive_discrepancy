from types import SimpleNamespace

import torch
import torch.nn.functional as F
import pytorch3d.ops as ops
from torch import nn
from tqdm import tqdm


class WarpField(nn.Module):
    """Interface shared by every deformation model in this framework.

    A model maps normalised coordinates ``[N, 3]`` in ``[-1, 1]`` to a
    displacement field ``[N, 3]`` expressed in the same normalised units.
    Implementing :meth:`forward` and :meth:`trained_parameters` is enough; the
    remaining hooks are optional and let a model change its own capacity while
    the registration runs.
    """

    def reinitialize(self, mask: torch.Tensor, generator: torch.Generator = None):
        """Re-instantiate the parameters given the ROI mask of the fixed image.

        ``generator`` carries this model's own randomness; using it instead of the
        global RNG keeps the initialisation independent of how many other objects
        were constructed first.
        """

    def trained_parameters(self, lr):
        """Return optimiser groups, each carrying a ``name`` and an ``lr``.

        The names are the keys looked up in ``configs/optimizer/*.yaml``.
        """
        return [{"params": list(self.parameters()), "name": "default", "lr": lr.default}]

    def adaptive_control(self, step, max_steps, optimizer):
        """Called after ``backward`` and before ``step`` to grow or prune capacity."""

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussianWarp(WarpField):
    """GaussianDIR: a displacement field blended from Gaussian primitives.

    Each primitive carries a translation (optionally a rotation) and a radius, and
    a query point takes the normalised Gaussian-weighted average of its ``K``
    nearest primitives. Model complexity is governed by ``num_gaussians``, the
    hyperparameter that Contrastive Discrepancy selects at testing time.
    """

    def __init__(self, num_gaussians=5000, K=20, sparsification=8,
                 with_quaternion=True, anisotropy=None, densify=None, **kwargs):
        super().__init__()
        self.num_gaussians = num_gaussians
        self.K = K
        self.sparsification = sparsification
        self.with_quaternion = with_quaternion
        self.anisotropy = SimpleNamespace(**(anisotropy or {}))
        self.densify = SimpleNamespace(**(densify or {"enabled": False}))
        self.max_nodes = num_gaussians

        self._build(*self._lattice_nodes())

    # -------------------------------------------------------------- lifecycle

    @property
    def _radius(self):
        """Initial kernel radius, the half-diagonal of a primitive's share of the volume."""
        return torch.tensor(3.0).sqrt() / self.num_gaussians ** (1 / 3)

    def _lattice_nodes(self):
        """A deterministic placeholder, so that constructing the model draws no RNG.

        ``reinitialize`` replaces these with ROI-restricted random samples. Keeping
        ``__init__`` free of random draws makes a model's initialisation depend only
        on the seed, not on how many other objects were constructed before it.
        """
        side = max(round(self.num_gaussians ** (1 / 3)), 1)
        axis = torch.linspace(-1, 1, side + 1)[:-1] + 1 / side
        grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1)
        return grid.reshape(-1, 3), self._radius

    def _sample_nodes(self, mask, generator=None):
        """Draw primitives uniformly in ``[-1, 1]^3``, keeping those inside the ROI."""
        positions = torch.rand(1, 1, 1, self.num_gaussians, 3, generator=generator) * 2.0 - 1.0
        inside = F.grid_sample(mask[None, None].cpu().float(), positions,
                               mode="nearest", align_corners=False).squeeze().bool()
        return positions.reshape(-1, 3)[inside], self._radius

    def _build(self, positions, radius):
        num = positions.shape[0]
        log_radius = torch.log(radius)
        identity_quat = torch.cat([torch.ones(num, 1), torch.zeros(num, 3)], dim=1)

        self.node_position = nn.Parameter(positions)
        self.translation = nn.Parameter(torch.zeros(num, 3))
        if self.anisotropy.quaternion:
            self.node_quaternion = nn.Parameter(identity_quat.clone())
        if self.anisotropy.scaling:
            self._node_scaling = nn.Parameter(torch.full((num, 3), log_radius.item()))
        else:
            self._node_radius = nn.Parameter(torch.full((num, 1), log_radius.item()))
        if self.with_quaternion:
            self.quaternion = nn.Parameter(identity_quat.clone())
        self.reset_stats()

    def reinitialize(self, mask, generator=None):
        self.generator = generator
        self._build(*self._sample_nodes(mask, generator))
        self.max_nodes = int(mask.sum()) // (self.sparsification ** 3)
        print(f"[GaussianWarp] {self.node_position.shape[0]} primitives inside the ROI "
              f"(sampled {self.num_gaussians}, densification cap {self.max_nodes}).")

    def reset_stats(self):
        num = self.node_position.shape[0]
        device = self.node_position.device
        self.register_buffer("contribution", torch.zeros(num, device=device),
                             persistent=False)
        self.register_buffer("counter", torch.zeros(num, device=device),
                             persistent=False)

    def trained_parameters(self, lr):
        groups = [
            {"params": [self.node_position], "name": "node_position", "lr": lr.node_position},
            {"params": [self.translation], "name": "translation", "lr": lr.translation},
        ]
        if self.anisotropy.quaternion:
            groups.append({"params": [self.node_quaternion], "name": "node_quaternion",
                           "lr": lr.node_quaternion})
        if self.anisotropy.scaling:
            groups.append({"params": [self._node_scaling], "name": "_node_scaling",
                           "lr": lr.node_scaling})
        else:
            groups.append({"params": [self._node_radius], "name": "_node_radius",
                           "lr": lr.node_radius})
        if self.with_quaternion:
            groups.append({"params": [self.quaternion], "name": "quaternion",
                           "lr": lr.quaternion})
        return groups

    # ----------------------------------------------------------------- kernel

    @property
    def node_radius(self):
        return torch.exp(self._node_radius)

    @property
    def node_scaling(self):
        return torch.exp(self._node_scaling)

    def cal_nn_weight(self, x, nodes, K=None):
        """Normalised Gaussian weights of the ``K`` primitives nearest to ``x``."""
        K = self.K if K is None else K
        _, nn_idxs, _ = ops.knn_points(x[None], nodes[None], None, None, K=K)
        nn_idxs = nn_idxs[0]
        local_coords = x[:, None] - nodes[nn_idxs]  # [M, K, 3]

        if self.anisotropy.scaling:
            if self.anisotropy.quaternion:
                rot_matrix = self.quaternion_to_matrix(self.node_quaternion)[nn_idxs]
                local_coords = torch.matmul(local_coords.unsqueeze(-2), rot_matrix).squeeze(-2)
            exponent = torch.sum((local_coords / self.node_scaling[nn_idxs]) ** 2, dim=-1)
            nn_weight = torch.exp(-0.5 * exponent)
            nn_weight = nn_weight / (torch.prod(self.node_scaling[nn_idxs], dim=-1) + 1e-7)
        else:
            exponent = torch.sum((local_coords / self.node_radius[nn_idxs]) ** 2, dim=-1)
            nn_weight = torch.exp(-0.5 * exponent)
            nn_weight = nn_weight / (self.node_radius[nn_idxs].squeeze(-1) ** 3 + 1e-7)

        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)
        return nn_weight, nn_idxs, local_coords

    def forward(self, x):
        nn_weight, nn_idxs, local_coords = self.cal_nn_weight(x, self.node_position)
        if self.with_quaternion:
            local_coords = local_coords.detach()
            rot_matrix = self.quaternion_to_matrix(self.quaternion)[nn_idxs]
            displacement = (torch.matmul(rot_matrix, local_coords.unsqueeze(-1)).squeeze(-1)
                            + self.translation[nn_idxs] - local_coords)
        else:
            displacement = self.translation[nn_idxs]
        return (displacement * nn_weight[..., None]).sum(dim=1)

    @staticmethod
    def quaternion_to_matrix(quaternions):
        """Convert (unnormalised) quaternions to rotation matrices."""
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack((
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ), -1)
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    # ------------------------------------------------------- adaptive control

    @torch.no_grad()
    def adaptive_control(self, step, max_steps, optimizer):
        if not self.densify.enabled or step > self.densify.until_ratio * max_steps:
            return
        self.contribution += torch.norm(self.translation.grad, dim=-1)
        self.counter += 1

        interval = max(int(self.densify.interval_ratio * max_steps), 1)
        if step >= self.densify.from_iter and step % interval == 0:
            self.densify_and_prune(optimizer)

    @torch.no_grad()
    def densify_and_prune(self, optimizer):
        score = self.contribution / self.counter
        score[score.isnan()] = 0.0

        extents = self._node_scaling if self.anisotropy.scaling else self._node_radius
        invalid = extents[:, 0].isnan()
        grow = torch.logical_and(score >= self.densify.max_contribution, ~invalid)
        prune = torch.logical_or(score <= self.densify.min_contribution, invalid)

        budget = self.max_nodes - self.node_position.shape[0] + int(prune.sum())
        if int(grow.sum()) > budget:
            grow = torch.zeros_like(grow)
            if budget > 0:
                grow[torch.topk(score, budget)[1]] = True

        tqdm.write(f"\nAdd {int(grow.sum())} and prune {int(prune.sum())} primitives.")

        if grow.any():
            if self.anisotropy.scaling:
                stds = self.node_scaling[grow]
            else:
                stds = self.node_radius[grow].repeat(1, 3)
            offset = torch.normal(torch.zeros_like(stds), stds,
                                  generator=getattr(self, "generator", None))

            extensions = {
                "node_position": self.node_position[grow] + offset,
                "translation": self.translation[grow],
            }
            if self.anisotropy.quaternion:
                extensions["node_quaternion"] = self.node_quaternion[grow]
            if self.anisotropy.scaling:
                extensions["_node_scaling"] = self._node_scaling[grow]
            else:
                extensions["_node_radius"] = self._node_radius[grow]
            if self.with_quaternion:
                extensions["quaternion"] = self.quaternion[grow]
            self._extend_params(optimizer, extensions)

        if prune.any():
            keep = torch.ones(self.node_position.shape[0], dtype=torch.bool,
                              device=prune.device)
            keep[: prune.shape[0]] = ~prune
            self._prune_params(optimizer, keep)

        self.reset_stats()
        tqdm.write(f"With {self.node_position.shape[0]} primitives left.")

    def _extend_params(self, optimizer, extensions):
        """Append new rows to every parameter group and to the Adam moments."""
        for group in optimizer.param_groups:
            param = group["params"][0]
            extension = extensions[group["name"]]
            state = optimizer.state.pop(param, None)
            if state is not None:
                zeros = torch.zeros_like(extension)
                state["exp_avg"] = torch.cat((state["exp_avg"], zeros), dim=0)
                state["exp_avg_sq"] = torch.cat((state["exp_avg_sq"], zeros), dim=0)
            new = nn.Parameter(torch.cat((param.data, extension), dim=0).requires_grad_(True))
            group["params"][0] = new
            if state is not None:
                optimizer.state[new] = state
            setattr(self, group["name"], new)

    def _prune_params(self, optimizer, keep):
        """Drop rows from every parameter group and from the Adam moments."""
        for group in optimizer.param_groups:
            param = group["params"][0]
            state = optimizer.state.pop(param, None)
            if state is not None:
                state["exp_avg"] = state["exp_avg"][keep]
                state["exp_avg_sq"] = state["exp_avg_sq"][keep]
            new = nn.Parameter(param.data[keep].requires_grad_(True))
            group["params"][0] = new
            if state is not None:
                optimizer.state[new] = state
            setattr(self, group["name"], new)
