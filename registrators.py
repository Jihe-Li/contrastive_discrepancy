import math

import torch
import torch.optim as optim
from tqdm import tqdm

import datasets
import losses
import metrics
import networks
import transforms
import utils


class CDRegistrator:
    """Computes Contrastive Discrepancy for one fixed/moving pair.

    Up to three registrations are run with the same model and hyperparameters:

    ``a``         ``F -> M_a``, the observed moving image;
    ``b``         ``F -> M_b`` with ``M_b = g * M_a``, a second observation drawn
                  from the same anatomical orbit;
    ``residual``  ``M_a o phi_a -> M_b o phi_b``, needed only for CD3.

    Both warped results live in the coordinate frame of the fixed image, which is
    what makes the two deformation fields comparable. CD2 is the masked MAE
    between them; CD3 is the mean norm of the residual field.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.device = self._resolve_device(cfg.device)
        self.wanted = {"CD2", "CD3"} if cfg.metric == "both" else {cfg.metric}
        self.seeds = self._derive_seeds(cfg)

        self.dataset = utils.create(datasets, cfg.datasets, seed=self.seeds["dataset"])
        self.transform = utils.create(transforms, cfg.transform, self.dataset.shape)
        self.dataset.build_orbit_sample(self.transform).to(self.device)

        self.criterion = getattr(losses, cfg.similarity)().to(self.device)
        self.branches = {}
        self.base_lr = None

    # ----------------------------------------------------------------- set-up

    @staticmethod
    def _derive_seeds(cfg):
        """One independent seed per component, all determined by ``cfg.seed``.

        Every source of randomness draws from its own generator, so the result
        depends on the seed alone and not on the order in which the dataset and
        the three branches happen to be constructed.
        """
        master = torch.Generator().manual_seed(cfg.seed)
        dataset, a, b, residual = torch.randint(0, 2 ** 31 - 1, (4,),
                                                generator=master).tolist()
        return dict(dataset=dataset, a=a, b=a if cfg.paired_init else b,
                    residual=residual)

    @staticmethod
    def _resolve_device(name):
        if name != "cpu" and not torch.cuda.is_available():
            print("[CDRegistrator] CUDA is unavailable, falling back to CPU.")
            return torch.device("cpu")
        return torch.device(name)

    def branch(self, name):
        """Return the (network, optimizer) pair of a branch, building it on first use."""
        if name in self.branches:
            return self.branches[name]

        generator = torch.Generator().manual_seed(self.seeds[name])
        network = utils.create(networks, self.cfg.network)
        network.reinitialize(self.dataset.fix_mask, generator)
        network.to(self.device)
        optimizer_cls = getattr(optim, self.cfg.optimizer.type)
        optimizer = optimizer_cls(network.trained_parameters(self.cfg.optimizer.lr))

        if self.base_lr is None:
            self.base_lr = {g["name"]: g["lr"] for g in optimizer.param_groups}
        self.branches[name] = (network, optimizer)
        return self.branches[name]

    def _set_lr(self, optimizer, step):
        """Linear warm-up followed by cosine decay, per parameter group."""
        warmup = max(self.cfg.optimizer.schedule.warmup_steps, 1)
        total = self.cfg.max_steps
        for group in optimizer.param_groups:
            base = self.base_lr[group["name"]]
            if step <= warmup:
                group["lr"] = step / warmup * base
            else:
                ratio = (step - warmup) / (total + 1 - warmup)
                group["lr"] = (math.cos(math.pi * ratio) + 1) / 2 * base

    def tv_regularizer(self, flow):
        """Mean displacement difference between each centre and its neighbours."""
        num_centers = flow.shape[0] // (self.dataset.neighs + 1)
        center = flow[:num_centers]
        neighs = flow[num_centers:].reshape(-1, self.dataset.neighs, 3)
        return torch.norm(neighs - center[:, None], dim=-1).mean()

    # ----------------------------------------------------------- registration

    def optimize(self, branch, samp_fix, samp_mov, desc):
        """Run one full registration of ``samp_mov`` onto ``samp_fix``."""
        network, optimizer = self.branch(branch)
        network.train()
        self.dataset.reseed(self.seeds[branch])
        iterator = iter(self.dataset)

        for step in tqdm(range(1, self.cfg.max_steps + 1), ncols=80, desc=desc):
            self._set_lr(optimizer, step)
            coords = next(iterator)
            with torch.no_grad():
                fix_val = samp_fix(coords)

            optimizer.zero_grad()
            flow = network(coords)
            warp_val = samp_mov(coords + flow)

            loss = self.criterion(warp_val, fix_val)
            if self.cfg.lambda_tv > 0 and self.dataset.neighs > 0:
                loss = loss + self.cfg.lambda_tv * self.tv_regularizer(flow)
            loss.backward()

            network.adaptive_control(step, self.cfg.max_steps, optimizer)
            optimizer.step()
        return network

    def _chunks(self):
        for i in range(0, len(self.dataset), self.cfg.chunk_size):
            yield self.dataset[i: i + self.cfg.chunk_size]

    @torch.no_grad()
    def warp_volume(self, branch, samp_mov):
        """Warp the moving image onto the fixed-image grid."""
        network, _ = self.branch(branch)
        network.eval()
        values = [samp_mov(coords + network(coords)) for coords in self._chunks()]
        return self.dataset.scatter_to_volume(torch.cat(values, dim=0))

    @torch.no_grad()
    def residual_norm(self, branch):
        """Mean norm of the residual deformation field over the ROI."""
        network, _ = self.branch(branch)
        network.eval()
        scale = self.dataset.image_size / 2
        if self.cfg.cd3_units == "mm":
            scale = scale * self.dataset.voxel_size

        total, count = 0.0, 0
        for coords in self._chunks():
            flow = network(coords) * scale
            total += metrics.field_norm(flow).item() * flow.shape[0]
            count += flow.shape[0]
        return total / count

    @torch.no_grad()
    def landmark_error(self, branch):
        """Reference TRE in mm; returns ``None`` when the dataset has no landmarks."""
        if not self.dataset.has_landmarks:
            return None
        network, _ = self.branch(branch)
        network.eval()

        coords = self.dataset.abs2rel(self.dataset.fix_marks)
        warp_marks = self.dataset.rel2abs(coords + network(coords))
        warp_marks = torch.round(warp_marks)  # DIRLab convention: voxel-level TRE
        return metrics.compute_landmark_accuracy(
            self.dataset.abs2phys(warp_marks),
            self.dataset.abs2phys(self.dataset.mov_marks),
        )

    # ------------------------------------------------------------------- main

    def run(self):
        """Register both orbit observations and return the requested CD values."""
        self.optimize("a", self.dataset.samp_fix, self.dataset.samp_mov, "F->Ma")
        tre = self.landmark_error("a")
        warp_a = self.warp_volume("a", self.dataset.samp_mov)

        self.optimize("b", self.dataset.samp_fix, self.dataset.samp_aug_mov, "F->Mb")
        warp_b = self.warp_volume("b", self.dataset.samp_aug_mov)

        result = {}
        if "CD2" in self.wanted:
            mask = self.dataset.fix_mask
            result["CD2"] = (warp_a[mask] - warp_b[mask]).abs().mean().item()

        if "CD3" in self.wanted:
            self.optimize("residual", self.dataset.sampler(warp_a),
                          self.dataset.sampler(warp_b), "Ma->Mb")
            result["CD3"] = self.residual_norm("residual")

        if tre is not None:
            result["TRE"] = tre["mean"]
            result["TRE_std"] = tre["std"]
        return result

    def summary(self, result):
        fields = ", ".join(f"{k}: {v:.4f}" for k, v in result.items() if k != "TRE_std")
        return f"{self.dataset.name} | {fields}"
