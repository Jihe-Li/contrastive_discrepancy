import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from datasets import CDDataset
import losses
import networks
import utils
import metrics


class CDRegistrator:
    def __init__(self, cfg):
        self.chunk_size = cfg.chunk_size
        self.max_steps = cfg.max_steps
        self.warmup_steps = cfg.warmup_steps
        torch.manual_seed(cfg.seed)

        self.enable_densify = cfg.enable_densify
        if self.enable_densify:
            self.densify_from_iter = cfg.densify_from_iter
            self.densify_until_ratio = cfg.densify_until_ratio
            self.densify_interval_ratio = cfg.densify_interval_ratio

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = CDDataset(cfg.case_idx)
        self.dataset.to(self.device)
        self.shape = self.dataset.shape

        self.criterion = losses.NCC().to(self.device)
        self.lambda_tv = 8

        self.network: nn.Module = utils.create(networks, cfg.network)
        self.network.reinitialize(self.dataset.fix_mask) 
        self.network.to(self.device)

        self.network2: networks.GaussianWarp = utils.create(networks, cfg.network)
        self.network2.reinitialize(self.dataset.fix_mask)
        self.network2.to(self.device)

        self.network3: networks.GaussianWarp = utils.create(networks, cfg.network)
        self.network3.reinitialize(self.dataset.fix_mask)
        self.network3.to(self.device)

        self.optimizer =  optim.Adam(self.network.trained_parameters(cfg.lr))
        self.optimizer2 = optim.Adam(self.network2.trained_parameters(cfg.lr))
        self.optimizer3 = optim.Adam(self.network3.trained_parameters(cfg.lr))

        self.lr = {}
        for group in self.optimizer.param_groups:
            self.lr[group["name"]] = group["lr"]

    @torch.no_grad()
    def adaptive_control(self, step, network, optimizer):
        if self.enable_densify and step <= self.densify_until_ratio * self.max_steps \
                and self.densify_from_iter < self.max_steps * self.densify_until_ratio:  
            network.add_densification_stats()   

            if step >= self.densify_from_iter and step % (self.densify_interval_ratio * self.max_steps) == 0:
                network.densify_and_prune(optimizer=optimizer)  

    def tv_regulizer(self, acc_flow):
        center_flow = acc_flow[:acc_flow.shape[0] // (self.dataset.neighs + 1)]
        neighs_flow = acc_flow[acc_flow.shape[0] // (self.dataset.neighs + 1):].reshape(-1, self.dataset.neighs, 3)
        diff_norm = torch.norm(neighs_flow - center_flow[:, None], dim=-1)
        return diff_norm.mean()

    def get_cur_lr(self, step, lr):
        if step <= self.warmup_steps:
            cur_lr = step / self.warmup_steps * lr
        else:
            ratio = (step - self.warmup_steps) / (self.max_steps + 1 - self.warmup_steps)
            cur_lr = (math.cos(math.pi * ratio) + 1) / 2 * lr
        return cur_lr

    def train_step(self, step):
        """Perform one iteration of training."""
        self.optimizer.zero_grad()
        coords = next(self.dataset_iter)
        with torch.no_grad():
            fix_val = self.dataset.samp_fix(coords)
        
        acc_flow = self.network(coords)
        tar_coords = acc_flow + coords
        warp_val = self.dataset.samp_mov(tar_coords)

        loss = self.criterion(warp_val, fix_val)
        loss += self.lambda_tv * self.tv_regulizer(acc_flow)
        loss.backward()

        self.adaptive_control(step, self.network, self.optimizer)
        self.optimizer.step()

    def train_step_aug(self, step):
        """Perform one iteration of training."""
        coords = next(self.dataset_iter)
        with torch.no_grad():
            fix_val = self.dataset.samp_fix(coords)

        self.network2.train()
        self.optimizer2.zero_grad()
        acc_flow = self.network2(coords)
        tar_coords = coords + acc_flow
        warp_val = self.dataset.samp_aug_mov(tar_coords)

        loss = self.criterion(warp_val, fix_val)
        # tv regularization
        if self.lambda_tv > 0 and self.dataset.neighs != 0:
            loss += self.lambda_tv * self.tv_regulizer(acc_flow)
        loss.backward()

        self.adaptive_control(step, self.network2, self.optimizer2)
        self.optimizer2.step()

    def train_step_triplet(self, step, warp_arr_fix, warp_arr_mov):
        """Perform one iteration of training."""
        coords = next(self.dataset_iter)
        with torch.no_grad():
            fix_val = self.dataset._sampling(coords, warp_arr_fix)

        self.network3.train()
        self.optimizer3.zero_grad()
        acc_flow = self.network3(coords)
        tar_coords = coords + acc_flow
        warp_val = self.dataset._sampling(tar_coords, warp_arr_mov)

        loss = self.criterion(warp_val, fix_val)
        # tv regularization
        if self.lambda_tv > 0 and self.dataset.neighs != 0:
            loss += self.lambda_tv * self.tv_regulizer(acc_flow)
        loss.backward()

        self.adaptive_control(step, self.network3, self.optimizer3)
        self.optimizer3.step()

    @torch.no_grad()
    def inf_volume_torch(self):
        """Return the image-values for the given input-coordinates."""
        self.network.eval()

        warp_vals = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            tar_coords_i = self.network(coords_i) + coords_i
            warp_val_i = self.dataset.samp_mov(tar_coords_i)
            warp_vals.append(warp_val_i)

        warp_val = torch.cat(warp_vals, dim=0)
        warp_val = self.dataset.masked_gather(warp_val, is_flow=False)
        warp_val = warp_val.reshape(*self.shape)
        return warp_val

    @torch.no_grad()
    def inf_volume_aug_torch(self):
        """Return the image-values for the given input-coordinates."""
        self.network2.eval()

        warp_vals = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            tar_coords_i = coords_i + self.network2(coords_i)
            warp_val_i = self.dataset.samp_aug_mov(tar_coords_i)
            warp_vals.append(warp_val_i)

        warp_val = torch.cat(warp_vals, dim=0)
        warp_val = self.dataset.masked_gather(warp_val, is_flow=False)
        warp_val = warp_val.reshape(*self.shape)
        return warp_val

    @torch.no_grad()
    def eval(self):
        self.network.eval()
        coords = self.dataset.abs2rel(self.dataset.fix_marks).to(self.device)

        tar_coords = self.network(coords) + coords
        warp_marks = self.dataset.rel2abs(tar_coords)
        warp_marks = torch.round(warp_marks)

        mean, std = metrics.compute_landmark_accuracy(
            self.dataset.abs2phys(warp_marks),
            self.dataset.abs2phys(self.dataset.mov_marks),
        )
        message = "Case{} landmarks (mm): {:.4f}±{:.4f}".format(
            self.dataset.case_idx, mean[0], std[0]
        )
        print(message)
        return mean, std

    @torch.no_grad()
    def val_triplet_roi(self):
        self.network3.eval()
        image_size = self.dataset.image_size.to(self.device)

        track = []
        for i in range(0, len(self.dataset), self.chunk_size):
            coords_i = self.dataset[i : i + self.chunk_size]
            track_i = self.network3(coords_i)
            track_i = track_i * image_size / 2
            track.append(track_i)

        track = torch.cat(track, dim=0)
        triplet = metrics.comp_triplet(track)
        return triplet

    def optimize_stage1(self):
        self.dataset_iter = iter(self.dataset)
        for step in tqdm(range(1, self.max_steps + 1), ncols=80):
            for group in self.optimizer.param_groups:
                group["lr"] = self.get_cur_lr(step, self.lr[group['name']]) 
            self.train_step(step)
        error_mean, error_std = self.eval()

        return self.inf_volume_torch(), error_mean, error_std

    def optimize_stage2(self):
        """Train the network."""
        self.dataset.shuffle()
        self.dataset_iter = iter(self.dataset)
        for step in tqdm(range(1, self.max_steps + 1), ncols=80):
            for group in self.optimizer2.param_groups:
                group["lr"] = self.get_cur_lr(step, self.lr[group['name']]) 
            self.train_step_aug(step)

        return self.inf_volume_aug_torch()
    
    def optimize_stage3(self, warp_arr_fix, warp_arr_mov):
        """Train the network."""
        self.dataset.shuffle()
        self.dataset_iter = iter(self.dataset)
        for step in tqdm(range(1, self.max_steps + 1), ncols=80):
            for group in self.optimizer3.param_groups:
                group["lr"] = self.get_cur_lr(step, self.lr[group['name']]) 
            self.train_step_triplet(step, warp_arr_fix, warp_arr_mov)

        return self.val_triplet_roi()

    def evaluate_cd(self):
        warp_arr_1, error_mean, error_std = self.optimize_stage1()
        warp_arr_2 = self.optimize_stage2()
        mask = self.dataset.fix_mask.reshape(-1)
        CD2 = F.l1_loss(self.dataset.fix_arr.reshape(-1)[mask], self.dataset.mov_arr.reshape(-1)[mask])
        CD3 = self.optimize_stage3(warp_arr_1.unsqueeze(0).unsqueeze(0), 
                                       warp_arr_2.unsqueeze(0).unsqueeze(0))

        return CD2, CD3, error_mean, error_std
