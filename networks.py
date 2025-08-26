import numpy as np
import torch
from types import SimpleNamespace
import pytorch3d.ops as ops
from torch import nn
from tqdm import tqdm


class GaussianWarp(nn.Module):
    def __init__(self, K, node_shape, max_densify_num, sparsification, max_contribution, min_contribution, 
                 with_quaternion=False, anisotropy=False, **kwargs) -> None:
        super().__init__()
        self.K = K
        self.node_shape = node_shape  
        self.node_initial_num = np.prod(node_shape)   
        self.max_densify_num = max_densify_num
        self.sparsification = sparsification 
        self.max_contribution = max_contribution
        self.min_contribution = min_contribution
        self.with_quaternion = with_quaternion
        self.anisotropy = SimpleNamespace(**anisotropy)
    
        node_coords, node_radius = self.make_coors(self.node_shape)
        self.node_position = nn.Parameter(node_coords)          
        self.translation = nn.Parameter(torch.zeros(self.node_initial_num, 3))   

        if self.anisotropy.quaternion:  
            self.node_quaternion = nn.Parameter(torch.concat([torch.ones(self.node_initial_num, 1), 
                                                torch.zeros(self.node_initial_num, 3)], dim=1))
        if self.anisotropy.scaling:
            self._node_scaling = nn.Parameter(torch.ones(self.node_initial_num, 3) * torch.log(node_radius))
        else:
            self._node_radius = nn.Parameter(torch.ones(self.node_initial_num, 1) * torch.log(node_radius)) 

        if with_quaternion:
            self.quaternion = nn.Parameter(torch.concat([torch.ones(self.node_initial_num, 1), 
                                                         torch.zeros(self.node_initial_num, 3)], dim=1))

        # 记录梯度
        self.contribution = torch.zeros(self.node_initial_num, device="cuda")
        self.counter = torch.zeros(self.node_initial_num, device="cuda")

    def reinitialize(self, mask: torch.Tensor):
        node_position = torch.rand((1, 1, 1, self.max_densify_num, 3)) * 2. - 1.
        node_radius = 1.732 / (torch.tensor(self.max_densify_num) ** (1/3))
        mask = mask.cpu().unsqueeze(0).unsqueeze(0).float()
        node_mask = torch.nn.functional.grid_sample(mask, node_position, mode='nearest', align_corners=False) \
                    .squeeze().bool()

        self.node_initial_num = node_mask.sum()
        self.node_position = nn.Parameter(node_position.squeeze()[node_mask])
        if self.anisotropy.quaternion:
            self.node_quaternion = nn.Parameter(torch.concat([torch.ones(self.node_initial_num, 1), 
                                                torch.zeros(self.node_initial_num, 3)], dim=1))
        if self.anisotropy.scaling:
            self._node_scaling = nn.Parameter(torch.ones(self.node_initial_num, 3) * torch.log(node_radius))
        else:
            self._node_radius = nn.Parameter(torch.ones(self.node_initial_num, 1) * torch.log(node_radius))
        self.translation = nn.Parameter(torch.zeros(self.node_initial_num, 3))
        if self.with_quaternion:
            self.quaternion = nn.Parameter(torch.concat([torch.ones(self.node_initial_num, 1), 
                                                         torch.zeros(self.node_initial_num, 3)], dim=1))
        
        self.max_densify_num = mask.sum().int().item() // (self.sparsification ** 3)
        self.contribution = torch.zeros(self.node_position.shape[0], device="cuda")
        self.counter = torch.zeros(self.node_position.shape[0], device="cuda")
        print('The initial number of Gaussians is %d.' % self.node_position.shape[0])

    @property
    def node_radius(self):
        return torch.exp(self._node_radius)
    
    @property
    def node_scaling(self):
        return torch.exp(self._node_scaling)

    def trained_parameters(self, lr):
        l = [{'params': [self.node_position],   'name': 'node_position',   'lr': lr.node_position},
             {'params': [self.translation],     'name': 'translation',     'lr': lr.translation}]
        
        if self.anisotropy.quaternion:
            l += [{'params': [self.node_quaternion], 'name': 'node_quaternion', 'lr': lr.node_quaternion}]
        
        if self.anisotropy.scaling:
            l += [{'params': [self._node_scaling],    'name': '_node_scaling',    'lr': lr.node_scaling}]
        else: 
            l += [{'params': [self._node_radius],     'name': '_node_radius',     'lr': lr.node_radius}]
        
        if self.with_quaternion:
            l += [{'params': [self.quaternion],      'name': 'quaternion',      'lr': lr.quaternion}]
        return l

    def make_coors(self, shape):
        """Make a coordinate tensor."""
  
        coords = [torch.linspace(-1, 1, size + 1)[:-1] + 1 / size for size in shape]
        coords = torch.meshgrid(*coords, indexing="ij")
        coords = torch.stack(coords[::-1], dim=-1)
        coords = coords.view(-1, 3)

        diagonal_point_index = shape[1]*shape[2] + shape[2] + 1  
        diagonal_point = coords[diagonal_point_index]
        radius = torch.linalg.vector_norm(diagonal_point - coords[0]) * 0.5

        return coords, radius

    def cal_nn_weight(self, x:torch.Tensor, nodes=None, K=None):
        K = self.K if K is None else K
        _, nn_idxs, _ = ops.knn_points(x[None], nodes[None], None, None, K=K) 
        nn_idxs = nn_idxs[0]  # both [M, K]
        local_coords = x[:, None] - nodes[nn_idxs]  # [M, K, 3]
        if self.anisotropy.scaling:
            if self.anisotropy.quaternion:
                rot_matrix = self.quaternion_to_matrix(self.node_quaternion)[nn_idxs]  # [M, K, 3, 3] 
                local_coords = torch.matmul(local_coords.unsqueeze(2), rot_matrix).squeeze()
            exponent = torch.sum((local_coords / self.node_scaling[nn_idxs]) ** 2, dim=-1)
            nn_weight = torch.exp(-0.5 * exponent)
            nn_weight = nn_weight / (torch.prod(self.node_scaling[nn_idxs], dim=-1) + 1e-7)

        else:
            exponent = torch.sum((local_coords / self.node_radius[nn_idxs]) ** 2, dim=-1)
            nn_weight = torch.exp(-0.5 * exponent)
            nn_weight = nn_weight / (self.node_radius[nn_idxs].squeeze() ** 3 + 1e-7)
        nn_weight = nn_weight / nn_weight.sum(dim=-1, keepdim=True)  # [M, K]
        return nn_weight, nn_idxs, local_coords

    def forward(self, x: torch.Tensor):
        nn_weight, nn_idxs, local_coords = self.cal_nn_weight(x, self.node_position, K=self.K)  
        if self.with_quaternion:
            local_coords = local_coords.detach()
            rot_matrix = self.quaternion_to_matrix(self.quaternion)[nn_idxs]
            self.gaussian_wise_translation = torch.matmul(rot_matrix, local_coords.unsqueeze(-1)).squeeze(-1) + self.translation[nn_idxs] - local_coords
        else:
            self.gaussian_wise_translation = self.translation[nn_idxs]
        acc_flow = (self.gaussian_wise_translation * nn_weight[..., None]).sum(dim=1)

        return acc_flow

    def quaternion_to_matrix(self, quaternions: torch.Tensor) -> torch.Tensor:
        '''transform quaternion to rotation matrix with normalization'''
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)
        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def densify_and_prune(self, optimizer=None):
        max_contribution = self.max_contribution
        nodes_contribution = self.contribution / self.counter
        nodes_contribution[nodes_contribution.isnan()] = 0.

        # Picking pts to densify
        if self.anisotropy.scaling:
            nan_nodes = self._node_scaling[:, 0].isnan()
        else:
            nan_nodes = self._node_radius[:, 0].isnan()
        densified_pts_mask = torch.logical_and(nodes_contribution >= max_contribution, nan_nodes.logical_not())
        # Picking pts to prune
        pruned_pts_mask = torch.logical_or(nodes_contribution <= self.min_contribution, nan_nodes.isnan())

        if self.node_position.shape[0] + densified_pts_mask.sum() - pruned_pts_mask.sum() > self.max_densify_num: 
            max_contribution = torch.inf  # 不进行 densify
            densify_num = self.max_densify_num - self.node_position.shape[0] + pruned_pts_mask.sum()
            idxs = torch.topk(nodes_contribution, densify_num)[1]
            densified_pts_mask = nodes_contribution >= torch.inf
            densified_pts_mask[idxs] = True

        tqdm.write(f'\nAdd {densified_pts_mask.sum()} nodes and prune {pruned_pts_mask.sum()} nodes.')

        # Densify and Prune
        if densified_pts_mask.sum() > 0:
            # densify
            if self.anisotropy.scaling:
                stds = self.node_scaling[densified_pts_mask]
            else:
                stds = self.node_radius[densified_pts_mask].repeat(1,3)  # gaussian 的标准差
            means = torch.zeros((stds.size(0), 3),device="cuda")
            samples = torch.normal(mean=means, std=stds)
            new_node_position = samples + self.node_position[densified_pts_mask] 

            new_param_list = {'node_position': new_node_position}
            if self.anisotropy.quaternion:
                new_param_list['node_quaternion'] = self.node_quaternion[densified_pts_mask]
            if self.anisotropy.scaling:
                new_param_list['_node_scaling'] = self._node_scaling[densified_pts_mask]
            else:
                new_param_list['_node_radius'] = self._node_radius[densified_pts_mask]

            new_param_list['translation'] = self.translation[densified_pts_mask]
            if self.with_quaternion:
                new_param_list['quaternion'] = self.quaternion[densified_pts_mask]
                # translation, quaternion 参数
            
            for group in optimizer.param_groups:
                stored_state = optimizer.state.get(group['params'][0], None) 
                extension_tensor = new_param_list[group['name']]   
                if stored_state is not None:    
                    stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0) 
                    stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0) 
                    del optimizer.state[group['params'][0]]   
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))  
                    optimizer.state[group['params'][0]] = stored_state 
                    setattr(self, group['name'], group["params"][0])   
                else:  
                    group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                    setattr(self, group['name'], group["params"][0])  
        
        # Prune
        if pruned_pts_mask.sum() > 0:  
            if pruned_pts_mask.shape[0] < self.node_position.shape[0]:  
                pruned_pts_mask = torch.cat([pruned_pts_mask, torch.zeros([self.node_position.shape[0] - pruned_pts_mask.shape[0]]).to(pruned_pts_mask.device).to(pruned_pts_mask.dtype)])
            pruned_pts_mask = ~pruned_pts_mask 

            for group in optimizer.param_groups:
                stored_state = optimizer.state.get(group['params'][0], None)  
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][pruned_pts_mask]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][pruned_pts_mask]
                    del optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][pruned_pts_mask].requires_grad_(True)))
                    optimizer.state[group['params'][0]] = stored_state  
                    setattr(self, group['name'], group["params"][0])  
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][pruned_pts_mask].requires_grad_(True))
                    setattr(self, group['name'], group["params"][0])  

        self.densification_postfix()  
        tqdm.write(f'With {self.node_position.shape[0]} nodes left.')

    def add_densification_stats(self):  
        self.contribution += torch.norm(self.translation.grad, dim=-1)  
        self.counter += 1

    def densification_postfix(self):
        self.contribution = torch.zeros(self.node_position.shape[0], device="cuda")
        self.counter = torch.zeros(self.node_position.shape[0], device="cuda")
