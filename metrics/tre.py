import torch


def compute_landmark_accuracy(coords_pred, coords_gt):
    """Target Registration Error between landmark sets given in physical units."""
    difference = coords_pred - coords_gt
    distance = torch.norm(difference, dim=1)
    per_axis = difference.abs()

    return dict(
        mean=distance.mean().item(),
        std=distance.std().item(),
        axis_mean=per_axis.mean(dim=0).tolist(),
        axis_std=per_axis.std(dim=0).tolist(),
    )
