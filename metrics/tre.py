import torch


def compute_landmark_accuracy(coords_pred, coords_gt):
    difference = (coords_pred - coords_gt).abs()

    means = torch.mean(difference, dim=0).tolist()
    stds = torch.std(difference, dim=0).tolist()

    difference = torch.norm(difference, dim=1)
    means.append(difference.mean().item())
    stds.append(difference.std().item())

    means = means[::-1]
    stds = stds[::-1]

    return means, stds
