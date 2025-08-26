import os

import numpy as np
import SimpleITK as sitk
import torch

def load_landmarks(path):
    """Load landmarks from a text file."""
    with open(path) as f:
        landmarks = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    return landmarks


def load_DIRLab_imgs(folder, case_idx=1, phase_idx=0, only_lung=True):
    ct_img = sitk.ReadImage(
        os.path.join(folder, "Images", f"case{case_idx}_T{phase_idx}0.nii.gz")
    )
    ct_arr = torch.FloatTensor(sitk.GetArrayFromImage(ct_img))
    params = dict(
        direction=ct_img.GetDirection(),
        origin=ct_img.GetOrigin(),
        size=ct_img.GetSize(),
        spacing=ct_img.GetSpacing(),
    )

    mask_folder = "Lungs" if only_lung else "Bodies"
    mask_img = sitk.ReadImage(os.path.join(folder, mask_folder, f"case{case_idx}_T{phase_idx}0.nii.gz"))
    mask_arr = torch.BoolTensor(sitk.GetArrayFromImage(mask_img))

    return ct_arr, mask_arr, params


def load_DIRLab_marks(folder, case_idx=1, phase_idx=0):
    path = os.path.join(folder, "ExtremePhases", f"Case{case_idx}_300_T{phase_idx}0_xyz.txt")
    marks = load_landmarks(path)
    return torch.FloatTensor(marks)


def load_DIRLab(folder="data/DIRLab", case_idx=1, only_lung=True):
    folder = os.path.join(folder, f"Case{case_idx}Pack")
    # Images
    fix_arr, fix_mask, params = load_DIRLab_imgs(folder, case_idx, 0, only_lung)
    mov_arr, mov_mask, _ = load_DIRLab_imgs(folder, case_idx, 5, only_lung)
    fix_marks = load_DIRLab_marks(folder, case_idx, 0)
    mov_marks = load_DIRLab_marks(folder, case_idx, 5)

    return dict(
        fix_arr=fix_arr,
        mov_arr=mov_arr,
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        fix_marks=fix_marks,
        mov_marks=mov_marks,
        params=params,
    )
