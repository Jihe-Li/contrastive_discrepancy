import os

import SimpleITK as sitk
import torch


def load_volume(path):
    """Read a volume and the geometry of its grid."""
    image = sitk.ReadImage(path)
    array = torch.FloatTensor(sitk.GetArrayFromImage(image))
    params = dict(
        direction=image.GetDirection(),
        origin=image.GetOrigin(),
        size=image.GetSize(),
        spacing=image.GetSpacing(),
    )
    return array, params


def load_mask(path, shape):
    """Read a binary ROI mask, or return an all-true mask when no path is given."""
    if path is None:
        return torch.ones(shape, dtype=torch.bool)
    return torch.BoolTensor(sitk.GetArrayFromImage(sitk.ReadImage(path)))


def load_landmarks(path):
    """Read one 1-indexed ``x y z`` landmark per line."""
    if path is None or not os.path.exists(path):
        return None
    with open(path) as f:
        marks = [[float(v) for v in line.split()[:3]] for line in f if line.strip()]
    return torch.FloatTensor(marks)


def load_pair(fix_image, mov_image, fix_mask=None, mov_mask=None,
              fix_marks=None, mov_marks=None):
    """Load a fixed/moving pair with optional ROI masks and landmarks."""
    fix_arr, params = load_volume(fix_image)
    mov_arr, _ = load_volume(mov_image)

    return dict(
        fix_arr=fix_arr,
        mov_arr=mov_arr,
        fix_mask=load_mask(fix_mask, fix_arr.shape),
        mov_mask=load_mask(mov_mask, mov_arr.shape),
        fix_marks=load_landmarks(fix_marks),
        mov_marks=load_landmarks(mov_marks),
        params=params,
    )


def load_DIRLab(root="data/DIRLab", case_idx=1, fix_phase=0, mov_phase=5,
                mask_folder="Lungs"):
    """Load one DIRLab 4DCT case; see the README for the expected layout."""
    folder = os.path.join(root, f"Case{case_idx}Pack")

    def phase(idx):
        return dict(
            image=os.path.join(folder, "Images", f"case{case_idx}_T{idx}0.nii.gz"),
            mask=os.path.join(folder, mask_folder, f"case{case_idx}_T{idx}0.nii.gz"),
            marks=os.path.join(folder, "ExtremePhases",
                               f"Case{case_idx}_300_T{idx}0_xyz.txt"),
        )

    fix, mov = phase(fix_phase), phase(mov_phase)
    return load_pair(
        fix_image=fix["image"], mov_image=mov["image"],
        fix_mask=fix["mask"], mov_mask=mov["mask"],
        fix_marks=fix["marks"], mov_marks=mov["marks"],
    )
