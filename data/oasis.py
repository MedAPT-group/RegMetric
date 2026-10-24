"""OASIS loading with the original sorted 70/30 training/validation split."""

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from pathlib import Path


def normalize_0_1(array):
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Image contains NaN or infinite voxel values.")
    return (array - array.min()) / (array.max() - array.min() + 1e-8)


class OASISRaw3DDataset(Dataset):
    def __init__(self, root_dir, mode="train"):
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'.")
        files = sorted(Path(root_dir).glob("OASIS_*/aligned_norm.nii.gz"))
        split = int(len(files) * 0.7)
        self.file_paths = files[:split] if mode == "train" else files[split:]
        if not self.file_paths:
            raise ValueError(f"No OASIS images in the {mode} split under {root_dir}.")
        print(f"[{mode}] {len(self.file_paths)} OASIS images")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        image = sitk.ReadImage(str(self.file_paths[index]))
        if image.GetDimension() != 3:
            raise ValueError("OASIS input must be a 3D volume.")
        # SimpleITK returns Z, Y, X; inference uses the same axis order.
        array = normalize_0_1(sitk.GetArrayFromImage(image))
        return torch.from_numpy(array).unsqueeze(0)
