# data/pair_dataset.py
import os
import glob
import re
import numpy as np
import torch
import SimpleITK as sitk
from torch.utils.data import Dataset
import itertools


def natural_sort_key(s):
    """
    Return a natural-sort key so ``10`` follows ``2``.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_split_files(root_dir, mode='train'):
    """
    Return the fixed 20/3/7 train, validation, and test split.
    """
    # Prefer the standard imagesTr directory.
    search_pattern = os.path.join(root_dir, "imagesTr", "*.nii.gz")
    all_files = glob.glob(search_pattern)

    # Fall back to images stored directly below the root.
    if not all_files:
        search_pattern = os.path.join(root_dir, "*.nii.gz")
        all_files = glob.glob(search_pattern)

    if not all_files:
        raise ValueError(f"No .nii.gz files found under {root_dir}")

    # Natural sorting preserves numeric case order.
    all_files = sorted(all_files, key=natural_sort_key)

    if len(all_files) < 30:
        print(f"Warning: found {len(all_files)} files; 30 were expected")

    # Apply the fixed dataset split.
    if mode == 'train':
        return all_files[:20]
    elif mode == 'val':
        return all_files[20:23]
    elif mode == 'test':
        return all_files[23:30]
    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")


class BaseAbdomenDataset:
    """Load and preprocess NIfTI volumes in ZYX order with SimpleITK."""

    def load_sitk_image(self, path):
        # SimpleITK arrays use ZYX order.
        itk_img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(itk_img).astype(np.float32)

        # Clip abdominal CT values to the selected HU window.
        min_hu = -200.0
        max_hu = 300.0
        data = np.clip(data, min_hu, max_hu)

        # Scale the clipped values to [0, 1].
        data = (data - min_hu) / (max_hu - min_hu)

        return data

    def load_sitk_seg(self, img_path):
        # Locate the paired label by replacing imagesTr with labelsTr.
        seg_path = img_path.replace("imagesTr", "labelsTr")

        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Matching segmentation not found: {seg_path}")

        itk_seg = sitk.ReadImage(seg_path)
        # Segmentation labels are returned unchanged in ZYX order.
        seg_data = sitk.GetArrayFromImage(itk_seg).astype(np.float32)
        return seg_data


class AbdomenPairDataset(Dataset, BaseAbdomenDataset):
    """
    Pair every ordered pair within each split.

    Train has 20 volumes and 380 pairs, validation has 3 and 6, and test has
    7 and 42 respectively.
    """

    def __init__(self, root_dir, mode='train', return_seg=True):
        self.mode = mode
        self.return_seg = return_seg
        self.file_paths = get_split_files(root_dir, mode=mode)

        # Generate ordered pairs without self-pairs.
        self.pairs = list(itertools.permutations(self.file_paths, 2))
        print(f"[{mode.capitalize()}] Loaded {len(self.file_paths)} volumes and {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path_moving, path_fixed = self.pairs[idx]

        # Load normalized images in ZYX order.
        moving = self.load_sitk_image(path_moving)
        fixed = self.load_sitk_image(path_fixed)

        # Add the channel dimension expected by 3D networks.
        m_img = torch.from_numpy(moving).unsqueeze(0)
        f_img = torch.from_numpy(fixed).unsqueeze(0)

        # Use labels to mask the background when available.
        if self.return_seg:
            # Load multiclass labels and add the channel dimension.
            m_seg = torch.from_numpy(self.load_sitk_seg(path_moving)).unsqueeze(0)
            f_seg = torch.from_numpy(self.load_sitk_seg(path_fixed)).unsqueeze(0)

            # ==========================================
            # Convert multiclass labels into binary foreground masks.
            # ==========================================
            # Treat every nonzero organ label as foreground.
            m_roi = (m_seg > 0).float()
            f_roi = (f_seg > 0).float()

            # Remove normalized air and scanner-bed voxels.
            m_img = m_img * m_roi
            f_img = f_img * f_roi

            # Keep original multiclass labels for downstream Dice metrics.
            return {
                'moving_img': m_img,
                'fixed_img': f_img,
                'moving_seg': m_seg,
                'fixed_seg': f_seg
            }
        else:
            # Unlabeled data returns only the normalized images.
            return {
                'moving_img': m_img,
                'fixed_img': f_img
            }
