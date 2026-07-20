import os
import glob
import re

import torch
import numpy as np
from torch.utils.data import Dataset
import SimpleITK as sitk


class OASISRaw3DDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        self.file_paths = self._get_file_paths(root_dir, mode)

        print(f"[{mode.upper()}] Loaded {len(self.file_paths)} OASIS volumes")
        print("    - Reader: SimpleITK in native ZYX order")
        print("    - Normalization: min-max to [0, 1]")

    def _get_file_paths(self, root_dir, mode):
        if mode not in {'train', 'val', 'test'}:
            raise ValueError("mode must be 'train', 'val', or 'test'")
        all_files = sorted(glob.glob(os.path.join(root_dir, "OASIS_*/aligned_norm.nii.gz")))
        if len(all_files) == 0:
            raise ValueError(f"No OASIS data found under {root_dir}")

        split_idx = int(len(all_files) * 0.7)
        return all_files[:split_idx] if mode == 'train' else all_files[split_idx:]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]

        # SimpleITK arrays use ZYX order.
        itk_img = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(itk_img).astype(np.float32)

        # Normalize each image to [0, 1].
        d_min = data.min()
        d_max = data.max()
        data = (data - d_min) / (d_max - d_min + 1e-8)

        # Convert to a tensor and add the channel dimension.
        tensor = torch.from_numpy(data).unsqueeze(0)

        return tensor


class COPDG_Dataset(Dataset):
    """
    Load 160-cubed COPDGene pairs for perceptual-metric training.

    Subjects 7, 8, and 9 form the validation split. Images use ZYX order and
    are normalized to [0, 1].
    """

    def __init__(self, root_dir, mode='train', only_lung=True):
        if mode not in {'train', 'val', 'test'}:
            raise ValueError("mode must be 'train', 'val', or 'test'")
        self.root_dir = root_dir
        self.only_lung = only_lung

        # Discover subject directories and split by numeric ID.
        all_folders = sorted(
            [f for f in os.listdir(root_dir) if f.startswith('copd') and os.path.isdir(os.path.join(root_dir, f))])

        train_subjects = []
        val_subjects = []

        for f in all_folders:
            match = re.search(r'\d+', f)
            if not match:
                continue
            subj_id = int(match.group())

            # Reserve subjects 7, 8, and 9 for validation.
            if subj_id in [7, 8, 9]:
                val_subjects.append(f)
            else:
                train_subjects.append(f)

        # Select the requested split.
        if mode == 'train':
            self.subjects = train_subjects
        else:
            self.subjects = val_subjects

        print(f"[{mode.upper()}] Loaded {len(self.subjects)} COPDGene subjects")
        print("    - Validation IDs: 7, 8, 9")
        print("    - Normalization: [0, 1]")

    def _load_and_mask(self, folder, subj, phase):
        """Load, normalize, and mask one respiratory phase."""
        # Read in ZYX order.
        ct_path = os.path.join(folder, f"{subj}_{phase}BHCT.nii.gz")
        ct_img = sitk.ReadImage(ct_path)
        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)

        # Normalize the selected HU range to [0, 1].
        MIN_HU = -1000.0
        MAX_HU = 1000.0
        ct_arr = np.clip(ct_arr, MIN_HU, MAX_HU)
        ct_arr = (ct_arr - MIN_HU) / (MAX_HU - MIN_HU)

        # Create a lung mask from the segmentation.
        seg_path = os.path.join(folder, f"{subj}_{phase}seg.nii.gz")
        seg_img = sitk.ReadImage(seg_path)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.float32)

        if self.only_lung:
            mask_arr = ((seg_arr == 1) | (seg_arr == 2)).astype(np.float32)
        else:
            mask_arr = seg_arr

        # Remove the non-lung background.
        ct_arr = ct_arr * mask_arr

        # Return (C, Z, Y, X).
        return torch.from_numpy(ct_arr).unsqueeze(0).float()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx):
        subj = self.subjects[idx]
        subj_dir = os.path.join(self.root_dir, subj)

        # Return expiratory and inspiratory phases.
        ebh_tensor = self._load_and_mask(subj_dir, subj, 'e')
        ibh_tensor = self._load_and_mask(subj_dir, subj, 'i')

        return ebh_tensor, ibh_tensor



def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def get_split_files(root_dir, mode='train'):
    """Return the fixed 20-volume training and 10-volume evaluation split."""
    search_pattern = os.path.join(root_dir, "imagesTr", "*.nii.gz")
    all_files = glob.glob(search_pattern)

    if not all_files:
        search_pattern = os.path.join(root_dir, "*.nii.gz")
        all_files = glob.glob(search_pattern)

    if not all_files:
        raise ValueError(f"No .nii.gz files found under {root_dir}")

    all_files = sorted(all_files, key=natural_sort_key)

    if mode == 'train':
        return all_files[:20]
    elif mode in ['val', 'test']:
        return all_files[20:30]
    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")


class AbdomenMetricDataset(Dataset):
    """
    Load single abdominal CT images for self-supervised metric training.
    """

    def __init__(self, root_dir, mode='train'):
        self.mode = mode
        self.file_paths = get_split_files(root_dir, mode=mode)

        print(f"[{mode.upper()}] Loaded {len(self.file_paths)} abdominal NIfTI volumes")
        print("    - Normalization: HU [-200, 300] to [0, 1]")

    def _load_and_mask(self, img_path):
        # Read in ZYX order.
        itk_img = sitk.ReadImage(img_path)
        data = sitk.GetArrayFromImage(itk_img).astype(np.float32)

        # Apply the abdominal HU window and normalize to [0, 1].
        min_hu = -200.0
        max_hu = 300.0
        data = np.clip(data, min_hu, max_hu)
        data = (data - min_hu) / (max_hu - min_hu)

        # Add the channel dimension.
        return torch.from_numpy(data).unsqueeze(0)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Each item is one image; augmentations create the training triplet.
        img_path = self.file_paths[idx]
        img_tensor = self._load_and_mask(img_path)
        return img_tensor
