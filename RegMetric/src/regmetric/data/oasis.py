# data/pair_dataset.py
import os
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset, Dataset
import nibabel as nib


def get_split_files(root_dir, mode='train', split_ratio=0.7, val_ratio=0.1):
    """
    Create a 70/10/20 split with one shared atlas for validation and test.

    ``mode`` must be ``train``, ``val``, or ``test``.
    """
    if not 0.0 < split_ratio < 1.0:
        raise ValueError("split_ratio must be in (0, 1)")
    if val_ratio < 0.0 or split_ratio + val_ratio >= 1.0:
        raise ValueError("val_ratio must be non-negative and leave room for a test split")

    search_pattern = os.path.join(root_dir, "OASIS_*", "aligned_norm.nii.gz")
    all_files = sorted(glob.glob(search_pattern))

    if not all_files:
        raise ValueError(f"No OASIS files found under {root_dir}")

    total = len(all_files)
    train_end = int(total * split_ratio)
    val_end = int(total * (split_ratio + val_ratio))

    if mode == 'train':
        # Use the first 70 percent for training.
        return all_files[:train_end]

    # Reserve the first post-training image as the shared atlas.
    atlas_file = all_files[train_end]

    if mode == 'val':
        # Validation entries follow the atlas and precede the test split.
        val_files = all_files[train_end + 1: val_end]
        return [atlas_file] + val_files

    elif mode == 'test':
        # Use the final 20 percent for testing.
        test_files = all_files[val_end:]
        return [atlas_file] + test_files

    else:
        raise ValueError("mode must be 'train', 'val', or 'test'")


class BaseOASISDataset:
    """Provide common NIfTI loading and preprocessing helpers."""

    def load_nii_image(self, path):
        nii = nib.load(path)
        data = nii.get_fdata().astype(np.float32)
        # Apply per-volume z-score normalization.
        data = (data - data.mean()) / (data.std() + 1e-8)

        return data

    def load_nii_seg(self, img_path):
        # OASIS segmentation files are expected to use this paired filename.
        seg_path = img_path.replace("aligned_norm.nii.gz", "aligned_seg35.nii.gz")
        if not os.path.exists(seg_path):
            raise FileNotFoundError(f"Matching segmentation not found: {seg_path}")
        nii = nib.load(seg_path)
        return nii.get_fdata().astype(np.float32)


class OASISTrainInfiniteDataset(IterableDataset, BaseOASISDataset):
    """Yield random pairs from the training split indefinitely."""

    def __init__(self, root_dir, split_ratio=0.7):
        self.file_paths = get_split_files(root_dir, mode='train', split_ratio=split_ratio)
        if len(self.file_paths) < 2:
            raise ValueError("The OASIS training split requires at least two volumes")
        print(f"[Train] Loaded {len(self.file_paths)} OASIS volumes for random pairing")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            np.random.seed(worker_info.seed % (2 ** 32 - 1))

        while True:
            idx1, idx2 = np.random.choice(len(self.file_paths), size=2, replace=False)
            moving = self.load_nii_image(self.file_paths[idx1])
            fixed = self.load_nii_image(self.file_paths[idx2])

            # Add the channel dimension.
            m_tensor = torch.from_numpy(moving).unsqueeze(0)
            f_tensor = torch.from_numpy(fixed).unsqueeze(0)
            yield m_tensor, f_tensor


class OASISTestAtlasDataset(Dataset, BaseOASISDataset):
    """Pair every validation or test image with a fixed atlas."""

    def __init__(self, root_dir, split_ratio=0.7, mode='test'):
        self.file_paths = get_split_files(root_dir, mode=mode, split_ratio=split_ratio)
        print(f"[{mode.capitalize()}] Loaded one atlas and {len(self.file_paths) - 1} moving images")

    def __len__(self):
        return len(self.file_paths) - 1

    def __getitem__(self, idx):
        path_fixed = self.file_paths[0]
        path_moving = self.file_paths[idx + 1]

        # Load tensors and add the channel dimension.
        m_img = torch.from_numpy(self.load_nii_image(path_moving)).unsqueeze(0)
        f_img = torch.from_numpy(self.load_nii_image(path_fixed)).unsqueeze(0)
        m_seg = torch.from_numpy(self.load_nii_seg(path_moving)).unsqueeze(0)
        f_seg = torch.from_numpy(self.load_nii_seg(path_fixed)).unsqueeze(0)

        return m_img, f_img, m_seg, f_seg


