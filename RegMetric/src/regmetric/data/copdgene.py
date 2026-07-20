import os
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
import re


class COPDGDataset(Dataset):
    """
    Load preprocessed 160-cubed COPDGene image pairs.

    Images and masks use (1, Z, Y, X). Landmarks and spacing are converted
    from XYZ to ZYX. When ``only_lung`` is true, non-lung voxels are masked.
    """

    def __init__(self, root_dir, mode='train', only_lung=True):
        self.root_dir = root_dir
        self.mode = mode
        self.only_lung = only_lung

        # Discover subject directories.
        all_folders = sorted(
            [f for f in os.listdir(root_dir) if f.startswith('copd') and os.path.isdir(os.path.join(root_dir, f))])

        test_subjects = []
        val_subjects = []
        train_subjects = []

        # Parse numeric IDs so both copd7 and copd07 are supported.
        for f in all_folders:
            match = re.search(r'\d+', f)
            if not match:
                continue
            subj_id = int(match.group())

            if subj_id in [7, 8]:
                test_subjects.append(f)
            elif subj_id == 9:
                val_subjects.append(f)
            else:
                train_subjects.append(f)

        # Select the requested split.
        if mode == 'train':
            target_subjects = train_subjects
        elif mode == 'val':
            target_subjects = val_subjects
        elif mode == 'test':
            target_subjects = test_subjects
        else:
            raise ValueError(f"Unsupported mode {mode!r}; use 'train', 'val', or 'test'")

        # Build one expiratory-to-inspiratory pair per subject.
        self.pairs = []
        for subj in target_subjects:
            self.pairs.append({'subj': subj, 'moving_phase': 'e', 'fixed_phase': 'i'})

        print(f"[{mode.upper()}] Loaded {len(self.pairs)} COPDGene e-to-i pairs")

    def load_imgs(self, folder, subj, phase):
        """Load a CT image and its segmentation mask."""
        ct_path = os.path.join(folder, f"{subj}_{phase}BHCT.nii.gz")
        ct_img = sitk.ReadImage(ct_path)
        ct_arr = sitk.GetArrayFromImage(ct_img).astype(np.float32)

        MIN_HU = -1000.0
        MAX_HU = 1000.0
        ct_arr = np.clip(ct_arr, MIN_HU, MAX_HU)
        ct_arr = (ct_arr - MIN_HU) / (MAX_HU - MIN_HU)

        seg_path = os.path.join(folder, f"{subj}_{phase}seg.nii.gz")
        seg_img = sitk.ReadImage(seg_path)
        seg_arr = sitk.GetArrayFromImage(seg_img).astype(np.float32)

        if self.only_lung:
            mask_arr = ((seg_arr == 1) | (seg_arr == 2)).astype(np.float32)
        else:
            mask_arr = seg_arr

        spacing_xyz = np.array(ct_img.GetSpacing(), dtype=np.float32)

        return ct_arr, mask_arr, spacing_xyz

    def load_marks(self, folder, subj, phase):
        """Load landmark coordinates in XYZ order."""
        path = os.path.join(folder, f"{subj}_300_{phase}BH_xyz_r1.txt")
        marks_xyz = np.loadtxt(path, dtype=np.float32)
        return marks_xyz

    def format_data(self, img_np, seg_np, kpts_xyz, spacing_xyz):
        """Convert arrays to tensors and coordinates from XYZ to ZYX."""
        img_t = torch.from_numpy(img_np).unsqueeze(0).float()
        seg_t = torch.from_numpy(seg_np).unsqueeze(0).float()

        kpts_zyx = kpts_xyz[:, [2, 1, 0]].copy()
        kpts_t = torch.from_numpy(kpts_zyx).float()

        spacing_zyx = spacing_xyz[[2, 1, 0]].copy()
        spacing_t = torch.from_numpy(spacing_zyx).float()

        return img_t, seg_t, kpts_t, spacing_t

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # Resolve the moving and fixed phases for this subject.
        pair_info = self.pairs[idx]
        subj = pair_info['subj']
        m_phase = pair_info['moving_phase']
        f_phase = pair_info['fixed_phase']

        subj_dir = os.path.join(self.root_dir, subj)

        # Load the fixed inspiratory image.
        f_img_raw, f_seg_raw, f_spacing_xyz = self.load_imgs(subj_dir, subj, f_phase)
        f_kpt_xyz = self.load_marks(subj_dir, subj, f_phase)

        # Load the moving expiratory image.
        m_img_raw, m_seg_raw, _ = self.load_imgs(subj_dir, subj, m_phase)
        m_kpt_xyz = self.load_marks(subj_dir, subj, m_phase)

        # Convert images, masks, landmarks, and spacing to network tensors.
        m_img, m_seg, m_kpts, _ = self.format_data(m_img_raw, m_seg_raw, m_kpt_xyz, f_spacing_xyz)
        f_img, f_seg, f_kpts, spacing_zyx = self.format_data(f_img_raw, f_seg_raw, f_kpt_xyz, f_spacing_xyz)

        # ==========================================
        # Mask the scanner bed and background air.
        # ==========================================
        m_img = m_img * m_seg
        f_img = f_img * f_seg

        return {
            'moving_img': m_img,  # (1, 160, 160, 160)
            'fixed_img': f_img,  # (1, 160, 160, 160)
            'moving_seg': m_seg,  # (1, 160, 160, 160)
            'fixed_seg': f_seg,  # (1, 160, 160, 160)
            'moving_kpts': m_kpts,  # (N, 3)
            'fixed_kpts': f_kpts,  # (N, 3)
            'spacing': spacing_zyx,  # (3,)
            'subject_id': f"{subj}_{m_phase}_to_{f_phase}"
        }
