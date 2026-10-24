"""Paths are relative to this directory unless an absolute path is supplied."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = "datasets/OASIS/neurite-oasis.v1.0"
DEFAULT_BACKBONE_DIR = "weights/nnInteractive_v1.0"
DEFAULT_METRIC_CHECKPOINT = "weights/metric/oasis_best.pth"
DEFAULT_FIXED_IMAGE = "picture/fixed.nii.gz"
DEFAULT_MOVING_IMAGE = "picture/moving.nii.gz"
DEFAULT_TRAIN_OUTPUT = "outputs/training"
DEFAULT_INFERENCE_OUTPUT = "outputs/inference"


def resolve_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def configure_nnunet():
    for name, directory in (
        ("nnUNet_raw", "raw"),
        ("nnUNet_preprocessed", "preprocessed"),
        ("nnUNet_results", "results"),
    ):
        os.environ.setdefault(name, str(PROJECT_ROOT / "nnUNet" / directory))
