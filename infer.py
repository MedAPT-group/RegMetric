"""Register an OASIS pair with SyN and save three PNG risk-map comparisons."""

import argparse
import math
import os
from tempfile import TemporaryDirectory

from config import (
    DEFAULT_BACKBONE_DIR,
    DEFAULT_METRIC_CHECKPOINT,
    DEFAULT_INFERENCE_OUTPUT,
    DEFAULT_FIXED_IMAGE,
    DEFAULT_MOVING_IMAGE,
    resolve_path,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", default=DEFAULT_FIXED_IMAGE, help="Fixed OASIS NIfTI image (default: picture/fixed.nii.gz)")
    parser.add_argument("--moving", default=DEFAULT_MOVING_IMAGE, help="Moving OASIS NIfTI image (default: picture/moving.nii.gz)")
    parser.add_argument("--backbone-dir", default=DEFAULT_BACKBONE_DIR)
    parser.add_argument("--metric-checkpoint", default=DEFAULT_METRIC_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_INFERENCE_OUTPUT)
    parser.add_argument("--device", default=None, help="Default: cuda:0 when available, otherwise cpu")
    parser.add_argument("--threads", type=int, default=16)
    # SyN settings from the existing OASIS SyN pipeline (no parameter sweep).
    parser.add_argument("--grad-step", type=float, default=0.5)
    parser.add_argument("--flow-sigma", type=float, default=11.0)
    parser.add_argument("--total-sigma", type=float, default=0.0)
    parser.add_argument("--reg-iterations", type=int, nargs="+", default=[200, 100, 70, 20])
    args = parser.parse_args()
    if not all(math.isfinite(value) for value in (args.grad_step, args.flow_sigma, args.total_sigma)):
        parser.error("registration parameters must be finite numbers")
    if args.threads < 1 or args.grad_step <= 0 or min(args.reg_iterations) < 0:
        parser.error("threads and grad-step must be positive; iterations must be nonnegative")
    if args.flow_sigma < 0 or args.total_sigma < 0:
        parser.error("smoothing sigmas must be nonnegative")
    return args


def run_inference(args):
    fixed_path = resolve_path(args.fixed)
    moving_path = resolve_path(args.moving)
    backbone_dir = resolve_path(args.backbone_dir)
    checkpoint_path = resolve_path(args.metric_checkpoint)
    missing = [path for path in (fixed_path, moving_path, checkpoint_path) if not path.is_file()]
    if missing:
        paths = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(
            f"Required input files are missing:\n{paths}\n"
            "Place two OASIS 3D NIfTI volumes in picture/fixed.nii.gz and "
            "picture/moving.nii.gz, or supply --fixed and --moving. "
            "Use --metric-checkpoint to select trained metric weights."
        )
    for relative_path in (
        "dataset.json", "plans.json", "inference_session_class.json", "fold_0/checkpoint_final.pth"
    ):
        if not (backbone_dir / relative_path).is_file():
            raise FileNotFoundError(f"Required nnInteractive model file missing: {backbone_dir / relative_path}")

    os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = str(args.threads)
    import ants
    import numpy as np
    import SimpleITK as sitk
    import torch

    from data.oasis import normalize_0_1
    from models.nnInteractive_metric import nnInteractiveMetric

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = nnInteractiveMetric(str(backbone_dir), requires_grad=False, device=device).to(device)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.lin_layers.load_state_dict(state, strict=True)
    model.eval()

    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    if fixed.dimension != 3 or moving.dimension != 3:
        raise ValueError("Fixed and moving inputs must be 3D OASIS volumes.")
    if not np.isfinite(fixed.numpy()).all() or not np.isfinite(moving.numpy()).all():
        raise ValueError("Input images contain NaN or infinite voxel values.")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Registering moving image to fixed image with SyN...")
    # ANTs needs transform files internally; discard them after registration.
    with TemporaryDirectory(prefix="oasis_syn_") as temporary_dir:
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform="SyN",
            outprefix=os.path.join(temporary_dir, "syn_"),
            grad_step=args.grad_step,
            flow_sigma=args.flow_sigma,
            total_sigma=args.total_sigma,
            aff_metric="mattes",
            aff_sampling=32,
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=tuple(args.reg_iterations),
            write_composite_transform=False,
            singleprecision=False,
            verbose=False,
        )
        warped = registration["warpedmovout"]
        if warped.shape != fixed.shape or not all(
            np.allclose(getattr(warped, name), getattr(fixed, name), rtol=0, atol=1e-5)
            for name in ("origin", "spacing", "direction")
        ):
            raise ValueError("SyN output must use the fixed image's physical grid.")
        warped_array = warped.numpy().transpose(2, 1, 0).copy()

    # Use the training loader's Z,Y,X order, rather than ANTs' X,Y,Z array order.
    fixed_image = sitk.ReadImage(str(fixed_path))
    if fixed_image.GetDimension() != 3 or fixed_image.GetNumberOfComponentsPerPixel() != 1:
        raise ValueError("Fixed input must be a scalar 3D image.")
    if not all((
        np.allclose(fixed_image.GetOrigin(), fixed.origin, rtol=0, atol=1e-5),
        np.allclose(fixed_image.GetSpacing(), fixed.spacing, rtol=0, atol=1e-5),
        np.allclose(np.array(fixed_image.GetDirection()).reshape(3, 3), fixed.direction, rtol=0, atol=1e-5),
    )):
        raise ValueError("ANTs and SimpleITK disagree on the fixed image's physical grid.")
    warped_image = sitk.GetImageFromArray(warped_array)
    if fixed_image.GetSize() != warped_image.GetSize():
        raise ValueError("SyN output size does not match the fixed image.")
    warped_image.CopyInformation(fixed_image)

    def to_tensor(image):
        array = normalize_0_1(sitk.GetArrayFromImage(image))
        return torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(device)

    fixed_tensor = to_tensor(fixed_image)
    warped_tensor = to_tensor(warped_image)
    print("Computing nnInteractive spatial difference map...")
    with torch.inference_mode():
        spatial_map = model.spatial_difference(fixed_tensor, warped_tensor)
    spatial_array = spatial_map[0, 0].cpu().numpy()
    if not np.isfinite(spatial_array).all():
        raise RuntimeError("nnInteractive produced a non-finite spatial difference map.")
    difference_image = sitk.GetImageFromArray(spatial_array)
    difference_image.CopyInformation(fixed_image)

    # Canonical anatomical axes for display only; model inputs stay unchanged.
    fixed_display = sitk.DICOMOrient(fixed_image, "LPS")
    warped_display = sitk.DICOMOrient(warped_image, "LPS")
    difference_display = sitk.DICOMOrient(difference_image, "LPS")
    save_comparisons(
        sitk.GetArrayFromImage(fixed_display),
        sitk.GetArrayFromImage(warped_display),
        sitk.GetArrayFromImage(difference_display),
        fixed_display.GetSpacing(),
        output_dir,
    )


def save_comparisons(fixed_array, warped_array, spatial_array, spacing, output_dir):
    """Save central axial, coronal and sagittal slices with a shared risk scale."""
    import numpy as np

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_min = min(float(fixed_array.min()), float(warped_array.min()))
    image_max = max(float(fixed_array.max()), float(warped_array.max()))
    risk_max = max(float(spatial_array.max()), np.finfo(np.float32).eps)
    sx, sy, sz = spacing
    for view, dimension, aspect, origin in (
        ("axial", 0, sy / sx, "upper"),
        ("coronal", 1, sz / sx, "lower"),
        ("sagittal", 2, sz / sy, "lower"),
    ):
        middle_slice = spatial_array.shape[dimension] // 2
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for axis, volume, title in zip(
            axes[:2], (fixed_array, warped_array), ("Fixed", "SyN warped moving")
        ):
            axis.imshow(
                np.take(volume, middle_slice, axis=dimension), cmap="gray",
                vmin=image_min, vmax=image_max, aspect=aspect, origin=origin,
            )
            axis.set_title(title)
        heatmap = axes[2].imshow(
            np.take(spatial_array, middle_slice, axis=dimension), cmap="jet",
            vmin=0, vmax=risk_max, aspect=aspect, origin=origin,
        )
        axes[2].set_title("nnInteractive risk map")
        fig.colorbar(heatmap, ax=axes[2], fraction=0.046, pad=0.04)
        for axis in axes:
            axis.axis("off")
        fig.suptitle(f"{view.capitalize()} - middle slice {middle_slice}")
        fig.tight_layout()
        path = output_dir / f"spatial_difference_{view}.png"
        fig.savefig(str(path), dpi=150)
        plt.close(fig)
        print(f"Saved comparison: {path}")


if __name__ == "__main__":
    run_inference(parse_args())
