"""Train the learnable channel weights of the nnInteractive perceptual metric."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from regmetric.data.metric import (
    AbdomenMetricDataset,
    COPDG_Dataset,
    OASISRaw3DDataset,
)
from regmetric.models import NNInteractiveMetric
from regmetric.utils.deformation_transform import apply_rigid_transform, random_deform
from regmetric.utils.loss import get_loss


DATASET_NAMES = ("oasis", "copdgene", "l2r-abdomen")


def parse_milestones(value: str) -> list[int]:
    """Parse a comma-separated learning-rate milestone list."""
    if not value.strip():
        return []
    milestones = [int(item.strip()) for item in value.split(",")]
    if any(item < 1 for item in milestones):
        raise argparse.ArgumentTypeError("Milestones must be positive epoch numbers")
    return milestones


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/metric"))
    parser.add_argument("--nnunet-base", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--steps-per-epoch", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--geo-spacing", type=int, default=32)
    parser.add_argument("--rigid-probability", type=float, default=0.5)
    parser.add_argument("--milestones", type=parse_milestones, default=[200, 275, 350, 425])
    parser.add_argument("--early-stop-patience", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true", help="Enable CUDA automatic mixed precision")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.steps_per_epoch < 1 or args.batch_size < 1:
        raise ValueError("epochs, steps-per-epoch, and batch-size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if not 0.0 <= args.rigid_probability <= 1.0:
        raise ValueError("rigid-probability must be in [0, 1]")
    if args.margin < 0.0:
        raise ValueError("margin must be non-negative")
    if args.geo_spacing < 1:
        raise ValueError("geo-spacing must be positive")
    if args.early_stop_patience < 1:
        raise ValueError("early-stop-patience must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be non-negative")
    if not args.data_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {args.data_root}")
    if not args.weights_dir.is_dir():
        raise FileNotFoundError(f"nnInteractive weights directory does not exist: {args.weights_dir}")
    if args.nnunet_base is not None and not args.nnunet_base.is_dir():
        raise FileNotFoundError(f"nnU-Net base directory does not exist: {args.nnunet_base}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(name: str, root: Path, mode: str) -> Dataset:
    root_string = str(root)
    if name == "oasis":
        return OASISRaw3DDataset(root_string, mode=mode)
    if name == "copdgene":
        return COPDG_Dataset(root_string, mode=mode)
    if name == "l2r-abdomen":
        return AbdomenMetricDataset(root_string, mode=mode)
    raise ValueError(f"Unsupported dataset: {name}")


def unpack_batch(dataset_name: str, batch: object) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return the anchor and an optional real negative image from a dataset batch."""
    if dataset_name in {"oasis", "l2r-abdomen"}:
        if not isinstance(batch, torch.Tensor):
            raise TypeError(f"Expected a tensor batch for {dataset_name}")
        return batch, None

    if not isinstance(batch, Sequence) or len(batch) != 2:
        raise TypeError(f"Expected a two-image batch for {dataset_name}")
    first, second = batch
    if not isinstance(first, torch.Tensor) or not isinstance(second, torch.Tensor):
        raise TypeError("Dataset batches must contain tensors")

    if dataset_name == "copdgene":
        expiratory, inspiratory = first, second
        return inspiratory, expiratory
    return first, second


def infinite_batches(loader: DataLoader) -> Iterator[object]:
    """Repeat a DataLoader without caching its large medical-image batches."""
    while True:
        yield from loader


def random_rigid_pair(anchor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if random.random() < 0.5:
        positive = apply_rigid_transform(
            anchor,
            "translation",
            translate_voxels=random.uniform(0.0, 1.0),
        )
        negative = apply_rigid_transform(
            anchor,
            "translation",
            translate_voxels=random.uniform(1.0, 10.0),
        )
        return positive, negative

    positive_angle = random.uniform(-3.0, 3.0)
    negative_angle = random.choice((-1.0, 1.0)) * random.uniform(3.0, 10.0)
    positive = apply_rigid_transform(anchor, "rotation", rotate_angle=positive_angle)
    negative = apply_rigid_transform(anchor, "rotation", rotate_angle=negative_angle)
    return positive, negative


def make_training_triplet(
    anchor: torch.Tensor,
    real_negative: torch.Tensor | None,
    geo_spacing: int,
    rigid_probability: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a small-deformation positive and a stronger negative."""
    if random.random() < rigid_probability:
        return random_rigid_pair(anchor)

    positive = random_deform(
        anchor,
        geo_mag=random.uniform(0.0, 2.0),
        geo_spacing=geo_spacing,
    )
    if real_negative is not None and random.random() < 0.5:
        negative = random_deform(
            real_negative,
            geo_mag=random.uniform(0.0, 4.0),
            geo_spacing=geo_spacing,
        )
    else:
        negative = random_deform(
            anchor,
            geo_mag=random.uniform(2.0, 8.0),
            geo_spacing=geo_spacing,
        )
    return positive, negative


def make_validation_triplet(
    anchor: torch.Tensor,
    real_negative: torch.Tensor | None,
    geo_spacing: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = random_deform(
        anchor,
        geo_mag=2.0,
        geo_spacing=geo_spacing,
    )
    negative_source = real_negative if real_negative is not None else anchor
    negative_magnitude = 2.0 if real_negative is not None else 4.0
    negative = random_deform(
        negative_source,
        geo_mag=negative_magnitude,
        geo_spacing=geo_spacing,
    )
    return positive, negative


def metric_distance(
    model: NNInteractiveMetric,
    anchor: torch.Tensor,
    candidate: torch.Tensor,
) -> torch.Tensor:
    anchor_features = model(anchor)
    candidate_features = model(candidate)
    distances = [
        model.calc_distance(anchor_features[layer], candidate_features[layer], layer)
        for layer in model.target_layers
    ]
    return torch.stack(distances).mean()


@torch.no_grad()
def validate(
    model: NNInteractiveMetric,
    loader: DataLoader,
    dataset_name: str,
    device: torch.device,
    geo_spacing: int,
) -> tuple[float, float]:
    model.eval()
    correct = 0
    gap_sum = 0.0
    sample_count = 0
    for batch in loader:
        anchor, real_negative = unpack_batch(dataset_name, batch)
        anchor = anchor.to(device, non_blocking=True)
        if real_negative is not None:
            real_negative = real_negative.to(device, non_blocking=True)
        positive, negative = make_validation_triplet(anchor, real_negative, geo_spacing)
        positive_distance = metric_distance(model, anchor, positive)
        negative_distance = metric_distance(model, anchor, negative)
        correct += int(positive_distance.item() < negative_distance.item())
        gap_sum += (negative_distance - positive_distance).item()
        sample_count += 1
    if sample_count == 0:
        raise RuntimeError("Validation dataset is empty")
    return correct / sample_count, gap_sum / sample_count


def save_checkpoint(
    path: Path,
    model: NNInteractiveMetric,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    args: argparse.Namespace,
    validation_accuracy: float,
    validation_gap: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "metric_state_dict": model.lin_layers.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "target_layers": model.target_layers,
        "validation_accuracy": validation_accuracy,
        "validation_gap": validation_gap,
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    torch.save(checkpoint, path)


def train(args: argparse.Namespace) -> None:
    validate_args(args)
    seed_everything(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device was requested, but this PyTorch build cannot use CUDA. "
            "Install a CUDA-enabled PyTorch build or pass --device cpu."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(
        json.dumps(
            {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            indent=2,
        ),
        encoding="utf-8",
    )

    train_dataset = build_dataset(args.dataset, args.data_root, "train")
    validation_dataset = build_dataset(args.dataset, args.data_root, "val")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise RuntimeError("Training and validation datasets must both be non-empty")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=args.num_workers > 0,
    )

    model = NNInteractiveMetric(
        weights_dir=args.weights_dir,
        requires_grad=True,
        device=device,
        nnunet_base=args.nnunet_base,
    ).to(device)
    optimizer = torch.optim.Adam(model.lin_layers.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.milestones,
        gamma=0.1,
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    batches = infinite_batches(train_loader)

    best_loss = float("inf")
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step in range(1, args.steps_per_epoch + 1):
            batch = next(batches)
            anchor, real_negative = unpack_batch(args.dataset, batch)
            anchor = anchor.to(device, non_blocking=True)
            if real_negative is not None:
                real_negative = real_negative.to(device, non_blocking=True)

            with torch.no_grad():
                positive, negative = make_training_triplet(
                    anchor,
                    real_negative,
                    args.geo_spacing,
                    args.rigid_probability,
                )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                anchor_features = model(anchor)
                positive_features = model(positive)
                negative_features = model(negative)
                loss = get_loss(
                    "soft_margin_triplet",
                    model,
                    anchor_features,
                    positive_features,
                    negative_features,
                    margin=args.margin,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                for layer in model.lin_layers.values():
                    layer.weight.clamp_(min=1e-5)

            running_loss += loss.item()
            if step == 1 or step % 10 == 0:
                print(f"Epoch {epoch:04d} step {step:04d}/{args.steps_per_epoch}: loss={loss.item():.6f}")

        average_loss = running_loss / args.steps_per_epoch
        validation_accuracy, validation_gap = validate(
            model,
            validation_loader,
            args.dataset,
            device,
            args.geo_spacing,
        )
        scheduler.step()
        print(
            f"Epoch {epoch:04d}: train_loss={average_loss:.6f}, "
            f"val_accuracy={validation_accuracy:.4f}, val_gap={validation_gap:.6f}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            args,
            validation_accuracy,
            validation_gap,
        )
        if average_loss < best_loss:
            best_loss = average_loss
            epochs_without_improvement = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                args,
                validation_accuracy,
                validation_gap,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.early_stop_patience:
                print(f"Early stopping after {epoch} epochs; best train loss was {best_loss:.6f}.")
                break


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
