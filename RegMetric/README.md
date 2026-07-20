# RegMetric

RegMetric contains reusable perceptual-metric and uncertainty-aware components
for 3D medical image registration. Experimental sweeps, generated figures,
private dataset paths, cached files, and large checkpoints are excluded from
this repository.

## Included code

- `regmetric.models`: nnInteractive metric, TransMorph, and TransMorph-Bayes.
- `regmetric.data`: loaders for OASIS, COPDGene, and L2R Abdomen.
- `regmetric.utils`: metric losses, TRE helpers, and deformation transforms.

## Installation

```bash
python -m venv .venv
python -m pip install -U pip
python -m pip install -e .
```

The pinned versions in `requirements.txt` are based on the local `regis`
environment. That environment currently contains the CPU build of PyTorch
2.8.0 even though CUDA 12.1 packages are present. For GPU training, replace it
with the CUDA-enabled PyTorch 2.8.0 build appropriate for the host driver, then
verify that `torch.cuda.is_available()` returns `True`.

Metric losses use all nnInteractive feature stages by default. Pass an explicit
`target_layers` sequence only when a layer ablation is required.

## Example

```python
from regmetric.models import NNInteractiveMetric

metric = NNInteractiveMetric(
    weights_dir="/path/to/nnInteractive_v1.0",
    nnunet_base="/path/to/nnUNet",  # optional if env vars are already set
    device="cuda",
)
```

Dataset locations and checkpoints must be supplied at runtime; the package does
not embed machine-specific paths.

## Training the metric

The training entry point freezes the nnInteractive backbone and optimizes only
the learnable per-stage channel weights. All nnInteractive feature stages are
used by default. Positives receive small deformations, negatives receive larger
deformations or use the paired acquisition, and optimization uses
`softplus(d_positive - d_negative + margin)`.

```bash
python scripts/train_metric.py \
  --dataset oasis \
  --data-root /path/to/OASIS \
  --weights-dir /path/to/nnInteractive_v1.0 \
  --nnunet-base /path/to/nnUNet \
  --output-dir outputs/oasis_metric \
  --device cuda
```

Supported dataset names are `oasis`, `copdgene`, and `l2r-abdomen`.
Run `python scripts/train_metric.py --help` for the complete configuration.
The output directory contains `config.json`, `last.pt`, and the checkpoint with
the lowest training loss as `best.pt`. Load trained channel weights with:

```python
import torch

checkpoint = torch.load("outputs/oasis_metric/best.pt", map_location="cpu")
metric.lin_layers.load_state_dict(checkpoint["metric_state_dict"])
```

Pass `--nnunet-base` unless the three nnU-Net path environment variables are
already configured. Direct inference from a trained-model folder can work
without them, but nnU-Net will print path warnings.

## Repository scope

Training sweeps and one-off evaluation scripts from the original `experiments/`
directory are not included. If exact paper reproduction is required, add a
small set of reviewed entry points under `scripts/` rather than restoring the
entire experiment tree.

## License and attribution

A repository license has not been selected yet. Review `THIRD_PARTY.md`, verify
the licenses of derived components, and add a compatible `LICENSE` file before
making the repository public.
