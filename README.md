# OASIS Registration Risk Maps with nnInteractive

Train an nnInteractive-based perceptual metric on OASIS and visualize spatial
differences after ANTs SyN registration.

## Example result

![Sagittal comparison](outputs/inference/spatial_difference_sagittal.png)

## Installation

Use Python 3.10+ and install the appropriate
[PyTorch build](https://pytorch.org/get-started/locally/) for your GPU, then run:

```bash
python -m pip install -r requirements.txt
```

The dependency list is not an exact lockfile. Keep your working package versions
when reproducing results.

## Model weights

The nnInteractive backbone is **not included**. Download the complete
`nnInteractive_v1.0` folder from the
[official model repository](https://huggingface.co/MIC-DKFZ/nnInteractive/tree/main/nnInteractive_v1.0)
and place it under `weights/`:

```bash
python -m pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='MIC-DKFZ/nnInteractive', allow_patterns=['nnInteractive_v1.0/**'], local_dir='weights')"
```

Expected files:

```text
weights/nnInteractive_v1.0/dataset.json
weights/nnInteractive_v1.0/plans.json
weights/nnInteractive_v1.0/inference_session_class.json
weights/nnInteractive_v1.0/fold_0/checkpoint_final.pth
weights/metric/oasis_best.pth
```

Both the backbone and the trained OASIS metric checkpoint are required. Keep all
model JSON files alongside `fold_0/`.

## Inference

Place two OASIS 3D `aligned_norm.nii.gz` volumes at:

```text
picture/fixed.nii.gz
picture/moving.nii.gz
```

Inputs must be 3D NIfTI volumes. Run:

```bash
python infer.py --device cuda:0
```

To select your own images and checkpoint:

```bash
python infer.py --fixed picture/fixed.nii.gz --moving picture/moving.nii.gz --metric-checkpoint weights/metric/oasis_best.pth --device cuda:0
```

Use `--device cpu` to select CPU execution, which has not been validated here.
Omitting `--device` selects CUDA when available, otherwise CPU. Whole-volume
inference can require substantial GPU memory.

The pipeline performs SyN registration, nnInteractive feature comparison, then
risk-map visualization. Results are saved under `outputs/inference/`:

```text
spatial_difference_axial.png
spatial_difference_coronal.png
spatial_difference_sagittal.png
```

Each PNG shows the fixed image, SyN warped moving image, and risk map at the middle
slice. Maps use the `jet` rainbow colormap with a shared color scale within each
run. Values represent feature differences.

## Training

Obtain the [Neurite OASIS v1.0 collection](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)
under its data-use terms. The loader expects files matching
`datasets/OASIS/neurite-oasis.v1.0/OASIS_*/aligned_norm.nii.gz`.

```bash
python train.py --data-root datasets/OASIS/neurite-oasis.v1.0 --device cuda:0 --batch-size 1 --epochs 500 --lr 0.001
```

Backbone parameters are frozen; training learns per-stage channel weights.

Checkpoints are saved under `outputs/training/`. The best model is selected by
training loss, with early-stopping patience 50. Use new weights with:

```bash
python infer.py --metric-checkpoint outputs/training/best_train_loss_model.pth --device cuda:0
```

Training does not overwrite `weights/metric/oasis_best.pth`. Checkpoints contain
channel weights only, not optimizer state. Seeds are not fixed.

## Acknowledgments

We thank the **nnInteractive team** for publicly releasing their code and pretrained
model. Our perceptual metric builds on their multi-stage volumetric features.

- [nnInteractive code](https://github.com/MIC-DKFZ/nnInteractive)
- [Official pretrained model](https://huggingface.co/MIC-DKFZ/nnInteractive)
- Isensee et al., *nnInteractive: Redefining 3D Promptable Segmentation* (2025),
  [arXiv:2503.08373](https://arxiv.org/abs/2503.08373).

Please cite nnInteractive when using its backbone in your research. We also thank
the [ANTs/SyN](https://github.com/ANTsX/ANTsPy) and
[Neurite/OASIS](https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md)
teams for their software and data; follow their citation instructions.

## License

Project-authored source code is released under the [MIT License](LICENSE).
