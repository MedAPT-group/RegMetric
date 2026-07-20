# Model weights

Model checkpoints are intentionally not stored in Git. Download the required
nnInteractive or registration checkpoints separately and pass their local path
to the model constructor.

The original `checkpoint_final.pth` is about 392 MB and exceeds GitHub's normal
100 MB per-file limit. Publish large checkpoints through GitHub Releases,
Hugging Face, Zenodo, or another artifact host.
