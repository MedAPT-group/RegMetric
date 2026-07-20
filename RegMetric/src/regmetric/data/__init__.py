"""Dataset loaders for the datasets used by RegMetric."""

from .copdgene import COPDGDataset
from .l2r_abdomen import AbdomenPairDataset
from .oasis import OASISTestAtlasDataset, OASISTrainInfiniteDataset

__all__ = [
    "COPDGDataset",
    "AbdomenPairDataset",
    "OASISTestAtlasDataset",
    "OASISTrainInfiniteDataset",
]
