"""Model implementations exposed by RegMetric."""

from .nninteractive_metric import NNInteractiveMetric, nnInteractiveMetric

__all__ = [
    "NNInteractiveMetric",
    "nnInteractiveMetric",
    "TransMorph",
    "TransMorphBayes",
]


def __getattr__(name):
    """Load optional registration baselines only when requested."""
    if name == "TransMorph":
        from .transmorph import TransMorph

        return TransMorph
    if name == "TransMorphBayes":
        from .transmorph_bayes import TransMorphBayes

        return TransMorphBayes
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
