"""Model registry for runtime model construction."""

from .majority import MajorityBaseline
from .tfidf_logreg import TfidfLogRegModel


def build_model(model_cfg: dict):
    """Build a model instance from configuration."""
    name = model_cfg.get("name")
    if name == "majority":
        return MajorityBaseline()
    if name == "tfidf_logreg":
        return TfidfLogRegModel(
            model_cfg.get("tfidf", {}),
            model_cfg.get("classifier", {}),
        )
    if name == "transformer":
        from .transformer_classifier import TransformerClassifier

        return TransformerClassifier(model_cfg)
    raise ValueError(f"unknown model: {name}")
