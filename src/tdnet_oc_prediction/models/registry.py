from .majority import MajorityBaseline
from .tfidf_logreg import TfidfLogRegModel
from .transformer_classifier import TransformerClassifier

def build_model(model_cfg: dict):
    name = model_cfg.get('name')
    if name == 'majority': return MajorityBaseline()
    if name == 'tfidf_logreg': return TfidfLogRegModel(model_cfg.get('tfidf',{}), model_cfg.get('classifier',{}))
    if name == 'transformer': return TransformerClassifier(model_cfg)
    raise ValueError(f'unknown model: {name}')
