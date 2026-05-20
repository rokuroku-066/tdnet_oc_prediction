import math
from tdnet_oc_prediction.evaluation.metrics import classification_metrics


def test_metrics():
    m = classification_metrics([0,1,1,0], [0.1,0.7,0.6,0.2], 0.5)
    for k in ['accuracy','balanced_accuracy','roc_auc','log_loss']:
        assert k in m


def test_metrics_single_class_split():
    m = classification_metrics([1,1,1], [0.8,0.9,0.7], 0.5)
    assert math.isnan(m['roc_auc'])
    assert math.isnan(m['log_loss'])
