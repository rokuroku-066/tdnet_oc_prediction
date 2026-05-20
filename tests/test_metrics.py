from tdnet_oc_prediction.evaluation.metrics import classification_metrics

def test_metrics():
    m = classification_metrics([0,1,1,0], [0.1,0.7,0.6,0.2], 0.5)
    for k in ['accuracy','balanced_accuracy','roc_auc','log_loss']:
        assert k in m
