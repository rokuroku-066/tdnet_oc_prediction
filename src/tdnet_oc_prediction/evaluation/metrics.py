import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, log_loss, brier_score_loss


def classification_metrics(y_true, y_proba, threshold=0.5):
    y_true_arr = np.array(y_true)
    y_proba_arr = np.array(y_proba)
    y_pred = (y_proba_arr >= threshold).astype(int)
    unique_labels = np.unique(y_true_arr)
    has_both_classes = unique_labels.size > 1

    return {
        'accuracy': float(accuracy_score(y_true_arr, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true_arr, y_pred)),
        'precision': float(precision_score(y_true_arr, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true_arr, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true_arr, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true_arr, y_proba_arr)) if has_both_classes else float('nan'),
        'pr_auc': float(average_precision_score(y_true_arr, y_proba_arr)),
        'log_loss': float(log_loss(y_true_arr, y_proba_arr, labels=[0, 1])) if has_both_classes else float('nan'),
        'brier_score': float(brier_score_loss(y_true_arr, y_proba_arr)),
    }
