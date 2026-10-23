"""Protein-level metrics at the fixed classification threshold 0.5."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

def metric_dict(y, p):
    prediction = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {'roc_auc': float(roc_auc_score(y, p)), 'pr_auc': float(average_precision_score(y, p)),
            'accuracy': float(accuracy_score(y, prediction)),
            'balanced_accuracy': float(balanced_accuracy_score(y, prediction)),
            'precision': float(precision_score(y, prediction, zero_division=0)),
            'recall': float(recall_score(y, prediction, zero_division=0)),
            'specificity': float(tn / (tn + fp)), 'f1': float(f1_score(y, prediction, zero_division=0)),
            'mcc': float(matthews_corrcoef(y, prediction)),
            'brier': float(np.mean((y - p) ** 2)),
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'threshold': 0.5}
