import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def compute_per_class_metrics(preds, targs, num_classes):
    """
    preds, targs: 1D numpy arrays of shape (N,)
    Returns a DataFrame with per-class metrics.

    Note:
    - per-class accuracy here is defined as TP / (TP + FN),
      which equals class-wise recall (row-wise accuracy in confusion matrix).
    """
    labels = list(range(num_classes))

    # Confusion matrix: rows = true, cols = pred
    cm = confusion_matrix(targs, preds, labels=labels)

    # Per-class accuracy (class-wise recall)
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.diag(cm) / cm.sum(axis=1)
        per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

    # Per-class precision/recall/f1
    prec, rec, f1, support = precision_recall_fscore_support(
        targs, preds, labels=labels, average=None, zero_division=0
    )

    df = pd.DataFrame({
        "label": labels,
        "support": support.astype(int),
        "acc_cls": np.round(per_class_acc, 4),
        "precision": np.round(prec, 4),
        "recall": np.round(rec, 4),
        "f1": np.round(f1, 4),
    })

    return df, cm


def save_per_class_metrics_to_csv(df, cm, settings, train_iter, result_path):
    csv_dir = os.path.join(result_path, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Per-class metrics
    per_class_path = os.path.join(
        csv_dir, f"{settings}_per_class_metrics_iter{train_iter}.csv"
    )
    df.to_csv(per_class_path, index=False)

    # Confusion matrix (optional but useful)
    cm_path = os.path.join(
        csv_dir, f"{settings}_confusion_matrix_iter{train_iter}.csv"
    )
    pd.DataFrame(cm).to_csv(cm_path, index=False)

def build_per_class_columns(num_classes):
    per_class_columns = []
    for i in range(num_classes):
        per_class_columns += [
            f'c{i}_acc', f'c{i}_prec', f'c{i}_rec', f'c{i}_f1', f'c{i}_sup'
        ]
    return per_class_columns


def build_per_class_values(per_class_df, num_classes):
    """
    per_class_df columns expected:
    ['label', 'support', 'acc_cls', 'precision', 'recall', 'f1']
    Returns a flat list aligned with build_per_class_columns().
    """
    df = per_class_df.sort_values("label").reset_index(drop=True)

    per_class_values = []
    for i in range(num_classes):
        row = df.iloc[i]
        per_class_values += [
            round(float(row['acc_cls']), 4),
            round(float(row['precision']), 4),
            round(float(row['recall']), 4),
            round(float(row['f1']), 4),
            int(row['support'])
        ]
    return per_class_values
