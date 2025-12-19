import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, precision_recall_fscore_support
)
import pandas as pd

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.targets = []
        self.predictions = []

    def update(self, preds, targs):
        preds = preds.cpu().numpy()
        targs = targs.cpu().numpy()
        self.predictions.append(preds)
        self.targets.append(targs)

    def _concat(self):
        preds = np.concatenate(self.predictions, axis=0) if len(self.predictions) else np.array([])
        targs = np.concatenate(self.targets, axis=0) if len(self.targets) else np.array([])
        return preds, targs

    def compute(self):
        preds, targs = self._concat()
        accuracy = round(accuracy_score(targs, preds), 4)
        precision = round(precision_score(targs, preds, average='macro', zero_division=0), 4)
        recall = round(recall_score(targs, preds, average='macro', zero_division=0), 4)
        f1 = round(f1_score(targs, preds, average='macro', zero_division=0), 4)
        return accuracy, precision, recall, f1

    def compute_per_class(self, num_classes):
        preds, targs = self._concat()
        labels = list(range(num_classes))

        cm = confusion_matrix(targs, preds, labels=labels)

        # class-wise accuracy = TP/(TP+FN) = class-wise recall
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_acc = np.diag(cm) / cm.sum(axis=1)
            per_class_acc = np.nan_to_num(per_class_acc, nan=0.0)

        prec, rec, f1, support = precision_recall_fscore_support(
            targs, preds, labels=labels, average=None, zero_division=0
        )

        per_class_df = pd.DataFrame({
            "label": labels,
            "support": support.astype(int),
            "acc_cls": np.round(per_class_acc, 4),
            "precision": np.round(prec, 4),
            "recall": np.round(rec, 4),
            "f1": np.round(f1, 4),
        })

        return per_class_df, cm
