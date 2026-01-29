from __future__ import annotations

from numpy.typing import ArrayLike
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    matrix = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_det": float(matrix.trace() / matrix.sum()),
    }
