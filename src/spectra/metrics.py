from __future__ import annotations

from collections import Counter
from typing import Sequence


def classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str] | None = None,
    rare_labels: Sequence[str] | None = None,
) -> dict[str, object]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("at least one sample is required")

    label_list = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    correct = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == pred)
    per_class_recall = {}
    per_class_precision = {}
    per_class_f1 = {}
    for label in label_list:
        tp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth == label and pred == label)
        fn = sum(1 for truth in y_true if truth == label) - tp
        fp = sum(1 for truth, pred in zip(y_true, y_pred, strict=True) if truth != label and pred == label)
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_recall[label] = recall
        per_class_precision[label] = precision
        per_class_f1[label] = f1

    rare = list(rare_labels or [])
    rare_recall = (
        sum(per_class_recall.get(label, 0.0) for label in rare) / len(rare)
        if rare
        else 0.0
    )
    support = Counter(y_true)
    weighted_f1 = sum(per_class_f1[label] * support.get(label, 0) for label in label_list) / len(y_true)
    return {
        "accuracy": correct / len(y_true),
        "balanced_accuracy": sum(per_class_recall.values()) / len(label_list) if label_list else 0.0,
        "macro_f1": sum(per_class_f1.values()) / len(label_list) if label_list else 0.0,
        "weighted_f1": weighted_f1,
        "rare_attack_recall": rare_recall,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "per_class_f1": per_class_f1,
    }
