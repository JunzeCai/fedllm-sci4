import math

from spectra.metrics import classification_metrics


def test_classification_metrics_matches_hand_calculation_without_torch():
    metrics = classification_metrics(
        y_true=["A", "A", "B", "B"],
        y_pred=["A", "B", "B", "B"],
        labels=["A", "B"],
        rare_labels=["A"],
    )

    assert math.isclose(metrics["accuracy"], 0.75)
    assert math.isclose(metrics["per_class_recall"]["A"], 0.5)
    assert math.isclose(metrics["per_class_recall"]["B"], 1.0)
    assert math.isclose(metrics["balanced_accuracy"], 0.75)
    assert math.isclose(metrics["rare_attack_recall"], 0.5)
