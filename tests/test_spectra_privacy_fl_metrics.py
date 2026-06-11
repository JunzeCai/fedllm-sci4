import math

import numpy as np
import pytest

from spectra.metrics import classification_metrics

torch = pytest.importorskip("torch")
from spectra.adapter import SpectralCoreAdapter
from spectra.fl import aggregate_client_uploads, flatten_core_state, load_core_state_, make_client_upload
from spectra.privacy import calibrate_noise_std, clip_by_l2, gaussian_rdp, make_privacy_ledger


def test_privacy_accountant_monotonicity_and_clipping():
    stats = clip_by_l2(torch.tensor([3.0, 4.0]), max_norm=2.5)
    assert math.isclose(float(torch.linalg.vector_norm(stats.clipped)), 2.5)
    assert stats.was_clipped is True

    eps_low_noise = gaussian_rdp(8, clip_norm=1.0, noise_std=1.0, adjacency="replace_one")
    eps_high_noise = gaussian_rdp(8, clip_norm=1.0, noise_std=2.0, adjacency="replace_one")
    assert eps_high_noise < eps_low_noise

    noise = calibrate_noise_std(target_epsilon=4.0, delta=1e-5, round_count=2, clip_norm=1.0)
    assert noise > 0
    ledger = make_privacy_ledger(
        orders=[2, 4, 8],
        round_count=2,
        clip_norm=1.0,
        noise_std=2.0,
        delta=1e-5,
        adjacency="add_remove",
    )
    assert ledger.epsilon > 0
    assert ledger.round_count == 2


def test_fl_flatten_load_and_weighted_aggregation():
    model = torch.nn.Sequential(
        SpectralCoreAdapter(torch.nn.Linear(2, 2, bias=False), U=torch.eye(2, 1), V=torch.eye(2, 1))
    )
    with torch.no_grad():
        model[0].core[:] = torch.tensor([[2.0]])
    flat = flatten_core_state(model)
    assert torch.equal(flat, torch.tensor([2.0]))
    load_core_state_(model, torch.tensor([3.0]))
    assert torch.equal(flatten_core_state(model), torch.tensor([3.0]))

    uploads = [
        make_client_upload(client_id="a", global_state=torch.tensor([0.0]), local_state=torch.tensor([1.0]), num_examples=1),
        make_client_upload(client_id="b", global_state=torch.tensor([0.0]), local_state=torch.tensor([3.0]), num_examples=3),
    ]
    aggregated, stats = aggregate_client_uploads(uploads)
    assert torch.equal(aggregated, torch.tensor([2.5]))
    assert stats.total_examples == 4


def test_classification_metrics_matches_hand_calculation():
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
