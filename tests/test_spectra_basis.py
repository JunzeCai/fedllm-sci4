import numpy as np
import pytest

torch = pytest.importorskip("torch")
from spectra.basis import allocate_layer_ranks, build_truncated_svd, retained_energy


def test_build_truncated_svd_shapes_and_energy():
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))

    basis = build_truncated_svd(weight, rank=2, layer_name="layer")

    assert basis.U.shape == (4, 2)
    assert basis.V.shape == (4, 2)
    assert basis.S.shape == (2,)
    assert retained_energy(torch.tensor([4.0, 3.0, 2.0, 1.0]), 1) < retained_energy(
        torch.tensor([4.0, 3.0, 2.0, 1.0]), 2
    )


def test_allocate_layer_ranks_respects_budget_and_prefers_energy_gain():
    spectra = {
        "strong": torch.tensor([5.0, 4.0, 1.0, 1.0]),
        "weak": torch.tensor([2.0, 1.0, 1.0, 1.0]),
    }

    allocation = allocate_layer_ranks(spectra, candidate_ranks=[1, 2, 4], param_budget=5)

    assert sum(rank * rank for rank in allocation.values()) <= 5
    assert allocation["strong"] >= allocation["weak"]
