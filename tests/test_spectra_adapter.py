import numpy as np
import pytest

torch = pytest.importorskip("torch")
from spectra.adapter import SpectralCoreAdapter


def test_spectral_core_adapter_delta_and_uploadable_state_exclude_residual():
    base = torch.nn.Linear(4, 3, bias=False)
    adapter = SpectralCoreAdapter(
        base,
        U=torch.eye(3, 2),
        V=torch.eye(4, 2),
        gamma=0.5,
        use_local_residual=True,
        residual_scale=0.2,
    )
    with torch.no_grad():
        adapter.core[:] = torch.ones((2, 2))
        adapter.local_residual[:] = torch.full((2, 2), 3.0)

    delta = adapter.delta_weight(include_local_residual=True)
    stats = adapter.stats()

    assert delta.shape == (3, 4)
    assert adapter.uploadable_core().shape == (2, 2)
    assert stats.uploaded_scalars == 4
