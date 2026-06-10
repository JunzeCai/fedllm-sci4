from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AdapterStats:
    trainable_parameters: int
    uploaded_scalars: int
    uploaded_bytes_float32: int


class SpectralCoreAdapter(nn.Module):
    """Frozen public-basis core adapter.

    For a frozen base linear map y = x W_0^T + b, SPECTRA-FedCore uses

        Delta W_l = gamma_l U_{l,p} C_l V_{l,p}^T,

    where U and V are public SVD bases and C_l is the only trainable/uploadable
    parameter. Optional local residual cores can be kept on-device and excluded
    from uploads to support personalization without spending communication or
    client-level release-DP budget.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        U: torch.Tensor,
        V: torch.Tensor,
        *,
        gamma: float = 1.0,
        use_local_residual: bool = False,
        residual_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if base_linear.weight.ndim != 2:
            raise ValueError("base_linear must have a 2D weight")
        if U.ndim != 2 or V.ndim != 2:
            raise ValueError("U and V must be 2D tensors")
        if U.shape[0] != base_linear.out_features:
            raise ValueError("U.shape[0] must match base layer out_features")
        if V.shape[0] != base_linear.in_features:
            raise ValueError("V.shape[0] must match base layer in_features")
        if U.shape[1] != V.shape[1]:
            raise ValueError("U and V must have the same rank")

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.gamma = float(gamma)
        self.residual_scale = float(residual_scale)
        self.use_local_residual = bool(use_local_residual)

        self.register_buffer("base_weight", base_linear.weight.detach().clone(), persistent=False)
        if base_linear.bias is None:
            self.register_buffer("base_bias", None, persistent=False)
        else:
            self.register_buffer("base_bias", base_linear.bias.detach().clone(), persistent=False)
        self.register_buffer("U", U.detach().float().clone(), persistent=False)
        self.register_buffer("V", V.detach().float().clone(), persistent=False)

        rank = U.shape[1]
        self.core = nn.Parameter(torch.zeros(rank, rank))
        if self.use_local_residual:
            self.local_residual = nn.Parameter(torch.zeros(rank, rank))
        else:
            self.register_parameter("local_residual", None)

    @property
    def rank(self) -> int:
        return int(self.core.shape[0])

    def uploadable_core(self) -> torch.Tensor:
        return self.core

    def delta_weight(self, *, include_local_residual: bool = True) -> torch.Tensor:
        core = self.core
        if include_local_residual and self.local_residual is not None:
            core = core + self.residual_scale * self.local_residual
        U = self.U.to(device=core.device, dtype=core.dtype)
        V = self.V.to(device=core.device, dtype=core.dtype)
        return self.gamma * (U @ core @ V.T)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.base_weight.to(device=x.device, dtype=x.dtype)
        delta = self.delta_weight(include_local_residual=True).to(device=x.device, dtype=x.dtype)
        bias = None if self.base_bias is None else self.base_bias.to(device=x.device, dtype=x.dtype)
        return torch.nn.functional.linear(x, base_weight + delta, bias)

    def stats(self, *, dtype_bytes: int = 4) -> AdapterStats:
        uploaded_scalars = int(self.core.numel())
        return AdapterStats(
            trainable_parameters=sum(param.numel() for param in self.parameters() if param.requires_grad),
            uploaded_scalars=uploaded_scalars,
            uploaded_bytes_float32=uploaded_scalars * dtype_bytes,
        )
