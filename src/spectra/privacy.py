from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch


DEFAULT_RDP_ORDERS = (1.25, 1.5, 2, 3, 4, 5, 8, 16, 32, 64, 128, 256)


@dataclass(frozen=True)
class ClippingResult:
    clipped: torch.Tensor
    original_norm: float
    clipped_norm: float
    was_clipped: bool
    scale: float


@dataclass(frozen=True)
class PrivacyLedger:
    epsilon: float
    best_order: float
    delta: float
    rdp: dict[float, float]
    adjacency: str
    round_count: int
    clip_norm: float
    noise_std: float
    sensitivity: float
    notes: str


def clip_by_l2(vector: torch.Tensor, max_norm: float) -> ClippingResult:
    """Clip a flat client update vector by its L2 norm."""
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    norm = float(torch.linalg.vector_norm(vector.detach().float()).item())
    scale = min(1.0, max_norm / max(norm, 1e-12))
    clipped = vector * scale
    return ClippingResult(
        clipped=clipped,
        original_norm=norm,
        clipped_norm=float(torch.linalg.vector_norm(clipped.detach().float()).item()),
        was_clipped=scale < 1.0,
        scale=scale,
    )


def gaussian_sensitivity(clip_norm: float, *, adjacency: str = "replace_one") -> float:
    """Sensitivity of a clipped client upload under the declared client adjacency.

    replace_one: two neighboring datasets replace one participating client, so
    sensitivity is 2R. add_remove: neighboring datasets differ by one participating
    client, so sensitivity is R.
    """
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    if adjacency == "replace_one":
        return 2.0 * clip_norm
    if adjacency == "add_remove":
        return clip_norm
    raise ValueError("adjacency must be 'replace_one' or 'add_remove'")


def gaussian_rdp(alpha: float, *, clip_norm: float, noise_std: float, adjacency: str = "replace_one") -> float:
    """RDP of one client-side Gaussian release without subsampling amplification.

    For a Gaussian mechanism with L2 sensitivity Delta and isotropic noise sigma,
    epsilon_RDP(alpha) = alpha * Delta^2 / (2 sigma^2).
    """
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    if noise_std <= 0:
        raise ValueError("noise_std must be positive")
    sensitivity = gaussian_sensitivity(clip_norm, adjacency=adjacency)
    return float(alpha * sensitivity**2 / (2.0 * noise_std**2))


def compose_gaussian_rdp(
    *,
    round_count: int,
    clip_norm: float,
    noise_std: float,
    orders: Iterable[float] = DEFAULT_RDP_ORDERS,
    adjacency: str = "replace_one",
) -> dict[float, float]:
    """Conservative sequential composition over participating rounds."""
    if round_count < 0:
        raise ValueError("round_count must be non-negative")
    return {
        float(order): round_count * gaussian_rdp(
            float(order), clip_norm=clip_norm, noise_std=noise_std, adjacency=adjacency
        )
        for order in orders
    }


def rdp_to_epsilon(rdp: Mapping[float, float], *, delta: float) -> tuple[float, float]:
    """Convert RDP values to the tightest epsilon over the provided orders."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if not rdp:
        raise ValueError("rdp must not be empty")
    best_epsilon = float("inf")
    best_order = float("nan")
    for order, rho in rdp.items():
        order = float(order)
        if order <= 1:
            continue
        epsilon = float(rho) + math.log(1.0 / delta) / (order - 1.0)
        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_order = order
    if not math.isfinite(best_epsilon):
        raise ValueError("no valid RDP order was provided")
    return best_epsilon, best_order


def make_privacy_ledger(
    *,
    round_count: int,
    clip_norm: float,
    noise_std: float,
    delta: float,
    orders: Iterable[float] = DEFAULT_RDP_ORDERS,
    adjacency: str = "replace_one",
) -> PrivacyLedger:
    """Build an auditable conservative privacy ledger for client-level DP."""
    rdp = compose_gaussian_rdp(
        round_count=round_count,
        clip_norm=clip_norm,
        noise_std=noise_std,
        orders=orders,
        adjacency=adjacency,
    )
    epsilon, best_order = rdp_to_epsilon(rdp, delta=delta)
    return PrivacyLedger(
        epsilon=epsilon,
        best_order=best_order,
        delta=delta,
        rdp=rdp,
        adjacency=adjacency,
        round_count=round_count,
        clip_norm=clip_norm,
        noise_std=noise_std,
        sensitivity=gaussian_sensitivity(clip_norm, adjacency=adjacency),
        notes="Conservative client-level Gaussian release accounting without subsampling amplification.",
    )


def calibrate_noise_std(
    *,
    target_epsilon: float,
    delta: float,
    round_count: int,
    clip_norm: float,
    orders: Iterable[float] = DEFAULT_RDP_ORDERS,
    adjacency: str = "replace_one",
    tol: float = 1e-4,
    max_steps: int = 80,
) -> float:
    """Binary-search a Gaussian noise std that reaches target epsilon or lower."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    low, high = 1e-12, max(1.0, gaussian_sensitivity(clip_norm, adjacency=adjacency))

    def epsilon_for(sigma: float) -> float:
        ledger = make_privacy_ledger(
            round_count=round_count,
            clip_norm=clip_norm,
            noise_std=sigma,
            delta=delta,
            orders=orders,
            adjacency=adjacency,
        )
        return ledger.epsilon

    while epsilon_for(high) > target_epsilon:
        high *= 2.0

    for _ in range(max_steps):
        mid = (low + high) / 2.0
        eps = epsilon_for(mid)
        if abs(eps - target_epsilon) <= tol:
            return mid
        if eps > target_epsilon:
            low = mid
        else:
            high = mid
    return high


def add_gaussian_noise(vector: torch.Tensor, noise_std: float, *, generator: torch.Generator | None = None) -> torch.Tensor:
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if noise_std == 0:
        return vector.clone()
    noise = torch.normal(
        mean=0.0,
        std=noise_std,
        size=vector.shape,
        generator=generator,
        device=vector.device,
        dtype=vector.dtype,
    )
    return vector + noise
