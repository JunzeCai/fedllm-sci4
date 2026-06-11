from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import nn

from spectra.privacy import add_gaussian_noise, clip_by_l2


@dataclass(frozen=True)
class ClientUpload:
    client_id: str
    update: torch.Tensor
    num_examples: int
    original_norm: float
    clipped_norm: float
    was_clipped: bool


@dataclass(frozen=True)
class AggregationStats:
    client_count: int
    total_examples: int
    uploaded_scalars: int
    uploaded_bytes_float32: int
    clipped_clients: int


def iter_uploadable_core_parameters(model: nn.Module) -> Iterable[tuple[str, nn.Parameter]]:
    """Yield only uploadable global-path core matrices.

    Local residual cores intentionally stay on-device and are not part of the
    release vector. This distinction is central to the client-level DP claim.
    """
    for name, parameter in model.named_parameters():
        if name.endswith(".core") and "local_residual" not in name:
            yield name, parameter


def flatten_core_state(model: nn.Module) -> torch.Tensor:
    chunks = [parameter.detach().reshape(-1).cpu() for _, parameter in iter_uploadable_core_parameters(model)]
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def load_core_state_(model: nn.Module, vector: torch.Tensor) -> None:
    """Overwrite uploadable core parameters from a flat vector."""
    offset = 0
    with torch.no_grad():
        for _, parameter in iter_uploadable_core_parameters(model):
            width = parameter.numel()
            if offset + width > vector.numel():
                raise ValueError("vector is too short for model core state")
            parameter.copy_(vector[offset : offset + width].view_as(parameter).to(parameter.device, parameter.dtype))
            offset += width
    if offset != vector.numel():
        raise ValueError("vector has unused values after loading model core state")


def add_core_delta_(model: nn.Module, delta: torch.Tensor) -> None:
    current = flatten_core_state(model)
    if current.numel() != delta.numel():
        raise ValueError("delta shape does not match model core state")
    load_core_state_(model, current + delta.detach().cpu())


def make_client_upload(
    *,
    client_id: str,
    global_state: torch.Tensor,
    local_state: torch.Tensor,
    num_examples: int,
    clip_norm: float | None = None,
    noise_std: float = 0.0,
    generator: torch.Generator | None = None,
) -> ClientUpload:
    """Build a release vector from one trained local model state."""
    if global_state.shape != local_state.shape:
        raise ValueError("global_state and local_state must have the same shape")
    update = local_state.detach().cpu() - global_state.detach().cpu()
    if clip_norm is None:
        original_norm = float(torch.linalg.vector_norm(update.float()).item())
        clipped = update
        clipped_norm = original_norm
        was_clipped = False
    else:
        clipping = clip_by_l2(update, clip_norm)
        clipped = clipping.clipped
        original_norm = clipping.original_norm
        clipped_norm = clipping.clipped_norm
        was_clipped = clipping.was_clipped
    released = add_gaussian_noise(clipped, noise_std, generator=generator)
    return ClientUpload(
        client_id=client_id,
        update=released,
        num_examples=int(num_examples),
        original_norm=original_norm,
        clipped_norm=clipped_norm,
        was_clipped=was_clipped,
    )


def weighted_average_updates(updates: Sequence[torch.Tensor], weights: Sequence[int | float] | None = None) -> torch.Tensor:
    if not updates:
        raise ValueError("updates must not be empty")
    first_shape = updates[0].shape
    if any(update.shape != first_shape for update in updates):
        raise ValueError("all updates must have the same shape")
    if weights is None:
        weights_tensor = torch.ones(len(updates), dtype=torch.float64)
    else:
        if len(weights) != len(updates):
            raise ValueError("weights length must match updates length")
        weights_tensor = torch.as_tensor(weights, dtype=torch.float64)
    total = torch.sum(weights_tensor).clamp_min(1e-12)
    result = torch.zeros_like(updates[0], dtype=torch.float64)
    for update, weight in zip(updates, weights_tensor):
        result += update.detach().cpu().double() * weight
    return (result / total).to(dtype=updates[0].dtype)


def aggregate_client_uploads(uploads: Sequence[ClientUpload], *, weight_by_examples: bool = True) -> tuple[torch.Tensor, AggregationStats]:
    if not uploads:
        raise ValueError("uploads must not be empty")
    weights = [upload.num_examples for upload in uploads] if weight_by_examples else None
    average = weighted_average_updates([upload.update for upload in uploads], weights=weights)
    total_examples = sum(upload.num_examples for upload in uploads)
    stats = AggregationStats(
        client_count=len(uploads),
        total_examples=total_examples,
        uploaded_scalars=int(sum(upload.update.numel() for upload in uploads)),
        uploaded_bytes_float32=int(sum(upload.update.numel() for upload in uploads) * 4),
        clipped_clients=sum(1 for upload in uploads if upload.was_clipped),
    )
    return average, stats


def summarize_uploads(uploads: Sequence[ClientUpload]) -> dict[str, float | int]:
    if not uploads:
        return {"client_count": 0, "mean_original_norm": 0.0, "mean_clipped_norm": 0.0, "clipped_clients": 0}
    return {
        "client_count": len(uploads),
        "mean_original_norm": float(sum(upload.original_norm for upload in uploads) / len(uploads)),
        "mean_clipped_norm": float(sum(upload.clipped_norm for upload in uploads) / len(uploads)),
        "clipped_clients": sum(1 for upload in uploads if upload.was_clipped),
    }


def state_dict_from_flat(model: nn.Module, vector: torch.Tensor) -> Mapping[str, torch.Tensor]:
    """Convert a flat core vector to a name->tensor mapping without mutating the model."""
    offset = 0
    state: dict[str, torch.Tensor] = {}
    for name, parameter in iter_uploadable_core_parameters(model):
        width = parameter.numel()
        state[name] = vector[offset : offset + width].view_as(parameter).detach().clone()
        offset += width
    if offset != vector.numel():
        raise ValueError("vector has unused values")
    return state
