"""Online dual-cache runtime for one scheduled Mamba revocation domain.

The runtime operates on token IDs so tokenization is fixed before a branch is
created. It maintains independent live and sanitized caches and independent
positions, then selects the sanitized cache and retires the live cache.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers.cache_utils import MambaCache


def clone_mamba_cache(cache: MambaCache, config) -> MambaCache:
    clone = MambaCache(
        config,
        max_batch_size=cache.max_batch_size,
        dtype=cache.ssm_states[0].dtype,
        device=cache.ssm_states[0].device,
    )
    for target, source in zip(clone.ssm_states, cache.ssm_states):
        target.copy_(source)
    for target, source in zip(clone.conv_states, cache.conv_states):
        target.copy_(source)
    return clone


def cache_equal(left: MambaCache, right: MambaCache) -> bool:
    left_tensors = left.ssm_states + left.conv_states
    right_tensors = right.ssm_states + right.conv_states
    return len(left_tensors) == len(right_tensors) and all(
        torch.equal(a, b) for a, b in zip(left_tensors, right_tensors)
    )


def cache_max_difference(left: MambaCache, right: MambaCache) -> float:
    return max(
        float((a.float() - b.float()).abs().max())
        for a, b in zip(
            left.ssm_states + left.conv_states,
            right.ssm_states + right.conv_states,
        )
    )


@dataclass
class Branch:
    cache: MambaCache
    position: int


class ShadowMambaSession:
    """One live branch and one sanitized branch with a scheduled expiry."""

    def __init__(self, model, common_ids: list[int], device: str):
        if not common_ids:
            raise ValueError("common_ids must contain at least one token")
        self.model = model
        self.device = device
        self.live: Branch | None = None
        self.shadow: Branch | None = None
        self.active: Branch | None = None
        self._start(common_ids)

    @torch.inference_mode()
    def _start(self, common_ids: list[int]) -> None:
        input_ids = torch.tensor([common_ids], dtype=torch.long, device=self.device)
        output = self.model(input_ids=input_ids, use_cache=True, return_dict=True)
        shared = output.cache_params
        position = len(common_ids)
        self.live = Branch(clone_mamba_cache(shared, self.model.config), position)
        self.shadow = Branch(clone_mamba_cache(shared, self.model.config), position)

    @torch.inference_mode()
    def advance(self, branch: Branch, token_ids: list[int]) -> None:
        for token_id in token_ids:
            input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
            cache_position = torch.tensor([branch.position], device=self.device)
            output = self.model(
                input_ids=input_ids,
                cache_params=branch.cache,
                cache_position=cache_position,
                use_cache=True,
                return_dict=True,
            )
            branch.cache = output.cache_params
            branch.position += 1

    def advance_live(self, token_ids: list[int]) -> None:
        if self.live is None:
            raise RuntimeError("live branch has already been retired")
        self.advance(self.live, token_ids)

    def advance_shadow(self, token_ids: list[int]) -> None:
        if self.shadow is None:
            raise RuntimeError("shadow branch is unavailable")
        self.advance(self.shadow, token_ids)

    @torch.inference_mode()
    def expire(self) -> dict:
        if self.live is None or self.shadow is None:
            raise RuntimeError("session has already expired")
        if self.device == "cuda":
            torch.cuda.synchronize()
        started_ns = time.perf_counter_ns()
        self.active = self.shadow
        selection_ns = time.perf_counter_ns() - started_ns
        if self.device == "cuda":
            torch.cuda.synchronize()
        retire_started = time.perf_counter()
        for tensor in self.live.cache.ssm_states + self.live.cache.conv_states:
            tensor.zero_()
        if self.device == "cuda":
            torch.cuda.synchronize()
        retirement_ms = 1000 * (time.perf_counter() - retire_started)
        retired_max = max(
            float(tensor.float().abs().max())
            for tensor in self.live.cache.ssm_states + self.live.cache.conv_states
        )
        self.live = None
        return {
            "selection_ns": selection_ns,
            "retirement_ms": retirement_ms,
            "retired_cache_max_absolute_value": retired_max,
        }

    def advance_active(self, token_ids: list[int]) -> None:
        if self.active is None:
            raise RuntimeError("expire must be called before advancing the active branch")
        self.advance(self.active, token_ids)

