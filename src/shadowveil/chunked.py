"""Multi-token continuation of a Mamba-1 cache from a NON-ZERO state.

HF Transformers 4.51 supports (a) prefill from an empty cache and (b) one-token decode.
Its prefill branch pads the conv input with zeros, so continuing a populated cache with
several tokens would silently ignore the previous K-1 conv inputs. The lookahead
family needs exactly that operation (burst from a snapshot s0 != 0), so we implement it
here from the HF submodules: conv window prepended from the cache, SSM initialised from
the cache, the same sequential scan arithmetic as HF `slow_forward`.

This is a research reimplementation of the unfused path, not a fused kernel.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers.cache_utils import MambaCache


def clone_cache(cache: MambaCache, config) -> MambaCache:
    clone = MambaCache(
        config, max_batch_size=cache.max_batch_size,
        dtype=cache.ssm_states[0].dtype, device=cache.ssm_states[0].device,
    )
    for dst, src in zip(clone.ssm_states, cache.ssm_states):
        dst.copy_(src)
    for dst, src in zip(clone.conv_states, cache.conv_states):
        dst.copy_(src)
    return clone


def cache_tensors(cache: MambaCache):
    return cache.ssm_states + cache.conv_states


def caches_equal(a: MambaCache, b: MambaCache) -> bool:
    return all(torch.equal(x, y) for x, y in zip(cache_tensors(a), cache_tensors(b)))


def cache_max_diff(a: MambaCache, b: MambaCache) -> float:
    return max(float((x.float() - y.float()).abs().max()) for x, y in zip(cache_tensors(a), cache_tensors(b)))


def cache_bytes(cache: MambaCache) -> int:
    return sum(t.numel() * t.element_size() for t in cache_tensors(cache))


@torch.inference_mode()
def prefill(model, ids, device):
    """Standard HF prefill from an empty state (the 'parallel' path)."""
    out = model(input_ids=torch.tensor([ids], dtype=torch.long, device=device), use_cache=True, return_dict=True)
    return out.cache_params, out.logits


@torch.inference_mode()
def step(model, cache, position, token_id, device):
    """Standard HF one-token decode (the 'recurrent' path). Updates `cache` in place."""
    out = model(
        input_ids=torch.tensor([[token_id]], dtype=torch.long, device=device),
        cache_params=cache, cache_position=torch.tensor([position], device=device),
        use_cache=True, return_dict=True,
    )
    return out.cache_params, out.logits


def _mixer_chunk(mixer, hidden, cache: MambaCache):
    """hidden: [B, n, D] already normed. Updates cache conv/ssm state in place."""
    dtype = hidden.dtype
    layer = mixer.layer_idx
    projected = mixer.in_proj(hidden).transpose(1, 2)               # [B, 2I, n]
    x, gate = projected.chunk(2, dim=1)                              # [B, I, n]
    K = mixer.conv_kernel_size
    conv_state = cache.conv_states[layer]                            # [B, I, K] = last K conv inputs
    ext = torch.cat([conv_state[..., 1:].to(x.dtype), x], dim=-1)    # previous K-1 inputs + new chunk
    bias = mixer.conv1d.bias if mixer.use_conv_bias else None
    conv_out = F.conv1d(ext, mixer.conv1d.weight, bias, groups=mixer.intermediate_size)  # [B, I, n]
    cache.conv_states[layer].copy_(torch.cat([conv_state, x.to(conv_state.dtype)], dim=-1)[..., -K:])
    hs = mixer.act(conv_out)

    ssm_parameters = mixer.x_proj(hs.transpose(1, 2))
    time_step, Bm, Cm = torch.split(
        ssm_parameters, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1
    )
    dt = F.softplus(mixer.dt_proj(time_step)).transpose(1, 2)        # [B, I, n]
    A = -torch.exp(mixer.A_log.float())
    discrete_A = torch.exp(A[None, :, None, :] * dt[:, :, :, None])
    discrete_B = dt[:, :, :, None] * Bm[:, None, :, :].float()
    deltaB_u = discrete_B * hs[:, :, :, None].float()

    ssm_state = cache.ssm_states[layer].clone()
    outs = []
    for i in range(hs.shape[-1]):
        ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
        outs.append(torch.matmul(ssm_state.to(dtype), Cm[:, i, :].unsqueeze(-1))[:, :, 0])
    scan = torch.stack(outs, dim=-1) + hs * mixer.D[None, :, None]
    scan = scan * mixer.act(gate)
    cache.ssm_states[layer].copy_(ssm_state)
    return mixer.out_proj(scan.transpose(1, 2))


@torch.inference_mode()
def chunk(model, cache: MambaCache, token_ids, device):
    """Continue `cache` (possibly non-zero) by len(token_ids) tokens in one call. In place."""
    ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    hidden = model.backbone.embeddings(ids)
    for block in model.backbone.layers:
        residual = hidden
        h = block.norm(hidden.to(dtype=block.norm.weight.dtype))
        if block.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden = residual + _mixer_chunk(block.mixer, h, cache)
    hidden = model.backbone.norm_f(hidden)
    return model.lm_head(hidden.to(model.lm_head.weight.dtype)).float()
