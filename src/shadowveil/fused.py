"""Fused Mamba-1 continuation (Triton selective scan) for HF Mamba checkpoints.

One code path serves every operation the lookahead family needs:
    chunk(cache, ids)  -> continue a (possibly non-zero) cache by n tokens in ONE call,
    n = 1              -> a decode step,
    n = L - delta      -> the burst that builds the shadow from the s0 snapshot.
The sequential part of the selective scan is a Triton kernel that loops over the n tokens with the SSM
state held in registers (the same recurrence as HF `slow_forward`, fp32 accumulation, state stored in the
cache dtype after the chunk). Projections, convolution and norms use HF's own modules, batched over the
chunk. Decode steps can additionally be captured in a CUDA graph to remove launch overhead.

Bitwise equality is relative to this code path (same schedule): chunking changes where the fp32 state is
rounded to fp16, so different chunkings agree numerically (~1e-3) but not bit for bit.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _ssm_scan_kernel(state_ptr, x_ptr, dt_ptr, a_ptr, b_ptr, c_ptr, d_ptr, z_ptr, y_ptr,
                     n_tokens, d_inner, N: tl.constexpr, BLOCK_D: tl.constexpr):
    pid_d = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < d_inner
    offs_n = tl.arange(0, N)
    sp = state_ptr + pid_b * d_inner * N + offs_d[:, None] * N + offs_n[None, :]
    st = tl.load(sp, mask=mask_d[:, None], other=0.0).to(tl.float32)
    a = tl.load(a_ptr + offs_d[:, None] * N + offs_n[None, :], mask=mask_d[:, None], other=0.0)
    dv = tl.load(d_ptr + offs_d, mask=mask_d, other=0.0)
    for t in range(n_tokens):
        row = pid_b * n_tokens + t
        x = tl.load(x_ptr + row * d_inner + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        dt = tl.load(dt_ptr + row * d_inner + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        dt = tl.where(dt <= 20.0, tl.log(1.0 + tl.exp(dt)), dt)
        z = tl.load(z_ptr + row * d_inner + offs_d, mask=mask_d, other=0.0).to(tl.float32)
        bv = tl.load(b_ptr + row * N + offs_n).to(tl.float32)
        cv = tl.load(c_ptr + row * N + offs_n).to(tl.float32)
        st = st * tl.exp(a * dt[:, None]) + (dt * x)[:, None] * bv[None, :]
        y = tl.sum(st * cv[None, :], axis=1) + dv * x
        y = y * (z * tl.sigmoid(z))
        tl.store(y_ptr + row * d_inner + offs_d, y.to(y_ptr.dtype.element_ty), mask=mask_d)
    tl.store(sp, st.to(state_ptr.dtype.element_ty), mask=mask_d[:, None])


class FusedMamba:
    def __init__(self, model):
        self.model = model
        self.cfg = model.config
        self.dev = next(model.parameters()).device
        self.A = [(-torch.exp(blk.mixer.A_log.float())).contiguous() for blk in model.backbone.layers]
        self.D = [blk.mixer.D.float().contiguous() for blk in model.backbone.layers]
        self.graphs = {}
        self.token_transitions = 0      # measured: token x row transitions executed by forward()

    @torch.inference_mode()
    def _mixer(self, layer, mixer, h, cache):
        Bsz, n, _ = h.shape
        proj = mixer.in_proj(h).transpose(1, 2)                          # [B, 2I, n]
        xs, gate = proj.chunk(2, dim=1)
        K = mixer.conv_kernel_size
        conv_state = cache.conv_states[layer]
        ext = torch.cat([conv_state[..., 1:].to(xs.dtype), xs], dim=-1)
        bias = mixer.conv1d.bias if mixer.use_conv_bias else None
        conv = F.conv1d(ext, mixer.conv1d.weight, bias, groups=mixer.intermediate_size)
        cache.conv_states[layer].copy_(torch.cat([conv_state, xs.to(conv_state.dtype)], dim=-1)[..., -K:])
        xa = F.silu(conv).transpose(1, 2).contiguous()                   # [B, n, I]
        ssm = mixer.x_proj(xa)
        ts, Bm, Cm = torch.split(ssm, [mixer.time_step_rank, mixer.ssm_state_size, mixer.ssm_state_size], dim=-1)
        dt = mixer.dt_proj(ts).contiguous()                              # pre-softplus, bias included
        z = gate.transpose(1, 2).contiguous()
        y = torch.empty_like(xa)
        I, N = mixer.intermediate_size, mixer.ssm_state_size
        BLOCK_D = 32
        _ssm_scan_kernel[(triton.cdiv(I, BLOCK_D), Bsz)](
            cache.ssm_states[layer], xa, dt, self.A[layer], Bm.contiguous(), Cm.contiguous(), self.D[layer], z, y,
            n, I, N=N, BLOCK_D=BLOCK_D)
        return mixer.out_proj(y)

    @torch.inference_mode()
    def forward(self, cache, ids):
        """ids: LongTensor [B, n]. Updates cache in place; returns logits [B, n, V] (fp32)."""
        m = self.model
        self.token_transitions += int(ids.numel())
        h = m.backbone.embeddings(ids)
        for layer, blk in enumerate(m.backbone.layers):
            residual = h
            x = blk.norm(h.to(dtype=blk.norm.weight.dtype))
            if blk.residual_in_fp32:
                residual = residual.to(torch.float32)
            h = residual + self._mixer(layer, blk.mixer, x, cache)
        h = m.backbone.norm_f(h)
        return m.lm_head(h.to(m.lm_head.weight.dtype)).float()

    # ---- CUDA-graph decode step (n = 1) ------------------------------------------------------------
    @torch.inference_mode()
    def capture_step(self, cache, batch):
        """Capture a one-token step bound to `cache`'s buffers. Returns (replay_fn, ids_buffer, logits_buffer)."""
        ids = torch.zeros((batch, 1), dtype=torch.long, device=self.dev)
        for _ in range(3):                                               # warm up kernels / allocator
            self.forward(cache, ids)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = self.forward(cache, ids)
        def replay(token_ids=None):
            with torch.inference_mode():
                if token_ids is not None:
                    ids.copy_(token_ids)
                g.replay()
            return out
        return replay, ids, out
