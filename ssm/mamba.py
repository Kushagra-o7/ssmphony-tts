"""Mamba (S6, selective state space) mixer in pure PyTorch.

Difference from S4: Delta, B and C are functions of the input x_t, so the
system is time-varying and the convolution view no longer exists.  The
recurrence

    h_t = exp(Delta_t A) h_{t-1} + Delta_t B_t x_t,     y_t = C_t h_t + D x_t

is still linear in h, so it is evaluated with the same associative parallel
scan (ssm/scan.py).  The official implementation fuses this into a CUDA kernel
that keeps h in SRAM; that kernel does not build on Windows, so here the scan
materialises h in GPU memory (B, D_inner, N, L) and memory use is higher.
"""
import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from .scan import scan_blelloch


LOG_A_MIN = -5.0  # per-step decay floor: exp(-5) = 0.007, i.e. an effectively full reset


def _scan_chunk(dt, x, Bm, Cm, A, h0):
    """Selective scan over one chunk with the Blelloch parallel scan.
    dt,x: (B,T,di), Bm,Cm: (B,T,N), A: (di,N), h0: (B,di,N)."""
    dA = torch.exp((dt.unsqueeze(-1) * A).clamp(min=LOG_A_MIN)).permute(0, 2, 3, 1)  # (B,di,N,T)
    dBx = ((dt * x).unsqueeze(-1) * Bm.unsqueeze(2)).permute(0, 2, 3, 1)            # (B,di,N,T)
    h = scan_blelloch(dA, dBx, h0)
    return torch.einsum("bdnt,btn->btd", h, Cm), h[..., -1]


def _scan_chunk_logspace(dt, x, Bm, Cm, A, h0):
    """Same recurrence via cumulative sums (fewer kernel launches than Blelloch):
        S_t = sum_{k<=t} log a_k,   h_t = exp(S_t) * (h0 + sum_{k<=t} exp(-S_k) b_k)
    The decay floor keeps exp(-S) finite in fp32 for chunks of <= 16 steps."""
    logA = (dt.unsqueeze(-1) * A).clamp(min=LOG_A_MIN)                             # (B,T,di,N)
    S = torch.cumsum(logA, dim=1)
    b = (dt * x).unsqueeze(-1) * Bm.unsqueeze(2)                                    # (B,T,di,N)
    h = torch.exp(S) * (h0.unsqueeze(1) + torch.cumsum(torch.exp(-S) * b, dim=1))
    return torch.einsum("btdn,btn->btd", h, Cm), h[:, -1]


SCAN_IMPL = {"blelloch": (_scan_chunk, 64), "logspace": (_scan_chunk_logspace, 16)}


def selective_scan(dt, x, Bm, Cm, A, impl="logspace"):
    """Chunked selective scan: parallel (Blelloch) scan inside each chunk, state
    carried between chunks. With grad enabled each chunk is checkpointed, so the
    (B, di, N, L) state tensor is never stored whole -- only chunk-boundary states
    are kept and each chunk is recomputed in the backward pass (the same
    recompute-instead-of-store idea as Mamba's fused kernel)."""
    fn, chunk = SCAN_IMPL[impl]
    Bsz, L, di = x.shape
    h = x.new_zeros(Bsz, di, A.shape[1])
    ys = []
    for s in range(0, L, chunk):
        args = (dt[:, s:s + chunk], x[:, s:s + chunk], Bm[:, s:s + chunk], Cm[:, s:s + chunk], A, h)
        if torch.is_grad_enabled():
            y, h = torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)
        else:
            y, h = fn(*args)
        ys.append(y)
    return torch.cat(ys, 1), h


class MambaMixer(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dt_min=1e-3, dt_max=1e-1, scan_impl="logspace"):
        super().__init__()
        di = expand * d_model
        self.di, self.N, self.d_conv, self.scan_impl = di, d_state, d_conv, scan_impl
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * di, bias=False)
        self.conv = nn.Conv1d(di, di, d_conv, groups=di, padding=d_conv - 1)
        self.x_proj = nn.Linear(di, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, di)
        dt = torch.exp(torch.rand(di) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        # S4D-Real init: A_n = -(n+1)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.A_log._ssm_param = True
        self.D._ssm_param = True
        self.out_proj = nn.Linear(di, d_model, bias=False)

    def _ssm_inputs(self, x):
        """x: (B, L, di) post-conv activations -> dt (B,L,di), Bm, Cm (B,L,N)."""
        dbc = self.x_proj(x)
        dt, Bm, Cm = dbc.split([self.dt_rank, self.N, self.N], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        return dt, Bm, Cm

    def forward(self, u, mask=None, return_state=False):
        Bsz, L, _ = u.shape
        xz = self.in_proj(u)
        x_in, z = xz.chunk(2, dim=-1)
        x = self.conv(x_in.transpose(1, 2))[..., :L].transpose(1, 2)
        x = F.silu(x)
        if mask is not None:
            x = x * mask.unsqueeze(-1)
        with torch.autocast("cuda", enabled=False):
            x32 = x.float()
            dt, Bm, Cm = self._ssm_inputs(x32)
            A = -torch.exp(self.A_log)                                     # (di,N)
            y, h_last = selective_scan(dt, x32, Bm, Cm, A, impl=self.scan_impl)
            y = y + x32 * self.D
        y = y.to(z.dtype) * F.silu(z)
        out = self.out_proj(y)
        if return_state:
            conv_buf = F.pad(x_in.transpose(1, 2), (self.d_conv - 1, 0))[..., -(self.d_conv - 1):]  # (B,di,d_conv-1)
            return out, (conv_buf, h_last)
        return out

    # generation API ----------------------------------------------------------
    def prefill(self, u):
        return self.forward(u, return_state=True)

    def step(self, u, state):
        """u: (B, d_model)."""
        conv_buf, h = state
        x_in, z = self.in_proj(u).chunk(2, dim=-1)
        window = torch.cat([conv_buf, x_in.unsqueeze(-1)], dim=-1)          # (B,di,d_conv)
        x = (window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias
        x = F.silu(x)
        with torch.autocast("cuda", enabled=False):
            x32 = x.float()
            dt, Bm, Cm = self._ssm_inputs(x32)
            A = -torch.exp(self.A_log)
            h = torch.exp((dt.unsqueeze(-1) * A).clamp(min=LOG_A_MIN)) * h + (dt * x32).unsqueeze(-1) * Bm.unsqueeze(1)
            y = (h * Cm.unsqueeze(1)).sum(-1) + x32 * self.D
        y = y.to(z.dtype) * F.silu(z)
        return self.out_proj(y), (window[..., 1:], h)


class BiMamba(nn.Module):
    """Bidirectional Mamba for the text encoder: a forward and a backward mixer.
    Right padding is handled by flipping each sequence within its own length."""

    def __init__(self, d_model, **kw):
        super().__init__()
        self.fwd = MambaMixer(d_model, **kw)
        self.bwd = MambaMixer(d_model, **kw)

    def forward(self, u, mask):
        lengths = mask.sum(1).long()
        idx = _reverse_index(lengths, u.shape[1])
        u_rev = u.gather(1, idx.unsqueeze(-1).expand_as(u))
        y_rev = self.bwd(u_rev, mask)
        return self.fwd(u, mask) + y_rev.gather(1, idx.unsqueeze(-1).expand_as(y_rev))


def _reverse_index(lengths, L):
    """Index that reverses each row's first `length` entries (padding stays put)."""
    ar = torch.arange(L, device=lengths.device).unsqueeze(0)
    rev = lengths.unsqueeze(1) - 1 - ar
    return torch.where(rev >= 0, rev, ar)
