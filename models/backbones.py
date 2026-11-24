"""Interchangeable sequence-mixing backbones with a shared interface.

Every block is pre-norm:  x = x + Mixer(LN(x));  x = x + FFN(LN(x))
Only the Mixer differs (S4 / Mamba / causal attention / LSTM), so the TTS
comparison isolates the sequence-modelling layer.

Interface of a Backbone:
    forward(x, mask)      full sequence (training).  x: (B,L,D), mask: (B,L) bool
    prefill(x) -> y, cache    batch of 1, whole prefix, returns per-layer state
    step(x_t, cache) -> y_t, cache    one new position
"""
import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ssm.s4 import S4Layer
from ssm.mamba import MambaMixer, BiMamba


# ----------------------------------------------------------------------------
# Mixers
# ----------------------------------------------------------------------------
class S4Mixer(nn.Module):
    def __init__(self, d, causal, d_state=64, dropout=0.0):
        super().__init__()
        self.layer = S4Layer(d, d_state, bidirectional=not causal, dropout=dropout)

    def forward(self, x, mask):
        return self.layer(x, mask)

    def prefill(self, x):
        return self.layer.prefill(x)

    def step(self, x, state):
        return self.layer.step(x, state)


class MambaBlockMixer(nn.Module):
    def __init__(self, d, causal, d_state=16, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.m = MambaMixer(d, d_state=d_state) if causal else BiMamba(d, d_state=d_state)

    def forward(self, x, mask):
        return self.m(x, mask)

    def prefill(self, x):
        return self.m.prefill(x)

    def step(self, x, state):
        return self.m.step(x, state)


def rope(x, pos):
    """Rotary position embedding. x: (B, heads, L, hd), pos: (L,)"""
    hd = x.shape[-1]
    inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, device=x.device).float() / hd))
    ang = pos.float().unsqueeze(-1) * inv                     # (L, hd/2)
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class AttentionMixer(nn.Module):
    """Multi-head self-attention with RoPE; causal (decoder) or bidirectional
    with key padding mask (encoder).  KV cache for generation."""

    def __init__(self, d, causal, n_heads=6, dropout=0.0):
        super().__init__()
        self.h, self.causal, self.p = n_heads, causal, dropout
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)

    def _qkv(self, x, pos):
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        return rope(q, pos), rope(k, pos), v

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self._qkv(x, torch.arange(L, device=x.device))
        p = self.p if self.training else 0.0
        if self.causal:  # right padding: real queries never see later pads
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :], dropout_p=p)
        return self.o(y.transpose(1, 2).reshape(B, L, D))

    def prefill(self, x):
        B, L, D = x.shape
        q, k, v = self._qkv(x, torch.arange(L, device=x.device))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(y.transpose(1, 2).reshape(B, L, D)), (k, v)

    def step(self, x, cache):
        k_c, v_c = cache
        pos = torch.tensor([k_c.shape[2]], device=x.device)
        q, k, v = self._qkv(x.unsqueeze(1), pos)
        k_c, v_c = torch.cat([k_c, k], 2), torch.cat([v_c, v], 2)
        y = F.scaled_dot_product_attention(q, k_c, v_c)
        return self.o(y.transpose(1, 2).reshape(x.shape[0], -1)), (k_c, v_c)


class LSTMMixer(nn.Module):
    def __init__(self, d, causal, dropout=0.0):
        super().__init__()
        self.causal = causal
        self.rnn = nn.LSTM(d, d if causal else d // 2, batch_first=True, bidirectional=not causal)

    def forward(self, x, mask):
        with torch.autocast("cuda", enabled=False):
            x = x.float()
            if self.causal:
                return self.rnn(x)[0]
            lengths = mask.sum(1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            y, _ = self.rnn(packed)
            return pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])[0]

    def prefill(self, x):
        with torch.autocast("cuda", enabled=False):
            y, st = self.rnn(x.float())
        return y, st

    def step(self, x, st):
        with torch.autocast("cuda", enabled=False):
            y, st = self.rnn(x.float().unsqueeze(1), st)
        return y.squeeze(1), st


MIXERS = {"s4": S4Mixer, "mamba": MambaBlockMixer, "transformer": AttentionMixer, "lstm": LSTMMixer}


# ----------------------------------------------------------------------------
# Block / backbone
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, kind, d, causal, ffn_mult=2, dropout=0.1):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mixer = MIXERS[kind](d, causal)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_mult * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        m = mask.unsqueeze(-1).to(x.dtype)
        x = x + self.drop(self.mixer(self.n1(x) * m, mask))
        x = x + self.drop(self.ffn(self.n2(x)))
        return x * m  # keep padding exactly zero

    def prefill(self, x):
        y, st = self.mixer.prefill(self.n1(x))
        x = x + y.to(x.dtype)
        return x + self.ffn(self.n2(x)), st

    def step(self, x, st):
        y, st = self.mixer.step(self.n1(x), st)
        x = x + y.to(x.dtype)
        return x + self.ffn(self.n2(x)), st


class Backbone(nn.Module):
    def __init__(self, kind, d, n_layers, causal, ffn_mult=2, dropout=0.1, checkpoint=False):
        super().__init__()
        self.blocks = nn.ModuleList([Block(kind, d, causal, ffn_mult, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d)
        self.checkpoint = checkpoint

    def forward(self, x, mask):
        for b in self.blocks:
            if self.checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(b, x, mask, use_reentrant=False)
            else:
                x = b(x, mask)
        return self.norm(x)

    @torch.no_grad()
    def prefill(self, x):
        cache = []
        for b in self.blocks:
            x, st = b.prefill(x)
            cache.append(st)
        return self.norm(x), cache

    @torch.no_grad()
    def step(self, x, cache):
        new = []
        for b, st in zip(self.blocks, cache):
            x, st = b.step(x, st)
            new.append(st)
        return self.norm(x), new
