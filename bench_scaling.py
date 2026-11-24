"""Sequence-length scaling benchmark for a single mixing layer.

(1) training cost: forward + backward time and peak GPU memory vs length L
(2) generation cost: latency of ONE autoregressive step given a context of L
Writes results/scaling.json.
"""
import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ssm.mamba import MambaMixer
from ssm.s4 import S4Layer
from models.backbones import AttentionMixer

dev = "cuda"
# Cap the allocator below physical VRAM: on Windows (WDDM) going over silently spills
# into system RAM and inflates timings, so report a clean OOM instead.
torch.cuda.set_per_process_memory_fraction(0.85)
D, BATCH = 256, 2
LENGTHS = [2 ** k for k in range(8, 18)]  # 256 .. 131072


class NaiveAttention(nn.Module):
    """Textbook causal attention that materialises the L x L score matrix."""

    def __init__(self, d, h=4):
        super().__init__()
        self.h, self.qkv, self.o = h, nn.Linear(d, 3 * d), nn.Linear(d, d)

    def forward(self, x):
        B, L, Dm = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, Dm // self.h).permute(2, 0, 3, 1, 4)
        s = q @ k.transpose(-1, -2) / math.sqrt(Dm // self.h)
        s = s.masked_fill(torch.ones(L, L, device=x.device, dtype=torch.bool).triu(1), float("-inf"))
        return self.o((s.softmax(-1) @ v).transpose(1, 2).reshape(B, L, Dm))


class SDPAttention(nn.Module):
    def __init__(self, d, h=4):
        super().__init__()
        self.m = AttentionMixer(d, causal=True, n_heads=h)

    def forward(self, x):
        return self.m(x, None)


class LSTMLayer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.rnn = nn.LSTM(d, d, batch_first=True)

    def forward(self, x):
        return self.rnn(x)[0]


def make(name):
    return {
        "S4 (FFT conv)": lambda: S4Layer(D, 64, mode="conv"),
        "S4 (parallel scan)": lambda: S4Layer(D, 64, mode="scan"),
        "Mamba (chunked log-space scan)": lambda: MambaMixer(D, d_state=16, scan_impl="logspace"),
        "Mamba (chunked Blelloch scan)": lambda: MambaMixer(D, d_state=16, scan_impl="blelloch"),
        "Attention (naive)": lambda: NaiveAttention(D),
        "Attention (PyTorch SDPA)": lambda: SDPAttention(D),
        "LSTM (cuDNN)": lambda: LSTMLayer(D),
    }[name]().to(dev)


def timed(fn, reps=5):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
    return sorted(ts)[len(ts) // 2]


def bench_train():
    names = ["S4 (FFT conv)", "S4 (parallel scan)", "Mamba (chunked log-space scan)", "Mamba (chunked Blelloch scan)", "Attention (naive)",
             "Attention (PyTorch SDPA)", "LSTM (cuDNN)"]
    res = []
    for name in names:
        m = make(name)
        for L in LENGTHS:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(BATCH, L, D, device=dev, requires_grad=True)
            base = torch.cuda.max_memory_allocated()

            def step():
                m.zero_grad(set_to_none=True)
                m(x).float().pow(2).mean().backward()
            try:
                t = timed(step, reps=3 if L >= 16384 else 5)
                mem = (torch.cuda.max_memory_allocated() - base) / 2 ** 20
                res.append({"model": name, "L": L, "ms": t * 1e3, "mem_MB": mem})
                print(f"{name:28s} L={L:6d}  {t * 1e3:9.2f} ms  {mem:9.1f} MB", flush=True)
                if t > 8:
                    break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:  # OOM, or cuDNN refusing very long inputs
                kind = "OOM" if isinstance(e, torch.cuda.OutOfMemoryError) else "unsupported"
                res.append({"model": name, "L": L, "ms": None, "mem_MB": None, "fail": kind})
                print(f"{name:28s} L={L:6d}  {kind}", flush=True)
                break
            finally:
                del x
        del m
    return res


@torch.no_grad()
def bench_step():
    """Per-token generation latency with a context of L tokens already processed."""
    res = []
    s4 = S4Layer(D, 64).to(dev).eval()
    mamba = MambaMixer(D, d_state=16).to(dev).eval()
    attn = AttentionMixer(D, causal=True, n_heads=4).to(dev).eval()
    lstm = nn.LSTM(D, D, batch_first=True).to(dev)
    x = torch.randn(1, D, device=dev)
    for L in [2 ** k for k in range(8, 18)]:
        s4_state = torch.zeros(1, D, 32, dtype=torch.cfloat, device=dev)
        m_state = (torch.zeros(1, 2 * D, 3, device=dev), torch.zeros(1, 2 * D, 16, device=dev))
        kv = (torch.randn(1, 4, L, D // 4, device=dev), torch.randn(1, 4, L, D // 4, device=dev))
        hc = (torch.zeros(1, 1, D, device=dev), torch.zeros(1, 1, D, device=dev))
        for name, fn, state_bytes in [
            ("S4 (recurrent)", lambda: s4.step(x, s4_state), s4_state.numel() * 8),
            ("Mamba (recurrent)", lambda: mamba.step(x, m_state), sum(t.numel() * 4 for t in m_state)),
            ("Transformer (KV cache)", lambda: attn.step(x, kv), sum(t.numel() * 4 for t in kv)),
            ("LSTM", lambda: lstm(x.unsqueeze(1), hc), 2 * D * 4),
        ]:
            t = timed(fn, reps=50)
            res.append({"model": name, "L": L, "us": t * 1e6, "state_KB": state_bytes / 1024})
        print(f"step bench L={L} done", flush=True)
    return res


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    out = {"config": {"d_model": D, "batch": BATCH, "gpu": torch.cuda.get_device_name(0)}}
    out["train"] = bench_train()
    json.dump(out, open("results/scaling.json", "w"), indent=1)  # save before the next stage
    out["step"] = bench_step()
    json.dump(out, open("results/scaling.json", "w"), indent=1)
