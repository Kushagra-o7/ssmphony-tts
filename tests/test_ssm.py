"""Correctness tests: run with  python -m pytest tests -q  (or python tests/test_ssm.py)."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ssm.scan import scan_blelloch, scan_hillis_steele, scan_sequential  # noqa: E402
from ssm.s4 import S4Kernel, S4Layer, hippo_legs, nplr_legs  # noqa: E402
from ssm.mamba import MambaMixer  # noqa: E402
from models.backbones import Backbone  # noqa: E402
from models.tts import ARTTS, PrefixARTTS  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)


def test_scans_match_sequential():
    for L in (1, 7, 64, 100, 257):
        a = torch.rand(3, 5, L, device=dev) * 0.99
        b = torch.randn(3, 5, L, device=dev)
        ref = scan_sequential(a, b)
        assert torch.allclose(scan_hillis_steele(a, b), ref, atol=1e-5)
        assert torch.allclose(scan_blelloch(a, b), ref, atol=1e-5)
        ac = torch.polar(torch.rand(2, L, device=dev), torch.randn(2, L, device=dev))
        bc = torch.randn(2, L, dtype=torch.cfloat, device=dev)
        assert torch.allclose(scan_blelloch(ac, bc), scan_sequential(ac, bc), atol=1e-5)


def test_scan_initial_state():
    a, b, h0 = torch.rand(4, 33) * 0.9, torch.randn(4, 33), torch.randn(4)
    assert torch.allclose(scan_blelloch(a, b, h0), scan_sequential(a, b, h0), atol=1e-5)


def test_hippo_nplr_reconstructs_A():
    N = 16
    A, _ = hippo_legs(N)
    Lam, P, _, V = nplr_legs(N)
    Pf = np.sqrt(np.arange(N) + 0.5)
    A_rec = (V @ np.diag(Lam) @ V.conj().T).real - np.outer(Pf, Pf)
    assert np.allclose(A_rec, A, atol=1e-8)
    assert np.allclose(Lam.real, -0.5)


def test_zoh_matches_matrix_exponential():
    k = S4Kernel(4, 16).to(dev)
    dtA, A_bar, B_bar, _ = k.discrete()
    A = -torch.exp(k.log_A_real) + 1j * k.A_imag
    dt = torch.exp(k.log_dt)
    A, dt, Bc = A.detach().cdouble(), dt.detach().double(), torch.view_as_complex(k.B).detach().cdouble()
    for h in range(4):  # dense reference in float64 (fp32 matrix_exp alone has ~5e-5 error)
        M = torch.diag(A[h]) * dt[h]
        Ab = torch.linalg.matrix_exp(M)
        Bb = torch.linalg.solve(M, (Ab - torch.eye(8, device=dev, dtype=Ab.dtype)) @ (dt[h] * Bc[h]))
        assert torch.allclose(torch.diagonal(Ab).cfloat(), A_bar[h], atol=1e-5)
        assert torch.allclose(Bb.cfloat(), B_bar[h], atol=1e-4)


def test_s4_conv_recurrent_scan_equivalent():
    k = S4Kernel(8, 32).to(dev)
    u = torch.randn(2, 8, 300, device=dev)
    y_conv = k.forward_conv(u)
    y_scan = k.forward_scan(u)
    x = torch.zeros(2, 8, 16, dtype=torch.cfloat, device=dev)
    ys = []
    for t in range(u.shape[-1]):
        y, x = k.step(u[..., t], x)
        ys.append(y)
    y_rec = torch.stack(ys, -1)
    assert torch.allclose(y_conv, y_rec, atol=1e-3), (y_conv - y_rec).abs().max()
    assert torch.allclose(y_scan, y_rec, atol=1e-3), (y_scan - y_rec).abs().max()
    assert torch.allclose(k.final_state(u), x, atol=1e-3)


def test_s4_causal():
    layer = S4Layer(8, 16).to(dev).eval()
    u = torch.randn(1, 50, 8, device=dev)
    u2 = u.clone()
    u2[:, 30:] += 5
    assert torch.allclose(layer(u)[:, :30], layer(u2)[:, :30], atol=1e-5)


def test_mamba_forward_matches_step():
    m = MambaMixer(16, d_state=8).to(dev).eval()
    u = torch.randn(1, 40, 16, device=dev)
    y_full = m(u)
    y_pre, st = m.prefill(u[:, :25])
    ys = [y_pre]
    for t in range(25, 40):
        y, st = m.step(u[:, t], st)
        ys.append(y.unsqueeze(1))
    assert torch.allclose(torch.cat(ys, 1), y_full, atol=1e-4)


def test_backbone_prefill_step_matches_forward():
    for kind in ("s4", "mamba", "transformer", "lstm"):
        bb = Backbone(kind, 48, 2, causal=True).to(dev).eval()
        x = torch.randn(1, 30, 48, device=dev)
        full = bb(x, torch.ones(1, 30, dtype=torch.bool, device=dev))
        out, cache = bb.prefill(x[:, :20])
        outs = [out]
        for t in range(20, 30):
            o, cache = bb.step(x[:, t], cache)
            outs.append(o.unsqueeze(1))
        assert torch.allclose(torch.cat(outs, 1), full, atol=2e-3), (kind, (torch.cat(outs, 1) - full).abs().max())


def test_padding_invariance():
    """A padded batch must give the same outputs as the sample alone (both TTS variants)."""
    for kind in ("s4", "mamba", "transformer", "lstm"):
        for cls in (ARTTS, PrefixARTTS):
            model = cls(kind, 40, d=48, n_enc=1, n_dec=2, r=2).to(dev).eval()
            model.prenet.p = 0.0  # prenet dropout is always on; disable it for a deterministic check
            torch.manual_seed(1)
            t1, t2 = torch.randint(1, 40, (1, 9)), torch.randint(1, 40, (1, 5))
            m1, m2 = torch.randn(1, 12, 80), torch.randn(1, 8, 80)
            d1, d2 = torch.tensor([[1, 2, 1, 1, 2, 1, 2, 1, 1]]), torch.tensor([[2, 1, 2, 2, 1]])
            text = torch.zeros(2, 9, dtype=torch.long); text[0] = t1; text[1, :5] = t2
            mel = torch.zeros(2, 12, 80); mel[0] = m1; mel[1, :8] = m2
            dur = torch.zeros(2, 9, dtype=torch.long); dur[0] = d1; dur[1, :5] = d2
            tl, ns = torch.tensor([9, 5]), torch.tensor([6, 4])
            args_b = [text.to(dev), tl.to(dev), mel.to(dev), ns.to(dev)]
            args_1 = [t2.to(dev), torch.tensor([5], device=dev), m2.to(dev), torch.tensor([4], device=dev)]
            if cls is ARTTS:
                args_b.append(dur.to(dev)); args_1.append(d2.to(dev))
            ob = model(*args_b)[1]
            o1 = model(*args_1)[1]
            assert torch.allclose(ob[1, :8], o1[0], atol=2e-3), (kind, cls.__name__, (ob[1, :8] - o1[0]).abs().max())


def test_duration_model_generates_requested_length():
    for kind in ("s4", "mamba", "transformer", "lstm"):
        model = ARTTS(kind, 40, d=48, n_enc=1, n_dec=2, r=4).to(dev).eval()
        text = torch.randint(1, 40, (1, 7), device=dev)
        dur = torch.tensor([[3, 1, 4, 1, 5, 9, 2]], device=dev)
        mel, steps = model.generate(text, dur=dur)
        assert mel.shape == (25, 80) and steps == 7, (kind, mel.shape, steps)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
