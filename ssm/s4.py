"""S4 (Structured State Space) layer, written from scratch in PyTorch.

Continuous-time SSM (per channel h = 1..H, state size N):
    x'(t) = A x(t) + B u(t)
    y(t)  = C x(t) + D u(t)

1. HiPPO initialization.  A is the HiPPO-LegS matrix, which makes x(t) the
   coefficients of an optimal Legendre-polynomial projection of the input
   history u(<=t).  A is decomposed as Normal Plus Low-Rank (NPLR):
       A = V (Lambda) V^* - P P^T
   We keep the diagonal (normal) part, i.e. the S4D-LegS parameterisation
   (Gu et al. 2022, "On the Parameterization and Initialization of Diagonal
   State Space Models"), which keeps the HiPPO spectrum while making every
   computation elementwise.

2. Zero-Order Hold (ZOH) discretization with a learned step size Delta:
       A_bar = exp(Delta A)
       B_bar = (Delta A)^{-1} (exp(Delta A) - I) Delta B  =  (exp(Delta A) - 1) / A * B
   (exact for diagonal A; derived by integrating the ODE with u held constant
   over each step).

3. Three equivalent ways to run the discrete system:
   * convolutional: y = K * u with kernel K_l = C A_bar^l B_bar, computed in
     O(N L) as a Vandermonde product and applied with an FFT in O(L log L).
     Used for training (fully parallel).
   * recurrent: x_k = A_bar x_{k-1} + B_bar u_k, y_k = C x_k + D u_k.
     O(1) per step, used for autoregressive generation.
   * parallel scan: the recurrence evaluated with the associative scan in
     ssm/scan.py, O(log L) depth.  Used to verify the other two and to
     benchmark the scan kernel.

Because A is real, its complex eigenvalues come in conjugate pairs; we store
only N/2 of them and take 2*Re(...) of the output.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .scan import scan_blelloch


# ----------------------------------------------------------------------------
# HiPPO
# ----------------------------------------------------------------------------
def hippo_legs(N):
    """HiPPO-LegS matrices (A, B), Gu et al. 2020.

    A_nk = -sqrt(2n+1) sqrt(2k+1)  if n > k
           -(n+1)                  if n = k
            0                      if n < k
    B_n  =  sqrt(2n+1)
    """
    q = np.sqrt(2 * np.arange(N) + 1.0)
    A = -np.tril(np.outer(q, q), -1) - np.diag(np.arange(N) + 1.0)
    B = q.copy()
    return A, B


def nplr_legs(N):
    """NPLR decomposition of HiPPO-LegS: A = V diag(Lambda) V^* - P P^T.

    A + P P^T = -1/2 I + S with S skew-symmetric, so its eigenvalues are
    -1/2 + i*omega and its eigenvectors are unitary.
    Returns Lambda (N,), P (N,), B (N,), V (N, N) as complex numpy arrays.
    """
    A, B = hippo_legs(N)
    P = np.sqrt(np.arange(N) + 0.5)
    AP = A + np.outer(P, P)                     # normal part
    S = (AP - AP.T) / 2                         # skew-symmetric
    lam_re = np.mean(np.diag(AP)) * np.ones(N)  # = -1/2
    lam_im, V = np.linalg.eigh(S * -1j)         # S = V (i lam_im) V^*
    Lambda = lam_re + 1j * lam_im
    return Lambda, V.conj().T @ P, V.conj().T @ B, V


def s4d_legs_init(N):
    """Diagonal HiPPO-LegS init: keep one eigenvalue of each conjugate pair."""
    Lambda, _, B, _ = nplr_legs(N)
    idx = np.where(Lambda.imag > 0)[0]
    idx = idx[np.argsort(Lambda.imag[idx])][: N // 2]
    return Lambda[idx], B[idx]


# ----------------------------------------------------------------------------
# S4 kernel
# ----------------------------------------------------------------------------
class S4Kernel(nn.Module):
    """Holds (Lambda, B, C, log_dt) for H channels and computes the SSM in
    convolutional, recurrent or scan form. Channels = H * n_dir (n_dir = 2 for
    a bidirectional layer: one forward and one backward kernel)."""

    def __init__(self, H, N=64, dt_min=1e-3, dt_max=1e-1, n_dir=1):
        super().__init__()
        self.H, self.N2, self.n_dir = H, N // 2, n_dir
        Lambda, B = s4d_legs_init(N)
        Lambda = torch.tensor(Lambda, dtype=torch.cfloat)
        B = torch.tensor(B, dtype=torch.cfloat)
        C = torch.randn(n_dir * H, self.N2, dtype=torch.cfloat) * (0.5 ** 0.5)
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)

        # real part kept negative (stable) via -exp(.)
        self.log_A_real = nn.Parameter(torch.log(-Lambda.real).repeat(H, 1))
        self.A_imag = nn.Parameter(Lambda.imag.repeat(H, 1))
        self.B = nn.Parameter(torch.view_as_real(B.repeat(H, 1)))
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)
        for p in (self.log_A_real, self.A_imag, self.B, self.log_dt):
            p._ssm_param = True  # gets its own optimizer group (no weight decay)

    # --- discretization -----------------------------------------------------
    def discrete(self):
        """ZOH: returns dtA (H,N2), A_bar (H,N2), B_bar (H,N2), C (n_dir*H, N2)."""
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        dtA = A * dt
        A_bar = torch.exp(dtA)
        B_bar = (A_bar - 1.0) / A * torch.view_as_complex(self.B)
        return dtA, A_bar, B_bar, torch.view_as_complex(self.C)

    # --- convolutional mode ---------------------------------------------------
    def kernel(self, L):
        """K[c, l] = 2 Re( sum_n C_n B_bar_n A_bar_n^l ),  shape (n_dir*H, L).
        Vandermonde: A_bar^l = exp(l * dtA)."""
        dtA, _, B_bar, C = self.discrete()
        vand = torch.exp(dtA.unsqueeze(-1) * torch.arange(L, device=dtA.device))  # (H,N2,L)
        CB = C.view(self.n_dir, self.H, self.N2) * B_bar                           # (dir,H,N2)
        K = torch.einsum("dhn,hnl->dhl", CB, vand).reshape(self.n_dir * self.H, L)
        return 2 * K.real

    def forward_conv(self, u):
        """u: (B, H, L) real -> y: (B, H, L). Causal FFT convolution
        (bidirectional: forward kernel + time-reversed backward kernel)."""
        L = u.shape[-1]
        K = self.kernel(L)
        # zero-pad to >= 2L so circular convolution == linear (causal) convolution;
        # a power of two keeps cuFFT on its fastest path
        n = 1 << (2 * L - 1).bit_length()
        if self.n_dir == 1:
            return torch.fft.irfft(torch.fft.rfft(u, n) * torch.fft.rfft(K, n), n)[..., :L]
        Kf, Kb = K[: self.H], K[self.H:]
        yf = torch.fft.irfft(torch.fft.rfft(u, n) * torch.fft.rfft(Kf, n), n)[..., :L]
        yb = torch.fft.irfft(torch.fft.rfft(u.flip(-1), n) * torch.fft.rfft(Kb, n), n)[..., :L]
        return yf + yb.flip(-1)

    def final_state(self, u):
        """State after the last input: x_L = sum_k A_bar^{L-1-k} B_bar u_k.  (B,H,N2)."""
        L = u.shape[-1]
        dtA, _, B_bar, _ = self.discrete()
        w = torch.exp(dtA.unsqueeze(-1) * torch.arange(L - 1, -1, -1, device=u.device))
        return torch.einsum("hnl,bhl->bhn", w * B_bar.unsqueeze(-1), u.to(w.dtype))

    # --- recurrent mode -------------------------------------------------------
    def step(self, u, x):
        """One recurrent step. u: (B,H), x: (B,H,N2) complex -> y (B,H), x."""
        _, A_bar, B_bar, C = self.discrete()
        x = A_bar * x + B_bar * u.unsqueeze(-1)
        y = 2 * (C[: self.H] * x).sum(-1).real
        return y, x

    # --- parallel-scan mode ---------------------------------------------------
    def forward_scan(self, u):
        """Recurrence evaluated with the work-efficient parallel scan.
        Causal (n_dir == 1) only. u: (B,H,L) -> y (B,H,L)."""
        assert self.n_dir == 1
        L = u.shape[-1]
        _, A_bar, B_bar, C = self.discrete()
        a = A_bar.unsqueeze(0).unsqueeze(-1).expand(u.shape[0], -1, -1, L)
        b = B_bar.unsqueeze(0).unsqueeze(-1) * u.unsqueeze(2)          # (B,H,N2,L)
        x = scan_blelloch(a, b)
        return 2 * torch.einsum("hn,bhnl->bhl", C, x).real


class S4Layer(nn.Module):
    """S4 sequence-mixing layer: SSM + D skip + GELU + GLU output projection.
    Input / output: (B, L, H)."""

    def __init__(self, H, N=64, bidirectional=False, dropout=0.0, mode="conv"):
        super().__init__()
        self.kernel = S4Kernel(H, N, n_dir=2 if bidirectional else 1)
        self.D = nn.Parameter(torch.randn(H))
        self.out = nn.Linear(H, 2 * H)
        self.drop = nn.Dropout(dropout)
        self.mode = mode

    def _post(self, y, u):
        y = y + self.D * u
        y = self.drop(F.gelu(y))
        return F.glu(self.out(y), dim=-1)

    def forward(self, u, mask=None):
        with torch.autocast("cuda", enabled=False):
            u32 = u.float()
            if mask is not None:
                u32 = u32 * mask.unsqueeze(-1)
            ut = u32.transpose(1, 2)
            y = self.kernel.forward_scan(ut) if self.mode == "scan" else self.kernel.forward_conv(ut)
            y = y.transpose(1, 2)
        return self._post(y, u32)

    # generation API ----------------------------------------------------------
    def prefill(self, u):
        with torch.autocast("cuda", enabled=False):
            u32 = u.float()
            ut = u32.transpose(1, 2)
            y = self.kernel.forward_conv(ut).transpose(1, 2)
            state = self.kernel.final_state(ut)
        return self._post(y, u32), state

    def step(self, u, state):
        """u: (B, H)."""
        with torch.autocast("cuda", enabled=False):
            u32 = u.float()
            y, state = self.kernel.step(u32, state)
        return self._post(y, u32), state
