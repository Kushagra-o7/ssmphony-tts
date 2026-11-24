"""Parallel scan kernels for linear recurrences.

All functions solve the first-order linear recurrence along the LAST dimension

    h_t = a_t * h_{t-1} + b_t,      h_{-1} = h0 (default 0)

which is exactly the recurrent form of a discretized state space model
(a_t = A_bar, b_t = B_bar u_t).  The operator

    (a1, b1) . (a2, b2) = (a2 * a1,  a2 * b1 + b2)

is associative, so the recurrence can be evaluated as a prefix scan in
O(log L) parallel depth instead of L sequential steps.

Three implementations are provided:
  * scan_sequential   - the reference O(L)-depth loop (ground truth for tests)
  * scan_hillis_steele- O(L log L) work, log2(L) depth (simple, step-efficient)
  * scan_blelloch     - O(L) work, 2*log2(L) depth (work-efficient, used in models)

They work for real or complex tensors and are differentiable with autograd.
"""
import torch


def scan_sequential(a, b, h0=None):
    """Reference implementation: plain loop over time."""
    L = b.shape[-1]
    h = torch.zeros_like(b[..., 0]) if h0 is None else h0
    out = []
    for t in range(L):
        h = a[..., t] * h + b[..., t]
        out.append(h)
    return torch.stack(out, dim=-1)


def scan_hillis_steele(a, b, h0=None):
    """Inclusive scan, Hillis & Steele (1986). log2(L) rounds, O(L log L) work.

    Round d combines every element with the element 2^d positions to its left.
    """
    if h0 is not None:
        b = torch.cat([b[..., :1] + a[..., :1] * h0.unsqueeze(-1), b[..., 1:]], dim=-1)
    L = b.shape[-1]
    d = 1
    while d < L:
        b = torch.cat([b[..., :d], b[..., d:] + a[..., d:] * b[..., :-d]], dim=-1)
        a = torch.cat([a[..., :d], a[..., d:] * a[..., :-d]], dim=-1)
        d *= 2
    return b


def scan_blelloch(a, b, h0=None):
    """Work-efficient recursive (odd/even) parallel scan, after Blelloch (1990).

    Up-sweep: combine adjacent pairs (2k, 2k+1) into one element, which halves
    the problem.  Recurse on the half-length problem to get h at odd indices.
    Down-sweep: recover h at even indices with one more combine step.
    Total work O(L), depth O(log L).
    """
    if h0 is not None:
        b = torch.cat([b[..., :1] + a[..., :1] * h0.unsqueeze(-1), b[..., 1:]], dim=-1)
    return _blelloch(a, b)


def _blelloch(a, b):
    L = b.shape[-1]
    if L == 1:
        return b
    if L % 2 == 1:  # pad with the identity element (a=1, b=0)
        a = torch.cat([a, torch.ones_like(a[..., :1])], dim=-1)
        b = torch.cat([b, torch.zeros_like(b[..., :1])], dim=-1)
        return _blelloch(a, b)[..., :L]
    a0, a1 = a[..., 0::2], a[..., 1::2]
    b0, b1 = b[..., 0::2], b[..., 1::2]
    # up-sweep: pair (2k, 2k+1) -> maps h_{2k-1} to h_{2k+1}
    h_odd = _blelloch(a1 * a0, a1 * b0 + b1)
    # down-sweep: h_{2k} = a_{2k} h_{2k-1} + b_{2k}
    h_prev = torch.cat([torch.zeros_like(h_odd[..., :1]), h_odd[..., :-1]], dim=-1)
    h_even = a0 * h_prev + b0
    return torch.stack([h_even, h_odd], dim=-1).flatten(-2)
