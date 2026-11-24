"""Render result figures (PNG) into results/figures from results/*.json and run logs."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data import DATA_ROOT  # noqa: E402

OUT = "results/figures"
COLORS = {"s4": "#2563eb", "mamba": "#7c3aed", "transformer": "#d97706", "lstm": "#6b7280"}
LABEL = {"s4": "S4", "mamba": "Mamba", "transformer": "Transformer", "lstm": "LSTM"}


def scaling():
    d = json.load(open("results/scaling.json"))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for name in dict.fromkeys(r["model"] for r in d["train"]):
        rows = [r for r in d["train"] if r["model"] == name and r.get("ms")]
        L = [r["L"] for r in rows]
        ax[0].loglog(L, [r["ms"] for r in rows], "o-", label=name)
        ax[1].loglog(L, [r["mem_MB"] for r in rows], "o-", label=name)
    for a, t in zip(ax[:2], ["Forward + backward time (ms)", "Peak extra GPU memory (MB)"]):
        a.set_xlabel("sequence length L"); a.set_title(t); a.grid(True, which="both", alpha=0.3)
    ax[0].legend(fontsize=8)
    for name in dict.fromkeys(r["model"] for r in d["step"]):
        rows = [r for r in d["step"] if r["model"] == name]
        ax[2].loglog([r["L"] for r in rows], [r["us"] for r in rows], "o-", label=name)
    ax[2].set_xlabel("context length already generated"); ax[2].set_title("Latency of one generation step (µs)")
    ax[2].grid(True, which="both", alpha=0.3); ax[2].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/scaling.png", dpi=150)


def curves():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    for k in ["s4", "transformer", "lstm", "mamba"]:
        p = os.path.join(DATA_ROOT, "runs", k, "log.jsonl")
        if not os.path.exists(p):
            continue
        recs = [json.loads(l) for l in open(p)]
        tr = [r for r in recs if "loss" in r]
        va = [r for r in recs if "val" in r]
        ax[0].plot([r["step"] for r in va], [r["val"]["l1"] for r in va], "o-", ms=3, color=COLORS[k], label=LABEL[k])
        ax[1].plot([r["time"] / 3600 for r in va], [r["val"]["l1"] for r in va], "o-", ms=3, color=COLORS[k], label=LABEL[k])
    ax[0].set_xlabel("training step"); ax[1].set_xlabel("wall-clock training time (h)")
    for a in ax:
        a.set_ylabel("validation L1 (normalized mel)"); a.grid(alpha=0.3); a.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/val_curves.png", dpi=150)


def mels(idx=0):
    names = ["ground_truth", "s4", "transformer", "lstm", "mamba"]
    files = [(n, f"results/samples/{idx:02d}_{n}_mel.npy") for n in names]
    files = [(n, f) for n, f in files if os.path.exists(f)]
    fig, ax = plt.subplots(len(files), 1, figsize=(10, 2.1 * len(files)))
    for a, (n, f) in zip(np.atleast_1d(ax), files):
        a.imshow(np.load(f).T, origin="lower", aspect="auto", cmap="magma", vmin=-11.5, vmax=2)
        a.set_ylabel(LABEL.get(n, "Ground truth"))
    fig.tight_layout(); fig.savefig(f"{OUT}/mels_{idx:02d}.png", dpi=130)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for fn in (scaling, curves, mels):
        try:
            fn(); print("ok", fn.__name__)
        except Exception as e:  # keep going if a stage has no data yet
            print("skip", fn.__name__, e)
