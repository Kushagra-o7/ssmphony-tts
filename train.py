"""Train an autoregressive TTS model with a given backbone on LJSpeech.

    python train.py --kind s4 --steps 40000
Resumes automatically from <out>/last.pt.
"""
import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from data import DATA_ROOT, N_VOCAB, BucketSampler, LJDataset, make_collate
from models.tts import ARTTS, PrefixARTTS

# Per-backbone layer counts chosen so all models have ~15-16M parameters.
CONFIGS = {
    "s4":          dict(d=384, n_enc=4, n_dec=10),
    "mamba":       dict(d=384, n_enc=2, n_dec=6),
    "transformer": dict(d=384, n_enc=4, n_dec=8),
    "lstm":        dict(d=384, n_enc=3, n_dec=5),
}


def build_model(kind, r=4, variant="duration"):
    cls = ARTTS if variant == "duration" else PrefixARTTS
    return cls(kind, N_VOCAB, r=r, **CONFIGS[kind])


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def mel_losses(mel_pred, mel_post, mel, fmask):
    m = fmask.unsqueeze(-1).float()
    denom = m.sum() * mel.shape[-1]
    l1 = ((mel_pred - mel).abs() * m).sum() / denom + ((mel_post - mel).abs() * m).sum() / denom
    l2 = (((mel_pred - mel) ** 2) * m).sum() / denom + (((mel_post - mel) ** 2) * m).sum() / denom
    return l1, l2


def run_batch(model, batch, dev):
    """Forward + loss for either model variant. Returns (loss, parts)."""
    text, tl, mel, ns, fm, dur = [t.to(dev, non_blocking=True) for t in batch]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(text, tl, mel, ns, dur) if isinstance(model, ARTTS) else model(text, tl, mel, ns)
    out = [o.float() for o in out]
    if isinstance(model, ARTTS):
        l1, l2 = mel_losses(out[0], out[1], mel, fm)
        tmask = (torch.arange(text.shape[1], device=dev)[None] < tl[:, None]).float()
        ld = (((out[2] - torch.log1p(dur.float())) ** 2) * tmask).sum() / tmask.sum()
        return l1 + l2 + ld, {"l1": l1.item() / 2, "mse": l2.item() / 2, "dur": ld.item()}
    return loss_fn(*out, mel, fm, ns)


def loss_fn(mel_pred, mel_post, stop_logits, mel, fmask, n_steps):
    l1, l2 = mel_losses(mel_pred, mel_post, mel, fmask)
    S = stop_logits.shape[1]
    ar = torch.arange(S, device=mel.device)[None]
    smask = (ar < n_steps[:, None]).float()
    target = (ar == (n_steps[:, None] - 1)).float()
    stop = (F.binary_cross_entropy_with_logits(stop_logits, target, pos_weight=torch.tensor(8.0, device=mel.device),
                                               reduction="none") * smask).sum() / smask.sum()
    return l1 + l2 + stop, {"l1": l1.item() / 2, "mse": l2.item() / 2, "stop": stop.item()}


def make_optimizer(model, lr, wd=0.01):
    ssm = [p for p in model.parameters() if getattr(p, "_ssm_param", False)]
    ids = {id(p) for p in ssm}
    decay = [p for n, p in model.named_parameters() if id(p) not in ids and p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if id(p) not in ids and p.ndim < 2]
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    if ssm:
        groups.append({"params": ssm, "weight_decay": 0.0, "lr": min(lr, 1e-3)})
    for g in groups:
        g.setdefault("lr", lr)
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(groups, betas=(0.9, 0.98))


def lr_scale(step, warmup, total):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))


@torch.no_grad()
def validate(model, loader, dev):
    model.eval()
    tot, n = {}, 0
    torch.manual_seed(0)
    for batch in loader:
        _, parts = run_batch(model, batch, dev)
        for k, v in parts.items():
            tot[k] = tot.get(k, 0.0) + v
        n += 1
    model.train()
    return {k: v / n for k, v in tot.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--max_frames", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--val_every", type=int, default=1000)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--bench", type=int, default=0, help="time N steps and exit")
    ap.add_argument("--out", default=None)
    ap.add_argument("--accum", type=int, default=1, help="micro-batches per optimizer step (same frames/step)")
    ap.add_argument("--stop_at", type=int, default=0, help="stop early at this step, keeping the --steps LR schedule")
    ap.add_argument("--snapshots", default="3000,8000,20000")
    ap.add_argument("--variant", default="duration", choices=["duration", "prefix"])
    args = ap.parse_args()

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = args.out or os.path.join(DATA_ROOT, "runs", args.kind)
    os.makedirs(out, exist_ok=True)

    train_ds, val_ds = LJDataset("train"), LJDataset("val")
    coll = make_collate(args.r)
    mb_frames = args.max_frames // args.accum
    sampler = BucketSampler([m.shape[0] for m in train_ds.mels], mb_frames, seed=0)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_sampler=sampler, collate_fn=coll, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_sampler=BucketSampler(
        [m.shape[0] for m in val_ds.mels], mb_frames, seed=0), collate_fn=coll)

    torch.manual_seed(0)
    model = build_model(args.kind, args.r, args.variant).to(dev)
    opt = make_optimizer(model, args.lr)
    step, t_offset = 0, 0.0
    ck = os.path.join(out, "last.pt")
    if os.path.exists(ck) and not args.bench:
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step, t_offset = s["step"], s.get("train_time", 0.0)
        sampler.epoch = s.get("epoch", 0)
        print(f"resumed from step {step}")
    n_params = count_params(model)
    print(f"{args.kind}: {n_params / 1e6:.2f}M params, {len(sampler)} batches/epoch")
    json.dump({"kind": args.kind, "params": n_params, **vars(args), "config": CONFIGS[args.kind]},
              open(os.path.join(out, "config.json"), "w"), indent=1)

    log = open(os.path.join(out, "log.jsonl"), "a")
    model.train()
    t0 = time.time()
    torch.cuda.reset_peak_memory_stats()
    done = False
    last_step = args.stop_at or args.steps
    snapshots = {int(x) for x in args.snapshots.split(",") if x}
    micro = []
    while not done:
        for batch in train_dl:
            micro.append(batch)
            if len(micro) < args.accum:
                continue
            sc = lr_scale(step, args.warmup, args.steps)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * sc
            opt.zero_grad(set_to_none=True)
            for batch in micro:
                loss, parts = run_batch(model, batch, dev)
                (loss / args.accum).backward()
            micro = []
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            step += 1

            if args.bench and step == 3:
                torch.cuda.synchronize(); tb = time.time()
            if args.bench and step == args.bench + 3:
                torch.cuda.synchronize()
                dt = (time.time() - tb) / args.bench
                print(json.dumps({"kind": args.kind, "params_M": round(n_params / 1e6, 2), "sec_per_step": round(dt, 4),
                                  "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                                  "batches_per_epoch": len(sampler) // args.accum}))
                return

            if step % 100 == 0:
                rec = {"step": step, "loss": loss.item(), **parts, "lr": opt.param_groups[0]["lr"],
                       "grad_norm": gn.item(), "time": t_offset + time.time() - t0,
                       "mem_GB": torch.cuda.max_memory_allocated() / 1e9}
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"[{args.kind}] step {step} loss {loss.item():.4f} " +
                      " ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f" {rec['time'] / 60:.1f} min", flush=True)
            if step in snapshots:
                torch.save({"model": model.state_dict(), "step": step, "kind": args.kind, "r": args.r,
                            "variant": args.variant}, os.path.join(out, f"snap_{step}.pt"))
            if step % args.val_every == 0 or step >= last_step:
                v = validate(model, val_dl, dev)
                log.write(json.dumps({"step": step, "val": v, "time": t_offset + time.time() - t0}) + "\n"); log.flush()
                print(f"[{args.kind}] VAL step {step}: {v}", flush=True)
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                            "epoch": sampler.epoch, "train_time": t_offset + time.time() - t0}, ck + ".tmp")
                os.replace(ck + ".tmp", ck)
            if step >= last_step:
                done = True
                break
    torch.save({"model": model.state_dict(), "step": step, "kind": args.kind, "r": args.r,
                "variant": args.variant}, os.path.join(out, "final.pt"))
    print("done")


if __name__ == "__main__":
    main()
