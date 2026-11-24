"""Postnet ablation: the same generated spectrogram scored before and after the postnet.

For each final model, every test sentence is generated once; the decoder output
(pre-postnet) and the refined output (post-postnet) come from the same run, so the
only difference is the postnet. Scores: teacher-forced L1, Whisper CER/WER, MCD-DTW.
Writes results/postnet_ablation.json.
"""
import json
import os

import jiwer
import numpy as np
import soundfile as sf
import torch

from data import DATA_ROOT, LJDataset, make_collate
from evaluate import ASR, mcd_dtw, norm_text
from train import build_model
from vocoder import mel_to_wav

dev = "cuda"
N = 100
KINDS = ["s4", "transformer", "lstm", "mamba"]


class Capture(torch.nn.Module):
    """Wraps the postnet and remembers its input (the decoder's raw spectrogram)."""

    def __init__(self, postnet):
        super().__init__()
        self.postnet, self.last_in = postnet, None

    def forward(self, x, mask=None):
        self.last_in = x.detach().clone()
        return self.postnet(x, mask)


def masked_l1(pred, target, fmask):
    m = fmask.float()
    if m.dim() == 2:
        m = m.unsqueeze(-1)
    return float((pred - target).abs().mul(m).sum() / (m.sum() * target.shape[-1]))


def bootstrap(diff, reps=5000, seed=0):
    rng = np.random.default_rng(seed)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(reps)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    test = LJDataset("test")
    mean, std = test.mean.to(dev), test.std.to(dev)
    asr = ASR()
    coll = make_collate(4)
    os.makedirs("results/samples_postnet", exist_ok=True)
    out = {}
    for kind in KINDS:
        state = torch.load(os.path.join(DATA_ROOT, "runs", kind, "final.pt"), map_location=dev, weights_only=False)
        model = build_model(kind, variant="duration").to(dev)
        model.load_state_dict(state["model"])
        model.eval()
        torch.manual_seed(0)

        # teacher-forced L1 before / after the postnet
        tf_pre, tf_post = [], []
        with torch.no_grad():
            for i in range(0, N, 10):
                text, tl, mel, n_steps, fmask, dur = [t.to(dev) for t in coll([test[j] for j in range(i, min(i + 10, N))])]
                mel_pred, mel_post, _ = model(text, tl, mel, n_steps, dur)
                tf_pre.append(masked_l1(mel_pred, mel, fmask))
                tf_post.append(masked_l1(mel_post, mel, fmask))

        cap = Capture(model.postnet)
        model.postnet = cap
        rec = {v: {"cer": [], "wer": [], "mcd": []} for v in ("pre", "post")}
        for i in range(N):
            text_ids, _, _ = test[i]
            gt = test.mels[i].float()
            post_n, _ = model.generate(text_ids.unsqueeze(0).to(dev))
            pre_n = cap.last_in[0, :post_n.shape[0]]
            for v, m_n in (("pre", pre_n), ("post", post_n)):
                mel = (m_n * std + mean).float()
                wav = mel_to_wav(mel)
                hyp = asr(wav)
                rec[v]["cer"].append(jiwer.cer(norm_text(test.texts[i]), norm_text(hyp) or "x"))
                rec[v]["wer"].append(jiwer.wer(norm_text(test.texts[i]), norm_text(hyp) or "x"))
                rec[v]["mcd"].append(mcd_dtw(mel.cpu().numpy(), gt.numpy()))
                if i < 3:
                    sf.write(f"results/samples_postnet/{i:02d}_{kind}_{v}.wav", wav.numpy(), 22050)
        res = {"tf_l1_pre": float(np.mean(tf_pre)), "tf_l1_post": float(np.mean(tf_post))}
        for v in ("pre", "post"):
            for k in ("cer", "wer", "mcd"):
                res[f"{k}_{v}"] = float(np.mean(rec[v][k]))
        for k in ("cer", "wer", "mcd"):
            d = np.array(rec["post"][k]) - np.array(rec["pre"][k])
            res[f"{k}_diff_ci"] = bootstrap(d)
        out[kind] = res
        print(kind, json.dumps(res), flush=True)
        json.dump(out, open("results/postnet_ablation.json", "w"), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
