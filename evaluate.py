"""Evaluate trained TTS models on the held-out test split.

For every model:
  * teacher-forced L1 on normalized mels (test split)
  * free-running generation of all test sentences (batch 1, recurrent / KV-cache)
      - stop rate, duration ratio, generation speed (ms per step, real-time factor)
      - MCD-DTW (mel-cepstral distortion vs ground truth, after DTW alignment)
      - intelligibility: Whisper ASR on the vocoded audio -> CER / WER vs the text
Ground-truth mels through the same vocoder give the upper bound.
Writes results/eval.json and results/samples/*.wav
"""
import argparse
import json
import os
import re
import time

import jiwer
import numpy as np
import soundfile as sf
import torch
import torchaudio

from data import DATA_ROOT, N_MELS, LJDataset, make_collate
from train import build_model, run_batch
from vocoder import mel_to_wav

dev = "cuda"
N_SAMPLES_SAVE = 6


def norm_text(s):
    s = s.lower().replace("-", " ")
    s = re.sub(r"[^a-z' ]", "", s)
    return re.sub(r"\s+", " ", s).strip()


class ASR:
    def __init__(self, name="openai/whisper-base"):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        self.proc = WhisperProcessor.from_pretrained(name)
        self.model = WhisperForConditionalGeneration.from_pretrained(name).to(dev).eval()

    @torch.no_grad()
    def __call__(self, wav22k):
        wav = torchaudio.functional.resample(wav22k, 22050, 16000).numpy()
        feats = self.proc(wav, sampling_rate=16000, return_tensors="pt").input_features.to(dev)
        ids = self.model.generate(feats, language="en", task="transcribe")
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0]


def mfcc_from_logmel(lm, n=13):
    """DCT-II of the natural-log mel amplitude -> cepstral coefficients 1..n (drop c0 = energy).

    Scaled as a cepstrum, c_n = (1/M) sum_k ln|X_k| cos(pi n (k+1/2) / M), so that
    ln|X_k| = c_0 + 2 sum_n c_n cos(...). This is the convention the MCD formula
    (10/ln10) * sqrt(2 * sum dc^2) assumes. An orthonormal DCT would inflate MCD by sqrt(2M)."""
    M = lm.shape[1]
    k = np.arange(M)
    basis = np.cos(np.pi / M * (k[None, :] + 0.5) * np.arange(n + 1)[:, None]) / M
    return (lm @ basis.T)[:, 1:]


def mcd_dtw(a, b):
    """Mel-cepstral distortion (dB) between log-mel a (T1,80) and b (T2,80), DTW-aligned."""
    ca, cb = mfcc_from_logmel(a), mfcc_from_logmel(b)
    d = np.sqrt(((ca[:, None, :] - cb[None, :, :]) ** 2).sum(-1))
    T1, T2 = d.shape
    acc = np.full((T1 + 1, T2 + 1), np.inf); acc[0, 0] = 0
    steps = np.zeros((T1 + 1, T2 + 1), dtype=np.int32)
    for i in range(1, T1 + 1):
        row = acc[i - 1]
        for j in range(1, T2 + 1):
            best = min(acc[i - 1, j - 1], row[j], acc[i, j - 1])
            acc[i, j] = d[i - 1, j - 1] + best
            steps[i, j] = (steps[i - 1, j - 1] if best == acc[i - 1, j - 1] else
                           steps[i - 1, j] if best == row[j] else steps[i, j - 1]) + 1
    return float(10 / np.log(10) * np.sqrt(2) * acc[T1, T2] / steps[T1, T2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kinds", default="s4,transformer,lstm,mamba,s4@8000,transformer@8000,lstm@8000,mamba@8000")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--runs", default=os.path.join(DATA_ROOT, "runs"))
    ap.add_argument("--tag", default="", help="suffix for result names, e.g. _prefix")
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    os.makedirs("results/samples", exist_ok=True)
    test = LJDataset("test")
    mean, std = test.mean.to(dev), test.std.to(dev)
    asr = ASR()
    results = json.load(open("results/eval.json")) if os.path.exists("results/eval.json") else {}

    if "ground_truth" not in results:
        cers, wers = [], []
        for i in range(args.n):
            wav = mel_to_wav(test.mels[i].float())
            hyp = asr(wav)
            cers.append(jiwer.cer(norm_text(test.texts[i]), norm_text(hyp)))
            wers.append(jiwer.wer(norm_text(test.texts[i]), norm_text(hyp)))
            if i < N_SAMPLES_SAVE:
                sf.write(f"results/samples/{i:02d}_ground_truth.wav", wav.numpy(), 22050)
        results["ground_truth"] = {"cer": float(np.mean(cers)), "wer": float(np.mean(wers))}
        print("ground truth (vocoded):", results["ground_truth"], flush=True)

    coll = make_collate(4)
    for name in args.kinds.split(","):
        kind, _, snap = name.partition("@")  # "s4" -> final model, "s4@8000" -> snapshot at step 8000
        name = kind + args.tag + (f"@{snap}" if snap else "")
        ck = os.path.join(args.runs, kind, f"snap_{snap}.pt" if snap else "final.pt")
        if not os.path.exists(ck):
            print("missing", ck); continue
        if name in results and not args.redo:
            print("skip (done)", name); continue
        state = torch.load(ck, map_location=dev, weights_only=False)
        # older checkpoints carry no tag: the prefix model is the one with a stop head
        variant = state.get("variant", "prefix" if any(k.startswith("stop_head") for k in state["model"]) else "duration")
        model = build_model(kind, variant=variant).to(dev)
        model.load_state_dict(state["model"])
        model.eval()
        torch.manual_seed(0)

        # teacher-forced loss
        tf = []
        with torch.no_grad():
            for i in range(0, args.n, 10):
                tf.append(run_batch(model, coll([test[j] for j in range(i, min(i + 10, args.n))]), dev)[1]["l1"])

        rec = {"tf_l1": float(np.mean(tf)), "cer": [], "wer": [], "mcd": [], "dur_ratio": [], "stopped": [],
               "ms_per_step": [], "rtf": [], "hyp": []}
        for i in range(args.n):
            text_ids, _, gt_dur = test[i]
            gt = test.mels[i].float()
            max_steps = int(gt.shape[0] / 4 * 2) + 20
            torch.cuda.synchronize(); t0 = time.perf_counter()
            if variant == "duration":
                mel_n, n_steps = model.generate(text_ids.unsqueeze(0).to(dev))
                max_steps = n_steps + 1  # length comes from the duration predictor; always "stops"
            else:
                mel_n, n_steps = model.generate(text_ids.unsqueeze(0).to(dev), max_steps=max_steps)
            torch.cuda.synchronize(); dt = time.perf_counter() - t0
            mel = (mel_n * std + mean).float()
            wav = mel_to_wav(mel)
            hyp = asr(wav)
            rec["cer"].append(jiwer.cer(norm_text(test.texts[i]), norm_text(hyp) or "x"))
            rec["wer"].append(jiwer.wer(norm_text(test.texts[i]), norm_text(hyp) or "x"))
            rec["mcd"].append(mcd_dtw(mel.cpu().numpy(), gt.numpy()))
            rec["dur_ratio"].append(mel.shape[0] / gt.shape[0])
            rec["stopped"].append(n_steps < max_steps)
            rec["ms_per_step"].append(dt / n_steps * 1e3)
            rec["rtf"].append(dt / (mel.shape[0] * 256 / 22050))
            rec["hyp"].append(hyp)
            if i < N_SAMPLES_SAVE:
                sf.write(f"results/samples/{i:02d}_{name}.wav", wav.numpy(), 22050)
                np.save(f"results/samples/{i:02d}_{name}_mel.npy", mel.cpu().numpy())
            if i % 20 == 0:
                print(f"[{name}] {i}: cer {rec['cer'][-1]:.3f} mcd {rec['mcd'][-1]:.2f} | {hyp[:70]}", flush=True)
        summary = {k: float(np.mean(v)) for k, v in rec.items() if k != "hyp"}
        summary["cer_median"] = float(np.median(rec["cer"]))
        # Whisper can loop on babble (CER > 1); clipped means bound each sentence at 100% error
        summary["cer_clipped"] = float(np.mean(np.minimum(rec["cer"], 1.0)))
        summary["wer_clipped"] = float(np.mean(np.minimum(rec["wer"], 1.0)))
        summary["wer_median"] = float(np.median(rec["wer"]))
        summary["hyp_examples"] = rec["hyp"][:N_SAMPLES_SAVE]
        summary["per_utt_cer"] = rec["cer"]
        results[name] = summary
        print(name, {k: v for k, v in summary.items() if not isinstance(v, list)}, flush=True)
        json.dump(results, open("results/eval.json", "w"), indent=1)
        del model; torch.cuda.empty_cache()

    for i in range(N_SAMPLES_SAVE):
        np.save(f"results/samples/{i:02d}_ground_truth_mel.npy", test.mels[i].float().numpy())
    json.dump({"texts": test.texts[:N_SAMPLES_SAVE]}, open("results/samples/texts.json", "w"), indent=1)
    json.dump(results, open("results/eval.json", "w"), indent=1)


if __name__ == "__main__":
    main()
