"""Compute log-mel spectrograms for LJSpeech and save train/val/test caches."""
import os
import random

import soundfile as sf
import torch
from tqdm import tqdm

from data import CACHE, LJ_DIR, SR, normalize_text, text_to_ids, wav_to_mel

N_VAL, N_TEST = 200, 100


def main():
    os.makedirs(CACHE, exist_ok=True)
    rows = [l.strip().split("|") for l in open(os.path.join(LJ_DIR, "metadata.csv"), encoding="utf-8")]
    random.Random(1234).shuffle(rows)
    splits = {"test": rows[:N_TEST], "val": rows[N_TEST:N_TEST + N_VAL], "train": rows[N_TEST + N_VAL:]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for name, rs in splits.items():
        out = {"ids": [], "texts": [], "text_ids": [], "mels": []}
        for r in tqdm(rs, desc=name):
            wav, sr = sf.read(os.path.join(LJ_DIR, "wavs", r[0] + ".wav"), dtype="float32")
            assert sr == SR
            mel = wav_to_mel(torch.from_numpy(wav).to(dev)).cpu()
            out["ids"].append(r[0])
            out["texts"].append(normalize_text(r[2]))
            out["text_ids"].append(text_to_ids(r[2]))
            out["mels"].append(mel.half())
        torch.save(out, os.path.join(CACHE, f"{name}.pt"))
        if name == "train":
            allm = torch.cat([m.float() for m in out["mels"]])
            torch.save({"mean": allm.mean(0), "std": allm.std(0)}, os.path.join(CACHE, "stats.pt"))
            n = allm.shape[0]
            print(f"train: {len(rs)} utts, {n} frames = {n * 256 / SR / 3600:.2f} h")


if __name__ == "__main__":
    main()
