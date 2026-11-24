"""LJSpeech data pipeline: text normalisation, mel extraction, batching."""
import os
import random
import unicodedata

import torch
import torchaudio

DATA_ROOT = os.environ.get("SSM_DATA", "C:/ssm_data")
LJ_DIR = os.path.join(DATA_ROOT, "LJSpeech-1.1")
CACHE = os.path.join(DATA_ROOT, "cache")

# Mel settings match the pretrained SpeechBrain HiFi-GAN (LJSpeech) vocoder
SR, N_FFT, HOP, WIN, N_MELS, FMIN, FMAX = 22050, 1024, 256, 1024, 80, 0.0, 8000.0

# ---------------------------------------------------------------- text
PAD, EOS = "_", "~"
CHARS = PAD + EOS + " abcdefghijklmnopqrstuvwxyz'.,!?-;:\"()"
CHAR2ID = {c: i for i, c in enumerate(CHARS)}
N_VOCAB = len(CHARS)


def normalize_text(s):
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace("[", "(").replace("]", ")")
    return "".join(c for c in s if c in CHAR2ID and c not in (PAD, EOS))


def text_to_ids(s):
    return [CHAR2ID[c] for c in normalize_text(s)] + [CHAR2ID[EOS]]


# ---------------------------------------------------------------- audio
_mel_fn = None


def wav_to_mel(wav):
    """wav: (T,) float tensor at 22.05 kHz -> log-mel (frames, 80)."""
    global _mel_fn
    if _mel_fn is None:
        _mel_fn = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, win_length=WIN, hop_length=HOP, n_mels=N_MELS,
            f_min=FMIN, f_max=FMAX, power=1.0, normalized=False, norm="slaney", mel_scale="slaney")
    _mel_fn = _mel_fn.to(wav.device)
    mel = _mel_fn(wav)
    return torch.log(torch.clamp(mel, min=1e-5)).T


# ---------------------------------------------------------------- dataset
def load_split(split):
    """Returns dict with lists: ids, texts, text_ids, mels (fp16), and stats."""
    return torch.load(os.path.join(CACHE, f"{split}.pt"), weights_only=False)


class LJDataset(torch.utils.data.Dataset):
    def __init__(self, split, stats=None):
        d = load_split(split)
        self.ids, self.texts, self.text_ids, self.mels = d["ids"], d["texts"], d["text_ids"], d["mels"]
        st = stats or torch.load(os.path.join(CACHE, "stats.pt"))
        self.mean, self.std = st["mean"], st["std"]
        dp = os.path.join(CACHE, f"{split}_dur.pt")  # per-character durations from align.py
        self.durs = torch.load(dp) if os.path.exists(dp) else None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        mel = (self.mels[i].float() - self.mean) / self.std
        dur = self.durs[i].long() if self.durs is not None else torch.zeros(len(self.text_ids[i]), dtype=torch.long)
        return torch.tensor(self.text_ids[i]), mel, dur


def make_collate(r):
    def collate(batch):
        texts, mels, durs = zip(*batch)
        tl = torch.tensor([len(t) for t in texts])
        n_steps = torch.tensor([(m.shape[0] + r - 1) // r for m in mels])
        T, S = int(tl.max()), int(n_steps.max())
        text = torch.zeros(len(batch), T, dtype=torch.long)
        mel = torch.zeros(len(batch), S * r, N_MELS)
        fmask = torch.zeros(len(batch), S * r, dtype=torch.bool)
        dur = torch.zeros(len(batch), T, dtype=torch.long)
        for i, (t, m, du) in enumerate(batch):
            text[i, : len(t)] = t
            mel[i, : m.shape[0]] = m
            fmask[i, : m.shape[0]] = True
            dur[i, : len(du)] = du
        return text, tl, mel, n_steps, fmask, dur
    return collate


class BucketSampler(torch.utils.data.Sampler):
    """Groups utterances of similar length; each batch holds <= max_frames mel frames
    (padded).  Deterministic per epoch given the seed, so every model sees the
    same batches in the same order."""

    def __init__(self, lengths, max_frames, seed=0):
        self.lengths, self.max_frames, self.seed, self.epoch = lengths, max_frames, seed, 0

    def _batches(self):
        rng = random.Random(self.seed + self.epoch)
        idx = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i] + rng.random() * 40)
        batches, cur, longest = [], [], 0
        for i in idx:
            longest_new = max(longest, self.lengths[i])
            if cur and longest_new * (len(cur) + 1) > self.max_frames:
                batches.append(cur)
                cur, longest_new = [], self.lengths[i]
            cur.append(i)
            longest = longest_new
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        return batches

    def __iter__(self):
        b = self._batches()
        self.epoch += 1
        return iter(b)

    def __len__(self):
        return len(self._batches())
