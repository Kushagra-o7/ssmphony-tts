"""Per-character durations (in mel frames) for LJSpeech via CTC forced alignment.

A pretrained wav2vec2 character CTC model (torchaudio WAV2VEC2_ASR_BASE_960H)
gives frame-level emissions; torchaudio.functional.forced_align finds the best
monotonic path through the transcript's letters.  Letter spans are converted from
20 ms aligner frames to 11.6 ms mel frames.  Non-letter characters (spaces,
punctuation) share the gap between the surrounding words; leading silence goes to
the first character, trailing silence to the end-of-text token.
Durations sum exactly to the utterance's mel length.  Saves cache/{split}_dur.pt.
"""
import os

import soundfile as sf
import torch
import torchaudio
from tqdm import tqdm

from data import CACHE, CHARS, HOP, LJ_DIR, SR, load_split

dev = "cuda"


def char_durations(text_ids, emission, n_mel, labels):
    """text_ids: our ids (chars + EOS). emission: (T_frames, n_labels) log-probs."""
    chars = [CHARS[i] for i in text_ids]
    lab = {c: i for i, c in enumerate(labels)}
    # aligner targets: letters/apostrophes only, '|' between words
    targets, owner = [], []          # owner[k] = index into chars for target k (None for '|')
    prev_letter = False
    for j, c in enumerate(chars):
        u = c.upper()
        if u in lab and u not in ("-", "|"):
            if not prev_letter and targets:
                targets.append(lab["|"]); owner.append(None)
            targets.append(lab[u]); owner.append(j)
            prev_letter = True
        else:
            prev_letter = False
    T = emission.shape[0]
    if not targets or len(targets) > T:
        return None
    ali, _ = torchaudio.functional.forced_align(emission.unsqueeze(0), torch.tensor([targets], device=emission.device), blank=0)
    spans = torchaudio.functional.merge_tokens(ali[0], torch.zeros_like(ali[0], dtype=torch.float))
    if len(spans) != len(targets):
        return None
    return spans, owner


def to_mel_durations(chars_n, spans, owner, frames_to_mel, n_mel):
    """Assign mel-frame boundaries to every character."""
    start = [None] * chars_n
    end = [None] * chars_n
    for sp, j in zip(spans, owner):
        if j is not None:
            start[j], end[j] = sp.start * frames_to_mel, sp.end * frames_to_mel
    letters = [j for j in range(chars_n) if start[j] is not None]
    bounds = [0.0] * (chars_n + 1)   # bounds[j] = start of char j
    # walk characters: letters take their span; runs of non-letters split the gap evenly
    j = 0
    cursor = 0.0
    while j < chars_n:
        if start[j] is not None:
            bounds[j] = cursor if j == letters[0] else start[j]
            cursor = end[j]
            j += 1
            continue
        k = j
        while k < chars_n and start[k] is None:
            k += 1
        gap_end = float(n_mel) if k == chars_n else start[k]
        if j == 0:                       # leading non-letters: start at 0
            cursor = 0.0
        n = k - j
        for t in range(n):
            bounds[j + t] = cursor + (gap_end - cursor) * t / n
        cursor = gap_end
        j = k
    bounds[chars_n] = float(n_mel)
    b = torch.tensor(bounds).round().long().clamp(0, n_mel)
    b = torch.cummax(b, 0).values
    b[-1] = n_mel
    return (b[1:] - b[:-1])


def main():
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(dev).eval()
    labels = bundle.get_labels()
    for split in ("test", "val", "train"):
        d = load_split(split)
        durs, failed = [], 0
        for uid, ids, mel in tqdm(list(zip(d["ids"], d["text_ids"], d["mels"])), desc=split):
            wav, sr = sf.read(os.path.join(LJ_DIR, "wavs", uid + ".wav"), dtype="float32")
            w = torchaudio.functional.resample(torch.from_numpy(wav).to(dev), SR, bundle.sample_rate)
            with torch.inference_mode():
                em, _ = model(w.unsqueeze(0))
                em = torch.log_softmax(em[0], -1)
            n_mel = mel.shape[0]
            res = char_durations(ids, em, n_mel, labels)
            frames_to_mel = (len(w) / bundle.sample_rate / em.shape[0]) * SR / HOP
            if res is None:
                failed += 1
                dur = torch.full((len(ids),), n_mel // len(ids), dtype=torch.long)
                dur[-1] += n_mel - dur.sum()
            else:
                dur = to_mel_durations(len(ids), res[0], res[1], frames_to_mel, n_mel)
            assert dur.sum() == n_mel and len(dur) == len(ids)
            durs.append(dur.short())
        torch.save(durs, os.path.join(CACHE, f"{split}_dur.pt"))
        print(f"{split}: {len(durs)} utts, {failed} alignment failures (uniform fallback)")


if __name__ == "__main__":
    main()
