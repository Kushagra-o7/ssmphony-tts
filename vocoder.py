"""Mel -> waveform. Pretrained HiFi-GAN (SpeechBrain, LJSpeech 22.05 kHz),
with a Griffin-Lim fallback."""
import os

import torch
import torchaudio

from data import DATA_ROOT, FMAX, FMIN, HOP, N_FFT, N_MELS, SR, WIN

_hifi = None


def hifigan(device="cuda"):
    global _hifi
    if _hifi is None:
        from speechbrain.inference.vocoders import HIFIGAN
        from speechbrain.utils.fetching import LocalStrategy
        _hifi = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                     savedir=os.path.join(DATA_ROOT, "pretrained", "hifigan"),
                                     run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    return _hifi


@torch.no_grad()
def mel_to_wav(log_mel, device="cuda"):
    """log_mel: (frames, 80) natural-log magnitude mel (un-normalized)."""
    wav = hifigan(device).decode_batch(log_mel.T.unsqueeze(0).to(device))
    return wav.squeeze().cpu()


@torch.no_grad()
def griffin_lim(log_mel, n_iter=60):
    inv = torchaudio.transforms.InverseMelScale(n_stft=N_FFT // 2 + 1, n_mels=N_MELS, sample_rate=SR,
                                                f_min=FMIN, f_max=FMAX, norm="slaney", mel_scale="slaney")
    gl = torchaudio.transforms.GriffinLim(n_fft=N_FFT, win_length=WIN, hop_length=HOP, power=1.0, n_iter=n_iter)
    return gl(inv(torch.exp(log_mel.T.float().cpu())))
