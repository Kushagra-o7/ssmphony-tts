"""Autoregressive decoder-only TTS with a swappable sequence backbone.

    text --embed--> bidirectional encoder (n_enc blocks)  -> E (T x d)
    mel frames, grouped r at a time --prenet--> M (S x d)
    decoder input  = [E_1 ... E_T, GO, M_1 ... M_{S-1}]      (one sequence)
    causal decoder (n_dec blocks) over that sequence
    outputs at the S audio positions -> r mel frames + stop logit per step

There is no attention between text and audio.  The audio positions can read
the text only through the decoder's running state (for S4/Mamba/LSTM) or
causal self-attention over the prefix (Transformer baseline).  For S4 this is
exactly "stacked S4 layers in place of attention": the HiPPO memory must hold
the whole sentence while it is being spoken.

A non-causal conv postnet refines the finished spectrogram (Tacotron 2 style).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import Backbone


class Prenet(nn.Module):
    """Bottleneck with dropout that is ON at inference too (Tacotron trick).
    Forces the decoder to rely on the text rather than copying the last frame."""

    def __init__(self, d_in, d_out, hidden=256, p=0.5):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(d_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, d_out)
        self.p = p

    def forward(self, x):
        x = F.dropout(F.relu(self.l1(x)), self.p, training=True)
        x = F.dropout(F.relu(self.l2(x)), self.p, training=True)
        return self.l3(x)


class Postnet(nn.Module):
    def __init__(self, n_mels, ch=256, k=5, n=5, dropout=0.5):
        super().__init__()
        layers = []
        for i in range(n):
            cin, cout = (n_mels if i == 0 else ch), (n_mels if i == n - 1 else ch)
            layers += [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout)]
            if i < n - 1:
                layers += [nn.Tanh(), nn.Dropout(dropout)]
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):  # x (B, T, n_mels), mask (B, T, 1)
        x = x.transpose(1, 2)
        m = None if mask is None else mask.transpose(1, 2)
        for layer in self.net:
            x = layer(x)
            if m is not None:
                x = x * m  # re-zero padding after every layer (conv bias / BN shift) so it never leaks
        return x.transpose(1, 2)


class PrefixARTTS(nn.Module):
    """Ablation: text as a causal prefix, no alignment mechanism at all (see report Sec. 8)."""
    def __init__(self, kind, n_vocab, d=384, n_enc=4, n_dec=8, n_mels=80, r=4, ffn_mult=2,
                 dropout=0.1, checkpoint=False):
        super().__init__()
        self.kind, self.r, self.n_mels, self.d = kind, r, n_mels, d
        self.embed = nn.Embedding(n_vocab, d, padding_idx=0)
        self.encoder = Backbone(kind, d, n_enc, causal=False, ffn_mult=ffn_mult, dropout=dropout,
                                checkpoint=checkpoint)
        self.prenet = Prenet(r * n_mels, d)
        self.go = nn.Parameter(torch.randn(d) * 0.02)
        self.seg = nn.Embedding(2, d)  # 0 = text segment, 1 = audio segment
        self.decoder = Backbone(kind, d, n_dec, causal=True, ffn_mult=ffn_mult, dropout=dropout,
                                checkpoint=checkpoint)
        self.mel_head = nn.Linear(d, r * n_mels)
        self.stop_head = nn.Linear(d, 1)
        self.postnet = Postnet(n_mels)

    # ------------------------------------------------------------------ train
    def encode(self, text, text_len):
        mask = torch.arange(text.shape[1], device=text.device)[None] < text_len[:, None]
        return self.encoder(self.embed(text), mask), mask

    def forward(self, text, text_len, mel, n_steps):
        """text (B,T), text_len (B,), mel (B, S*r, n_mels) normalized, n_steps (B,)
        Returns mel_pred, mel_post (B, S*r, n_mels), stop_logits (B, S)."""
        B, S = mel.shape[0], mel.shape[1] // self.r
        enc, tmask = self.encode(text, text_len)
        enc = enc + self.seg.weight[0]

        steps = mel.view(B, S, self.r * self.n_mels)
        aud = torch.cat([self.go.expand(B, 1, -1), self.prenet(steps[:, :-1])], 1) + self.seg.weight[1]

        # place audio right after each sample's own text: [text_i ; audio_i ; pad]
        L = int((text_len + n_steps).max())
        seq = torch.zeros(B, L, self.d, device=mel.device, dtype=enc.dtype)
        seq[:, : enc.shape[1]] = enc * tmask.unsqueeze(-1)
        pos = (text_len[:, None] + torch.arange(S, device=mel.device)[None]).clamp(max=L - 1)  # (B,S)
        amask = torch.arange(S, device=mel.device)[None] < n_steps[:, None]
        seq = seq.scatter_add(1, pos.unsqueeze(-1).expand(-1, -1, self.d),
                              (aud * amask.unsqueeze(-1)).to(seq.dtype))
        smask = torch.arange(L, device=mel.device)[None] < (text_len + n_steps)[:, None]

        out = self.decoder(seq, smask)
        h = out.gather(1, pos.unsqueeze(-1).expand(-1, -1, self.d))                  # (B,S,d)
        mel_pred = self.mel_head(h).view(B, S * self.r, self.n_mels)
        return mel_pred, self.post(mel_pred, n_steps), self.stop_head(h).squeeze(-1)

    def post(self, mel_pred, n_steps):
        """Postnet on real frames only (its convolutions must not see padding)."""
        fm = (torch.arange(mel_pred.shape[1], device=mel_pred.device)[None] < (n_steps * self.r)[:, None])
        fm = fm.unsqueeze(-1).to(mel_pred.dtype)
        mel_pred = mel_pred * fm
        return mel_pred + self.postnet(mel_pred, fm) * fm

    # -------------------------------------------------------------- generate
    @torch.no_grad()
    def generate(self, text, max_steps=250, stop_threshold=0.5):
        """text: (1, T) LongTensor. Returns mel (frames, n_mels) normalized, n_steps."""
        assert text.shape[0] == 1
        enc, _ = self.encode(text, torch.tensor([text.shape[1]], device=text.device))
        prefix = torch.cat([enc + self.seg.weight[0], (self.go + self.seg.weight[1]).view(1, 1, -1)], 1)
        h, cache = self.decoder.prefill(prefix)
        h = h[:, -1]
        frames = []
        for _ in range(max_steps):
            f = self.mel_head(h)                                  # (1, r*n_mels)
            frames.append(f)
            if torch.sigmoid(self.stop_head(h)).item() > stop_threshold:
                break
            x = self.prenet(f) + self.seg.weight[1]
            h, cache = self.decoder.step(x, cache)
        mel = torch.cat(frames, 0).view(1, -1, self.n_mels)
        mel = mel + self.postnet(mel)
        return mel[0], len(frames)


class DurationPredictor(nn.Module):
    """Predicts log(1 + frames) per character from encoder states (FastSpeech style)."""

    def __init__(self, d, hidden=256, k=3, dropout=0.1):
        super().__init__()
        self.c1, self.c2 = nn.Conv1d(d, hidden, k, padding=k // 2), nn.Conv1d(hidden, hidden, k, padding=k // 2)
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):  # x (B,T,d)
        m = mask.unsqueeze(-1).to(x.dtype)
        x = self.drop(self.n1(F.relu(self.c1((x * m).transpose(1, 2)).transpose(1, 2))))
        x = self.drop(self.n2(F.relu(self.c2((x * m).transpose(1, 2)).transpose(1, 2))))
        return self.out(x).squeeze(-1) * mask


class ARTTS(nn.Module):
    """Duration-guided autoregressive TTS (the main model).

        text -> bidirectional encoder -> E (T x d) -> duration predictor
        E stretched by durations to frame rate, averaged per step of r frames -> U (S x d)
        decoder input at step s = prenet(frames of step s-1) + U_s   (step 0 uses GO)
        causal decoder (stacked S4 / Mamba / attention / LSTM blocks) -> r mel frames per step

    The decoder never attends to the text: alignment is given by durations (ground
    truth from CTC forced alignment in training, predicted at inference), and the
    causal backbone models the frame-to-frame acoustics autoregressively.
    """

    def __init__(self, kind, n_vocab, d=384, n_enc=4, n_dec=8, n_mels=80, r=4, ffn_mult=2,
                 dropout=0.1, checkpoint=False):
        super().__init__()
        self.kind, self.r, self.n_mels, self.d = kind, r, n_mels, d
        self.embed = nn.Embedding(n_vocab, d, padding_idx=0)
        self.encoder = Backbone(kind, d, n_enc, causal=False, ffn_mult=ffn_mult, dropout=dropout,
                                checkpoint=checkpoint)
        self.dur = DurationPredictor(d)
        self.prenet = Prenet(r * n_mels, d)
        self.go = nn.Parameter(torch.randn(d) * 0.02)
        self.decoder = Backbone(kind, d, n_dec, causal=True, ffn_mult=ffn_mult, dropout=dropout,
                                checkpoint=checkpoint)
        self.mel_head = nn.Linear(d, r * n_mels)
        self.postnet = Postnet(n_mels)

    def encode(self, text, text_len):
        mask = torch.arange(text.shape[1], device=text.device)[None] < text_len[:, None]
        enc = self.encoder(self.embed(text), mask)
        return enc, mask, self.dur(enc, mask)

    def regulate(self, enc, dur, S):
        """Length regulator: repeat each character's state for its duration, then
        average every r frames into one decoder step. enc (B,T,d), dur (B,T) -> (B,S,d)."""
        B, T, d = enc.shape
        cum = dur.cumsum(1)
        frames = torch.arange(S * self.r, device=enc.device).unsqueeze(0).expand(B, -1).contiguous()
        idx = torch.searchsorted(cum.contiguous(), frames, right=True).clamp(max=T - 1)
        up = enc.gather(1, idx.unsqueeze(-1).expand(-1, -1, d))
        return up.view(B, S, self.r, d).mean(2)

    def forward(self, text, text_len, mel, n_steps, dur):
        B, S = mel.shape[0], mel.shape[1] // self.r
        enc, tmask, log_dur = self.encode(text, text_len)
        U = self.regulate(enc, dur, S)
        steps = mel.view(B, S, self.r * self.n_mels)
        x = torch.cat([self.go.expand(B, 1, -1), self.prenet(steps[:, :-1])], 1) + U
        smask = torch.arange(S, device=mel.device)[None] < n_steps[:, None]
        h = self.decoder(x, smask)
        mel_pred = self.mel_head(h).view(B, S * self.r, self.n_mels)
        return mel_pred, PrefixARTTS.post(self, mel_pred, n_steps), log_dur

    @torch.no_grad()
    def generate(self, text, length_scale=1.0, dur=None):
        """text (1,T). Uses predicted durations unless `dur` is given. Returns mel (frames, n_mels), steps."""
        enc, _, log_dur = self.encode(text, torch.tensor([text.shape[1]], device=text.device))
        if dur is None:
            dur = torch.clamp(torch.round((torch.exp(log_dur) - 1) * length_scale), min=0).long()
        n_frames = max(int(dur.sum()), self.r)
        S = (n_frames + self.r - 1) // self.r
        U = self.regulate(enc, dur, S)
        h, cache = self.decoder.prefill((self.go + U[:, 0]).unsqueeze(1))
        h = h[:, -1]
        frames = []
        for s in range(S):
            f = self.mel_head(h)
            frames.append(f)
            if s + 1 < S:
                h, cache = self.decoder.step(self.prenet(f) + U[:, s + 1], cache)
        mel = torch.cat(frames, 0).view(1, -1, self.n_mels)
        mel = mel + self.postnet(mel)
        return mel[0, :n_frames], S
