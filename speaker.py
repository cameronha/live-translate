"""Guess which voice said a phrase. Used only for text colors.

A small "voiceprint" model (WeSpeaker ResNet34, 26 MB, runs on onnxruntime)
turns a phrase into 256 numbers. Phrases with similar numbers get the same
speaker id. This is a guess, and it works per phrase, not per word.
"""
from pathlib import Path

import numpy as np

SR = 16000
MODEL = Path(__file__).resolve().parent / "models" / "speaker" / "voxceleb_resnet34_LM.onnx"


def _mel_filters(n_mels=80, n_fft=512, low=20.0, high=8000.0):
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)
    edges = np.linspace(mel(low), mel(high), n_mels + 2)
    fft_mel = mel(np.arange(n_fft // 2 + 1) * SR / n_fft)
    left, center, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    up = (fft_mel - left) / (center - left)
    down = (right - fft_mel) / (right - center)
    return np.maximum(0.0, np.minimum(up, down)).astype(np.float32)


_FILTERS = _mel_filters()
_WINDOW = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(400) / 399)) ** 0.85


def fbank(audio):
    """Kaldi-style 80-band log mel features, mean-normalized. audio: float32 in [-1, 1]."""
    x = audio.astype(np.float64) * 32768.0
    n = 1 + (len(x) - 400) // 160
    idx = np.arange(400)[None, :] + 160 * np.arange(n)[:, None]
    frames = x[idx]
    frames -= frames.mean(axis=1, keepdims=True)
    frames = np.concatenate([frames[:, :1] * 0.03, frames[:, 1:] - 0.97 * frames[:, :-1]], axis=1)
    power = np.abs(np.fft.rfft(frames * _WINDOW, 512)) ** 2
    feats = np.log(np.maximum(power @ _FILTERS.T, 1.19e-7)).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerGuesser:
    def __init__(self, threshold=0.45, max_speakers=6, min_seconds=1.2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
        self.threshold, self.max_speakers = threshold, max_speakers
        self.min_samples = int(min_seconds * SR)
        self.sums = []     # one running sum of voiceprints per speaker
        self.last = 0

    def embed(self, audio):
        emb = self.session.run(None, {"feats": fbank(audio)[None]})[0][0]
        return emb / (np.linalg.norm(emb) + 1e-9)

    def identify(self, audio):
        """Return a speaker number (0, 1, 2...) for this phrase."""
        if len(audio) < self.min_samples:
            return self.last  # too short to judge: keep the current speaker
        emb = self.embed(audio[-10 * SR:])
        sims = [float(emb @ (s / np.linalg.norm(s))) for s in self.sums]
        best = int(np.argmax(sims)) if sims else -1
        if best >= 0 and (sims[best] >= self.threshold or len(self.sums) >= self.max_speakers):
            if sims[best] >= self.threshold:
                self.sums[best] = self.sums[best] + emb
            self.last = best
        else:
            self.sums.append(emb.copy())
            self.last = len(self.sums) - 1
        return self.last
