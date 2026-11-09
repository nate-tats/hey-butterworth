"""Butterworth features: audio -> log-mel spectrograms, augmentation, training batches.

This file is everything between a WAV file and the CNN's input. The CNN and its
training live in train.py. Design and rationale: Butterworth.md,
sections 2 (front end) and 4 (augmentation).

FRONT END (16 kHz audio -> log-mel frames), mirrored by the device
  STFT: 480-sample periodic Hann window (30 ms), hop 160 (10 ms), n_fft=512
  -> power spectrum -> 40 mel bins (125-7500 Hz, HTK mel) -> log(x + 1e-6)
  Frames from N samples (no centre padding): T = 1 + (N - 480) // 160
    1.50 s recorded clip = 24000 samples -> 148 frames
    model input          = 20160 samples -> 124 frames (1.26 s)

AUGMENTATION + BATCHES (training)
  rec (manifest row) + int16 audio
    -> speed perturb (ww, sp)            -> sub-frame shift + fit to 1.5 s
    -> channel EQ (toward ESP32 MEMS)    -> reverb (mostly dry TTS)
    -> background / babble mix at SNR    -> gain -> MEMS noise floor -> clip
    -> logmel (148 x 40)                 -> random 124-frame crop (word kept whole)
    -> SpecAugment                       -> (124, 40) float32, label

Reproducible: every example draws from its own generator, example_rng(seed,
batch, slot), so batches(seed=s) gives the same stream on every run and in any
worker, and start=b resumes at batch b. build_augmenter(seed) fixes the EQ
banks; eval_set() gives a fixed unaugmented val set.

Wake phrase: "hey Butterworth" only. "Butterworth" alone, "hi/hello Butterworth",
"hey butter" etc. are negatives.

Manifest: dataset/manifest_v1.csv (dataset README): path, label (wake|speech|background),
label_id (bg=0, sp=1, ww=2), source, speaker, domain, phrase, split (train|val).
Optional on, off = phrase span in samples for wake clips: fill with
annotate_spans(), hand-check the real MEMS clips, save with write_manifest().

COMMAND LINE
  python ml/features.py export [--manifest ...] [--float16] [--preview K]
      clean (unaugmented) (N, 148, 40) log-mels -> dataset/spectrograms/
  python ml/features.py selftest
      pipeline smoke test on fake clips
"""
import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parent.parent   # repo root; manifest paths start here

# ---------------------------------------------------------------- constants
SR = 16000
WIN = 480
HOP = 160
N_FFT = 512
N_MELS = 40
FMIN, FMAX = 125.0, 7500.0
LOG_EPS = 1e-6

# Model window. train.py checks that the CNN's receptive field is N_FRAMES and
# its total stride is DECISION_HOP, so the two files cannot drift apart.
N_FRAMES = 124                                            # model input frames
DECISION_HOP = 24                                         # frames between streaming decisions
N_SAMPLES = WIN + (N_FRAMES - 1) * HOP                    # 20160 (1.26 s)
HOP_SAMPLES = DECISION_HOP * HOP                          # 3840 (240 ms)

CLIP_SAMPLES = 24000                                      # recorded clips, 1.5 s
CLIP_FRAMES = 1 + (CLIP_SAMPLES - WIN) // HOP             # 148
N_CROPS = CLIP_FRAMES - N_FRAMES + 1                      # 25 (offsets 0-24)
assert N_CROPS >= DECISION_HOP + 1, "crops must cover a full streaming step"


# ---------------------------------------------------------------- front end
def _hz_to_mel(f):
    return 1127.0 * np.log1p(np.asarray(f) / 700.0)


def mel_matrix():
    """(n_fft//2+1, N_MELS) triangular HTK mel filterbank."""
    n_bins = N_FFT // 2 + 1
    bin_mel = _hz_to_mel(np.linspace(0.0, SR / 2, n_bins))
    edges = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    lo, ctr, hi = edges[:-2], edges[1:-1], edges[2:]
    m = bin_mel[:, None]
    up = (m - lo) / (ctr - lo)
    down = (hi - m) / (hi - ctr)
    w = np.maximum(0.0, np.minimum(up, down))
    w[0] = 0.0                                            # drop DC bin
    return w.astype(np.float32)


_HANN = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WIN) / WIN)).astype(np.float32)
_MEL = mel_matrix()


def logmel(audio):
    """Batch front end: 1-D float audio -> (T, N_MELS), T = 1 + (N-480)//160."""
    audio = np.asarray(audio, np.float32)
    T = 1 + (len(audio) - WIN) // HOP
    if T <= 0:
        return np.zeros((0, N_MELS), np.float32)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(T)[:, None]
    spec = np.abs(np.fft.rfft(audio[idx] * _HANN, n=N_FFT)) ** 2
    return np.log(spec @ _MEL + LOG_EPS).astype(np.float32)


class StreamingFrontend:
    """Push audio chunks of any size; get back only the newly completed frames.

    Keeps at most WIN - 1 leftover samples (320 once the stream is aligned to
    the hop). Pushing 3840 samples yields exactly 24 frames.
    """

    def __init__(self):
        self.buf = np.zeros(0, np.float32)

    def push(self, chunk):
        self.buf = np.concatenate([self.buf, np.asarray(chunk, np.float32)])
        frames = logmel(self.buf)
        self.buf = self.buf[len(frames) * HOP:]
        return frames


# ---------------------------------------------------------------- training data
def random_crop(mel_clip, rng, min_offset=0, max_offset=None):
    """Random N_FRAMES crop of a (CLIP_FRAMES, N_MELS) clip.

    Offsets are drawn uniformly from [min_offset, max_offset] (default: all
    N_CROPS offsets). Narrow the range for positive clips whose word sits near
    an edge, so every crop contains the whole word.
    """
    T = len(mel_clip)
    if max_offset is None:
        max_offset = T - N_FRAMES
    assert 0 <= min_offset <= max_offset <= T - N_FRAMES, (min_offset, max_offset, T)
    off = int(rng.integers(min_offset, max_offset + 1))
    return mel_clip[off:off + N_FRAMES]


CLASSES = ("ww", "sp", "bg")
LABEL = {"ww": 1.0, "sp": 0.0, "bg": 0.0}
CLS_OF_LABEL = {"wake": "ww", "speech": "sp", "background": "bg"}

# Longest phrase that some streaming window always contains whole, whatever its
# alignment: window 20160 samples, decisions every 3840, frames on a 160 grid.
# 16480 samples = 1.03 s. Longer phrases can fall between decisions.
MAX_PHRASE = N_SAMPLES - (HOP_SAMPLES - HOP)

# Generated negatives closest to "hey Butterworth" are drawn more often within
# their batch group (relative weight; unlisted phrases = 1).
PHRASE_WEIGHT = {
    "hey.butter": 3, "butterworth": 3, "hello.butterworth": 3, "hi.butterworth": 3,
    "hey.butler": 2, "hey.buttercup": 2, "hey": 2, "hey.wordsworth": 2,
}

# Share of each training batch per (class, domain). Groups absent from the
# manifest are dropped and the rest renormalised.
GROUP_SHARE = {
    ("ww", "generated"): 0.21, ("ww", "mems"): 0.09,
    ("sp", "generated"): 0.14, ("sp", "public"): 0.12,
    ("sp", "mems"): 0.08, ("sp", "phone"): 0.06,
    ("bg", "phone"): 0.14, ("bg", "mems"): 0.14, ("bg", "public"): 0.02,
}


@dataclass
class AugConfig:
    speeds: tuple = (0.9, 0.95, 1.05, 1.1)
    p_speed: float = 0.5                  # ww, sp
    p_eq: float = 0.7                     # generated / phone / public -> MEMS
    p_eq_mems: float = 0.2                # mild random EQ on real MEMS clips
    p_reverb_generated: float = 0.4
    p_reverb_other: float = 0.1           # real recordings already have a room
    rt60: tuple = (0.15, 0.7)             # s
    drr_db: tuple = (0.0, 15.0)           # direct-to-reverberant ratio
    p_bg: float = 0.6                     # mix background into ww / sp
    snr_bg: tuple = (0.0, 20.0)           # sp
    snr_bg_ww: tuple = (5.0, 20.0)        # ww: at 0 dB a soft "hey" can vanish
    p_babble: float = 0.15                # mix other speech into ww
    snr_babble: tuple = (10.0, 25.0)
    p_bg_bg: float = 0.3                  # mix two backgrounds
    snr_bg_bg: tuple = (-5.0, 10.0)
    gain_db: tuple = (-18.0, 6.0)
    noise_floor_dbfs: tuple = (-75.0, -55.0)   # placeholder: calibrate on MEMS bg
    p_specaug: float = 0.5
    freq_masks: int = 2
    max_freq_mask: int = 4                # of 40 mel bins
    time_masks: int = 2
    max_time_mask: int = 8                # frames (80 ms); never on ww (could erase "hey")


# ---------------------------------------------------------------- I/O
def load_wav(path):
    """Any WAV -> float32 mono at 16 kHz in [-1, 1]."""
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.dtype == np.uint8:
        x = (x.astype(np.float32) - 128.0) / 128.0
    elif x.dtype.kind == "i":
        x = x.astype(np.float32) / (np.iinfo(x.dtype).max + 1)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = signal.resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def to_int16(x):
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def to_float(x):
    return x.astype(np.float32) / 32768.0


def read_manifest(path, root=ROOT):
    """manifest_v1.csv -> list of dicts with cls (ww|sp|bg) and span added."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["cls"] = CLS_OF_LABEL[r["label"]]
        assert int(r["label_id"]) == {"bg": 0, "sp": 1, "ww": 2}[r["cls"]], r
        r["span"] = (int(r["on"]), int(r["off"])) if r.get("on") else None
        r["file"] = f"{root}/{r['path']}"
    return rows


def write_manifest(rows, path):
    """Write rows back with on/off span columns (bump the file name, e.g. v1_spans)."""
    cols = ["path", "label", "label_id", "source", "speaker", "domain", "phrase",
            "split", "on", "off"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            sp = r.get("span")
            w.writerow({**r, "on": sp[0] if sp else "", "off": sp[1] if sp else ""})


def split(manifest):
    """Group rows by the manifest's split column (train / val / test)."""
    out = {"train": [], "val": [], "test": []}
    for r in manifest:
        out[r["split"]].append(r)
    return out


class AudioStore:
    """Loads each clip once and keeps it as int16 (27k clips ~ 1.3 GB)."""

    def __init__(self):
        self._c = {}

    def __call__(self, path):
        if path not in self._c:
            self._c[path] = to_int16(load_wav(path))
        return to_float(self._c[path])


# ---------------------------------------------------------------- word span
def word_span(x, rel_db=30.0, floor_db=12.0, min_frames=5, max_gap=25):
    """Sample span [a, b) of the spoken phrase in a clip.

    Energy is measured per log-mel frame, i.e. only in the 125-7500 Hz speech
    band: the MEMS clips carry DC drift / rumble below 125 Hz that the model
    never sees but that would otherwise swamp a waveform-energy measure.
    A frame is active if it is within rel_db of the loudest frame AND floor_db
    above the clip's noise floor (10th-percentile frame). Active runs shorter
    than min_frames (clicks) are dropped, runs closer than max_gap frames
    (250 ms: the pause between "hey" and "Butterworth") are merged, and the
    group with the most energy is the phrase. Check MEMS spans by eye
    (python ml/features.py spans writes review sheets).
    """
    e = 10 * np.log10(np.exp(logmel(x)).sum(axis=1))
    act = e > max(e.max() - rel_db, np.percentile(e, 10) + floor_db)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], act.astype(np.int8), [0]])))
    runs = [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_frames]
    if not runs:
        return 0, len(x)
    groups = [list(runs[0])]
    for a, b in runs[1:]:
        if a - groups[-1][1] <= max_gap:
            groups[-1][1] = b
        else:
            groups.append([a, b])
    a, b = max(groups, key=lambda g: np.sum(10 ** (e[g[0]:g[1]] / 10)))
    return int(a * HOP), int(min(len(x), (b - 1) * HOP + WIN))


def annotate_spans(manifest, store, **kw):
    """Fill missing wake spans; return phrase-length stats and the clips over MAX_PHRASE."""
    long = []
    for r in manifest:
        if r["cls"] == "ww" and r["span"] is None:
            r["span"] = word_span(store(r["file"]), **kw)
        if r["cls"] == "ww" and r["span"][1] - r["span"][0] > MAX_PHRASE:
            long.append(r["path"])
    lens = np.array([r["span"][1] - r["span"][0] for r in manifest if r["cls"] == "ww"])
    stats = {} if not len(lens) else {
        "median_s": float(np.median(lens)) / SR,
        "p95_s": float(np.percentile(lens, 95)) / SR,
        "max_s": float(lens.max()) / SR,
        "over_limit": len(long), "limit_s": MAX_PHRASE / SR}
    return stats, long


def crop_range(span):
    """Crop offsets (frames) whose 20160-sample window contains span [a, b).

    Returns None if no window contains the whole phrase: that clip must not be
    used as a positive (a cut "hey Butterworth" looks like a negative).
    """
    a, b = span
    lo = max(0, -(-(b - N_SAMPLES) // HOP))
    hi = min(N_CROPS - 1, a // HOP)
    return (lo, hi) if lo <= hi else None


# ---------------------------------------------------------------- waveform ops
def noise(n, rng, dbfs=-70.0):
    return (rng.standard_normal(n) * 10 ** (dbfs / 20)).astype(np.float32)


def shift_span(span, d, n):
    if span is None:
        return None
    return int(np.clip(span[0] + d, 0, n)), int(np.clip(span[1] + d, 0, n))


def fit_length(x, span, rng, n=CLIP_SAMPLES):
    """Trim/pad to n samples, keeping span inside where possible.

    Pads with low-level noise, never digital zero: log(0 + 1e-6) frames would
    tell the model a clip is synthetic.
    """
    if len(x) > n:
        lo, hi = 0, len(x) - n
        if span is not None:
            lo, hi = max(0, span[1] - n), min(len(x) - n, span[0])
            if lo > hi:
                lo = hi = int(np.clip((span[0] + span[1] - n) // 2, 0, len(x) - n))
        s = int(rng.integers(lo, hi + 1))
        return x[s:s + n], shift_span(span, -s, n)
    if len(x) < n:
        s = int(rng.integers(0, n - len(x) + 1))
        x = np.concatenate([noise(s, rng), x, noise(n - len(x) - s, rng)])
        return x, shift_span(span, s, n)
    return x, span


def speed_perturb(x, span, rate):
    """Resample so the clip plays `rate` x faster (pitch moves too)."""
    up, down = 100, int(round(100 * rate))
    y = signal.resample_poly(x, up, down).astype(np.float32)
    if span is not None:
        span = (span[0] * up // down, span[1] * up // down)
    return y, span


def apply_fir(x, h):
    d = (len(h) - 1) // 2                       # linear phase: keep alignment
    return signal.fftconvolve(x, h)[d:d + len(x)].astype(np.float32)


def ltas_db(clips, nfft=N_FFT):
    """Long-term average spectrum (dB, nfft//2+1 bins) over a list of clips."""
    P = np.mean([signal.welch(c, SR, nperseg=nfft)[1] for c in clips], axis=0)
    return 10 * np.log10(P + 1e-12)


def fit_channel_eq(src_clips, mems_clips, smooth=9, limit_db=20.0):
    """dB curve mapping a source domain's LTAS onto the MEMS mic's.

    Compare like with like (e.g. generated speech vs MEMS speech) so the curve
    captures the channel, not the content. Level is left to the gain aug.
    """
    d = ltas_db(mems_clips) - ltas_db(src_clips)
    p = smooth // 2
    d = np.convolve(np.pad(d, p, mode="edge"), np.ones(smooth) / smooth, "valid")
    return np.clip(d - np.median(d), -limit_db, limit_db)


def make_eq_bank(eq_db, rng, n=32, ntaps=255, alpha=(0.5, 1.2), tilt=3.0):
    """n linear-phase FIRs: alpha * eq_db plus a random +/-tilt dB/octave."""
    f = np.linspace(0.0, 1.0, len(eq_db))
    octs = np.log2(np.maximum(f * SR / 2, 125.0) / 1000.0)
    bank = []
    for _ in range(n):
        g_db = rng.uniform(*alpha) * eq_db + rng.uniform(-tilt, tilt) * octs
        bank.append(signal.firwin2(ntaps, f, 10 ** (g_db / 20)).astype(np.float32))
    return bank


def synth_rir(rng, rt60, drr_db):
    n = int(rt60 * SR)
    t = np.arange(n) / SR
    tail = rng.standard_normal(n) * np.exp(-6.91 * t / rt60)   # -60 dB at rt60
    tail[: int(0.003 * SR)] = 0.0                              # 3 ms pre-delay
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(tail ** 2))
    tail[0] = 1.0                                                # direct path
    return tail.astype(np.float32)


def reverb(x, rng, cfg):
    h = synth_rir(rng, rng.uniform(*cfg.rt60), rng.uniform(*cfg.drr_db))
    y = signal.fftconvolve(x, h)[:len(x)]
    return (y * np.sqrt(np.mean(x ** 2) / (np.mean(y ** 2) + 1e-12))).astype(np.float32)


def mix_at_snr(x, other, snr_db, rng, span=None):
    """x + other scaled to snr_db; signal power measured over span if given."""
    if len(other) < len(x):
        other = np.tile(other, -(-len(x) // len(other)))
    s = int(rng.integers(0, len(other) - len(x) + 1))
    other = other[s:s + len(x)]
    ref = x[span[0]:span[1]] if span is not None and span[1] > span[0] else x
    px, po = np.mean(ref ** 2), np.mean(other ** 2) + 1e-12
    return (x + other * np.sqrt(px / (po * 10 ** (snr_db / 10)))).astype(np.float32)


# ---------------------------------------------------------------- augmenter
class Augmenter:
    """Waveform augmentation for one clip. Returns (24000 samples, span).

    Holds no random state: each call draws only from the rng it is given, so
    an example is fixed by its rng (see example_rng), not by call order.
    """

    def __init__(self, cfg, bg_pool, speech_pool, eq_banks):
        self.cfg = cfg
        self.bg = bg_pool              # int16 (n, CLIP_SAMPLES), train split only
        self.speech = speech_pool      # int16, public + real speech (no hard negatives)
        self.eq = eq_banks             # {domain: [fir, ...]}

    def __call__(self, x, rec, rng):
        c, r = self.cfg, rng
        pick = lambda pool: to_float(pool[r.integers(len(pool))])
        cls, dom, span = rec["cls"], rec["domain"], rec["span"]

        if cls != "bg" and r.random() < c.p_speed:
            rate = r.choice(c.speeds)
            # never slow a wake phrase past the length streaming always sees whole
            if span is None or (span[1] - span[0]) / rate <= MAX_PHRASE:
                x, span = speed_perturb(x, span, rate)
        k = int(r.integers(0, HOP))                      # sub-frame shift
        x, span = np.concatenate([noise(k, r), x]), shift_span(span, k, len(x) + k)
        x, span = fit_length(x, span, r)

        if r.random() < (c.p_eq_mems if dom == "mems" else c.p_eq) and dom in self.eq:
            bank = self.eq[dom]
            x = apply_fir(x, bank[r.integers(len(bank))])
        if cls != "bg" and r.random() < (c.p_reverb_generated if dom == "generated"
                                         else c.p_reverb_other):
            x = reverb(x, r, c)

        if cls in ("ww", "sp") and len(self.bg) and r.random() < c.p_bg:
            snr = c.snr_bg_ww if cls == "ww" else c.snr_bg
            x = mix_at_snr(x, pick(self.bg), r.uniform(*snr), r, span)
        if cls == "ww" and len(self.speech) and r.random() < c.p_babble:
            x = mix_at_snr(x, pick(self.speech), r.uniform(*c.snr_babble), r, span)
        if cls == "bg" and len(self.bg) and r.random() < c.p_bg_bg:
            x = mix_at_snr(x, pick(self.bg), r.uniform(*c.snr_bg_bg), r)

        x = x * 10 ** (r.uniform(*c.gain_db) / 20)
        x = x + noise(len(x), r, r.uniform(*c.noise_floor_dbfs))
        return np.clip(x, -1.0, 1.0).astype(np.float32), span


def spec_augment(mel, rng, cfg, time_masks=True):
    mel = mel.copy()
    fill = mel.mean()
    for _ in range(cfg.freq_masks):
        w = int(rng.integers(0, cfg.max_freq_mask + 1))
        f0 = int(rng.integers(0, N_MELS - w + 1))
        mel[:, f0:f0 + w] = fill
    for _ in range(cfg.time_masks if time_masks else 0):
        w = int(rng.integers(0, cfg.max_time_mask + 1))
        t0 = int(rng.integers(0, len(mel) - w + 1))
        mel[t0:t0 + w] = fill
    return mel


# ---------------------------------------------------------------- examples
def example_rng(seed, *keys):
    """Independent generator for one example, fixed by (seed, *keys).

    Same keys -> same draws, whatever was generated before or in another
    worker, so a batch can be regenerated or resumed on its own.
    """
    return np.random.default_rng([seed, *keys])


def make_example(rec, audio, rng, aug=None):
    """One (124, 40) log-mel crop + label, or None if a wake phrase can't fit.

    aug=None -> eval (no augmentation). All randomness comes from rng.
    """
    ww = rec["cls"] == "ww"
    assert not ww or rec["span"] is not None, f"wake clip without span: {rec['path']}"
    if aug is not None:
        x, span = aug(audio, rec, rng)
    else:
        x, span = fit_length(audio, rec["span"], rng)
    rng_off = crop_range(span) if ww else (0, N_CROPS - 1)
    if rng_off is None:
        return None
    crop = random_crop(logmel(x), rng, *rng_off)       # (148, 40) -> (124, 40)
    if aug is not None and rng.random() < aug.cfg.p_specaug:
        crop = spec_augment(crop, rng, aug.cfg, time_masks=not ww)
    return crop, LABEL[rec["cls"]]


def sample_weights(manifest, share=GROUP_SHARE, phrase_weight=PHRASE_WEIGHT):
    """Per-clip draw probability: batch share per (cls, domain) group, split
    within the group by phrase weight (closest near-misses drawn more)."""
    keys = [(r["cls"], r["domain"]) for r in manifest]
    pw = np.array([phrase_weight.get(r.get("phrase") or "", 1.0) for r in manifest])
    tot = {}
    for k, v in zip(keys, pw):
        tot[k] = tot.get(k, 0.0) + v
    w = np.array([share.get(k, 0.0) * v / tot[k] for k, v in zip(keys, pw)])
    assert w.sum() > 0, "no manifest group has a batch share"
    return w / w.sum()


class BatchMaker:
    """make(b) -> class/domain-balanced training batch b: X (B, 124, 40), y (B,).

    Reproducible: batch b depends only on (seed, b) given the same manifest,
    augmenter and config, so batches can be made in any order or in parallel.
    """

    def __init__(self, manifest, store, aug, batch_size=128, seed=0):
        self.manifest, self.store, self.aug = manifest, store, aug
        self.batch_size, self.seed = batch_size, seed
        self.cdf = np.cumsum(sample_weights(manifest))

    def make(self, b):
        ex = []
        for j in range(self.batch_size):
            rng = example_rng(self.seed, int(b), j)
            e = None
            while e is None:               # redraw if a wake phrase can't fit
                i = min(int(np.searchsorted(self.cdf, rng.random() * self.cdf[-1], side="right")),
                        len(self.manifest) - 1)
                r = self.manifest[i]
                e = make_example(r, self.store(r["file"]), rng, self.aug)
            ex.append(e)
        return (np.stack([e[0] for e in ex]).astype(np.float32),
                np.array([e[1] for e in ex], np.float32))


def batches(manifest, store, aug, batch_size=128, seed=0, start=0):
    """Endless stream of BatchMaker batches start, start+1, ... (start=b resumes at b)."""
    maker = BatchMaker(manifest, store, aug, batch_size, seed)
    b = start
    while True:
        yield maker.make(b)
        b += 1


def eval_set(manifest, store, seed=0):
    """Fixed, unaugmented eval examples: one crop per clip, same on every call.

    Returns X (N, 124, 40), y (N,), and the rows used (wake clips whose phrase
    can't fit a crop are left out, as in training).
    """
    X, y, rows = [], [], []
    for n, r in enumerate(manifest):
        e = make_example(r, store(r["file"]), example_rng(seed, n))
        if e is not None:
            X.append(e[0])
            y.append(e[1])
            rows.append(r)
    return np.stack(X).astype(np.float32), np.array(y, np.float32), rows


def build_augmenter(train, store, cfg=None, seed=0, n_eq_ref=300):
    """Pools and EQ banks from the TRAIN split only (EQ banks fixed by seed)."""
    cfg, rng = cfg or AugConfig(), np.random.default_rng(seed)
    def pool(keep):
        xs = [to_int16(store(r["file"])) for r in train if keep(r)]
        return np.stack(xs) if xs else np.zeros((0, CLIP_SAMPLES), np.int16)

    bg = pool(lambda r: r["cls"] == "bg")
    sp = pool(lambda r: r["cls"] == "sp" and r["domain"] != "generated")

    def clips(domain, cls="sp"):
        rs = [r for r in train if r["domain"] == domain and r["cls"] == cls]
        rs = [rs[i] for i in rng.permutation(len(rs))[:n_eq_ref]]
        return [store(r["file"]) for r in rs]

    mems = clips("mems")
    banks = {"mems": make_eq_bank(np.zeros(N_FFT // 2 + 1), rng, alpha=(0, 0))}
    for dom in ("generated", "phone", "public"):
        src = clips(dom)
        if src and mems:
            banks[dom] = make_eq_bank(fit_channel_eq(src, mems), rng)
    return Augmenter(cfg, bg, sp, banks)


# ---------------------------------------------------------------- export
LABEL_ID = {"bg": 0, "sp": 1, "ww": 2}


def folder_rows(dataset):
    """Every wav in <dataset>/processed/{ww,sp,bg}, as manifest-like rows (no split)."""
    rows = []
    for cls in CLASSES:
        for wav in sorted((dataset / "processed" / cls).glob("*.wav")):
            rows.append({"path": wav.relative_to(dataset.parent).as_posix(), "cls": cls,
                         "split": "", "file": str(wav)})
    return rows


def save_previews(feats, rows, ok, k, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for cls in CLASSES:
        idx = [i for i, r in enumerate(rows) if r["cls"] == cls and ok[i]][:k]
        if not idx:
            continue
        fig, axes = plt.subplots(len(idx), 1, figsize=(8, 1.6 * len(idx)), squeeze=False)
        for ax, i in zip(axes[:, 0], idx):
            ax.imshow(np.asarray(feats[i], np.float32).T, origin="lower", aspect="auto",
                      cmap="magma", extent=(0, CLIP_FRAMES * HOP / SR, 0, N_MELS))
            ax.set_title(Path(rows[i]["path"]).name, fontsize=8)
            ax.set_ylabel("mel")
        axes[-1, 0].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(out / f"preview_{cls}.png", dpi=110)
        plt.close(fig)


def export(dataset, manifest=None, out=None, float16=False, preview=0):
    """Clean (unaugmented) log-mel for every clip -> out/features.npy (N, 148, 40),
    index.csv (row, path, cls, label_id, target, split, ok) and frontend.json."""
    out = Path(out or dataset / "spectrograms")
    rows = read_manifest(manifest, root=dataset.parent) if manifest else folder_rows(dataset)
    if not rows:
        sys.exit(f"no clips found under {dataset / 'processed'} (or in the manifest)")

    out.mkdir(parents=True, exist_ok=True)
    dtype = np.float16 if float16 else np.float32
    feats = np.lib.format.open_memmap(out / "features.npy", mode="w+", dtype=dtype,
                                      shape=(len(rows), CLIP_FRAMES, N_MELS))
    print(f"{len(rows)} clips -> {out} ({feats.nbytes / 1e6:.0f} MB)")

    ok = np.ones(len(rows), bool)
    refit = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        try:
            x = load_wav(r["file"])
            if len(x) != CLIP_SAMPLES:     # trim / pad with -70 dBFS noise, never zeros
                refit += 1
                x, _ = fit_length(x, None, example_rng(0, i))
            feats[i] = logmel(x)
        except Exception as e:
            feats[i] = np.log(LOG_EPS)
            ok[i] = False
            print(f"  {r['path']}: failed ({e})")
        if (i + 1) % 2000 == 0 or i + 1 == len(rows):
            print(f"  {i + 1}/{len(rows)}  {time.time() - t0:.0f} s", flush=True)
    feats.flush()

    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "path", "cls", "label_id", "target", "split", "ok"])
        for i, r in enumerate(rows):
            w.writerow([i, r["path"], r["cls"], LABEL_ID[r["cls"]], int(r["cls"] == "ww"),
                        r["split"], int(ok[i])])
    with open(out / "frontend.json", "w") as f:
        json.dump({"sr": SR, "clip_samples": CLIP_SAMPLES, "win": WIN, "hop": HOP,
                   "n_fft": N_FFT, "window": "periodic hann", "n_mels": N_MELS,
                   "f_lo": FMIN, "f_hi": FMAX, "mel": "htk", "log_eps": LOG_EPS,
                   "shape": [len(rows), CLIP_FRAMES, N_MELS], "dtype": np.dtype(dtype).name},
                  f, indent=2)
    if preview > 0:
        save_previews(feats, rows, ok, preview, out)

    counts = {c: int(sum(r["cls"] == c and ok[i] for i, r in enumerate(rows))) for c in CLASSES}
    print(f"done in {time.time() - t0:.0f} s: " + ", ".join(f"{c} {n}" for c, n in counts.items())
          + (f"; {refit} not 1.5 s (trimmed/padded)" if refit else "")
          + (f"; {int((~ok).sum())} failed (rows marked ok=0)" if not ok.all() else ""))


# ---------------------------------------------------------------- span review
def read_overrides(path):
    """Hand-checked spans: CSV with path, on_s, off_s (seconds) -> {path: (on, off) samples}."""
    if not path or not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {r["path"]: (round(float(r["on_s"]) * SR), round(float(r["off_s"]) * SR))
                for r in csv.DictReader(f)}


def review_sheets(rows, store, out, per_sheet=24, cols=4):
    """Spectrogram grids with the span marked; numbered panels listed in out/index.csv."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.png"):
        old.unlink()
    with open(out / "index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet", "panel", "path", "on_s", "off_s", "len_s", "source"])
        for s in range(0, len(rows), per_sheet):
            chunk = rows[s:s + per_sheet]
            nrow = -(-len(chunk) // cols)
            fig, axes = plt.subplots(nrow, cols, figsize=(4.2 * cols, 1.7 * nrow), squeeze=False)
            for ax in axes.flat[len(chunk):]:
                ax.axis("off")
            sheet = s // per_sheet + 1
            for k, (ax, r) in enumerate(zip(axes.flat, chunk), start=1):
                a, b = r["span"]
                ax.imshow(logmel(store(r["file"])).T, origin="lower", aspect="auto", cmap="magma",
                          extent=(0, CLIP_FRAMES * HOP / SR, 0, N_MELS))
                for t in (a / SR, b / SR):
                    ax.axvline(t, color="cyan", lw=1.5)
                long = b - a > MAX_PHRASE
                ax.set_title(f"{k}. {Path(r['path']).stem}  {(b - a) / SR:.2f}s" + ("  TOO LONG" if long else ""),
                             fontsize=8, color="red" if long else "black")
                ax.set_yticks([])
                w.writerow([sheet, k, r["path"], f"{a / SR:.2f}", f"{b / SR:.2f}", f"{(b - a) / SR:.2f}", r["source"]])
            fig.tight_layout()
            fig.savefig(out / f"sheet_{sheet:02d}.png", dpi=80)
            plt.close(fig)


def spans(manifest, out_manifest, overrides, review):
    """Detect wake-phrase spans, apply hand overrides, write the spans manifest and review sheets."""
    rows = read_manifest(manifest)
    store = AudioStore()
    ov = read_overrides(overrides)
    for r in rows:
        if r["cls"] == "ww":
            r["span"] = ov.get(r["path"]) or word_span(store(r["file"]))
    stats, long = annotate_spans(rows, store)
    write_manifest(rows, out_manifest)
    print(f"wake spans: {stats}")
    print(f"{len(ov)} hand overrides from {overrides}; wrote {out_manifest}")
    if review:
        ww = [r for r in rows if r["cls"] == "ww"]
        check = [r for r in ww if r["domain"] != "generated" or r["path"] in long]
        review_sheets(check, store, review)
        print(f"review: {len(check)} clips (all real recordings + any over {MAX_PHRASE / SR:.2f} s) "
              f"-> {review}/sheet_*.png, panel list in index.csv")
    for p in long:
        print(f"  over limit, not used as a positive: {p}")


# ---------------------------------------------------------------- self-test

def selftest():
    """Pipeline smoke test on fake clips (no dataset needed)."""
    import os
    import tempfile

    rng = np.random.default_rng(0)
    n = CLIP_SAMPLES
    tmp = tempfile.mkdtemp()

    def fake(cls, dom, plen=14000):
        x = noise(n, rng, -60)
        if dom == "generated":
            x[:] = 0.0                                      # TTS digital silence
        if cls != "bg":
            a = int(rng.integers(0, n - plen))
            t = np.arange(plen) / SR
            env = np.minimum(1.0, np.minimum(t, t[::-1]) / 0.05)      # flat, 50 ms ramps
            x[a:a + plen] += 0.3 * np.sin(2 * np.pi * 300 * t) * env
        else:
            x += noise(n, rng, -30)
        return x

    # fake manifest_v1 with real WAV files
    name = {"ww": "wake", "sp": "speech", "bg": "background"}
    lid = {"bg": 0, "sp": 1, "ww": 2}
    rows = []
    combos = [(c, d) for c in CLASSES for d in ("generated", "mems", "phone", "public")]
    for i, (cls, dom) in enumerate(combos * 5):
        plen = {0: 20000, 1: 23900}.get(i, 14000) if cls == "ww" else 14000
        p = f"processed/{cls}/{cls}_{dom}_x_{i:04d}.wav"
        os.makedirs(os.path.dirname(f"{tmp}/{p}"), exist_ok=True)
        wavfile.write(f"{tmp}/{p}", SR, to_int16(fake(cls, dom, plen)))
        phrase = ("hey.butterworth" if cls == "ww" else "hey.butter") if dom == "generated" else ""
        rows.append(dict(path=p, label=name[cls], label_id=lid[cls], source=f"{dom}_x",
                         speaker="s", domain=dom, phrase=phrase,
                         split="val" if i % 4 == 3 else "train"))
    with open(f"{tmp}/manifest_v1.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    store = AudioStore()
    man = read_manifest(f"{tmp}/manifest_v1.csv", root=tmp)
    stats, long = annotate_spans(man, store)
    assert len(long) == 2, long                    # the 20000- and 23900-sample phrases
    write_manifest(man, f"{tmp}/manifest_v1_spans.csv")
    man = read_manifest(f"{tmp}/manifest_v1_spans.csv", root=tmp)
    assert all(r["span"] for r in man if r["cls"] == "ww")
    parts = split(man)

    aug = build_augmenter(parts["train"], store, n_eq_ref=5)
    for r in parts["train"]:
        x, span = aug(store(r["file"]), r, rng)
        assert x.shape == (n,) and np.isfinite(x).all() and np.abs(x).max() <= 1
        if r["cls"] == "ww" and crop_range(span):
            lo, hi = crop_range(span)
            for o in (lo, hi):          # every allowed crop keeps the phrase whole
                assert o * HOP <= span[0] and span[1] <= o * HOP + N_SAMPLES
    too_long = next(r for r in man if r["cls"] == "ww" and r["span"][1] - r["span"][0] > N_SAMPLES)
    assert make_example(too_long, store(too_long["file"]), rng, aug) is None

    X, y = next(batches(parts["train"], store, aug, batch_size=64))
    assert X.shape == (64, N_FRAMES, N_MELS) and np.isfinite(X).all()
    assert X.min() > np.log(LOG_EPS) + 0.05, "digital-silence frames leaked through"
    v = parts["val"][0]
    Xe, _ = make_example(v, store(v["file"]), rng)

    # reproducibility: same seed -> same batches, fresh augmenter or not;
    # start=b resumes at batch b; a different seed gives different batches
    def first(k, seed=0, start=0, a=aug):
        g = batches(parts["train"], store, a, batch_size=16, seed=seed, start=start)
        return [next(g) for _ in range(k)]
    run1, run2 = first(3), first(3, a=build_augmenter(parts["train"], store, n_eq_ref=5))
    assert all(np.array_equal(p[0], q[0]) and np.array_equal(p[1], q[1]) for p, q in zip(run1, run2))
    assert np.array_equal(first(1, start=2)[0][0], run1[2][0])
    assert not np.array_equal(first(1, seed=1)[0][0], run1[0][0])
    E1, E2 = eval_set(parts["val"], store), eval_set(parts["val"], store)
    assert np.array_equal(E1[0], E2[0]) and len(E1[2]) <= len(parts["val"])

    print(f"spans: {stats}")
    print(f"batch {X.shape}, positives {int(y.sum())}/64, eval crop {Xe.shape}, "
          f"eval set {E1[0].shape}, reproducible  OK")


# ---------------------------------------------------------------- command line
def main():
    ap = argparse.ArgumentParser(description="Butterworth features: spectrogram export and pipeline self-test.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="clean (unaugmented) log-mel for every clip -> dataset/spectrograms/")
    ex.add_argument("--dataset", type=Path, default=ROOT / "dataset",
                    help="dataset root holding processed/ (default: dataset/)")
    ex.add_argument("--manifest", type=Path,
                    help="manifest CSV; paths relative to the folder above --dataset (the repo root). "
                         "Default: every wav in processed/ww, sp, bg")
    ex.add_argument("--out", type=Path, help="output folder (default: <dataset>/spectrograms)")
    ex.add_argument("--float16", action="store_true", help="store features as float16 (half the disk)")
    ex.add_argument("--preview", type=int, default=0, metavar="K",
                    help="also save K example spectrogram images per class (needs matplotlib)")
    sp = sub.add_parser("spans", help="wake-phrase spans -> manifest with on/off columns + review sheets")
    sp.add_argument("--manifest", type=Path, default=ROOT / "dataset" / "manifest_v1.csv")
    sp.add_argument("--out", type=Path, default=ROOT / "dataset" / "manifest_v1_spans.csv")
    sp.add_argument("--overrides", type=Path, default=ROOT / "dataset" / "span_overrides.csv",
                    help="hand-checked spans: path, on_s, off_s (seconds); used instead of detection")
    sp.add_argument("--review", type=Path, default=ROOT / "dataset" / "span_review",
                    help="folder for review sheets (needs matplotlib)")
    sp.add_argument("--no-review", action="store_true", help="skip the review sheets")
    sub.add_parser("selftest", help="pipeline smoke test on fake clips")
    args = ap.parse_args()

    if args.cmd == "export":
        export(args.dataset.resolve(), args.manifest, args.out, args.float16, args.preview)
    elif args.cmd == "spans":
        spans(args.manifest, args.out, args.overrides, None if args.no_review else args.review)
    else:
        selftest()


if __name__ == "__main__":
    main()
