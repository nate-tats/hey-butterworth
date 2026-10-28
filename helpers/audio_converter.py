import argparse
import hashlib
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 16000  # for the mic
LABELS = ("ww", "sp", "bg")  # same tags as dataset_recording.py
CLIP_SEC = 1.5
CLIP_LEN = int(SR * CLIP_SEC)
ROOT = Path(__file__).parent.parent / "dataset"
RECORDINGS, LONG_CLIPS = ROOT / "raw" / "recordings", ROOT / "raw" / "long_clips"
PROCESSED = ROOT / "processed"


def load(path):
    audio, _ = librosa.load(path, sr=SR, mono=True)  # resample + mono
    return audio


def to_int16(audio):
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)


def loudest_window(audio, n):
    """Start index of the n-sample window with the most energy."""
    if len(audio) <= n:
        return 0
    c = np.concatenate(([0.0], np.cumsum(audio.astype(np.float64) ** 2)))
    return int(np.argmax(c[n:] - c[:-n]))


def fit_to_length(audio, n):
    """Loudest n samples, zero-padded if the clip is shorter."""
    start = loudest_window(audio, n)
    clip = audio[start:start + n]
    return np.pad(clip, (0, n - len(clip)))


def chop(audio, n, hop):
    """Fixed-size clips every `hop` samples; the leftover tail is dropped."""
    return [audio[s:s + n] for s in range(0, len(audio) - n + 1, hop)]


def digest(pcm):
    return hashlib.sha1(pcm.tobytes()).hexdigest()


class Store:
    """One processed/<label> folder: content hashes of what's there and the next free index per prefix."""

    def __init__(self, folder):
        self.folder = folder
        self.hashes = set()
        self.next_idx = {}
        for f in sorted(folder.glob("*.wav")):
            pcm, _ = sf.read(f, dtype="int16")
            self.hashes.add(digest(pcm))
            m = re.fullmatch(r"(.+)_(\d+)", f.stem)
            if m:
                p, i = m.group(1), int(m.group(2))
                self.next_idx[p] = max(self.next_idx.get(p, 0), i + 1)

    def add(self, prefix, pcm):
        """Returns the new file name, or None if the content is a duplicate."""
        h = digest(pcm)
        if h in self.hashes:
            return None
        self.hashes.add(h)
        i = self.next_idx.get(prefix, 0)
        self.next_idx[prefix] = i + 1
        name = f"{prefix}_{i:04d}.wav"
        self.folder.mkdir(parents=True, exist_ok=True)
        sf.write(self.folder / name, pcm, SR, subtype="PCM_16")
        return name


def label_of(wav):
    """ww / sp / bg from the filename prefix, or None."""
    m = re.match(rf"({'|'.join(LABELS)})_", wav.stem)
    return m.group(1) if m else None


def save(clips, prefix, store):
    saved = dupes = 0
    for clip in clips:
        if store.add(prefix, to_int16(clip)):
            saved += 1
        else:
            dupes += 1
    return saved, dupes


def run(folder, make_clips, keep_raw):
    """Process every wav in folder; the label comes from its ww_/sp_/bg_ prefix."""
    wavs = sorted(folder.glob("*.wav"))
    print(f"{len(wavs)} files in {folder.relative_to(ROOT.parent)}")
    stores = {}
    for wav in wavs:
        label = label_of(wav)
        if label is None:
            print(f"  {wav.name}: no ww_/sp_/bg_ prefix, skipped")
            continue
        try:
            clips = make_clips(load(wav))
            if clips is None:
                print(f"  {wav.name}: shorter than {CLIP_SEC}s, skipped")
                continue
            store = stores.setdefault(label, Store(PROCESSED / label))
            # bg_mems_viroom_0086 -> bg_mems_viroom; the clips get fresh numbers
            prefix = re.sub(r"_\d+$", "", wav.stem)
            saved, dupes = save(clips, prefix, store)
        except Exception as e:
            print(f"  {wav.name}: failed ({e}), raw kept")
            continue
        print(f"  {wav.name} -> processed/{label}: {saved} saved, {dupes} duplicate")
        if not keep_raw:
            wav.unlink()


def main():
    ap = argparse.ArgumentParser(description="Convert raw audio to 16 kHz mono 16-bit 1.5 s clips in dataset/processed/<label>.\n"
                                 "raw/recordings: one clip each (loudest 1.5 s), raw deleted.\n"
                                 "raw/long_clips: chopped into 1.5 s clips, raw deleted unless --keep-raw.",
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--window", type=float, default=CLIP_SEC,
                    help="long clips only: seconds between clip starts, e.g. 0.5 gives overlapping clips (default 1.5)")
    ap.add_argument("--keep-raw", action="store_true", help="long clips only: don't delete them after chopping")
    args = ap.parse_args()
    if args.window <= 0:
        ap.error("--window must be positive")

    run(RECORDINGS, lambda a: [fit_to_length(a, CLIP_LEN)], keep_raw=False)
    run(LONG_CLIPS, lambda a: chop(a, CLIP_LEN, int(SR * args.window)) if len(a) >= CLIP_LEN else None, args.keep_raw)


if __name__ == "__main__":
    main()
