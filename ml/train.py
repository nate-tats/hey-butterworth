"""Butterworth CNN: the network, its exact streaming version, and training.

Input features (log-mel, augmentation, batches) come from features.py.
Design and rationale: Butterworth.md, section 2.

NETWORK (all convs VALID padding, BN + ReLU after each)
  layer  k  stride  out.spacing  RF(frames)  T_in -> T_out
    1    4    3          3            4       124 -> 41
    2    3    2          6           10        41 -> 20
    3    2    2         12           16        20 -> 10
    4    2    2         24           28        10 -> 5
    5    3    1         24           76         5 -> 3
    6    3    1         24          124         3 -> 1
  Head: Flatten (1 x C) -> Dense(1, sigmoid) = P(Butterworth)

  out.spacing = product of strides so far; RF = 1 + sum_i (k_i - 1) * J_i
  where J_i is layer i's input spacing (1, 3, 6, 12, 24, 24). The input is
  exactly one receptive field, so the network emits exactly one timestep.
  With valid padding every activation depends only on real audio, so it is
  identical whichever window it falls in. That is what makes streaming reuse
  exact.

COST (C=64)
  Full 124-frame window:            ~0.84 M MACs, ~63.5 K weights
  Streaming, per 24-frame hop:      ~0.18 M MACs  (one new decision / 240 ms)
  Streaming state (BN folded):      <= sum_i (k_i - 1) * C_in_i floats
                                    = 120 + 128 + 64 + 64 + 128 + 128 = 632
STREAMING TIMELINE
  Audio arrives in 3840-sample (240 ms) chunks -> 24 new mel frames per chunk.
  First decision after 20160 samples (1.26 s); then one every 3840 samples.
  Decision j covers mel frames [24j, 24j + 124), i.e. exactly what the Keras
  model would see if you cut that window out and ran it.

TRAINING
  python ml/train.py                         # defaults: 40 epochs x 200 batches of 128
  python ml/train.py --epochs 2 --steps 20   # quick check
  Batches are features.BatchMaker batches 0, 1, 2, ... (augmented, class/domain
  balanced, train split only); val is features.eval_set (fixed, unaugmented).
  Same --seed -> same batches, same weight init, same result.
  Writes runs/<time>/: best.keras, history.csv, val_report.json, config.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402

# ---------------------------------------------------------------- architecture
KERNELS = [4, 3, 2, 2, 3, 3]
STRIDES = [3, 2, 2, 2, 1, 1]
TOTAL_STRIDE = int(np.prod(STRIDES))                     # 24 frames = 240 ms
RF = 1 + sum((k - 1) * int(np.prod(STRIDES[:i]))
             for i, k in enumerate(KERNELS))              # 124 frames
assert RF == F.N_FRAMES and TOTAL_STRIDE == F.DECISION_HOP, \
    "CNN receptive field / stride must match features.N_FRAMES / DECISION_HOP"


# ---------------------------------------------------------------- Keras model
def build_model(C=64):
    from tensorflow import keras
    from tensorflow.keras import layers

    x = inp = keras.Input(shape=(F.N_FRAMES, F.N_MELS), name="logmel")
    for i, (k, s) in enumerate(zip(KERNELS, STRIDES), start=1):
        x = layers.Conv1D(C, k, strides=s, padding="valid", use_bias=False,
                          name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.ReLU(name=f"relu{i}")(x)
    x = layers.Flatten(name="feat")(x)                    # (1, C) -> (C,)
    out = layers.Dense(1, activation="sigmoid", name="p_butterworth")(x)
    return keras.Model(inp, out, name="butterworth_tcnn")


# ---------------------------------------------------------------- streaming
class _StreamConv:
    """Valid, strided 1-D conv over an unbounded stream, BN folded in, + ReLU.

    Holds only input frames that a future output still needs (< k of them in
    steady state). On a device this is a ring buffer of k-1 (+burst) frames.
    """

    def __init__(self, W, b, stride):
        self.W = W.reshape(-1, W.shape[2])               # (k*Cin, Cout)
        self.b = b
        self.k, self.cin = W.shape[0], W.shape[1]
        self.s = stride
        self.buf = np.zeros((0, self.cin), np.float32)

    def push(self, x):
        self.buf = np.concatenate([self.buf, x])
        n = 0 if len(self.buf) < self.k else 1 + (len(self.buf) - self.k) // self.s
        if n == 0:
            return np.zeros((0, len(self.b)), np.float32)
        idx = np.arange(self.k)[None, :] + self.s * np.arange(n)[:, None]
        y = self.buf[idx].reshape(n, -1) @ self.W + self.b
        self.buf = self.buf[n * self.s:]
        return np.maximum(y, 0.0)

    def macs_per_output(self):
        return self.W.size


class StreamingButterworth:
    """Incremental inference that exactly matches build_model() on each window.

    push(mel_frames) returns an array of logits, one per new final-layer frame
    (normally one per 24 new mel frames). sigmoid(logit) = P(Butterworth).
    """

    def __init__(self, model):
        self.layers = []
        for i, s in enumerate(STRIDES, start=1):
            W = model.get_layer(f"conv{i}").get_weights()[0]
            gamma, beta, mean, var = model.get_layer(f"bn{i}").get_weights()
            eps = model.get_layer(f"bn{i}").epsilon
            scale = gamma / np.sqrt(var + eps)
            self.layers.append(_StreamConv((W * scale).astype(np.float32),
                                           (beta - mean * scale).astype(np.float32), s))
        Wd, bd = model.get_layer("p_butterworth").get_weights()
        self.wd, self.bd = Wd[:, 0], bd[0]

    def push(self, mel):
        x = np.asarray(mel, np.float32)
        for layer in self.layers:
            x = layer.push(x)
        return (x @ self.wd + self.bd).astype(np.float32)

    def macs_per_hop(self):
        total, cum = 0, 1
        for layer in self.layers:
            cum *= layer.s
            total += (TOTAL_STRIDE // cum) * layer.macs_per_output()
        return total + len(self.wd)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------- training
def make_dataset(maker, start, workers):
    """tf.data stream of maker.make(start), make(start + 1), ... in order."""
    import tensorflow as tf
    B = maker.batch_size

    def load(b):
        x, y = tf.numpy_function(maker.make, [b], [tf.float32, tf.float32])
        return tf.ensure_shape(x, (B, F.N_FRAMES, F.N_MELS)), tf.ensure_shape(y, (B,))

    ds = tf.data.Dataset.range(start, 2 ** 62)
    return ds.map(load, num_parallel_calls=workers, deterministic=True).prefetch(2)


def val_report(p, y, rows, threshold=0.5):
    """Per-group error rates on val at one threshold (wake: misses, others: false accepts)."""
    def rate(mask, wrong):
        n = int(mask.sum())
        return {"n": n, "rate": float(wrong[mask].mean()) if n else None}
    fire = p >= threshold
    cls = np.array([r["cls"] for r in rows])
    dom = np.array([r["domain"] for r in rows])
    phrase = np.array([r.get("phrase") or "" for r in rows])
    spk = np.array([r.get("speaker") or "" for r in rows])
    rep = {"threshold": threshold, "false_reject": {}, "false_reject_by_speaker": {}, "false_accept": {},
           "false_accept_by_phrase": {}}
    for d in sorted(set(dom[cls == "ww"])):
        rep["false_reject"][d] = rate((cls == "ww") & (dom == d), ~fire)
    for s in sorted(set(spk[(cls == "ww") & (dom == "mems")])):
        rep["false_reject_by_speaker"][s] = rate((cls == "ww") & (dom == "mems") & (spk == s), ~fire)
    for c in ("sp", "bg"):
        for d in sorted(set(dom[cls == c])):
            rep["false_accept"][f"{c}/{d}"] = rate((cls == c) & (dom == d), fire)
    for ph in sorted(set(phrase[(cls == "sp") & (phrase != "")])):
        rep["false_accept_by_phrase"][ph] = rate((cls == "sp") & (phrase == ph), fire)
    return rep


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=F.ROOT,
                              capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def train(args):
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(args.seed)              # weight init, Python, numpy, TF
    tf.config.experimental.enable_op_determinism()
    out = Path(args.out or F.ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    man = F.read_manifest(args.manifest, root=args.root)
    store = F.AudioStore()
    stats, long = F.annotate_spans(man, store)
    print(f"wake phrase spans: {stats}")
    if long:
        print(f"  {len(long)} wake clips longer than {F.MAX_PHRASE / F.SR:.2f} s are skipped as positives")
    parts = F.split(man)
    print(f"train {len(parts['train'])} clips, val {len(parts['val'])} clips")

    aug = F.build_augmenter(parts["train"], store, seed=args.seed)
    Xv, yv, vrows = F.eval_set(parts["val"], store, seed=args.seed)
    maker = F.BatchMaker(parts["train"], store, aug, args.batch, args.seed)
    ds = make_dataset(maker, 0, args.workers)

    model = build_model(args.channels)
    model.compile(optimizer=keras.optimizers.Adam(args.lr), loss="binary_crossentropy",
                  metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="acc")])
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "out": str(out), "git_commit": git_commit(), "spans": stats,
              "n_train": len(parts["train"]), "n_val": len(vrows), "params": model.count_params()}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    model.fit(ds, steps_per_epoch=args.steps, epochs=args.epochs, validation_data=(Xv, yv),
              shuffle=False, verbose=2, callbacks=[
                  keras.callbacks.ModelCheckpoint(out / "best.keras", monitor="val_auc", mode="max",
                                                  save_best_only=True),
                  keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5,
                                                    patience=3),
                  keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=args.patience,
                                                restore_best_weights=True),
                  keras.callbacks.CSVLogger(out / "history.csv")])

    p = model.predict(Xv, batch_size=512, verbose=0)[:, 0]
    rep = val_report(p, yv, vrows)
    (out / "val_report.json").write_text(json.dumps(rep, indent=2))
    print(f"\nval at threshold {rep['threshold']}:")
    for k, v in rep["false_reject"].items():
        print(f"  false reject  ww/{k:<14} {v['rate']:.3f}  (n={v['n']})")
    for k, v in rep["false_reject_by_speaker"].items():
        print(f"  false reject  ww/mems/{k:<9} {v['rate']:.3f}  (n={v['n']})")
    for k, v in rep["false_accept"].items():
        print(f"  false accept  {k:<17} {v['rate']:.3f}  (n={v['n']})")
    print(f"saved {out}")
    return model, out


def main():
    ap = argparse.ArgumentParser(description="Train the Butterworth wake-word CNN.")
    ap.add_argument("--manifest", type=Path, default=F.default_manifest(),
                    help=f"default: manifest_{F.MANIFEST_VERSION}_spans.csv (checked wake spans) if present, "
                         f"else manifest_{F.MANIFEST_VERSION}.csv")
    ap.add_argument("--root", type=Path, default=F.ROOT, help="folder the manifest paths start from (repo root)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=64, help="conv channels C")
    ap.add_argument("--patience", type=int, default=8, help="early stopping on val AUC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="batches built in parallel")
    ap.add_argument("--out", type=Path, help="run folder (default: runs/<time>)")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
