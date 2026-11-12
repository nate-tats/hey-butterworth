# ml

Two files, one per stage, plus a test and an analysis script. Design and rationale: [`Butterworth.md`](../Butterworth.md).
Run from anywhere (`python ml/train.py`); manifest paths are resolved from the repo root.

| File | Stage | What's in it |
|---|---|---|
| `features.py` | **Spectrogram conversion + augmentation** | Front end (`logmel`: 16 kHz audio -> 40 log-mel bins every 10 ms, `StreamingFrontend`), manifest I/O, wake-phrase spans, waveform augmentation toward the ESP32 MEMS mic, SpecAugment, training batches (`BatchMaker`, `batches`), fixed val set (`eval_set`), spectrogram export |
| `train.py` | **CNN + training** | 1D temporal CNN (`build_model`, Keras), exact streaming inference (`StreamingButterworth`), training loop and val report |
| `test_streaming.py` | test | Streaming front end and CNN match the batch/Keras versions exactly |
| `analyze_run.py` | analysis | Loss/accuracy/AUC curves (train vs val) and a text summary for a run |

`train.py` imports `features.py`, never the other way round. `train.py` also checks that the CNN's receptive field
(124 frames) and stride (24 frames) match the window `features.py` cuts, so the two can't drift apart.

## Commands

```
python ml/features.py selftest                     # pipeline smoke test on fake clips
python ml/features.py export --preview 4           # clean spectrograms -> dataset/spectrograms/
python ml/features.py spans                        # wake-phrase spans -> dataset/manifest_v2_spans.csv + review sheets
python ml/train.py                                 # train: 40 epochs x 200 batches of 128 -> runs/<time>/
python ml/train.py --epochs 2 --steps 20           # quick check
python ml/test_streaming.py                        # needs TensorFlow
python ml/analyze_run.py [runs/<time> ...]         # curves -> runs/<time>/training_curves.png (default: latest run)
```

## Training (train.py)

Batches are augmented, class/domain balanced, and drawn from the train split only; val is a fixed, unaugmented crop
per val clip. Wake-phrase spans come from `manifest_v2_spans.csv` (see below), and wake clips longer than 1.03 s
are skipped as positives. Checkpoints keep the best val AUC; the learning rate halves after 3 epochs without
improvement, and training stops after `--patience` (8) epochs without improvement.

Each run writes `runs/<time>/` (ignored by git):

| File | Contents |
|---|---|
| `best.keras` | model with the best val AUC |
| `history.csv` | loss, AUC, accuracy per epoch (train and val) |
| `val_report.json` | at threshold 0.5: false-reject rate per wake domain and per real (mems) speaker, false-accept rate per class/domain and per generated phrase |
| `config.json` | all arguments, git commit, span stats, clip counts, parameter count |
| `training_curves.png` | written by `analyze_run.py`: loss, accuracy, AUC per epoch; best-val-AUC epoch and LR halvings marked |

| Flag | Default | Description |
|---|---|---|
| `--manifest` | `manifest_v2_spans.csv` if present, else `manifest_v2.csv` (`MANIFEST_VERSION` in `features.py`) | Manifest to train from |
| `--epochs` / `--steps` / `--batch` | 40 / 200 / 128 | Epochs, batches per epoch, batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--channels` | `64` | Conv channels C (64 -> ~65 K parameters) |
| `--patience` | `8` | Early stopping on val AUC |
| `--seed` | `0` | Seeds batches, augmentation EQ banks, val crops and weight init |
| `--workers` | `4` | Batches built in parallel (12 is ~20% faster on this machine; results are identical) |
| `--out` | `runs/<time>` | Run folder |

Memory: every clip is cached once as int16 (~1.3 GB for 27k clips) plus the mixing pools (~0.8 GB).
Still to do (doc section 7): threshold choice by false accepts per hour with `StreamingButterworth`.

## Results

### full-v2-20261111-1544 (manifest v2: +81 darrell and +4 seth wake clips in train)

`python ml/train.py --workers 12 --manifest dataset/manifest_v2_spans.csv`, same settings as the first full run.
Early stopping ended it at epoch 34; `best.keras` is epoch 26 (val AUC 0.9996). 26 wake clips are over the 1.03 s
limit (15 of them darrell, who speaks slowly) and are not used as positives.

Both models scored on the same fixed v2 val set (`compare.json` in the run folder). The val crops of the `nat`
clips shift slightly with the extra rows, so the old model's numbers here differ a little from its own run report.

| v2 val | full-20261109-1509 (v1) | full-v2-20261111-1544 |
|---|---|---|
| AUC, all val | 0.9968 | 0.9996 |
| AUC, real wake vs all negatives | 0.9902 | 0.9991 |
| **Real wake missed, threshold set for 0.5% false accepts** | **10.0%** (t=0.58) | **3.3%** (t=0.89) |
| Real wake missed, threshold set for 0.1% false accepts | 35.6% (t=0.92) | 16.7% (t=0.99) |
| At 0.5: real wake missed (n=90) | 8.9% | 1.1% |
| At 0.5: "hi butterworth" false accepts (n=300) | 4.7% | 16.7% |
| At 0.5: sp/mems false accepts (n=156) | 0% | 4.5% |

On the v1 val subset alone the picture is the same (real wake missed at 0.5% false accepts: 13.1% -> 4.9%).
The new model separates real wake clips from everything else much better, but its scores sit higher overall, so 0.5 is
now too low a threshold: "hi butterworth" is the main false accept. Pick the threshold by false accepts per hour, not 0.5.
This is one seed per model, and the real val set is 90 clips from mostly one voice, so treat the differences as
indicative.

### full-20261109-1509 (first full run)

`python ml/train.py --workers 12`: defaults (40 epochs x 200 batches of 128, lr 1e-3, C=64, seed 0) on
`manifest_v1_spans.csv` (spans after review: median 0.62 s, 11 wake clips over the 1.03 s limit), commit `9ee6f77`
plus the uncommitted span files. CPU only (16 cores, WSL2), about 45 minutes.

- Early stopping ended it at epoch 38 of 40. `best.keras` is epoch 30: **val AUC 0.9942**, val accuracy 0.992,
  val loss 0.028 (train: AUC 0.9991, loss 0.024).
- Val metrics swung widely until about epoch 20 and settled as the learning rate was halved (epochs 12, 20, 26, 34, 37).
  Train and val loss end close together: no sign of overfitting.

Val report at threshold 0.5, against the 2-epoch `test-run` (before spans):

| | full run | test-run |
|---|---|---|
| False reject, ww/generated (n=200) | 1.0% | 4.5% |
| **False reject, ww/mems: real voices on the ESP32 mic (n=61)** | **18.0%** | 50% (n=2) |
| False accept, sp/generated: near-miss phrases (n=950) | 1.7% | 29.3% |
| False accept, sp/public (n=2297) | 0.6% | 15.3% |
| False accept, sp/mems and all bg | 0% | up to 5.8% |
| Worst phrase: "hi butterworth" (n=300) | 4.7% | 58.3% |

The weak point is real wake clips: 18% are missed at threshold 0.5, against 1% of generated ones, so the model doesn't
yet carry over well to the real mic and voices. False accepts are close to zero everywhere, so there is room to lower
the threshold; choose it by false accepts per hour with `StreamingButterworth` (above). More wake recordings on
the ESP32 mic would help most, since only about 170 of the 2,372 wake clips are real.

## Wake-phrase spans (features.py spans)

Training needs to know where "hey Butterworth" sits in each wake clip, so a crop never cuts it. `spans` finds the
phrase in every wake clip, writes `dataset/manifest_v2_spans.csv` (`--manifest`, default manifest_v2, plus `on`/`off` columns, in samples),
and draws review sheets of every real recording (plus any clip over the 1.03 s limit) into `dataset/span_review/`.

- Detection uses energy in the 125-7500 Hz speech band (the MEMS clips have rumble below 125 Hz), relative to each
  clip's noise floor; clicks shorter than 50 ms are ignored.
- **span_review/** (not tracked): `sheet_NN.png` spectrograms with the detected start/end as cyan lines, title in red if
  the phrase is over the limit; `index.csv` maps sheet + panel number to the file and span.
- Fix a wrong span by adding a row to `dataset/span_overrides.csv` (`path,on_s,off_s,note`, seconds), then rerun
  `spans`. Overrides win over detection.
- Clips over 1.03 s are not used as positives (no streaming window would contain the whole phrase).
- `train.py` uses `manifest_v2_spans.csv` by default when it exists. `manifest_v1_spans.csv` is kept for the v1 runs;
  its rows are identical in v2.

## Reproducibility

Every training example draws from its own generator, `features.example_rng(seed, batch, slot)`, so batch `b` is the
same on every run, in any worker and in any order (`BatchMaker.make(b)`; `batches(..., start=b)` resumes at `b`).
`build_augmenter(seed)` fixes the EQ banks and `eval_set(seed)` the val crops. `train.py` also seeds TensorFlow and
turns on deterministic ops, so the same `--seed` gives identical weights (checked). The split itself is fixed by
`SEED` in `helpers/build_manifest.py`.

## Spectrogram export (features.py export)

Converts clips to (148, 40) log-mel spectrograms with `logmel()`, the same code training uses, without augmentation:
for baselines, evaluation and inspection.

```
python ml/features.py export
python ml/features.py export --manifest dataset/manifest_v1.csv --float16 --preview 4
```

Writes to `dataset/spectrograms/`: `features.npy` (N, 148, 40), `index.csv` (row, path, cls, label_id, target, split, ok)
and `frontend.json`. Load with `np.load("features.npy", mmap_mode="r")`. Clips that aren't 1.5 s are trimmed or padded
with -70 dBFS noise; files that fail to load are reported and marked `ok=0`. About 2 minutes and 640 MB (320 MB with
`--float16`) for the full 27k-clip dataset.

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `dataset` | Dataset root holding `processed/` |
| `--manifest` | none | Manifest CSV, paths relative to the repo root. Default: all wavs in `processed/` |
| `--out` | `<dataset>/spectrograms` | Output folder |
| `--float16` | off | Store features as float16 |
| `--preview` | `0` | Save K example spectrogram images per class |
