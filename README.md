# hey-butterworth

Embedded voice assistant project: an ESP32 with an I2S MEMS mic that listens for the wake phrase "hey Butterworth".
Design doc: [`Butterworth.md`](Butterworth.md).

```
Butterworth.md  design doc
docs/           Word original of the design doc
dataset/        README + manifest_v1.csv (tracked); audio in processed/ and raw/ (download, not tracked)
helpers/        dataset tools: record from the ESP32, convert, generate Piper speech, build the manifest, report
ml/             features.py (spectrograms + augmentation), train.py (CNN + training), streaming test
firmware/       ESP32 sketches (recording, I2S probes) and a serial test script
```

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Download the dataset (link in `dataset/README.md`) and unzip it so the clips land in `dataset/processed/{ww,sp,bg}/`.

## Pipeline

1. Record / generate audio: `helpers/dataset_recording.py`, `helpers/generate_piper.py` -> `dataset/raw/`
2. Convert to 1.5 s, 16 kHz clips: `helpers/audio_converter.py` -> `dataset/processed/<label>/`
3. Split: `helpers/build_manifest.py --write` -> `dataset/manifest_v1.csv`
4. Wake-phrase spans: `python ml/features.py spans` -> `dataset/manifest_v1_spans.csv` (check `dataset/span_review/`)
5. Spectrograms + augmentation: `ml/features.py`; CNN + training: `python ml/train.py` -> `runs/`; see `ml/README.md`
