# helpers

Five scripts for building and checking the dataset. They work on the `dataset/` folder at the repo root:

```
dataset/
  raw/recordings/      short recordings straight from the mic (all labels together)
  raw/long_clips/      long files you add by hand, for chopping
  raw/generated/       Piper-generated clips (zero-padded to 1.5 s, no speech check)
  processed/<label>/   16 kHz, mono, 16-bit, 1.5 s clips, sorted by label
  manifest_v1.csv      every processed clip with its label and train/val split
```

See `dataset/README.md` for the filename pattern, manifest columns and how the split works.

Labels: `ww` (wake word), `sp` (other speech), `bg` (background).

Requires `pyserial` (recording), `librosa`, `soundfile`, `numpy`, `webrtcvad` (converting; `pip install webrtcvad-wheels` if `webrtcvad` won't build), `ffmpeg` for mp3/m4a input, and [piper-sample-generator](https://github.com/rhasspy/piper-sample-generator) in `~/piper-sample-generator` (generating).

## dataset_recording.py

Records a batch of fixed-length clips from the ESP32 mic over serial into `dataset/raw/recordings/`.
Files are named `<label>_mems[_<note>]_NNNN.wav`, continuing from the highest existing number.

```
python helpers/dataset_recording.py --num 20 --label ww
python helpers/dataset_recording.py --num 50 --length 2 --label bg --note kitchen
```

| Flag | Default | Description |
|---|---|---|
| `--num` | required | How many recordings |
| `--label` | required | `ww`, `sp` or `bg` |
| `--length` | `1.5` | Seconds per recording, 0.1 to 10 in 0.1 steps |
| `--note` | none | Describe the setting of recording, added to the filename (letters, digits, hyphens) |
| `--gap` | `0.0` | Countdown seconds before each recording |
| `--port` | auto | Serial port, e.g. `/dev/cu.usbserial-0001` |
| `--out` | `dataset/raw/recordings` | Folder to save clips in |

## generate_piper.py

Generates speech clips with Piper (the LibriTTS-R multi-speaker model in `~/piper-sample-generator/models/`, speakers capped at 200)
into `dataset/raw/generated/`. It runs Piper through `~/piper-sample-generator/.venv`, so any Python can start it.
Files are named `<label>_generated[_<note>]_NNNN.wav`, continuing from the highest existing number for that prefix
(checked in `raw/generated` and `processed/<label>`). Clips are 22.05 kHz; run `audio_converter.py` afterwards.

```
python helpers/generate_piper.py --phrase "turn on the lights" --label sp --note kitchen
python helpers/generate_piper.py --phrase "hey butterworth" --label ww --num 50
```

| Flag | Default | Description |
|---|---|---|
| `--phrase` | required | What Piper should say |
| `--label` | required | `ww`, `sp` or `bg` |
| `--note` | none | Added to the filename (letters, digits, dots, hyphens) |
| `--num` | `1` | How many clips; each uses a different blended voice, speed and noise setting |
| `--out` | `dataset/raw/generated` | Folder to save clips in |

## audio_converter.py

Run it with no arguments. Raw files can be `.wav`, `.flac`, `.mp3` or `.m4a` (mp3/m4a need `ffmpeg` installed); output is always `.wav`. It processes all three raw folders, writes clips to `dataset/processed/<label>/`,
and deletes the raw files it handled. The label comes from the filename prefix (`ww_`, `sp_`, `bg_`).

```
python helpers/audio_converter.py
python helpers/audio_converter.py --window 0.5 --keep-raw
```

- `raw/recordings/`: resampled to 16 kHz, mono, 16-bit; keeps the loudest 1.5 s (zero-padded if shorter).
  Output keeps the name prefix with a new number, e.g. `bg_mems_viroom_0086` becomes `bg_mems_viroom_NNNN`.
  Always deleted after processing. Put only short recordings here; anything longer than 1.5 s is trimmed, not chopped.
- `raw/generated/`: handled like `raw/recordings/` (zeros added at the end up to 1.5 s, or the loudest 1.5 s kept if longer),
  but with no speech check, even for `sp`. Always deleted after processing.
- `raw/long_clips/`: same conversion, then cut into 1.5 s clips; the leftover tail is dropped.
  Name files like `bg_mems_cafeteria.wav`; clips become `bg_mems_cafeteria_0000.wav`, `_0001`, and so on.
  A number already at the end of the name is dropped, so `sp_mems_vivoice_0002.wav` also gives `sp_mems_vivoice_NNNN`.

Clips whose audio already exists in `processed/<label>/` are skipped (content match, not filename);
the raw file is still deleted in that case. Numbering continues from what is already in `processed`.
Speech check (`sp` only; `ww` and `bg` are never checked). An `sp` clip is kept only if it contains speech: its loudest
100 ms is above `SP_MIN_DBFS` (-45 dBFS) and the WebRTC VAD (mode 3, the most aggressive) hears speech in at least
`VAD_MIN_SPEECH` (10%) of 30 ms frames. Rejected clips are relabeled `bg` (`sp_mems_x_NNNN` becomes `bg_mems_x_NNNN`) and saved to `processed/bg/` with the usual numbering and duplicate check.
The thresholds are constants at the top of `audio_converter.py`.
Files with no label prefix, long clips shorter than 1.5 s, and files that fail to load are skipped and kept.

| Flag | Default | Description |
|---|---|---|
| `--window` | `1.5` | Long clips only. Seconds between clip starts; `0.5` gives overlapping clips |
| `--keep-raw` | off | Long clips only. Don't delete them after chopping (recordings are always deleted) |

## dataset_report.py

Read-only. Counts the clips in `dataset/processed/` per label and name, where the name is everything between the label prefix and the trailing number
(`ww_generated_hey.butterworth_0012.wav` is `generated_hey.butterworth`). Names are sorted by count, with a total at the end.

```
python helpers/dataset_report.py
python helpers/dataset_report.py --label ww
python helpers/dataset_report.py --label sp --note kitchen
```

| Flag | Default | Description |
|---|---|---|
| `--label` | all | Only report `ww`, `sp` or `bg` |
| `--note` | none | Only count names containing this text |
| `--dir` | `dataset/processed` | Processed folder to read |

## build_manifest.py

Builds `dataset/manifest_v1.csv` (path, label, label_id, source, speaker, domain, phrase, split) from `dataset/processed/`. It reads audio only.
Default is a dry run: it prints the train/val split summary and checks (shared sessions have the same split in every class, no duplicate or missing paths,
every clip is 16 kHz mono 1.5 s) and writes nothing. `--write` writes the CSV, and refuses to if a check finds a problem.
Clip names must look like `<bg|sp|ww>_<mems|phone|generated|sample>_<name>_NNNN.wav`, otherwise it exits.

```
python helpers/build_manifest.py
python helpers/build_manifest.py --write
```

The split is by whole source, speaker or phrase, not by clip, with one seeded random choice (`SEED = 42`).
The rules (`VAL_SOURCES`, `VAL_PHRASES`, the generated wake block settings) are constants at the top of the script.

| Flag | Default | Description |
|---|---|---|
| `--write` | off | Write the CSV instead of a dry run |
