# helpers

Two scripts for building the dataset. Both work on the `dataset/` folder at the repo root:

```
dataset/
  raw/recordings/      short recordings straight from the mic (all labels together)
  raw/long_clips/      long files you add by hand, for chopping
  processed/<label>/   16 kHz, mono, 16-bit, 1.5 s clips, sorted by label
```

Labels: `ww` (wake word), `sp` (other speech), `bg` (background).

Requires `pyserial` (recording) and `librosa`, `soundfile`, `numpy`, `webrtcvad` (converting; `pip install webrtcvad-wheels` if `webrtcvad` won't build).

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
| `--gap` | `1.0` | Countdown seconds before each recording |
| `--port` | auto | Serial port, e.g. `/dev/cu.usbserial-0001` |
| `--out` | `dataset/raw/recordings` | Folder to save clips in |

## audio_converter.py

Run it with no arguments. It processes both raw folders, writes clips to `dataset/processed/<label>/`,
and deletes the raw files it handled. The label comes from the filename prefix (`ww_`, `sp_`, `bg_`).

```
python helpers/audio_converter.py
python helpers/audio_converter.py --window 0.5 --keep-raw
```

- `raw/recordings/`: resampled to 16 kHz, mono, 16-bit; keeps the loudest 1.5 s (zero-padded if shorter).
  Output keeps the name prefix with a new number, e.g. `bg_mems_viroom_0086` becomes `bg_mems_viroom_NNNN`.
  Always deleted after processing. Put only short recordings here; anything longer than 1.5 s is trimmed, not chopped.
- `raw/long_clips/`: same conversion, then cut into 1.5 s clips; the leftover tail is dropped.
  Name files like `bg_mems_cafeteria.wav`; clips become `bg_mems_cafeteria_0000.wav`, `_0001`, and so on.
  A number already at the end of the name is dropped, so `sp_mems_vivoice_0002.wav` also gives `sp_mems_vivoice_NNNN`.

Clips whose audio already exists in `processed/<label>/` are skipped (content match, not filename);
the raw file is still deleted in that case. Numbering continues from what is already in `processed`.
Speech check (`sp` only; `ww` and `bg` are never checked). An `sp` clip is kept only if it contains speech: its loudest
100 ms is above `SP_MIN_DBFS` (-45 dBFS) and the WebRTC VAD (mode 3, the most aggressive) hears speech in at least
`VAD_MIN_SPEECH` (10%) of 30 ms frames. Rejected clips are not saved (their raw file is still deleted).
The thresholds are constants at the top of `audio_converter.py`.
Files with no label prefix, long clips shorter than 1.5 s, and files that fail to load are skipped and kept.

| Flag | Default | Description |
|---|---|---|
| `--window` | `1.5` | Long clips only. Seconds between clip starts; `0.5` gives overlapping clips |
| `--keep-raw` | off | Long clips only. Don't delete them after chopping (recordings are always deleted) |
