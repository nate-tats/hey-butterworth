# Dataset

- `processed/` holds the clips: `bg/`, `sp/`, `ww/`. Every clip is 16 kHz, mono, 16-bit, 1.5 s.
- `raw/` holds the unprocessed recordings.
- `manifest_v2.csv` lists every processed clip with its label and train/val split (current). `manifest_v1.csv` is the
  earlier split, kept unchanged so older runs stay reproducible; see [Split versions](#split-versions).
- `manifest_vN_spans.csv` is the same manifest plus wake-phrase `on`/`off` columns (`python ml/features.py spans`).

Rebuild the manifest with `python helpers/build_manifest.py` (dry run of the latest version: prints the split summary and checks) and add `--write` to write the CSV. The script reads audio only and fails rather than write if a check finds a problem. Every split version's rules live in `CONFIGS` at the top of that script; `--version v1 --check` rebuilds v1's clips under v1's rules and confirms the CSV still matches.

## Download

The full dataset is on Google Drive: [dataset.zip](https://drive.google.com/file/d/1nP777gvYfJU4ylIDqT5QBjz_slohyPHs/view?usp=sharing).

## Dataset size

**27,194 processed clips in total** (`python helpers/dataset_report.py` prints the per-source breakdown).

| Section | Folder | Clips | Sources |
|---|---|---|---|
| Wake word | `ww/` | 2,486 | 9 |
| Speech | `sp/` | 16,726 | 35 |
| Background | `bg/` | 7,982 | 12 |
| **Total** | | **27,194** | |

Largest sources per section:
- `ww`: `generated_hey.butterworth` (2,200), then real voices `mems_vivoice` (105), `mems_darrellvoice` (81), `mems_natvoice` (54) + `mems_natvoice1` (29, a second session of the same speaker), `mems_micvoice` (7), `mems_jivoice` (4), `mems_sethvoice` (4), `mems_shrvoice` (2).
- `sp`: `sample_librispeech` (7,562), `sample_motispeech` (2,297), `generated_hi.butter` (750), `generated_hello.butter` (550), plus many generated near-miss and far phrases (150-300 each).
- `bg`: `phone_natrehearsal` (3,843), `phone_nathome` (1,365), `phone_geisel` (1,064), `mems_viroom` (1,032).

## Filename pattern

`<bg|sp|ww>_<domain>_<name>_<NNNN>.wav`, e.g. `ww_generated_hey.butterworth_0012.wav`. The index restarts at 0000 within each source.

## Manifest columns

Columns appear in this order.

| Column | Meaning |
|---|---|
| `path` | Clip path relative to the project root. |
| `label` | `wake`, `speech` or `background`. |
| `label_id` | Fixed ids, never change: bg=0, sp=1, ww=2. |
| `source` | Recording or dataset name from the filename, e.g. `mems_vivoice`, `phone_natrehearsal`, `generated_hey.butter`, `sample_librispeech`. |
| `speaker` | `mems_<name>voice[N]` sources: the person (`vi`, `darrell`, `nat`, `mic`, `ji`, `seth`, `shr`); a trailing number is a later session of the same person, so `natvoice1` is `nat`. Generated clips: `synthetic`. Everything else: the source name. |
| `domain` | From the filename prefix: `mems`, `phone`, `generated`, or `public` (for `sample_`). |
| `phrase` | Generated clips only: the text spoken, e.g. `hey.butter`. Empty otherwise. |
| `split` | `train` or `val`. `test` is reserved for a later held-out set; no test rows exist yet. |

## Counts (manifest_v2)

| Class | Train | Val | % val |
|---|---|---|---|
| wake | 2196 | 290 | 11.7% |
| speech | 13323 | 3403 | 20.3% |
| background | 6417 | 1565 | 19.6% |

## Split versions

| Version | Date | Clips | Wake voices in train | Wake voices in val | Real wake clips train / val |
|---|---|---|---|---|---|
| `v1` | 2026-11-08 | 27,080 | `vi`, `ji`, `shr` | `nat`, `mic` | 111 / 61 |
| `v2` | 2026-11-11 | 27,194 | `vi`, `darrell`, `ji`, `seth`, `shr` | `nat` (both sessions), `mic` | 196 / 90 |

v2 only adds the 114 new real wake clips; every v1 row keeps its split, so v2 val is a superset of v1 val and the two can be compared directly. New voices went to train because real wake training data was the weak point of v1 (18% of real wake clips missed). `natvoice1` is the same speaker as `natvoice`, so it stays held out with it. A real voice that isn't listed in `CONFIGS` stops the build until it is assigned to a side.

## How the split works

Splits are grouped, never random by clip. Neighbouring clips from the same recording sound alike, so a random split would leak between train and val and make val scores look better than they are. Instead whole sources, speakers or phrases go to one side. The one random choice is seeded (`SEED = 42` in the script, printed on every run).

**Wake** (v2)
- Val voices: `mic` (7 clips) and `nat` (54 + 29 clips over two sessions). They are held out completely, including their speech clips, so neither voice is in train.
- Val generated: 200 `generated_hey.butterworth` clips, taken as 4 seeded blocks of 50 consecutive indices.
- Train: `vi`, `darrell`, `ji`, `seth`, `shr`, and the other 2,000 generated clips.

**Speech**
- Isolated false words (generated hard negatives): `hey.butter` and `hi.butterworth` (near-misses of the wake phrase), `hello` and `hello.mother.earth` (far ones). All other phrases stay in train.
- Whole sources in val: `sample_motispeech`, `mems_shrclass`, `mems_tamclass`, and the `mic` and `nat` speech clips.
- Everything else is train, including all of `sample_librispeech`.

**Background**
- Whole sources in val: `phone_geisel`, `mems_library`, `mems_tamclass`, `mems_shrclass`, and `sample_motispeech`.
- Everything else is train, including the large `phone_natrehearsal`, `phone_nathome` and `mems_viroom`.

**Rule for sources in several classes.** A source that appears in more than one class has the same split in all of them (`tamclass`, `shrclass`, `motispeech` in val; `envision`, `vinatyap`, `librispeech` in train). The script checks this.

## Caveats

- **Wake val is small and mostly synthetic.** It has 90 real clips from 2 voices (83 of them `nat`) plus 200 generated. Report metrics separately by `domain` (`mems` vs `generated`) so the synthetic number cannot hide a weak real-voice one.
- **Generated overlap.** Filenames carry no TTS voice id, so the held-out generated wake clips may share voices with train and that score will be optimistic. `generated_hey.butterworth` and the speaker `synthetic` are therefore the only things in both splits. The script treats this as expected.
- **Real wake training data is thin.** `vi` has 105 clips and `darrell` 81 (15 of those are spoken slowly enough to exceed the 1.03 s phrase limit and aren't used as positives); `ji`, `seth` and `shr` have 4, 4 and 2. The 2,000 generated clips dominate wake training.
- **Train never sees motispeech.** It is held out whole, so it is a test on an unseen corpus. If you would rather train on it, `val_sources` in the script's `CONFIGS` is the place to change that.
- **Val is about 20% of speech and background** because whole sources were isolated and then the remainder filled to roughly that share. It will shift if new recordings arrive.

## Changing the split

Add a new entry to `CONFIGS` at the top of `helpers/build_manifest.py` (copy the latest one and change `train_voices`, `val_voices`, `val_sources` or `val_phrases`), set `LATEST`, run it without `--write` to read the new summary, then rerun with `--write`. It writes `manifest_<version>.csv`; never edit an old entry, so earlier manifests stay reproducible. Then run `python ml/features.py spans` for the spans manifest and bump `MANIFEST_VERSION` in `ml/features.py` so training uses it by default.
