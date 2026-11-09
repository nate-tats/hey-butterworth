# Dataset

- `processed/` holds the clips: `bg/`, `sp/`, `ww/`. Every clip is 16 kHz, mono, 16-bit, 1.5 s.
- `raw/` holds the unprocessed recordings.
- `manifest_v1.csv` lists every processed clip with its label and train/val split.

Rebuild the manifest with `python helpers/build_manifest.py` (dry run: prints the split summary and checks) and add `--write` to write the CSV. The script reads audio only and fails rather than write if a check finds a problem. Split rules and the seed live at the top of that script.

## Filename pattern

`<bg|sp|ww>_<domain>_<name>_<NNNN>.wav`, e.g. `ww_generated_hey.butterworth_0012.wav`. The index restarts at 0000 within each source.

## manifest_v1.csv columns

Columns appear in this order.

| Column | Meaning |
|---|---|
| `path` | Clip path relative to the project root. |
| `label` | `wake`, `speech` or `background`. |
| `label_id` | Fixed ids, never change: bg=0, sp=1, ww=2. |
| `source` | Recording or dataset name from the filename, e.g. `mems_vivoice`, `phone_natrehearsal`, `generated_hey.butter`, `sample_librispeech`. |
| `speaker` | `mems_*voice` sources: the person (`vi`, `ji`, `mic`, `nat`, `shr`). Generated clips: `synthetic`. Everything else: the source name. |
| `domain` | From the filename prefix: `mems`, `phone`, `generated`, or `public` (for `sample_`). |
| `phrase` | Generated clips only: the text spoken, e.g. `hey.butter`. Empty otherwise. |
| `split` | `train` or `val`. `test` is reserved for a later held-out set; no test rows exist yet. |

## Counts (manifest_v1)

| Class | Train | Val | % val |
|---|---|---|---|
| wake | 2111 | 261 | 11.0% |
| speech | 13323 | 3403 | 20.3% |
| background | 6417 | 1565 | 19.6% |

## How the split works

Splits are grouped, never random by clip. Neighbouring clips from the same recording sound alike, so a random split would leak between train and val and make val scores look better than they are. Instead whole sources, speakers or phrases go to one side. The one random choice is seeded (`SEED = 42` in the script, printed on every run).

**Wake**
- Val voices: `mic` (7 clips) and `nat` (54 clips). They are held out completely, including their speech clips, so neither voice is in train.
- Val generated: 200 `generated_hey.butterworth` clips, taken as 4 seeded blocks of 50 consecutive indices.
- Train: `vi`, `ji`, `shr`, and the other 2,000 generated clips.

**Speech**
- Isolated false words (generated hard negatives): `hey.butter` and `hi.butterworth` (near-misses of the wake phrase), `hello` and `hello.mother.earth` (far ones). All other phrases stay in train.
- Whole sources in val: `sample_motispeech`, `mems_shrclass`, `mems_tamclass`, and the `mic` and `nat` speech clips.
- Everything else is train, including all of `sample_librispeech`.

**Background**
- Whole sources in val: `phone_geisel`, `mems_library`, `mems_tamclass`, `mems_shrclass`, and `sample_motispeech`.
- Everything else is train, including the large `phone_natrehearsal`, `phone_nathome` and `mems_viroom`.

**Rule for sources in several classes.** A source that appears in more than one class has the same split in all of them (`tamclass`, `shrclass`, `motispeech` in val; `envision`, `vinatyap`, `librispeech` in train). The script checks this.

## Caveats

- **Wake val is small and mostly synthetic.** It has 61 real clips from 2 voices plus 200 generated. Report metrics separately by `domain` (`mems` vs `generated`) so the synthetic number cannot hide a weak real-voice one.
- **Generated overlap.** Filenames carry no TTS voice id, so the held-out generated wake clips may share voices with train and that score will be optimistic. `generated_hey.butterworth` and the speaker `synthetic` are therefore the only things in both splits. The script treats this as expected.
- **Real wake training data is thin.** `vi` has 105 clips, and `ji` and `shr` have 4 and 2. The 2,000 generated clips dominate wake training.
- **Train never sees motispeech.** It is held out whole, so it is a test on an unseen corpus. If you would rather train on it, the script's `VAL_SOURCES` table is the place to change that.
- **Val is about 20% of speech and background** because whole sources were isolated and then the remainder filled to roughly that share. It will shift if new recordings arrive.

## Changing the split

Edit `VAL_SOURCES`, `VAL_PHRASES` and the wake-block settings at the top of `helpers/build_manifest.py`, run it without `--write` to read the new summary, then rerun with `--write`. Bump the file name (`manifest_v2.csv`) if experiments already depend on v1.
