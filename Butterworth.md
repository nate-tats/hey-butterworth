# Butterworth

**Goal:** Build an embedded voice assistant - Butterworth.

**Thrust 1:** Use ML model to identify the wake word “Butterworth” when spoken, and acknowledge the response in some way.

**Code (project):** `model.py` (front end, CNN, streaming), `claude/augment.py` (data pipeline), `claude/test_streaming.py` (streaming == Keras check).

## Contents

1. [ML pipeline overview](#1-ml-pipeline-overview)
2. [Model and front end](#2-model-and-front-end)
3. [Dataset](#3-dataset)
4. [Data augmentation pipeline](#4-data-augmentation-pipeline-draft-augmentpy)
5. [Train/val split: two schemes](#5-trainval-split-two-schemes)
6. [Wake phrase “hey Butterworth”](#6-wake-phrase-hey-butterworth-consequences-for-the-cnn-and-augmentation)
7. [Open items](#7-open-items)

---

## 1. ML pipeline overview

1. **Data:** 1.5 s, 16 kHz mono WAV clips in three classes: wake word (ww), speech (sp), background (bg). See section 3.
2. **Augmentation** (`augment.py`, draft): waveform augmentations aimed at the ESP32 MEMS microphone, then WAV → log-mel spectrogram, then a random 124-frame crop. See section 4.
3. **Model** (`model.py`): 1D temporal CNN over log-mel frames, outputs P(Butterworth). See section 2.
4. **Training:** Keras, binary cross-entropy, class- and domain-balanced batches from `augment.batches()`.
5. **Streaming inference:** `StreamingButterworth` gives one decision every 240 ms and matches the Keras model exactly (checked by `test_streaming.py`).
6. **Deployment:** port the front end and the streaming CNN (BN folded, 632 floats of state) to the ESP32; acknowledge on detection.

---

## 2. Model and front end

### Front end (16 kHz audio → log-mel frames)

- STFT: 480-sample periodic Hann window (30 ms), hop 160 (10 ms), `n_fft = 512`.
- Power spectrum → 40 mel bins (125–7500 Hz, HTK mel) → `log(x + 1e-6)`.
- Frames from N samples (no centre padding): `T = 1 + (N − 480) // 160`.
  - 1.50 s recorded clip = 24,000 samples → 148 frames
  - model input = 20,160 samples → 124 frames (1.26 s)

### Network

All convolutions use VALID padding, with BatchNorm + ReLU after each.

| Layer | Kernel | Stride | Out spacing | RF (frames) | T in → out |
|---:|---:|---:|---:|---:|---:|
| 1 | 4 | 3 | 3 | 4 | 124 → 41 |
| 2 | 3 | 2 | 6 | 10 | 41 → 20 |
| 3 | 2 | 2 | 12 | 16 | 20 → 10 |
| 4 | 2 | 2 | 24 | 28 | 10 → 5 |
| 5 | 3 | 1 | 24 | 76 | 5 → 3 |
| 6 | 3 | 1 | 24 | 124 | 3 → 1 |

**Head:** Flatten (1 × C) → Dense(1, sigmoid) = P(Butterworth).

Out spacing is the product of strides so far; `RF = 1 + Σᵢ (kᵢ − 1) · Jᵢ`, where Jᵢ is layer i’s input spacing (1, 3, 6, 12, 24, 24). The input is exactly one receptive field, so the network emits exactly one timestep.

With valid padding every activation depends only on real audio, so it is identical whichever window it falls in. That is what makes streaming reuse exact.

### Training: random crops

Each 148-frame clip holds `CLIP_FRAMES − N_FRAMES + 1 = 25` crops of 124 frames (offsets 0–24). Picking one at random each epoch covers every alignment of the word against the 24-frame (240 ms) streaming grid, using only real audio. Positive crops must contain the whole phrase: pass `min_offset`/`max_offset` to `random_crop()` for clips where it sits near an edge (`augment.crop_range()` computes these from the phrase span).

### Cost (C = 64)

| Item | Cost |
|---|---|
| Full 124-frame window | ~0.84 M MACs, ~63.5 K weights |
| Streaming, per 24-frame hop | ~0.18 M MACs (one new decision / 240 ms) |
| Streaming state (BN folded) | ≤ Σᵢ (kᵢ − 1) · C_inᵢ = 120 + 128 + 64 + 64 + 128 + 128 = 632 floats |
| Front-end state | 320 audio samples (480 − 160 overlap) |

### Streaming timeline

- Audio arrives in 3,840-sample (240 ms) chunks → 24 new mel frames per chunk.
- First decision after 20,160 samples (1.26 s); then one every 3,840 samples.
- Decision j covers mel frames `[24j, 24j + 124)`, i.e. exactly what the Keras model would see if you cut that window out and ran it.

---

## 3. Dataset

All clips are 1.5 s, 16 kHz, mono. Domains: **generated** = Piper synthetic speech; **mems** = recorded on the ESP32 I2S MEMS microphone; **phone** = recorded on a phone; **public** = downloaded sample datasets.

| Domain | Wake | Speech | Background | Total |
|---|---:|---:|---:|---:|
| generated | 2,200 | 6,300 | 0 | 8,500 |
| mems | 172 | 430 | 1,661 | 2,263 |
| phone | 0 | 137 | 6,279 | 6,416 |
| public | 0 | 9,859 | 42 | 9,901 |
| **Total** | **2,372** | **16,726** | **7,982** | **27,080** |
| Share | 8.8% | 61.8% | 29.5% | 100% |

Total audio is about 11.3 h (ww 1.0 h, sp 7.0 h, bg 3.3 h).

- **Wake word sources:** generated_hey.butterworth 2,200; mems_vivoice 105, mems_natvoice 54, mems_micvoice 7, mems_jivoice 4, mems_shrvoice 2.
- **Speech sources:** public LibriSpeech 7,562 and MotiSpeech 2,297; 6,300 generated hard negatives across 23 near-miss phrases (e.g. “hi butter” 750, “hello butter” 550, “butterworth” alone 300, “hey buttercup”, “hey wordsworth”, “hey mother earth”, “better work”, “butterworth filter”); real MEMS 430 (envision 123, librandos 107, shrclass 78, natvoice 43, …); phone 137 (vinatyap).
- **Background sources:** phone 6,279 (natrehearsal 3,843, nathome 1,365, geisel 1,064, vinatyap 7); MEMS 1,661 (viroom 1,032, tamclass 272, envision 132, shrclass 125, library 100); public 42.

### What the numbers mean for training

- **Domain gap is the main risk.** The device hears through the MEMS mic, but only 8.4% of clips (2,263) were recorded on it, and 93% of wake-word clips (2,200 of 2,372) are synthetic. Augmentation must push generated, phone and public audio toward the MEMS channel.
- **Real wake words are scarce:** 172 clips from 5 speakers, 105 of them from one speaker (vivoice). Recording more real speakers on the MEMS mic is the highest-value data work.
- **The hard negatives are well designed:** they use the same Piper voices as the positives, so the model cannot learn “synthetic = wake word”. Augmentations must therefore be applied the same way to generated ww and generated sp.
- **Background is 79% phone.** Without channel augmentation the model could learn “phone colouring = background”.
- **Class imbalance** (8.8% positives) is handled by batch sampling rather than by duplicating files.
- **Decided (9 Nov 2026):** the wake phrase is “hey Butterworth” only. “Butterworth” alone, “hi/hello Butterworth” and every other variant are negatives for now. Consequences are in section 6.

---

## 4. Data augmentation pipeline (draft, augment.py)

### Stages, per training example

1. **Load WAV** → float32 mono 16 kHz in [−1, 1] (resample/downmix if needed). Clips are cached in memory as int16 (about 1.3 GB for all 27k).
2. **Speed perturbation** (ww, sp; p = 0.5): rate 0.9 / 0.95 / 1.05 / 1.1 by resampling. Widens the speaker and speaking-rate variety of the 2,200 synthetic positives. Never slows a wake phrase past 1.03 s (section 6).
3. **Sub-frame shift** (all): 0–159 samples, so the phrase is not always aligned to the 10 ms frame grid. Then fit back to exactly 24,000 samples, keeping the phrase inside.
4. **Channel EQ toward the MEMS mic** (generated / phone / public, p = 0.7; real MEMS clips get only a mild random tilt, p = 0.2). The EQ curve is learned from the data: long-term average spectrum of MEMS speech minus that of the source domain’s speech, smoothed, then randomised (scale 0.5–1.2, tilt ±3 dB/octave) into a bank of 32 linear-phase FIR filters.
5. **Reverb** (ww, sp; p = 0.4 for generated, which is dry; p = 0.1 for real recordings): synthetic room impulse response, RT60 0.15–0.7 s, direct-to-reverberant ratio 0–15 dB.
6. **Mixing.**
   - Background (ww, sp; p = 0.6): a random train-split background clip at SNR 0–20 dB for speech and 5–20 dB for wake clips (measured over the phrase span).
   - Babble (ww only, p = 0.15): real or public speech underneath at SNR 10–25 dB, so a wake word with someone else talking still triggers.
   - Background on background (bg; p = 0.3): two backgrounds at −5 to 10 dB.
7. **Gain, noise floor, clipping:** gain −18 to +6 dB, then a noise floor (−75 to −55 dBFS placeholder, to be calibrated from the quietest frames of real MEMS background), then hard clip to [−1, 1] to simulate ADC saturation. The noise floor also removes the digital silence of TTS clips, which would otherwise give `log(1e-6)` frames that only synthetic clips have.
8. **WAV → spectrogram:** `model.logmel()` on the 24,000 samples → 148 × 40 log-mel. This is the same code the device front end mirrors, so training and inference features match exactly.
9. **Random crop:** 124 of the 148 frames. For ww clips the offset range comes from `crop_range(phrase span)`, so every crop contains the whole phrase; a wake clip whose phrase cannot fit is skipped. Negatives use all 25 offsets.
10. **SpecAugment** (p = 0.5): up to 2 frequency masks of ≤ 4 mel bins and 2 time masks of ≤ 8 frames (80 ms), filled with the crop mean. Wake clips get frequency masks only.

**Output:** (124, 40) float32 and a label (ww = 1, sp/bg = 0). Eval examples skip steps 2–7 and 10.

### Phrase spans

Positives need the phrase’s start and end so crops never cut it. `augment.annotate_spans()` finds them with an energy threshold (frames within 30 dB of the loudest). This is reliable on clean TTS; the 172 real MEMS clips should be checked by hand. Spans are stored in the manifest (columns `on`, `off`, in samples) with `write_manifest()`.

### Batch sampling

| Class | Groups (share of each batch) | Total |
|---|---|---:|
| ww | generated 21%, MEMS 9% | 30% |
| sp | generated hard negatives 14%, public 12%, MEMS 8%, phone 6% | 40% |
| bg | phone 14%, MEMS 14%, public 2% | 30% |

Real MEMS audio is over-represented relative to its 8.4% share, and the batch is 30% positive instead of 8.8%. Within the generated-speech group the closest near misses are drawn more often (section 6); other groups are drawn uniformly. If false accepts are too high, move share from positives to hard negatives.

### Splits

Two schemes are compared in section 5; scheme B (manifest_v1) was chosen. Background and babble pools for mixing come from the train split only, so held-out noise never leaks in.

### Metrics

- False-reject rate on held-out real MEMS wake words.
- False accepts per hour: run `StreamingButterworth` over held-out sp + bg clips concatenated into a long stream (about 2.1 h of negatives in manifest_v1 val).
- Pick the threshold for a target false-accept rate (e.g. ≤ 1 per hour), then report false rejects at that threshold.

### Compute cost

About 0.23 s per 128-example batch on one CPU core (numpy/scipy), so one pass over 27k clips takes under a minute. Fine for Keras with a Python generator; parallelise with `tf.data.Dataset.from_generator` or worker processes if it becomes a bottleneck.

---

## 5. Train/val split: two schemes

Both schemes split by whole source or speaker, never by random clip, because neighbouring clips from one recording sound alike and a random split would leak.

### Scheme A: source holdout (earlier augment.py draft, now removed)

| Split | Sources | Clips |
|---|---|---|
| Test | mems_natvoice, mems_library, phone_geisel | 54 ww, 43 sp, 1,164 bg |
| Val | mems_micvoice, mems_jivoice, mems_shrvoice, mems_envision | 13 ww, 150 sp, 132 bg |
| Train | everything else | 2,305 ww (2,200 generated + 105 real), 16,533 sp, 6,686 bg |

### Scheme B: manifest_v1.csv (dataset README)

Built by `helpers/build_manifest.py` (`SEED = 42`). A source that appears in several classes gets the same split in all of them; the script checks this.

| Class | Val holds out |
|---|---|
| Wake | Voices mic (7) and nat (54), with their speech clips too; 200 generated_hey.butterworth clips as 4 seeded blocks of 50 consecutive indices |
| Speech | Phrases hey.butter, hi.butterworth (near misses), hello, hello.mother.earth (far) = 950 clips; whole sources sample_motispeech, mems_shrclass, mems_tamclass; mic and nat speech |
| Background | phone_geisel, mems_library, mems_tamclass, mems_shrclass, sample_motispeech |
| Test | Reserved for a later held-out set; no rows yet |

| Class | Train | Val | % val |
|---|---:|---:|---:|
| wake | 2,111 | 261 | 11.0% |
| speech | 13,323 | 3,403 | 20.3% |
| background | 6,417 | 1,565 | 19.6% |

### Side by side

| Item | Scheme A (source holdout) | Scheme B (manifest_v1) |
|---|---|---|
| Sets | train / val / test | train / val (test reserved) |
| Real wake in val | 13 (mic, ji, shr) | 61 (mic, nat) |
| Real wake in test | 54 (nat) | none yet |
| Real wake in train | 105 (vi) | 111 (vi, ji, shr) |
| Generated wake held out | none (10% suggested) | 200, seeded blocks of 50 |
| Hard-negative phrases held out | none | 4 phrases, 950 clips |
| Unseen public corpus in val | no | yes, MotiSpeech (2,297) |
| Negatives in val | 282 clips, about 7 min | 4,968 clips, about 2.1 h |
| Negatives in test | 1,207 clips, about 30 min | none yet |
| Implementation | dict in augment.py, no checks | script with checks, seeded, reproducible |

### Comparison

- **Wake evaluation: B is stronger.** 61 real val wake words from two voices against 13 in A, and the 200 held-out generated clips give a second, larger (if optimistic) number. A spends its best real voice (nat) on a test set and leaves val almost empty for model selection.
- **False accepts: B is much stronger.** About 2.1 h of held-out negatives gives a usable false-accepts-per-hour estimate at a 1 per hour target; A’s 7 min of val negatives cannot.
- **Generalisation tests:** only B checks the things most likely to fail on the device: unseen near-miss phrasings (hi.butterworth is the hardest one) and an unseen speech corpus.
- **Training data cost:** scheme B removes more from train (MotiSpeech, tamclass, shrclass, 950 hard negatives, 200 generated wake clips). Real wake training is the same in both (about 110 clips), so the cost is mostly negatives, which are plentiful.
- **Missing in B: a true test set.** Tuning the model and the threshold on the same val will make the val numbers optimistic. A has a test set, but it is too small (54 real wake words) to be worth the val it costs.
- **Shared weakness:** generated wake clips have no voice id, so in both schemes any generated-wake score may share Piper voices with train. Report mems and generated separately, as the README says.

### Decision (9 Nov 2026): scheme B

Scheme B (manifest_v1) is the split. `augment.py` now reads `manifest_v1.csv` directly: `label` (wake/speech/background) maps to ww/sp/bg, the `split` column decides train/val, and the old HOLDOUT dict is gone. The binary training target is `label_id == 2`.

Next step: when new MEMS wake-word speakers are recorded, put them (with matching speech and a fresh room of background) straight into a test set and do not look at it while tuning. Until then, choose the threshold on val and treat val numbers as optimistic.

---

## 6. Wake phrase “hey Butterworth”: consequences for the CNN and augmentation

### Phrase length vs. the model window (main concern)

The model sees 1.26 s (20,160 samples) and decides every 240 ms. A phrase is guaranteed to fall whole inside some decision window, whatever its alignment, only if it is at most 20,160 − (3,840 − 160) = 16,480 samples, about **1.03 s**. A longer “hey Butterworth” can land so that every window cuts it, and a cut phrase looks like the negatives “hey butter” or “Butterworth”: the model is trained to reject it, so the wake word would be missed.

Typical “hey Butterworth” is 0.8–1.0 s, so this is probably fine, but it is close and must be measured. `augment.annotate_spans()` reports median, 95th percentile and maximum phrase length and lists clips over 1.03 s. Run it on manifest_v1 before training.

If the 95th percentile is over about 1.0 s, a drop-in option is a 120 ms decision hop:

| Variant | Kernels / strides | Decision hop | Max phrase | Weights | Streaming |
|---|---|---:|---:|---:|---:|
| Current | [4,3,2,2,3,3] / [3,2,2,2,1,1] | 240 ms | 1.03 s | 63.5 K | 0.75 M MACs/s |
| 120 ms hop | [4,3,2,4,4,4] / [3,2,2,1,1,1] | 120 ms | 1.15 s | 79.9 K | 1.02 M MACs/s |

Both keep the 124-frame receptive field and input; the cost increase is still small for the ESP32. The 1.5 s clips leave no room to make the window much longer.

### Partial windows are now hard negatives

“hey”, “hey butter” and “Butterworth” are prefixes and suffixes of the wake phrase, so in streaming the windows just before and after the right one look like negatives. That is correct behaviour, but it means a real wake word may fire in only one 240 ms window. Do not require several consecutive positive decisions; use a single threshold (optionally with a short refractory period after a detection).

### Changes made to augment.py

- Reads `manifest_v1.csv` (label mapping, split column) and writes spans back with `write_manifest()` as `on`/`off` columns.
- Speed perturbation never slows a wake phrase past 1.03 s.
- A wake clip whose phrase cannot fit whole in any training crop is skipped, not cropped (the old fallback would have labelled a cut phrase as positive).
- No SpecAugment time masks on wake clips: an 80 ms mask can erase most of “hey” and turn the clip into “Butterworth”, a negative. Frequency masks stay.
- Background SNR for wake clips is 5–20 dB (was 0–20): at 0 dB the short, quiet “hey” can disappear under noise, leaving a mislabelled “Butterworth”.
- Within the generated-speech batch group, the closest near misses are drawn more often: hey.butter, butterworth, hello.butterworth, hi.butterworth (×3); hey.butler, hey.buttercup, hey, hey.wordsworth (×2).

### Concerns that need a decision or a check

- **The closest negative is not in training.** manifest_v1 holds out hey.butter for val, and it is literally the first part of the wake phrase. The model still learns from hey.butler, hey.buttercup and “Butterworth” alone, but never trains on the exact prefix. Suggest manifest_v2: move hey.butter into train and hold out hey.butler (or hey.buttercup) instead. hi.butterworth stays held out; hello.butterworth in train covers that pattern.
- **Check the real wake recordings.** Every real MEMS wake clip must actually say “hey Butterworth”. Any clip with “Butterworth” alone must be relabelled as speech, or it will teach the opposite of the new rule.
- **The CNN itself needs no change** for the stricter phrase: the receptive field covers the whole window, so the “hey” at the start is visible. Watch the val confusion on hi.butterworth: telling “hi” from “hey” is the hardest part of the task.

---

## 7. Open items

- [ ] Run `annotate_spans()` on manifest_v1 and check the phrase-length statistics against the 1.03 s limit; switch to the 120 ms hop if needed.
- [ ] Hand-check spans and the spoken phrase for the 172 real MEMS wake clips.
- [ ] Decide on manifest_v2 (hey.butter into train, hold out hey.butler instead).
- [ ] Calibrate the noise-floor range from real MEMS background.
- [ ] Record more wake-word speakers on the MEMS mic (largest gap), and use them as the test set.
- [ ] Write the training script (`model.build_model` + `augment.batches`) and the false-accepts-per-hour evaluation.
