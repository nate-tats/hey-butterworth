"""Build dataset/manifest_<version>.csv from dataset/processed (read-only on audio).

Every split version is kept in CONFIGS below, so old manifests stay documented and checkable.
Default is a dry run of the latest version: prints the split summary and validation checks, writes nothing.

    python helpers/build_manifest.py                      # dry run, latest version (v2)
    python helpers/build_manifest.py --write              # write dataset/manifest_v2.csv
    python helpers/build_manifest.py --version v1 --check # rebuild v1's clips with v1's rules, diff vs the CSV
"""
import argparse
import csv
import random
import re
import sys
import wave
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "dataset" / "processed"

COLUMNS = ["path", "label", "label_id", "source", "speaker", "domain", "phrase", "split"]
LABELS = {"bg": ("background", 0), "sp": ("speech", 1), "ww": ("wake", 2)}  # ids are fixed
DOMAINS = {"mems": "mems", "phone": "phone", "generated": "generated", "sample": "public"}
NAME_RE = re.compile(r"^(bg|sp|ww)_(mems|phone|generated|sample)_(.+)_(\d{4})\.wav$")
VOICE_RE = re.compile(r"^(.+)voice\d*$")   # mems_<speaker>voice[N]: session N of a real voice

SEED = 42
WW_GEN_BLOCK = 50         # generated wake clips are held out in blocks of consecutive indices
WW_GEN_VAL_BLOCKS = 4     # -> 200 clips

# One entry per split version. A real voice (mems_<name>voice[N]) is split by speaker, in every class it
# appears in, and must be listed under train_voices or val_voices: a new voice stops the build until it
# is assigned. val_sources holds out other whole sources; val_phrases the generated false-word phrases.
_V1_VAL_SOURCES = {
    "ww": set(),
    "sp": {"sample_motispeech", "mems_shrclass", "mems_tamclass"},
    "bg": {"phone_geisel", "mems_library", "mems_tamclass", "mems_shrclass", "sample_motispeech"},
}
_V1_VAL_PHRASES = {"hey.butter", "hi.butterworth", "hello", "hello.mother.earth"}
CONFIGS = {
    # 2026-11-08: first split (27,080 clips)
    "v1": {"train_voices": {"vi", "ji", "shr"}, "val_voices": {"mic", "nat"},
           "val_sources": _V1_VAL_SOURCES, "val_phrases": _V1_VAL_PHRASES},
    # 2026-11-11: +114 real wake clips. New voices darrell (81) and seth (4) go to train; natvoice1 (29)
    # is a second session of nat and stays held out with nat. Everything else as v1, so v2 val is a
    # superset of v1 val.
    "v2": {"train_voices": {"vi", "ji", "shr", "darrell", "seth"}, "val_voices": {"mic", "nat"},
           "val_sources": _V1_VAL_SOURCES, "val_phrases": _V1_VAL_PHRASES},
}
LATEST = "v2"

# sessions that appear in two classes and must get the same split in both
SHARED_SESSIONS = ["mems_tamclass", "mems_shrclass", "mems_envision", "phone_vinatyap",
                   "sample_motispeech", "sample_librispeech"]
# intentional partial-source holdouts, so overlap there is expected, not an error
EXPECTED_SOURCE_OVERLAP = {"generated_hey.butterworth"}
EXPECTED_SPEAKER_OVERLAP = {"synthetic"}


def parse(path):
    m = NAME_RE.match(path.name)
    if not m:
        sys.exit(f"unparseable filename: {path}")
    prefix, dom, name, idx = m.groups()
    source = f"{dom}_{name}"
    if dom == "generated":
        speaker, phrase = "synthetic", name
    else:
        phrase = ""
        v = VOICE_RE.match(name) if dom == "mems" else None
        speaker = v.group(1) if v else source  # natvoice1 is a second session of nat
    label, label_id = LABELS[prefix]
    return {
        "path": path.relative_to(ROOT).as_posix(), "label": label, "label_id": label_id,
        "source": source, "speaker": speaker, "domain": DOMAINS[dom], "phrase": phrase,
        "split": "train", "_prefix": prefix, "_idx": int(idx), "_voice": bool(v) if dom == "mems" else False,
    }


def pick_held_out(rng):
    """Seeded random choice of generated wake index blocks to hold out."""
    blocks = sorted({r["_idx"] // WW_GEN_BLOCK for r in ROWS
                     if r["_prefix"] == "ww" and r["source"] == "generated_hey.butterworth"})
    return set(rng.sample(blocks, WW_GEN_VAL_BLOCKS))


def assign_split(cfg, ww_gen_val):
    unassigned = sorted({r["speaker"] for r in ROWS if r["_voice"]}
                        - cfg["train_voices"] - cfg["val_voices"])
    if unassigned:
        sys.exit(f"voices not assigned to train or val in CONFIGS: {unassigned}")
    for r in ROWS:
        p, src = r["_prefix"], r["source"]
        if r["_voice"]:
            val = r["speaker"] in cfg["val_voices"]
        elif src in cfg["val_sources"][p]:
            val = True
        elif p == "ww" and src == "generated_hey.butterworth":
            val = r["_idx"] // WW_GEN_BLOCK in ww_gen_val
        elif p == "sp" and src.startswith("generated_"):
            val = r["phrase"] in cfg["val_phrases"]
        else:
            val = False
        r["split"] = "val" if val else "train"


def table(title, counter, keys):
    print(f"\n{title}")
    for k in sorted(keys):
        t, v = counter[(k, "train")], counter[(k, "val")]
        print(f"  {str(k):<40} train={t:<6} val={v:<6}")


def validate():
    print("\n=== VALIDATION ===")
    by_label = Counter((r["label"], r["split"]) for r in ROWS)
    table("clips per label x split", by_label, {r["label"] for r in ROWS})
    by_src = Counter(((r["label"], r["source"]), r["split"]) for r in ROWS)
    table("clips per (label, source) x split", by_src, {(r["label"], r["source"]) for r in ROWS})

    print("\n% validation per class")
    for label in ("wake", "speech", "background"):
        t, v = by_label[(label, "train")], by_label[(label, "val")]
        print(f"  {label:<11} {100 * v / (t + v):5.1f}%  ({v}/{t + v})")

    problems = 0
    for field, expected in (("source", EXPECTED_SOURCE_OVERLAP), ("speaker", EXPECTED_SPEAKER_OVERLAP)):
        splits = defaultdict(set)
        for r in ROWS:
            splits[r[field]].add(r["split"])
        both = sorted(k for k, s in splits.items() if len(s) > 1)
        print(f"\n{field}s in both train and val: {both or 'none'}")
        unexpected = [k for k in both if k not in expected]
        if unexpected:
            problems += 1
            print(f"  UNEXPECTED: {unexpected}")
        else:
            print(f"  (all intentional: {sorted(expected)})")

    print("\nshared-session split consistency across classes")
    for s in SHARED_SESSIONS:
        per = {(r["label"], r["split"]) for r in ROWS if r["source"] == s}
        splits = {sp for _, sp in per}
        ok = len(splits) == 1
        problems += not ok
        print(f"  {s:<18} {sorted(per)} {'OK' if ok else 'MISMATCH'}")

    paths = Counter(r["path"] for r in ROWS)
    dups = [p for p, n in paths.items() if n > 1]
    missing = [r["path"] for r in ROWS if not (ROOT / r["path"]).is_file()]
    bad_fmt = []
    for r in ROWS:
        try:
            with wave.open(str(ROOT / r["path"])) as w:
                fmt = (w.getframerate(), w.getnchannels(), w.getnframes())
        except Exception as e:
            fmt = ("error", str(e))
        if fmt != (16000, 1, 24000):
            bad_fmt.append((r["path"], fmt))
    print(f"\nduplicate paths: {len(dups)}  missing files: {len(missing)}  "
          f"not 16kHz/mono/1.5s: {len(bad_fmt)}")
    for item in (dups + missing + bad_fmt)[:10]:
        print("  ", item)
    problems += bool(dups or missing or bad_fmt)
    print(f"\nrows: {len(ROWS)}  unexpected problems: {problems}")
    return problems


def main():
    global ROWS
    ap = argparse.ArgumentParser(description="Build dataset/manifest_<version>.csv (dry run unless --write).")
    ap.add_argument("--version", default=LATEST, choices=sorted(CONFIGS), help=f"split version (default {LATEST})")
    ap.add_argument("--write", action="store_true", help="write the CSV (default: dry run)")
    ap.add_argument("--check", action="store_true",
                    help="rebuild only the clips listed in the existing CSV and report any split differences")
    args = ap.parse_args()
    cfg, out = CONFIGS[args.version], ROOT / "dataset" / f"manifest_{args.version}.csv"

    ROWS = [parse(p) for d in LABELS for p in sorted((PROCESSED / d).glob("*.wav"))]
    if args.check:
        with out.open(newline="") as f:
            old = {r["path"]: r for r in csv.DictReader(f)}
        ROWS = [r for r in ROWS if r["path"] in old]
    rng = random.Random(SEED)
    ww_gen_val = pick_held_out(rng)
    assign_split(cfg, ww_gen_val)

    print(f"version={args.version}  seed={SEED}")
    print(f"generated wake val blocks (of {WW_GEN_BLOCK} clips): {sorted(ww_gen_val)}")
    print(f"wake voices: train {sorted(cfg['train_voices'])}, val {sorted(cfg['val_voices'])}")
    print(f"false-word phrases in val: {sorted(cfg['val_phrases'])}")
    problems = validate()

    if args.check:
        diff = [p for r in ROWS for p in [r["path"]]
                if any(str(r[c]) != old[p][c] for c in COLUMNS)]
        missing = sorted(set(old) - {r["path"] for r in ROWS})
        print(f"\ncheck vs {out.name}: {len(ROWS)} rows rebuilt, {len(diff)} differ, {len(missing)} files missing")
        for p in (diff + missing)[:10]:
            print("  ", p)
        sys.exit(1 if diff or missing else 0)
    if args.write:
        if problems:
            sys.exit("not writing: unexpected problems above")
        with out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(ROWS)
        print(f"\nwrote {out.relative_to(ROOT)} ({len(ROWS)} rows)")
    else:
        print("\ndry run: nothing written (use --write after approval)")


if __name__ == "__main__":
    main()
