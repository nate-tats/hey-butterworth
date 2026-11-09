"""Build dataset/manifest_v1.csv from dataset/processed (read-only on audio).

Default is a dry run: prints the split summary and validation checks, writes nothing.
Pass --write to write dataset/manifest_v1.csv.

    python helpers/build_manifest.py
    python helpers/build_manifest.py --write
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
OUT = ROOT / "dataset" / "manifest_v1.csv"

COLUMNS = ["path", "label", "label_id", "source", "speaker", "domain", "phrase", "split"]
LABELS = {"bg": ("background", 0), "sp": ("speech", 1), "ww": ("wake", 2)}  # ids are fixed
DOMAINS = {"mems": "mems", "phone": "phone", "generated": "generated", "sample": "public"}
NAME_RE = re.compile(r"^(bg|sp|ww)_(mems|phone|generated|sample)_(.+)_(\d{4})\.wav$")

SEED = 42
WW_GEN_BLOCK = 50         # generated wake clips are held out in blocks of consecutive indices
WW_GEN_VAL_BLOCKS = 4     # -> 200 clips

# false-word phrases (generated hard negatives) isolated in val
VAL_PHRASES = {"hey.butter", "hi.butterworth", "hello", "hello.mother.earth"}

# whole sources held out entirely, in every class they appear in (label -> sources)
VAL_SOURCES = {
    "ww": {"mems_micvoice", "mems_natvoice"},
    "sp": {"mems_micvoice", "mems_natvoice", "sample_motispeech", "mems_shrclass", "mems_tamclass"},
    "bg": {"phone_geisel", "mems_library", "mems_tamclass", "mems_shrclass", "sample_motispeech"},
}
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
        speaker = name[:-len("voice")] if dom == "mems" and name.endswith("voice") else source
    label, label_id = LABELS[prefix]
    return {
        "path": path.relative_to(ROOT).as_posix(), "label": label, "label_id": label_id,
        "source": source, "speaker": speaker, "domain": DOMAINS[dom], "phrase": phrase,
        "split": "train", "_prefix": prefix, "_idx": int(idx),
    }


def pick_held_out(rng):
    """Seeded random choice of generated wake index blocks to hold out."""
    blocks = sorted({r["_idx"] // WW_GEN_BLOCK for r in ROWS
                     if r["_prefix"] == "ww" and r["source"] == "generated_hey.butterworth"})
    return set(rng.sample(blocks, WW_GEN_VAL_BLOCKS))


def assign_split(ww_gen_val):
    for r in ROWS:
        p, src = r["_prefix"], r["source"]
        if src in VAL_SOURCES[p]:
            val = True
        elif p == "ww" and src == "generated_hey.butterworth":
            val = r["_idx"] // WW_GEN_BLOCK in ww_gen_val
        elif p == "sp" and src.startswith("generated_"):
            val = r["phrase"] in VAL_PHRASES
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
    ap = argparse.ArgumentParser(description="Build dataset/manifest_v1.csv (dry run unless --write).")
    ap.add_argument("--write", action="store_true", help="write the CSV (default: dry run)")
    args = ap.parse_args()

    ROWS = [parse(p) for d in LABELS for p in sorted((PROCESSED / d).glob("*.wav"))]
    rng = random.Random(SEED)
    ww_gen_val = pick_held_out(rng)
    assign_split(ww_gen_val)

    print(f"seed={SEED}")
    print(f"generated wake val blocks (of {WW_GEN_BLOCK} clips): {sorted(ww_gen_val)}")
    print(f"held-out wake voices: {sorted(VAL_SOURCES['ww'])}")
    print(f"false-word phrases in val: {sorted(VAL_PHRASES)}")
    problems = validate()

    if args.write:
        if problems:
            sys.exit("not writing: unexpected problems above")
        with OUT.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(ROWS)
        print(f"\nwrote {OUT.relative_to(ROOT)} ({len(ROWS)} rows)")
    else:
        print("\ndry run: nothing written (use --write after approval)")


if __name__ == "__main__":
    main()
