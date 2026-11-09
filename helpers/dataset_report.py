"""Report how many clips are in dataset/processed, grouped by label and name.

A clip's name is everything between the label prefix and the trailing number,
e.g. ww_generated_hey.butterworth_0012.wav -> generated_hey.butterworth

    python helpers/dataset_report.py
    python helpers/dataset_report.py --label ww
    python helpers/dataset_report.py --label sp --note kitchen
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

PROCESSED = Path(__file__).resolve().parent.parent / "dataset" / "processed"
LABELS = ["ww", "sp", "bg"]
NUM_SUFFIX = re.compile(r"_\d+$")


def clip_name(path, label):
    stem = NUM_SUFFIX.sub("", path.stem)
    prefix = label + "_"
    return stem[len(prefix):] if stem.startswith(prefix) else stem


def main():
    parser = argparse.ArgumentParser(description="Report processed clip counts per label and name.")
    parser.add_argument("--label", choices=LABELS, help="only report this label (default: all)")
    parser.add_argument("--note", help="only count names containing this text")
    parser.add_argument("--dir", type=Path, default=PROCESSED, help="processed folder")
    args = parser.parse_args()

    if not args.dir.is_dir():
        sys.exit(f"Folder not found: {args.dir}")

    grand = 0
    for label in [args.label] if args.label else LABELS:
        counts = Counter(clip_name(p, label) for p in (args.dir / label).glob("*.wav"))
        if args.note:
            counts = Counter({n: c for n, c in counts.items() if args.note in n})
        total = sum(counts.values())
        grand += total

        print(f"\n{label}  ({total} clips, {len(counts)} names)")
        if not counts:
            print("  (none)")
            continue
        width = max(len(n) for n in counts)
        for name, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  {name:<{width}}  {n:>6}")

    if not args.label:
        print(f"\ntotal: {grand} clips")


if __name__ == "__main__":
    main()
