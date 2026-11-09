import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PIPER_DIR = Path.home() / "piper-sample-generator"
PIPER_PYTHON = PIPER_DIR / ".venv" / "bin" / "python"
MODEL = PIPER_DIR / "models" / "en_US-libritts_r-medium.pt"
MAX_SPEAKERS = 200  # later LibriTTS speakers have few samples and can sound artifacty
DATASET = Path(__file__).parent.parent / "dataset"
OUT_DIR = DATASET / "raw" / "generated"
SOURCE = "generated"  # constant in every filename


def next_number(prefix, folders):
    highest = -1
    for folder in folders:
        for f in folder.glob(f"{prefix}_*.wav"):
            m = re.fullmatch(re.escape(prefix) + r"_(\d+)\.wav", f.name)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def main():
    ap = argparse.ArgumentParser(description="Generate speech clips with Piper into dataset/raw/generated.")
    ap.add_argument("--phrase", required=True, help="What Piper should say")
    ap.add_argument("--label", required=True, choices=["ww", "sp", "bg"])
    ap.add_argument("--note", help="Added to the filename (letters, digits, dots, hyphens)")
    ap.add_argument("--num", type=int, default=1, help="How many clips (different voices/speeds)")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="Folder to save clips in")
    args = ap.parse_args()

    if args.num < 1:
        sys.exit("--num must be at least 1")
    if args.note and not re.fullmatch(r"[A-Za-z0-9.-]+", args.note):
        sys.exit("--note may only contain letters, digits, dots and hyphens")
    for path in (PIPER_PYTHON, MODEL):
        if not path.exists():
            sys.exit(f"Missing {path}")

    prefix = f"{args.label}_{SOURCE}" + (f"_{args.note}" if args.note else "")
    args.out.mkdir(parents=True, exist_ok=True)
    start = next_number(prefix, [args.out, DATASET / "processed" / args.label])

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [str(PIPER_PYTHON), "-m", "piper_sample_generator", args.phrase,
               "--model", str(MODEL), "--max-samples", str(args.num),
               "--max-speakers", str(MAX_SPEAKERS), "--output-dir", tmp]
        result = subprocess.run(cmd, cwd=PIPER_DIR, capture_output=True, text=True)
        clips = sorted(Path(tmp).glob("*.wav"), key=lambda p: int(p.stem))
        if result.returncode != 0 or len(clips) != args.num:
            sys.exit(f"Piper failed:\n{result.stderr[-2000:]}")
        for i, clip in enumerate(clips):
            dest = args.out / f"{prefix}_{start + i:04d}.wav"
            clip.replace(dest)
            print(f"saved {dest}")


if __name__ == "__main__":
    main()
