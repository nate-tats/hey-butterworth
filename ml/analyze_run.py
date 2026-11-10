"""Training analysis for a train.py run: loss / accuracy / AUC curves plus a short text summary.

  python ml/analyze_run.py                       # latest folder in runs/
  python ml/analyze_run.py runs/full-20261109-1509
  python ml/analyze_run.py runs/a runs/b         # one figure per run

Reads <run>/history.csv (and val_report.json / config.json if present) and writes
<run>/training_curves.png. The dashed line marks the epoch with the best val AUC,
which is the checkpoint train.py keeps in best.keras; dotted lines mark the epochs
where ReduceLROnPlateau halved the learning rate.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANELS = [("loss", "Loss (binary cross-entropy)"), ("acc", "Accuracy"), ("auc", "AUC")]


def load_history(run):
    with open(run / "history.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{run}/history.csv has no epochs yet")
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def latest_run():
    runs = [p for p in (ROOT / "runs").glob("*") if (p / "history.csv").exists()]
    if not runs:
        sys.exit(f"no runs with history.csv under {ROOT / 'runs'}")
    return max(runs, key=lambda p: (p / "history.csv").stat().st_mtime)


def plot(run, h, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [int(e) + 1 for e in h["epoch"]]                 # 1-based for display
    best = max(range(len(ep)), key=lambda i: h["val_auc"][i])
    lr = h.get("learning_rate", [])
    drops = [ep[i] for i in range(1, len(lr)) if lr[i] < lr[i - 1]]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, PANELS):
        ax.plot(ep, h[key], "o-", ms=3, label="train (augmented)")
        ax.plot(ep, h[f"val_{key}"], "o-", ms=3, label="val")
        ax.axvline(ep[best], color="0.4", ls="--", lw=1,
                   label=f"best val AUC (epoch {ep[best]})")
        for d in drops:
            ax.axvline(d, color="0.7", ls=":", lw=1)
        ax.set(title=title, xlabel="epoch")
        ax.grid(alpha=0.3)
    if drops:
        axes[0].plot([], [], color="0.7", ls=":", lw=1, label="LR halved")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{run.name}: best val AUC {h['val_auc'][best]:.4f} at epoch {ep[best]} "
                 f"of {len(ep)}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return ep, best, drops


def summary(run, h, ep, best, drops, out):
    print(f"== {run}")
    cfg = run / "config.json"
    if cfg.exists():
        c = json.loads(cfg.read_text())
        print(f"   config: {c.get('epochs')} epochs x {c.get('steps')} steps x batch {c.get('batch')}, "
              f"lr {c.get('lr')}, C={c.get('channels')}, seed {c.get('seed')}, "
              f"commit {c.get('git_commit')}, manifest {Path(c.get('manifest', '?')).name}")
    planned = json.loads(cfg.read_text()).get("epochs", 0) if cfg.exists() else 0
    rep = run / "val_report.json"                 # written when train.py finishes
    state = (" (in progress)" if not rep.exists()
             else " (stopped early)" if len(ep) < planned else "")
    print(f"   epochs run: {len(ep)}{state}; LR halved at epochs: {drops or 'none'}")
    for i, tag in ((best, "best"), (len(ep) - 1, "last")):
        print(f"   {tag:4s} epoch {ep[i]:3d}: "
              f"loss {h['loss'][i]:.4f} / val {h['val_loss'][i]:.4f}   "
              f"acc {h['acc'][i]:.4f} / val {h['val_acc'][i]:.4f}   "
              f"AUC {h['auc'][i]:.4f} / val {h['val_auc'][i]:.4f}")

    if rep.exists():
        r = json.loads(rep.read_text())
        print(f"   val report (best.keras, threshold {r['threshold']}):")
        for name, d in r["false_reject"].items():
            print(f"     false reject  ww/{name:12s} {d['rate']:.3f}  (n={d['n']})")
        for name, d in r["false_accept"].items():
            print(f"     false accept  {name:15s} {d['rate']:.3f}  (n={d['n']})")
        worst = sorted(r.get("false_accept_by_phrase", {}).items(), key=lambda kv: -kv[1]["rate"])[:5]
        if worst:
            print("     worst phrases: " + ", ".join(f"{k} {v['rate']:.3f}" for k, v in worst))
    print(f"   figure: {out}")


def main():
    ap = argparse.ArgumentParser(description="Plot training curves for train.py runs.")
    ap.add_argument("runs", nargs="*", type=Path, help="run folders (default: latest in runs/)")
    ap.add_argument("--out", type=Path, help="figure path (single run only; default <run>/training_curves.png)")
    args = ap.parse_args()

    runs = args.runs or [latest_run()]
    if args.out and len(runs) > 1:
        sys.exit("--out only works with a single run")
    for run in runs:
        h = load_history(run)
        out = args.out or run / "training_curves.png"
        ep, best, drops = plot(run, h, out)
        summary(run, h, ep, best, drops, out)


if __name__ == "__main__":
    main()
