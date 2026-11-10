"""Logistic-regression baseline for the Butterworth wake-word CNN.

Same data, same augmentation, same validation set and same report as train.py,
but the model is a single logistic regression on time-pooled log-mel features:

  (124, 40) log-mel window -> per mel bin: mean, std, max over time -> 120 numbers
  -> standardise -> logistic regression (L2) -> P(Butterworth)

Pooling throws away word order, so this shows how much of the CNN's score comes
from temporal structure and how much from "speech-like spectrum vs not".
Needs only numpy + scipy (no TensorFlow, no scikit-learn).

  python ml/log_regression_baseline.py                  # 100 batches of 128
  python ml/log_regression_baseline.py --batches 400    # more training examples

Writes runs/baseline-<time>/: model.npz, val_report.json (same format as the
CNN's), config.json (arguments, git commit, train/val AUC and accuracy).
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features as F  # noqa: E402
import train as T  # noqa: E402  (only for val_report / git_commit; TensorFlow is not imported)


def pool(X):
    """(B, 124, 40) log-mel windows -> (B, 120): mean, std, max over time per mel bin."""
    return np.concatenate([X.mean(axis=1), X.std(axis=1), X.max(axis=1)], axis=1).astype(np.float64)


def auc(p, y):
    """ROC AUC from ranks (ties averaged)."""
    n1 = float((y == 1).sum())
    n0 = float(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def fit(Z, y, l2, max_iter=500):
    """Logistic regression by L-BFGS. Minimises mean log-loss + 0.5 * l2 * |w|^2 (bias not penalised)."""
    n, d = Z.shape

    def objective(wb):
        w, b = wb[:-1], wb[-1]
        z = Z @ w + b
        loss = np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w)
        g = expit(z) - y
        return loss, np.concatenate([Z.T @ g / n + l2 * w, [g.mean()]])

    res = minimize(objective, np.zeros(d + 1), jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter})
    return res.x[:-1], res.x[-1], res


def run(args):
    out = Path(args.out or F.ROOT / "runs" / time.strftime("baseline-%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    man = F.read_manifest(args.manifest, root=args.root)
    store = F.AudioStore()
    stats, long = F.annotate_spans(man, store)
    print(f"wake phrase spans: {stats}")
    if long:
        print(f"  {len(long)} wake clips longer than {F.MAX_PHRASE / F.SR:.2f} s are skipped as positives")
    parts = F.split(man)
    print(f"train {len(parts['train'])} clips, val {len(parts['val'])} clips")

    # Same objects train.py builds, so both models see the same data distribution.
    aug = F.build_augmenter(parts["train"], store, seed=args.seed)
    Xv, yv, vrows = F.eval_set(parts["val"], store, seed=args.seed)
    maker = F.BatchMaker(parts["train"], store, aug, args.batch, args.seed)

    print(f"building {args.batches} augmented batches of {args.batch} ...")
    t0 = time.time()

    def make_pooled(b):                      # pool right away: no need to keep the spectrograms
        X, y = maker.make(b)
        return pool(X), y

    with ThreadPoolExecutor(args.workers) as ex:
        made = list(ex.map(make_pooled, range(args.batches)))
    Ztr = np.concatenate([m[0] for m in made])
    ytr = np.concatenate([m[1] for m in made]).astype(np.float64)
    print(f"  {len(ytr)} examples ({ytr.mean():.1%} wake) in {time.time() - t0:.0f} s")

    mu, sd = Ztr.mean(axis=0), Ztr.std(axis=0) + 1e-6
    w, b, res = fit((Ztr - mu) / sd, ytr, args.l2)
    print(f"fit: {res.nit} L-BFGS iterations, converged={res.success}")

    def predict(X):
        return expit(((pool(X) - mu) / sd) @ w + b)

    p_tr = expit(((Ztr - mu) / sd) @ w + b)
    p_va = predict(Xv)
    metrics = {
        "train_auc": auc(p_tr, ytr), "train_acc": float(((p_tr >= 0.5) == ytr).mean()),
        "val_auc": auc(p_va, yv), "val_acc": float(((p_va >= 0.5) == yv).mean()),
    }

    np.savez(out / "model.npz", w=w, b=b, mu=mu, sd=sd)
    rep = T.val_report(p_va, yv, vrows)
    (out / "val_report.json").write_text(json.dumps(rep, indent=2))
    config = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              "out": str(out), "git_commit": T.git_commit(), "spans": stats,
              "n_train_examples": int(len(ytr)), "n_train_clips": len(parts["train"]),
              "n_val": len(vrows), "n_features": int(Ztr.shape[1]), "params": int(len(w) + 1),
              "lbfgs_iterations": int(res.nit), "converged": bool(res.success), **metrics}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\ntrain  AUC {metrics['train_auc']:.4f}  acc {metrics['train_acc']:.4f}")
    print(f"val    AUC {metrics['val_auc']:.4f}  acc {metrics['val_acc']:.4f}")
    print(f"\nval at threshold {rep['threshold']}:")
    for k, v in rep["false_reject"].items():
        print(f"  false reject  ww/{k:<14} {v['rate']:.3f}  (n={v['n']})")
    for k, v in rep["false_accept"].items():
        print(f"  false accept  {k:<17} {v['rate']:.3f}  (n={v['n']})")
    print(f"saved {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Logistic-regression baseline on time-pooled log-mels.")
    spans = F.ROOT / "dataset" / "manifest_v1_spans.csv"
    ap.add_argument("--manifest", type=Path, default=spans if spans.exists() else F.ROOT / "dataset" / "manifest_v1.csv",
                    help="default: manifest_v1_spans.csv (checked wake spans) if present, else manifest_v1.csv")
    ap.add_argument("--root", type=Path, default=F.ROOT, help="folder the manifest paths start from (repo root)")
    ap.add_argument("--batches", type=int, default=100, help="augmented training batches to generate")
    ap.add_argument("--batch", type=int, default=128, help="examples per batch")
    ap.add_argument("--l2", type=float, default=1e-3, help="L2 penalty on the weights")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="batches built in parallel")
    ap.add_argument("--out", type=Path, help="run folder (default: runs/baseline-<time>)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
