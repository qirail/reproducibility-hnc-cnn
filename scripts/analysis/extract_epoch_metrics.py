"""
Read the train / validation / test ROC AUC (with 95% bootstrap CI) of the
epoch whose weights were saved, directly from a training log.

How the training code works (hn_cnn/fit.py):
  * every epoch, the metrics for TRAIN_METRICS, VALIDATION and TESTING are
    written to the log under a line "Epoch <n>/<total>";
  * a checkpoint "model_<id>_<n>.pth.tar" is saved only when the validation
    AUC improves AND is above THRESHOLD AND |train AUC - val AUC| < MAX_DIFFERENCE.
  => the saved file with the HIGHEST epoch number is the selected model.

Usage (run from the repository root):

  # one log + an explicit epoch
  python scripts/analysis/extract_epoch_metrics.py --log LOG.txt --epoch 370

  # let the script find the selected (highest) epoch in a models folder
  python scripts/analysis/extract_epoch_metrics.py --log LOG.txt \
      --models results/statistical_reproducibility/seeds/seed_11/models \
      --prefix model_dm_5-fold_cv_fold0_

  # all 5 folds of a cross-validation run in one go
  python scripts/analysis/extract_epoch_metrics.py --cv \
      --logs results/statistical_reproducibility/seeds/seed_11/logs \
      --models results/statistical_reproducibility/seeds/seed_11/models \
      --log-stem log_dm_5-fold_cv --model-stem model_dm_5-fold_cv
"""
import argparse
import os
import re
import sys

import numpy as np

SUBSETS = ["TRAIN_METRICS", "VALIDATION", "TESTING"]
ROC_RE = re.compile(
    r"'ROC':\s*\{'AUC':\s*([0-9.eE+-]+).*?'LOWER_BOUND':\s*([0-9.eE+-]+),\s*'HIGHER_BOUND':\s*([0-9.eE+-]+)"
)


def selected_epoch(models_dir, prefix):
    """Highest epoch among files named <prefix><epoch>.pth.tar (None if none saved)."""
    if not os.path.isdir(models_dir):
        return None
    pat = re.compile(re.escape(prefix) + r"(\d+)\.pth\.tar$")
    epochs = [int(m.group(1)) for f in os.listdir(models_dir) if (m := pat.match(f))]
    return max(epochs) if epochs else None


def metrics_at_epoch(log_path, epoch):
    """Return {subset: (auc, ci_low, ci_high)} for the given epoch of a log file."""
    with open(log_path) as fh:
        lines = fh.read().splitlines()
    header = re.compile(rf"^Epoch {epoch}/\d+$")
    try:
        start = next(i for i, l in enumerate(lines) if header.match(l))
    except StopIteration:
        sys.exit(f"Epoch {epoch} not found in {log_path}")
    out, i = {}, start + 1
    while i < len(lines) and not lines[i].startswith("Epoch "):
        if lines[i] in SUBSETS and i + 1 < len(lines):
            m = ROC_RE.search(lines[i + 1])
            if m:
                out[lines[i]] = tuple(float(x) for x in m.groups())
        i += 1
    return out


def fmt(v):
    return f"{v[0]:.4f} [{v[1]:.4f}-{v[2]:.4f}]" if v else "n/a"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--models")
    ap.add_argument("--prefix", help="checkpoint file prefix, e.g. model_dm_ or model_dm_5-fold_cv_fold0_")
    ap.add_argument("--cv", action="store_true", help="summarise all folds of a cross-validation run")
    ap.add_argument("--logs", help="[--cv] folder with log_<stem>_fold<k>.txt")
    ap.add_argument("--log-stem", help="[--cv] e.g. log_dm_5-fold_cv")
    ap.add_argument("--model-stem", help="[--cv] e.g. model_dm_5-fold_cv")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    if not a.cv:
        epoch = a.epoch if a.epoch is not None else selected_epoch(a.models, a.prefix)
        if epoch is None:
            sys.exit("No saved checkpoint found: this run did not select a model.")
        m = metrics_at_epoch(a.log, epoch)
        print(f"Selected epoch: {epoch}")
        for s in SUBSETS:
            print(f"  {s:<14} ROC AUC {fmt(m.get(s))}")
        return

    rows = []
    print(f"{'Fold':<5} {'Saved':<6} {'Epoch':<6} {'Train AUC':<10} {'Val AUC':<10} {'Test AUC':<10}")
    for k in range(a.folds):
        epoch = selected_epoch(a.models, f"{a.model_stem}_fold{k}_")
        if epoch is None:
            print(f"{k:<5} {'No':<6} {'-':<6} {'excluded (no checkpoint saved)'}")
            continue
        m = metrics_at_epoch(os.path.join(a.logs, f"{a.log_stem}_fold{k}.txt"), epoch)
        r = [m[s][0] for s in SUBSETS]
        rows.append(r)
        print(f"{k:<5} {'Yes':<6} {epoch:<6} {r[0]:<10.4f} {r[1]:<10.4f} {r[2]:<10.4f}")
    if not rows:
        sys.exit("No fold saved a checkpoint.")
    arr = np.array(rows)
    print(f"\nFolds with a saved model: {len(rows)}/{a.folds}")
    for j, name in enumerate(["Train", "Validation", "Test"]):
        print(f"  {name:<11} mean {arr[:, j].mean():.4f}  range [{arr[:, j].min():.4f}, {arr[:, j].max():.4f}]")


if __name__ == "__main__":
    main()
