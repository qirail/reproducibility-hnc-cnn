"""Mean ROC plots (train / validation / external) for 5-fold cross-validation models (external validation).

Usage:
    python make_plots_cv_clean.py <predictions folder> <output folder>

Expected prediction files (only folds whose weights were saved need to exist):
    predictions_<run>_fold<k>_<SET>.csv      columns: label, prob
    e.g. predictions_dm_img_fold0_CMC_EXTERNAL.csv

Output, per model: ROC_<model>_5fold-CV-mean.png  (one mean ROC curve per cohort in SETS; shaded min-max band across folds for BAND_SETS)
Plus captions.txt and metrics.txt (per-fold AUCs, mean, SD). No calibration plots, no per-fold curves.
The mean ROC is the sensitivity averaged over the available folds on a common 1-specificity grid;
the legend gives the mean AUC with the range (min-max) of the fold AUCs; no 95% CI.
"""

import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ============================================================================
# 1. SETTINGS
# ============================================================================
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/statistical_reproducibility_weights/5_fold_cv"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/statistical_reproducibility/plots/5_fold_cv"

ROC_GRID = np.linspace(0, 1, 201)

# run name -> (file tag, outcome description, outcome short name, time frame, model input)
RUNS = {
    "dm_ann":  ("DM_clinical",          "distant metastasis at 2 years",   "DM",  "2 years", "clinical ANN"),
    "dm_img":  ("DM_imaging",           "distant metastasis at 2 years",   "DM",  "2 years", "imaging-only CNN"),
    "dm_cd":   ("DM_imaging+clinical",  "distant metastasis at 2 years",   "DM",  "2 years", "imaging + clinical CNN"),
    "lrf_ann": ("LRF_clinical",         "locoregional failure at 2 years", "LRF", "2 years", "clinical ANN"),
    "lrf_img": ("LRF_imaging",          "locoregional failure at 2 years", "LRF", "2 years", "imaging-only CNN"),
    "lrf_cd":  ("LRF_imaging+clinical", "locoregional failure at 2 years", "LRF", "2 years", "imaging + clinical CNN"),
    "os_ann":  ("OS_clinical",          "overall survival at 4 years", "OS", "4 years", "clinical ANN"),
    "os_img":  ("OS_imaging",           "overall survival at 4 years", "OS", "4 years", "imaging-only CNN"),
    "os_cd":   ("OS_imaging+clinical",  "overall survival at 4 years", "OS", "4 years", "imaging + clinical CNN"),
}

# csv suffix -> legend name (the key must match the end of the predictions file name)
SETS = {
    "TRAIN_METRICS": "HN-PET-CT training",
    "VALIDATION":    "HN-PET-CT validation",
    "HN1_TEST":    "HN1 test",   # not plotted
    # "CMC_EXTERNAL":  "HNC-RADIOMICS external",
}
BAND_SETS = ("HN1_TEST",)   # sets that get a shaded range band (add "VALIDATION"/"TRAIN_METRICS" to show more)
BAND_ALPHA = 0.14
BAND_SMOOTH = 15
COLOR = {"TRAIN_METRICS": "#23956F", "VALIDATION": "#1F5F99", "HN1_TEST": "#C4472B", "CMC_EXTERNAL": "#7B3FA0"}
INK, MUTED, GRID, DIAGONAL = "#1f2933", "#6b7280", "#e8eaed", "#9aa1ab"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.edgecolor": DIAGONAL, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
})


# ============================================================================
# 2. DATA AND STATISTICS
# ============================================================================
def load_folds(run, key):
    ## Return {fold number: (labels, probabilities)} for the folds that exist and contain both classes.
    folds = {}
    for path in glob.glob(os.path.join(IN_DIR, f"predictions_{run}_fold*_{key}.csv")):
        k = int(re.search(r"_fold(\d+)_", os.path.basename(path)).group(1))
        df = pd.read_csv(path)
        if df["label"].nunique() == 2:
            folds[k] = (df["label"].values.astype(int), df["prob"].values.astype(float))
    return dict(sorted(folds.items()))


def fold_tprs(folds, idx=None):
    ## Per-fold sensitivity on ROC_GRID and per-fold AUCs (optionally on a patient resample idx).
    tprs, aucs = [], []
    for y, p in folds.values():
        yy, pp = (y, p) if idx is None else (y[idx], p[idx])
        fpr, tpr, _ = roc_curve(yy, pp)
        t = np.interp(ROC_GRID, fpr, tpr)
        t[0] = 0.0
        tprs.append(t)
        aucs.append(roc_auc_score(yy, pp))
    return np.array(tprs), np.array(aucs)


def mean_roc(folds):
    ## Mean ROC over folds, per-fold AUCs, and the min/max sensitivity across folds (the range band).
    tprs, aucs = fold_tprs(folds)
    mean = tprs.mean(0)
    mean[-1] = 1.0
    return mean, aucs, tprs.min(0), tprs.max(0)


def smooth_band(v):
    pad = BAND_SMOOTH // 2
    return np.clip(np.convolve(np.pad(v, pad, mode="edge"), np.ones(BAND_SMOOTH) / BAND_SMOOTH, mode="valid"), 0, 1)


# ============================================================================
# 3. FIGURE
# ============================================================================
def plot_mean_roc(curves, title, name):
    ## curves = {set key: (mean tpr, per-fold AUCs, fold numbers, range lo, range hi)}
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)
    ax.plot([0, 1], [0, 1], color=DIAGONAL, lw=1, ls=(0, (1.5, 3)), zorder=1)

    for key, (mean, aucs, fold_ids, lo, hi) in curves.items():
        if key in BAND_SETS and len(aucs) > 1:
            ax.fill_between(ROC_GRID, smooth_band(lo), smooth_band(hi), color=COLOR[key], alpha=BAND_ALPHA, lw=0, zorder=2)
        auc_text = (f"Mean AUC {aucs.mean():.2f} (range {aucs.min():.2f}\u2013{aucs.max():.2f}; {len(aucs)} folds)"
                    if len(aucs) > 1 else f"AUC {aucs[0]:.2f} (1 fold)")
        ax.plot(ROC_GRID, mean, color=COLOR[key], lw=2.0, solid_capstyle="round", zorder=3,
                label=f"{SETS[key]}\n{auc_text}")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="center", fontweight="bold", fontsize=9, pad=8)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=False, edgecolor=GRID,
                       framealpha=0.96, labelspacing=0.9, handlelength=2.2)
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    captions, metrics = [], []

    for run, (tag, outcome, short, time_frame, model_input) in RUNS.items():
        curves = {}
        for key in SETS:
            folds = load_folds(run, key)
            if not folds:
                continue
            mean, aucs, lo, hi = mean_roc(folds)
            curves[key] = (mean, aucs, list(folds), lo, hi)
            y = next(iter(folds.values()))[0]
            metrics.append(f"{tag:22s} {key:13s} folds={list(folds)} n={len(y)} events={int(y.sum())} "
                           f"fold AUCs={[round(float(a), 3) for a in aucs]} mean AUC={aucs.mean():.3f} range={aucs.min():.3f}-{aucs.max():.3f}  "
                           f"table: {aucs.mean():.2f} ({aucs.min():.2f}-{aucs.max():.2f})")
        if not curves:
            continue

        name = f"ROC_{tag}_5fold-CV-mean"
        plot_mean_roc(curves, f"{short} at {time_frame}  ·  {model_input}", name)
        used = "; ".join(f"{SETS[k]}: folds {', '.join(map(str, v[2]))}" for k, v in curves.items())
        captions.append(f"{name}.png\nFigure S[x]. Mean ROC curve of the {model_input} for {outcome} from 5-fold "
                        f"cross-validation, evaluated on the HN-PET-CT training, HN-PET-CT validation and external cohorts. The curve is the sensitivity averaged "
                        f"across the folds with saved weights ({used}). The shaded band (external set) spans the minimum to maximum sensitivity across folds. Legend: mean AUC across folds with the range (minimum–maximum) of the fold AUCs. "
                        f"The dotted diagonal indicates chance performance.\n")

    with open(os.path.join(OUT_DIR, "captions.txt"), "w") as f:
        f.write("\n".join(captions))
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics) + "\n")
    print("Done")


if __name__ == "__main__":
    main()


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

"""
ROC plots (HN-PET-CT training / validation and HN1 test) for the statistical-reproducibility models.

Usage:
    python make_plots_stat.py <predictions folder> <output folder>
        arg 1 = folder with predictions_<run>_<set>.csv   (set = TRAIN_METRICS, VALIDATION, HN1_TEST)
        arg 2 = output folder

Output, per model:
    ROC_<model>_cohort-split_statistical-reproducibility.png
Plus captions.txt and metrics.txt in the output folder.


import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

warnings.filterwarnings("ignore")


# ============================================================================
# 1. SETTINGS
# ============================================================================
IN_DIR = sys.argv[1] if len(sys.argv) > 1 else "results/statistical_reproducibility_weights"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "results/statistical_reproducibility/plots"

WEIGHTS_TEXT = "models retrained with a different random seed (seed 11; statistical reproducibility)"
FILE_TAG = "statistical-reproducibility"

N_BOOT = 1000                       # bootstrap resamples for the 95% bands
SEED = 7651962
ROC_GRID = np.linspace(0, 1, 201)   # common 1-specificity grid for the ROC band

# run name -> (file tag, outcome description, outcome short name, time frame, model input)
RUNS = {
    "dm_img":  ("DM_imaging",           "distant metastasis at 2 years",   "DM",  "2 years", "imaging-only"),
    "dm_cd":   ("DM_imaging+clinical",  "distant metastasis at 2 years",   "DM",  "2 years", "imaging + clinical"),
    "lrf_img": ("LRF_imaging",          "locoregional failure at 2 years", "LRF", "2 years", "imaging-only"),
    "lrf_cd":  ("LRF_imaging+clinical", "locoregional failure at 2 years", "LRF", "2 years", "imaging + clinical"),
    "os_img":  ("OS_imaging",           "overall survival at 4 years",     "OS",  "4 years", "imaging-only"),
    "os_cd":   ("OS_imaging+clinical",  "overall survival at 4 years",     "OS",  "4 years", "imaging + clinical"),
}

# csv suffix -> panel title (the key must match the end of the predictions file name)
SETS = {
    "TRAIN_METRICS": "HN-PET-CT training",
    "VALIDATION":    "HN-PET-CT validation",
    "HN1_TEST":      "HN1 test",
}

# colour-blind-safe palette (all lines solid)
COLOR = {"TRAIN_METRICS": "#23956F", "VALIDATION": "#1F5F99", "HN1_TEST": "#C4472B"}
LINESTYLE = {k: "-" for k in SETS}
INK, MUTED, GRID, DIAGONAL = "#1f2933", "#6b7280", "#e8eaed", "#9aa1ab"

PANEL_WIDTH = 3.1   # inches per panel
BAND_SETS = ("HN1_TEST",)   # sets that get a shaded 95% band in the ROC graph; add "VALIDATION" / "TRAIN_METRICS" to show more
BAND_ALPHA = 0.14   # opacity of the 95% bands (lower = lighter)
BAND_SMOOTH = 15    # moving-average window (grid points) used only to smooth the drawn band edges

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.labelsize": 9.5, "axes.edgecolor": DIAGONAL, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "pdf.fonttype": 42, "savefig.facecolor": "white", "figure.facecolor": "white",
})

rng = np.random.default_rng(SEED)


# ============================================================================
# 2. STATISTICS
# ============================================================================
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def inv_logit(z):
    return 1 / (1 + np.exp(-z))


def fit_recalibration(y, p):
    ## Logistic recalibration  logit P(event) = intercept + slope * logit(p).  Returns (slope, intercept).
    model = LogisticRegression(C=1e6, max_iter=500).fit(logit(p).reshape(-1, 1), y)
    return float(model.coef_[0][0]), float(model.intercept_[0])


def wilson_interval(events, n, z=1.96):
    ## Wilson 95% confidence interval for a proportion.
    phat = events / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def bootstrap_indices(y, n_boot):
    ## Yield bootstrap resample indices that contain both classes.
    n = len(y)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            yield idx


def smooth_band(values):
    ## Moving average of a band edge (display only); the bootstrap values themselves are unchanged.
    pad = BAND_SMOOTH // 2
    padded = np.pad(values, pad, mode="edge")
    return np.clip(np.convolve(padded, np.ones(BAND_SMOOTH) / BAND_SMOOTH, mode="valid"), 0, 1)


def roc_bootstrap(y, p):
    ## 95% band for sensitivity on ROC_GRID, and 95% CI for the AUC.
    tprs, aucs = [], []
    for idx in bootstrap_indices(y, N_BOOT):
        fpr, tpr, _ = roc_curve(y[idx], p[idx])
        tprs.append(np.interp(ROC_GRID, fpr, tpr))
        aucs.append(roc_auc_score(y[idx], p[idx]))
    tprs = np.array(tprs)
    return np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), np.percentile(aucs, [2.5, 97.5])


def calibration_bootstrap(y, p, x):
    ## 95% band for the recalibration curve evaluated at predicted probabilities x.
    curves = []
    for idx in bootstrap_indices(y, N_BOOT // 2):
        slope, intercept = fit_recalibration(y[idx], p[idx])
        curves.append(inv_logit(intercept + slope * logit(x)))
    curves = np.array(curves)
    return np.percentile(curves, 2.5, 0), np.percentile(curves, 97.5, 0)


def calibration_bins(y, p):
    ## Mean prediction, observed fraction and Wilson CI in groups ordered by predicted probability.
    n_bins = 5 if len(y) >= 100 else 3
    groups = np.array_split(np.argsort(p), n_bins)
    mean_pred = np.array([p[g].mean() for g in groups])
    observed = np.array([y[g].mean() for g in groups])
    ci = np.array([wilson_interval(y[g].sum(), len(g)) for g in groups])
    return mean_pred, observed, ci


# ============================================================================
# 3. PLOT HELPERS
# ============================================================================
def style_axes(ax, grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(True, color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=3, width=0.8)


def square_unit_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def add_chance_line(ax):
    ax.plot([0, 1], [0, 1], color=DIAGONAL, lw=1, ls=(0, (1.5, 3)), zorder=1)


def add_stats_box(ax, text, x, y, ha, va):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=8, linespacing=1.5, zorder=5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=GRID, lw=0.8))


def save_figure(fig, name):
    fig.savefig(os.path.join(OUT_DIR, name + ".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 4. FIGURES
# ============================================================================
def plot_roc(data, title, name):
    ## One ROC graph with all sets. Returns {set key: (auc_lo, auc_hi)} so the metrics file matches the figure.
    auc_cis = {}
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    style_axes(ax)
    add_chance_line(ax)

    for key, (y, p) in data.items():
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        lower, upper, (auc_lo, auc_hi) = roc_bootstrap(y, p)
        auc_cis[key] = (auc_lo, auc_hi)

        if key in BAND_SETS:
            ax.fill_between(ROC_GRID, smooth_band(lower), smooth_band(upper), color=COLOR[key],
                            alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(fpr, tpr, color=COLOR[key], ls=LINESTYLE[key], lw=2.0, solid_capstyle="round", zorder=3,
                label=f"{SETS[key]}\nAUC {auc:.2f} (95% CI {auc_lo:.2f}–{auc_hi:.2f})")

    square_unit_axes(ax)
    ax.set_xlabel("1 − Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, loc="center", fontweight="bold", fontsize=9, pad=8)
    legend = ax.legend(loc="lower right", fontsize=8, frameon=True, fancybox=False, edgecolor=GRID,
                       framealpha=0.96, labelspacing=0.9, handlelength=2.2)
    legend.get_frame().set_linewidth(0.8)
    fig.tight_layout()
    save_figure(fig, name)
    return auc_cis


def plot_calibration(data, title, name):
    n = len(data)
    fig = plt.figure(figsize=(PANEL_WIDTH * n, 3.9))
    grid = GridSpec(2, n, height_ratios=[12, 1], hspace=0.05, wspace=0.12, figure=fig,
                    left=0.07, right=0.99, top=0.86, bottom=0.14)

    for col, (key, (y, p)) in enumerate(data.items()):
        ax = fig.add_subplot(grid[0, col])
        rug = fig.add_subplot(grid[1, col], sharex=ax)
        style_axes(ax)
        style_axes(rug, grid=False)
        add_chance_line(ax)

        # recalibration curve with bootstrap band, only over the range of observed predictions
        x = np.linspace(np.percentile(p, 1), np.percentile(p, 99), 100)
        slope, intercept = fit_recalibration(y, p)
        lower, upper = calibration_bootstrap(y, p, x)
        ax.fill_between(x, lower, upper, color=COLOR[key], alpha=BAND_ALPHA + 0.04, lw=0, zorder=2)
        ax.plot(x, inv_logit(intercept + slope * logit(x)), color=COLOR[key], ls=LINESTYLE[key], lw=2.0, zorder=3)

        # observed fractions in groups of patients, with Wilson 95% CI
        mean_pred, observed, ci = calibration_bins(y, p)
        ax.errorbar(mean_pred, observed, yerr=np.clip([observed - ci[:, 0], ci[:, 1] - observed], 0, None), fmt="o",
                    color=COLOR[key], elinewidth=1.1, capsize=2.5, ms=6, mec="white", mew=1.2, zorder=4)

        square_unit_axes(ax)
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_title(SETS[key], loc="center", fontweight="bold", fontsize=9, pad=6)
        add_stats_box(ax, f"Slope {slope:.2f}\nIntercept {intercept:.2f}\nBrier {brier_score_loss(y, p):.2f}",
                      0.04, 0.96, "left", "top")
        if col == 0:
            ax.set_ylabel("Observed event fraction")
        else:
            ax.set_yticklabels([])

        # ticks marking where each patient's prediction falls
        rug.vlines(p, 0, 1, color=COLOR[key], alpha=0.35, lw=0.7)
        rug.set_ylim(0, 1)
        rug.set_yticks([])
        rug.spines["left"].set_visible(False)
        rug.set_xlim(0, 1)
        rug.set_xticks(np.arange(0, 1.01, 0.2))
        rug.set_xlabel("Predicted probability")

    fig.suptitle(title, x=0.53, ha="center", fontweight="bold", fontsize=9.5, y=0.99)
    save_figure(fig, name)


# ============================================================================
# 5. CAPTIONS AND METRICS
# ============================================================================
def names_list(keys):
    ## 'A set', 'A set and B set' or 'A set, B set and C set'.
    names = [f"{SETS[k]} set" for k in keys]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def roc_caption(name, model_input, outcome, keys):
    bands = [k for k in keys if k in BAND_SETS]
    s = "s" if len(bands) > 1 else ""
    band_text = (f"The shaded band shows the bootstrap 95% confidence interval of the {names_list(bands)} "
                 f"curve{s} (1,000 resamples) and the dotted diagonal indicates chance performance. "
                 if bands else "The dotted diagonal indicates chance performance. ")
    return (f"{name}.png\nFigure S[x]. ROC curves of the {model_input} CNN for {outcome} (cohort split), using "
            f"{WEIGHTS_TEXT}, on the {names_list(keys)}. {band_text}AUC with 95% confidence interval is given "
            f"in the legend. The stepped shape of the curves reflects the limited number of events.\n")


def calibration_caption(name, model_input, outcome, keys):
    return (f"{name}.png\nFigure S[x]. Calibration of the {model_input} CNN for {outcome} (cohort split), using "
            f"{WEIGHTS_TEXT}, on the {names_list(keys)}. The solid line is the logistic recalibration curve of "
            f"observed event fraction against predicted probability, drawn over the range of observed "
            f"predictions, with its bootstrap 95% confidence band (shaded). Points show observed event fractions "
            f"in groups of patients ordered by predicted probability (5 groups; 3 where fewer than 100 patients) "
            f"with Wilson 95% confidence intervals. The dotted diagonal indicates perfect calibration. Ticks below "
            f"each panel mark the predicted probability of each patient. The models were trained with a "
            f"class-weighted loss, so predicted probabilities are not expected to match observed event rates.\n")


def metrics_line(tag, key, y, p, auc_ci):
    auc = roc_auc_score(y, p)
    auc_lo, auc_hi = auc_ci
    slope, intercept = fit_recalibration(y, p)
    return (f"{tag:22s} {key:13s} n={len(y):4d} events={int(y.sum()):3d} AUC={auc:.3f} "
            f"(95% CI {auc_lo:.3f}-{auc_hi:.3f}) "
            f"slope={slope:.2f} intercept={intercept:.2f} Brier={brier_score_loss(y, p):.3f}")


# ============================================================================
# 6. MAIN
# ============================================================================
def load_predictions(run):
    ## Return {set key: (labels, probabilities)} for the sets that exist and contain both classes.
    data = {}
    for key in SETS:
        path = os.path.join(IN_DIR, f"predictions_{run}_{key}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        if df["label"].nunique() == 2:
            data[key] = (df["label"].values.astype(int), df["prob"].values.astype(float))
    return data


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    captions, metrics = [], []

    for run, (tag, outcome, short, time_frame, model_input) in RUNS.items():
        data = load_predictions(run)
        if not data:
            continue
        title = f"{short} at {time_frame}  ·  {model_input} CNN"

        roc_name = f"ROC_{tag}_cohort-split_{FILE_TAG}"
        auc_cis = plot_roc(data, title, roc_name)
        captions.append(roc_caption(roc_name, model_input, outcome, list(data)))


        metrics += [metrics_line(tag, key, y, p, auc_cis[key]) for key, (y, p) in data.items()]

    with open(os.path.join(OUT_DIR, "captions.txt"), "w") as f:
        f.write("\n".join(captions))
    with open(os.path.join(OUT_DIR, "metrics.txt"), "w") as f:
        f.write("\n".join(metrics) + "\n")
    print("Done")


if __name__ == "__main__":
    main()
"""