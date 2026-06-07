"""
make_figures.py — Thesis/report figures for the supervised favela model.

Reads artifacts written by train_supervised.py (output/oof_eval.npz) and
evaluate_rmm.py (output/rmm_object_scores.csv) and produces:

  output/fig_roc_pr.png            ROC + precision-recall (window-level, OOF)
  output/fig_calibration.png       reliability diagram (are probabilities honest?)
  output/fig_feature_importance.png top-15 RandomForest feature importances
  output/fig_object_operating.png  object precision/recall vs threshold (raw+filtered)

Run (after train_supervised.py and evaluate_rmm.py):  python make_figures.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                             average_precision_score)
from sklearn.calibration import calibration_curve

import config

NPZ = os.path.join(config.OUTPUT_DIR, "oof_eval.npz")
OBJ = os.path.join(config.OUTPUT_DIR, "rmm_object_scores.csv")
DPI = 150


def load_labeled():
    d = np.load(NPZ, allow_pickle=True)
    oof, y, tr = d["oof_prob"], d["y"], d["train_idx"]
    p = oof[tr]; yt = y[tr]
    m = np.isfinite(p)
    return yt[m].astype(int), p[m], d["importances"], list(d["feat_cols"])


def fig_roc_pr(y, p):
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="ROC — favela vs. sampled negatives (out-of-fold)")
    ax[0].legend(loc="lower right"); ax[0].grid(alpha=.3)
    ax[1].plot(rec, prec, color="#2c3e50", lw=2, label=f"PR (AP={ap:.3f})")
    ax[1].axhline(y.mean(), ls="--", color="gray", lw=1,
                  label=f"base rate={y.mean():.2f}")
    ax[1].set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall (window-level)")
    ax[1].legend(loc="lower left"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "fig_roc_pr.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); return out


def fig_calibration(y, p):
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#c0392b", lw=2, label="RandomForest")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed favela fraction",
           title="Calibration (reliability diagram)")
    ax.legend(loc="upper left"); ax.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "fig_calibration.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); return out


def fig_importance(importances, feat_cols, k=15):
    order = np.argsort(importances)[-k:]
    names = [feat_cols[i] for i in order]
    vals = importances[order]
    colors = ["#27ae60" if any(s in n for s in
              ("vuln", "lowincome", "flood", "landslide", "has_socio"))
              else "#e67e22" if "DSM" in n or "slope" in n
              else "#34495e" for n in names]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set(xlabel="Mean RF importance", title="Top feature importances")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#27ae60", label="socio/env"),
                       Patch(color="#e67e22", label="terrain"),
                       Patch(color="#34495e", label="spectral")],
              loc="lower right")
    ax.grid(alpha=.3, axis="x"); fig.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "fig_feature_importance.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); return out


def fig_object_operating():
    if not os.path.isfile(OBJ):
        print("  (skip object figure — run evaluate_rmm.py first)"); return None
    df = pd.read_csv(OBJ).sort_values("threshold")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df.threshold, df.precision*100, "o-", color="#c0392b",
            label="precision (raw)")
    if "precision_filt" in df:
        ax.plot(df.threshold, df.precision_filt*100, "s--", color="#c0392b",
                alpha=.6, label="precision (filtered)")
    rec_all = (df.recall_core + df.recall_peri) / 2 * 100
    ax.plot(df.threshold, df.recall_core*100, "o-", color="#2980b9",
            label="recall (core)")
    ax.plot(df.threshold, df.recall_peri*100, "^-", color="#16a085",
            label="recall (periphery)")
    ax.set(xlabel="Probability threshold", ylabel="%",
           title="Object-level precision & recall inside the RMM")
    ax.legend(); ax.grid(alpha=.3); ax.set_ylim(0, 100)
    fig.tight_layout()
    out = os.path.join(config.OUTPUT_DIR, "fig_object_operating.png")
    fig.savefig(out, dpi=DPI); plt.close(fig); return out


def main():
    print("Loading out-of-fold artifacts ...")
    y, p, importances, feat_cols = load_labeled()
    print(f"  labeled windows: {len(y):,} (base rate {y.mean():.3f})")
    outs = [fig_roc_pr(y, p), fig_calibration(y, p),
            fig_importance(importances, feat_cols), fig_object_operating()]
    print("\nSaved figures:")
    for o in outs:
        if o:
            print(f"  {o}")
    print("\nDone.")


if __name__ == "__main__":
    main()
