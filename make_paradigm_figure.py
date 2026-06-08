"""
make_paradigm_figure.py — bar chart of the paradigm comparison (for slides/paper).

Recomputes ROC for the three approaches (cosine / One-Class SVM / RandomForest)
on the same labeled windows + spatial CV, then plots them side by side with a
chance line. Output: output/fig_paradigm_comparison.png

Run:  python make_paradigm_figure.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

import config
from compare_models import build_design, oof_scores, report, SEED
from compare_paradigms import oof_oneclass


def main():
    X, y, train_idx, hard_neg, fold_of, n_pos = build_design()
    sim = pd.read_csv(config.FEATURES_CSV)["similarity"].to_numpy(np.float64)
    rf = lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, class_weight="balanced",
        n_jobs=-1, random_state=SEED)

    # report() returns (roc_all, ap_all, roc_hard, ap_hard) and prints a line
    methods = [
        ("Cosine similarity\n(unsupervised)",  sim),
        ("One-Class SVM\n(positives only)",     oof_oneclass(X, y, train_idx, fold_of,
                                                             nu=config.SVM_NU)),
        ("RandomForest\n(supervised)",          oof_scores(rf, X, y, train_idx, fold_of)),
    ]
    roc_all, roc_hard = [], []
    for name, oof in methods:
        ra, _, rh, _ = report(name.replace("\n", " "), oof, y, train_idx, hard_neg)
        roc_all.append(ra); roc_hard.append(rh)

    labels = [m[0] for m in methods]
    x = np.arange(len(labels)); w = 0.36
    colors_all, colors_hard = "#9ecae1", "#d6604d"

    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar(x - w/2, roc_all,  w, label="favela vs sampled negatives",
                color=colors_all, edgecolor="black", linewidth=0.6)
    b2 = ax.bar(x + w/2, roc_hard, w, label="favela vs HARD urban (the real task)",
                color=colors_hard, edgecolor="black", linewidth=0.6)
    ax.axhline(0.5, ls="--", c="gray", lw=1)
    ax.text(len(labels) - 0.5, 0.515, "chance (0.5)", color="gray",
            fontsize=9, ha="right")

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.012, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Favela vs other built-up: why supervision matters\n"
                 "(same labels, same spatial cross-validation)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = f"{config.OUTPUT_DIR}/fig_paradigm_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
