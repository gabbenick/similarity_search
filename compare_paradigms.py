"""
compare_paradigms.py — the paper's headline contrast, on equal footing.

Unsupervised  vs  supervised, all scored on the SAME labeled windows (AGSN
positives + sampled hard/easy negatives) and the SAME spatial-CV folds:

  1. Cosine similarity to a reference favela   (the original approach; the
     'similarity' column already in features.csv — no labels used)
  2. One-Class SVM (RBF), fit on POSITIVES ONLY per training fold, scored on
     held-out labeled windows  (unsupervised novelty detection, on satellite —
     this is what the old 'SVM ROC ~0.5' claim never actually measured here)
  3. RandomForest (supervised)                  — the production model

The point: supervised classification with negatives is what makes favela-vs-
other-urban separable. Higher ROC/PR = better; the 'hard urban' column matters.

Run:  python compare_paradigms.py
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import config
from compare_models import build_design, oof_scores, report, N_FOLDS, SEED


def oof_oneclass(X, y, train_idx, fold_of, nu=0.1):
    """One-Class SVM fit on training-fold POSITIVES only; score held-out labels."""
    oof = np.full(len(y), np.nan, np.float64)
    for f in range(N_FOLDS):
        test_block = fold_of == f
        pos_tr = np.where((y == 1) & (~test_block))[0]      # positives, not in test
        te = train_idx[test_block[train_idx]]               # all labeled in test fold
        if len(te) == 0 or len(pos_tr) < 10:
            continue
        scaler = StandardScaler().fit(X[pos_tr])
        oc = OneClassSVM(nu=nu, kernel="rbf", gamma="scale")
        oc.fit(scaler.transform(X[pos_tr]))
        oof[te] = oc.decision_function(scaler.transform(X[te]))  # higher = favela-like
    return oof


def main():
    print("=" * 76)
    print("  PARADIGM COMPARISON — unsupervised vs supervised "
          "(same labels, same spatial CV)")
    print("=" * 76)
    X, y, train_idx, hard_neg, fold_of, n_pos = build_design()
    sim = pd.read_csv(config.FEATURES_CSV)["similarity"].to_numpy(np.float64)
    print(f"  {n_pos:,} positives | {len(train_idx):,} labeled windows\n")

    print("           favela vs sampled negatives        favela vs HARD urban")
    print("  " + "-" * 70)
    # 1. cosine similarity (no fitting to labels → evaluate directly)
    report("Cosine (unsup.)", sim, y, train_idx, hard_neg)
    # 2. one-class SVM, positives-only, spatial CV
    report("One-Class SVM", oof_oneclass(X, y, train_idx, fold_of, nu=config.SVM_NU),
           y, train_idx, hard_neg)
    # 3. supervised RandomForest (the production model)
    rf = lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, class_weight="balanced",
        n_jobs=-1, random_state=SEED)
    report("RandomForest", oof_scores(rf, X, y, train_idx, fold_of),
           y, train_idx, hard_neg)
    print("\nDone.  Supervision + negatives is the jump; the 'hard urban' column "
          "is the\n      favela-vs-other-urban task the unsupervised methods can't do.")


if __name__ == "__main__":
    main()
