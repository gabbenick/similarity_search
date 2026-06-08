"""
compare_models.py — RandomForest vs supervised SVM (RBF), apples to apples.

Answers the advisor question "why not an SVM?" with evidence instead of a
legacy claim. Reuses the EXACT same labels, negative sampling, and spatial-CV
folds as train_supervised.py — the only thing that changes is the classifier:

  • RandomForest (the production model)
  • SVC, RBF kernel, StandardScaler + class_weight='balanced' (a fair supervised
    SVM — NOT the old unsupervised One-Class SVM)

Metrics are out-of-fold on the labeled windows (favela vs sampled negatives, and
favela vs HARD urban negatives — the task that actually matters). Fast: only the
~n_pos*5 labeled windows are scored, not all 671k.

Run:  python compare_models.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

import config
from train_supervised import (rasterize_layer, window_overlap_fraction,
                              AGSN_TRUTH, SEED, POS_OVERLAP, NEG_HARD_FRAC,
                              NEG_RATIO, BLOCK_PX, N_FOLDS)


def build_design():
    """Replicate train_supervised's labels + sampling + folds (one source of truth)."""
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(config.FEATURES_CSV)
    feat_cols = [c for c in df.columns
                 if c not in ("row", "col", "map_x", "map_y", "similarity")]
    X = df[feat_cols].to_numpy(np.float32)
    rows = df["row"].to_numpy(); cols = df["col"].to_numpy()
    ws = config.WINDOW_SIZE
    has_socio = (df["has_socio"].to_numpy().astype(bool)
                 if "has_socio" in df else np.ones(len(df), bool))
    rmm_only = getattr(config, "RMM_ONLY", False) and "has_socio" in df
    np.nan_to_num(X, copy=False, nan=-1.0)

    with rasterio.open(config.HEATMAP_PATH) as src:
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)

    agsn_mask = rasterize_layer(AGSN_TRUTH, transform, crs, shape)
    frac = window_overlap_fraction(agsn_mask, rows, cols, ws)
    is_pos = frac >= POS_OVERLAP
    is_neg_eligible = frac == 0
    if rmm_only:
        is_pos = is_pos & has_socio
        is_neg_eligible = is_neg_eligible & has_socio

    band_means = [c for c in feat_cols if c.endswith("_mean")
                  and c.split("_")[0] in config.BAND_ORDER]
    brightness = df[band_means].to_numpy().mean(1)
    ndbi = df["NDBI_mean"].to_numpy() if "NDBI_mean" in df else brightness
    built = (pd.Series(brightness).rank(pct=True).to_numpy()
             + pd.Series(ndbi).rank(pct=True).to_numpy())

    n_pos = int(is_pos.sum()); n_neg = n_pos * NEG_RATIO
    n_hard = int(n_neg * NEG_HARD_FRAC)
    neg_all = np.where(is_neg_eligible)[0]
    hard_pool = neg_all[np.argsort(-built[neg_all])[: n_hard * 4]]
    hard_neg = rng.choice(hard_pool, size=min(n_hard, len(hard_pool)), replace=False)
    easy_pool = np.setdiff1d(neg_all, hard_neg)
    easy_neg = rng.choice(easy_pool, size=min(n_neg - len(hard_neg), len(easy_pool)),
                          replace=False)
    pos_idx = np.where(is_pos)[0]
    neg_idx = np.concatenate([hard_neg, easy_neg])
    train_idx = np.concatenate([pos_idx, neg_idx])
    y = np.zeros(len(df), np.int8); y[pos_idx] = 1

    n_block_cols = int(np.ceil(shape[1] / BLOCK_PX))
    block_id = (rows // BLOCK_PX) * n_block_cols + (cols // BLOCK_PX)
    uniq = np.unique(block_id); rng.shuffle(uniq)
    block_fold = {b: i % N_FOLDS for i, b in enumerate(uniq)}
    fold_of = np.vectorize(block_fold.get)(block_id)
    return X, y, train_idx, hard_neg, fold_of, n_pos


def oof_scores(model_factory, X, y, train_idx, fold_of):
    """Out-of-fold scores for the labeled windows, same CV as production."""
    oof = np.full(len(y), np.nan, np.float64)
    for f in range(N_FOLDS):
        test_block = fold_of == f
        tr = train_idx[~test_block[train_idx]]
        te = train_idx[test_block[train_idx]]
        if len(te) == 0:
            continue
        clf = model_factory()
        clf.fit(X[tr], y[tr])
        try:
            oof[te] = clf.predict_proba(X[te])[:, 1]
        except (AttributeError, NotImplementedError):
            oof[te] = clf.decision_function(X[te])   # SVM: rank scores are enough
    return oof


def report(name, oof, y, train_idx, hard_neg):
    m = ~np.isnan(oof[train_idx])
    yt = y[train_idx][m]; pp = oof[train_idx][m]
    roc_all = roc_auc_score(yt, pp); ap_all = average_precision_score(yt, pp)
    hard_set = set(hard_neg.tolist())
    sel = np.array([i in hard_set or y[i] == 1 for i in train_idx])
    idxs = train_idx[sel]; ok = ~np.isnan(oof[idxs])
    roc_h = roc_auc_score(y[idxs][ok], oof[idxs][ok])
    ap_h = average_precision_score(y[idxs][ok], oof[idxs][ok])
    print(f"  {name:<16} ROC {roc_all:.4f} | PR {ap_all:.4f}   ||   "
          f"vs HARD urban: ROC {roc_h:.4f} | PR {ap_h:.4f}")
    return roc_all, ap_all, roc_h, ap_h


def main():
    print("=" * 72)
    print("  RandomForest  vs  supervised RBF-SVM   (same labels, same spatial CV)")
    print("=" * 72)
    X, y, train_idx, hard_neg, fold_of, n_pos = build_design()
    print(f"  {n_pos:,} positives | {len(train_idx):,} labeled training windows\n")

    rf = lambda: RandomForestClassifier(
        n_estimators=300, min_samples_leaf=4, class_weight="balanced",
        n_jobs=-1, random_state=SEED)
    svc = lambda: make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced",
            random_state=SEED))

    print("           favela vs sampled negatives        favela vs HARD urban")
    print("  " + "-" * 68)
    report("RandomForest", oof_scores(rf, X, y, train_idx, fold_of),
           y, train_idx, hard_neg)
    report("RBF-SVM", oof_scores(svc, X, y, train_idx, fold_of),
           y, train_idx, hard_neg)
    print("\nDone.  (higher = better; the 'HARD urban' column is the one that matters)")


if __name__ == "__main__":
    main()
