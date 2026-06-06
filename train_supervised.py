"""
train_supervised.py — Supervised favela classifier (the real fix).

Why this exists
───────────────
The unsupervised similarity/One-Class approach can only learn "what favelas
look like" with no counter-examples, so it collapses into a generic
bright/built-up detector and confuses favelas with other urban areas.

With the IBGE AGSN layer we now have 273 comprehensive favela polygons, which
means "built-up but NOT in AGSN" is a TRUSTWORTHY negative. That lets us train
a binary classifier on the real, hard task: favela vs. other-urban.

What it does
────────────
  1. Reuses output/features.csv (671k windows × 25 features) — no re-extraction
  2. Labels each window by its overlap fraction with AGSN (summed-area table)
  3. Samples HARD negatives (bright/built-up, non-AGSN) + easy negatives
  4. Trains a RandomForest with SPATIAL block cross-validation (no leakage)
  5. Reports honest favela-vs-all and favela-vs-urban metrics
  6. Independently tests on hand-drawn communities AGSN does not cover
  7. Writes a full-image, fully out-of-fold probability GeoTIFF for QGIS

Run:  python train_supervised.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score

import config

AGSN_TRUTH   = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
HANDDRAWN    = os.path.join(config.DATA_DIR, "handdrawn_test.gpkg")  # independent test (all hand-drawn communities)
PROB_MAP_OUT = os.path.join(config.OUTPUT_DIR, "favela_probability.tif")
SEED         = 42

POS_OVERLAP    = 0.50   # window is favela if >=50% inside AGSN
NEG_HARD_FRAC  = 0.50   # half of negatives drawn from bright/built-up areas
NEG_RATIO      = 4      # negatives per positive
BLOCK_PX       = 800    # spatial CV block size (~8 km) — bigger than any polygon
N_FOLDS        = 5


# ─────────────────────────────────────────────────────────────────────────────
def rasterize_layer(path, transform, crs, shape):
    gdf = gpd.read_file(path)
    if gdf.crs is not None and str(gdf.crs) != str(crs):
        gdf = gdf.to_crs(crs)
    shapes = [(g, 1) for g in gdf.geometry if g is not None and not g.is_empty]
    return rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype="uint8").astype(np.uint8)


def window_overlap_fraction(mask, rows, cols, ws):
    """Exact fraction of each window covered by mask, via summed-area table."""
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), dtype=np.int64)
    ii[1:, 1:] = mask.cumsum(0).cumsum(1)
    r0, c0 = rows, cols
    r1, c1 = rows + ws, cols + ws
    counts = (ii[r1, c1] - ii[r0, c1] - ii[r1, c0] + ii[r0, c0])
    return counts / float(ws * ws)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)
    print("=" * 60)
    print("  Supervised favela classifier  (AGSN-trained, spatial CV)")
    print("=" * 60)

    # ── features + grid ───────────────────────────────────────────────────
    print("\nLoading features.csv ...")
    df = pd.read_csv(config.FEATURES_CSV)
    feat_cols = [c for c in df.columns
                 if c not in ("row", "col", "map_x", "map_y", "similarity")]
    X = df[feat_cols].to_numpy(np.float32)
    rows = df["row"].to_numpy(); cols = df["col"].to_numpy()
    ws = config.WINDOW_SIZE
    print(f"  {len(df):,} windows × {len(feat_cols)} features")

    with rasterio.open(config.HEATMAP_PATH) as src:
        transform, crs, shape, profile = (src.transform, src.crs,
                                          (src.height, src.width), src.profile.copy())

    # ── labels from AGSN ──────────────────────────────────────────────────
    print("\nLabelling windows against AGSN ...")
    agsn_mask = rasterize_layer(AGSN_TRUTH, transform, crs, shape)
    frac = window_overlap_fraction(agsn_mask, rows, cols, ws)
    is_pos  = frac >= POS_OVERLAP
    is_amb  = (frac > 0) & (frac < POS_OVERLAP)     # ambiguous edge — exclude
    is_neg_eligible = frac == 0
    print(f"  positives (>= {POS_OVERLAP:.0%} AGSN): {is_pos.sum():,}")
    print(f"  ambiguous edge (excluded)           : {is_amb.sum():,}")
    print(f"  negative-eligible (0% AGSN)         : {is_neg_eligible.sum():,}")

    # ── hard negatives: bright / built-up, non-AGSN ───────────────────────
    band_means = [c for c in feat_cols if c.endswith("_mean")
                  and c.split("_")[0] in config.BAND_ORDER]
    brightness = df[band_means].to_numpy().mean(1)
    ndbi = df["NDBI_mean"].to_numpy() if "NDBI_mean" in df else brightness
    built_score = (pd.Series(brightness).rank(pct=True).to_numpy()
                   + pd.Series(ndbi).rank(pct=True).to_numpy())

    n_pos = int(is_pos.sum())
    n_neg = n_pos * NEG_RATIO
    n_hard = int(n_neg * NEG_HARD_FRAC)

    neg_idx_all = np.where(is_neg_eligible)[0]
    # hard: most built-up non-favela windows
    hard_pool = neg_idx_all[np.argsort(-built_score[neg_idx_all])[: n_hard * 4]]
    hard_neg  = rng.choice(hard_pool, size=min(n_hard, len(hard_pool)), replace=False)
    # easy: random non-favela (water/veg/rural)
    easy_pool = np.setdiff1d(neg_idx_all, hard_neg, assume_unique=False)
    easy_neg  = rng.choice(easy_pool, size=min(n_neg - len(hard_neg), len(easy_pool)),
                           replace=False)
    neg_idx = np.concatenate([hard_neg, easy_neg])
    pos_idx = np.where(is_pos)[0]
    print(f"  training negatives: {len(hard_neg):,} hard + {len(easy_neg):,} easy")

    train_idx = np.concatenate([pos_idx, neg_idx])
    y = np.zeros(len(df), np.int8); y[pos_idx] = 1
    y_train = y[train_idx]

    # ── spatial blocks (group id per window) ──────────────────────────────
    n_block_cols = int(np.ceil(shape[1] / BLOCK_PX))
    block_id = (rows // BLOCK_PX) * n_block_cols + (cols // BLOCK_PX)
    uniq_blocks = np.unique(block_id)
    rng.shuffle(uniq_blocks)
    block_fold = {b: i % N_FOLDS for i, b in enumerate(uniq_blocks)}
    fold_of = np.vectorize(block_fold.get)(block_id)
    print(f"\nSpatial CV: {len(uniq_blocks)} blocks of {BLOCK_PX}px → {N_FOLDS} folds")

    # ── cross-validated training: predict EVERY window out-of-fold ────────
    oof_prob = np.full(len(df), np.nan, np.float32)
    importances = np.zeros(len(feat_cols))
    for f in range(N_FOLDS):
        test_block = fold_of == f
        tr = train_idx[~test_block[train_idx]]          # labeled, not in test blocks
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        # predict ALL windows whose block is in this test fold
        oof_prob[test_block] = clf.predict_proba(X[test_block])[:, 1]
        importances += clf.feature_importances_
        print(f"  fold {f+1}/{N_FOLDS}: trained on {len(tr):,} | "
              f"predicted {test_block.sum():,} windows")
    importances /= N_FOLDS

    # ── honest metrics on labeled windows (out-of-fold) ───────────────────
    m = ~np.isnan(oof_prob[train_idx])
    yt = y_train[m]; pp = oof_prob[train_idx][m]
    print("\n" + "=" * 60)
    print("  SPATIAL-CV METRICS  (favela vs sampled negatives)")
    print("=" * 60)
    print(f"  ROC-AUC : {roc_auc_score(yt, pp):.4f}")
    print(f"  PR-AUC  : {average_precision_score(yt, pp):.4f}   "
          f"(base rate {yt.mean():.3f})")

    # favela vs HARD (urban) negatives only — the task that actually matters
    hard_set = set(hard_neg.tolist())
    sel = np.array([i in hard_set or yi == 1
                    for i, yi in zip(train_idx, y_train)])
    selm = sel & np.r_[m] if False else sel  # keep simple
    idxs = train_idx[sel]
    yv = y[idxs]; pv = oof_prob[idxs]
    ok = ~np.isnan(pv)
    print(f"\n  favela vs HARD urban negatives only:")
    print(f"  ROC-AUC : {roc_auc_score(yv[ok], pv[ok]):.4f}")
    print(f"  PR-AUC  : {average_precision_score(yv[ok], pv[ok]):.4f}   "
          f"(base rate {yv[ok].mean():.3f})")

    # ── independent test: hand-drawn communities AGSN misses ──────────────
    hd_mask = rasterize_layer(HANDDRAWN, transform, crs, shape)
    hd_frac = window_overlap_fraction(hd_mask, rows, cols, ws)
    hd_pos = hd_frac >= POS_OVERLAP
    # only score where we have OOF preds and exclude AGSN-overlapping ones
    hd_only = hd_pos & (frac == 0) & ~np.isnan(oof_prob)
    if hd_only.sum() > 5:
        bg = (frac == 0) & ~hd_pos & ~np.isnan(oof_prob)
        bg_samp = rng.choice(np.where(bg)[0], size=min(50000, bg.sum()), replace=False)
        yy = np.r_[np.ones(hd_only.sum()), np.zeros(len(bg_samp))]
        pp2 = np.r_[oof_prob[hd_only], oof_prob[bg_samp]]
        print(f"\n  INDEPENDENT TEST — hand-drawn favelas NOT in AGSN "
              f"({hd_only.sum()} windows):")
        print(f"  ROC-AUC : {roc_auc_score(yy, pp2):.4f}   "
              f"(recall of these unseen settlements)")

    # ── feature importances ───────────────────────────────────────────────
    print("\n  Top features (RF importance):")
    for i in np.argsort(-importances)[:10]:
        print(f"    {feat_cols[i]:<18} {importances[i]:.3f}")

    # ── full-image probability map (out-of-fold) ──────────────────────────
    print("\nBuilding probability heatmap ...")
    H, W = shape
    psum = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    pf = np.nan_to_num(oof_prob, nan=0.0)
    for (r, c), p in zip(zip(rows, cols), pf):
        psum[r:r+ws, c:c+ws] += p; cnt[r:r+ws, c:c+ws] += 1
    prob_map = psum / np.maximum(cnt, 1.0)
    profile.update(count=1, dtype="float32", compress="lzw", nodata=None)
    with rasterio.open(PROB_MAP_OUT, "w", **profile) as dst:
        dst.write(prob_map.astype(np.float32), 1)
    print(f"  Saved → {PROB_MAP_OUT}")
    print("\nDone.")


if __name__ == "__main__":
    main()
