"""
tune_precision.py — In-memory experiment to improve in-RMM object precision.

Scope = RMM only. Tests two precision levers WITHOUT touching the production
probability map:
  1. RMM-scoped training  — sample positives AND negatives only inside the RMM
     (has_socio==1), so the "hard" negatives become the RMM's own non-favela
     built-up (the actual confusers), not irrelevant interior-Alagoas land.
  2. Lower favela prior    — raise NEG_RATIO (negatives per positive) and/or drop
     class_weight="balanced"; training base rate (20% at ratio 4) is far above
     the ~0.2% real prevalence, so the model over-flags.

For each config it runs the same spatial-block CV as train_supervised.py, builds
the out-of-fold probability map, masks to the RMM, and reports in-RMM object
precision/recall (matching evaluate_rmm.py) plus AGSN detection.

Run:  python tune_precision.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from sklearn.ensemble import RandomForestClassifier
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

import config
from train_supervised import rasterize_layer, window_overlap_fraction

AGSN     = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
PROB_RMM = os.path.join(config.OUTPUT_DIR, "favela_probability_rmm.tif")
SEED = 42
POS_OVERLAP = 0.50
NEG_HARD_FRAC = 0.50
BLOCK_PX = 800
N_FOLDS = 5
MIN_REGION_HA = 0.5
HIT_FRAC = 0.25
SENTINEL = -1.0

RMM_CORE = {"Maceió"}
RMM_PERIPHERY = {"Marechal Deodoro", "Rio Largo", "Barra de Santo Antônio",
                 "Santa Luzia Do Norte", "Satuba", "Paripueira"}


def run_cv(X, y, fold_of, hard_neg, neg_idx, pos_idx, rng,
           neg_ratio, class_weight):
    """Spatial-CV out-of-fold probabilities for one sampling config."""
    n_pos = len(pos_idx)
    n_neg = min(n_pos * neg_ratio, len(neg_idx))
    n_hard = int(n_neg * NEG_HARD_FRAC)
    hard = hard_neg[:min(n_hard, len(hard_neg))]
    easy_pool = np.setdiff1d(neg_idx, hard, assume_unique=False)
    easy = rng.choice(easy_pool, size=min(n_neg - len(hard), len(easy_pool)),
                      replace=False)
    tr_idx = np.concatenate([pos_idx, hard, easy])

    oof = np.full(len(X), np.nan, np.float32)
    for f in range(N_FOLDS):
        test_block = fold_of == f
        tr = tr_idx[~test_block[tr_idx]]
        clf = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=4,
            class_weight=class_weight, n_jobs=-1, random_state=SEED)
        clf.fit(X[tr], y[tr])
        oof[test_block] = clf.predict_proba(X[test_block])[:, 1]
    return oof, len(tr_idx)


def object_metrics(prob_map, rmm_mask, truth, core_geoms, peri_geoms,
                   transform, shape, px_ha, thresholds=(0.70, 0.90)):
    probz = np.where(rmm_mask, np.nan_to_num(prob_map, nan=0.0), 0.0)
    min_px = max(int(MIN_REGION_HA / px_ha), 1)
    out = {}
    for t in thresholds:
        lab, n = ndimage.label(probz >= t)
        if n == 0:
            out[t] = (0, 0.0, 0.0, 0.0); continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = set((np.where(sizes >= min_px)[0] + 1).tolist())
        keepmask = np.isin(lab, list(keep))
        hit = set(np.unique(lab[truth & keepmask]).tolist()) - {0}
        prec = len(hit) / max(len(keep), 1)

        def recall(geoms):
            det = 0
            for g in geoms:
                if g is None or g.is_empty:
                    continue
                pm = rasterize([(g, 1)], out_shape=shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)
                if pm.any() and (keepmask & pm).sum() / pm.sum() >= HIT_FRAC:
                    det += 1
            return det / max(len(geoms), 1)
        out[t] = (len(keep), prec, recall(core_geoms), recall(peri_geoms))
    return out


def main():
    rng0 = np.random.default_rng(SEED)
    print("Loading features + grid ...")
    df = pd.read_csv(config.FEATURES_CSV)
    feat_cols = [c for c in df.columns
                 if c not in ("row", "col", "map_x", "map_y", "similarity")]
    X = df[feat_cols].to_numpy(np.float32)
    np.nan_to_num(X, copy=False, nan=SENTINEL)
    rows = df["row"].to_numpy(); cols = df["col"].to_numpy()
    has_socio = df["has_socio"].to_numpy().astype(bool)
    ws = config.WINDOW_SIZE

    with rasterio.open(config.HEATMAP_PATH) as src:
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    H, W = shape
    with rasterio.open(PROB_RMM) as src:
        rmm_mask = np.isfinite(src.read(1))

    # labels
    agsn = gpd.read_file(AGSN).to_crs(crs).reset_index(drop=True)
    agsn_mask = rasterize_layer(AGSN, transform, crs, shape)
    frac = window_overlap_fraction(agsn_mask, rows, cols, ws)
    is_pos = frac >= POS_OVERLAP
    is_neg_elig = frac == 0

    muni = agsn["AGSN_NM_MU"].fillna("?")
    sc = np.where(muni.isin(RMM_CORE), "core",
          np.where(muni.isin(RMM_PERIPHERY), "periphery", "outside"))
    rmm_agsn = agsn[sc != "outside"]
    truth = rasterize([(g, 1) for g in rmm_agsn.geometry if g and not g.is_empty],
                      out_shape=shape, transform=transform, fill=0,
                      dtype="uint8").astype(bool)
    core_geoms = list(agsn[sc == "core"].geometry)
    peri_geoms = list(agsn[sc == "periphery"].geometry)

    lat = abs(transform.f)
    import math
    px_ha = (abs(transform.a) * 111_320 * math.cos(math.radians(lat))
             * abs(transform.e) * 110_540) / 10_000

    # built-up score for hard negatives
    band_means = [c for c in feat_cols if c.endswith("_mean")
                  and c.split("_")[0] in config.BAND_ORDER]
    brightness = df[band_means].to_numpy().mean(1)
    ndbi = df["NDBI_mean"].to_numpy() if "NDBI_mean" in df else brightness
    built = (pd.Series(brightness).rank(pct=True).to_numpy()
             + pd.Series(ndbi).rank(pct=True).to_numpy())

    # spatial folds
    nbc = int(np.ceil(W / BLOCK_PX))
    block_id = (rows // BLOCK_PX) * nbc + (cols // BLOCK_PX)
    ub = np.unique(block_id); rng0.shuffle(ub)
    bf = {b: i % N_FOLDS for i, b in enumerate(ub)}
    fold_of = np.vectorize(bf.get)(block_id)

    configs = [
        dict(name="baseline (full img, r4, bal)", rmm=False, ratio=4, cw="balanced"),
        dict(name="RMM-scope, r4, bal",           rmm=True,  ratio=4, cw="balanced"),
        dict(name="RMM-scope, r8, bal",           rmm=True,  ratio=8, cw="balanced"),
        dict(name="RMM-scope, r15, bal",          rmm=True,  ratio=15, cw="balanced"),
        dict(name="RMM-scope, r8, none",          rmm=True,  ratio=8, cw=None),
    ]

    print(f"\n  windows={len(df):,} | in-RMM={has_socio.sum():,} | "
          f"positives(all)={is_pos.sum():,} | in-RMM pos={int((is_pos&has_socio).sum()):,}")
    print("\n" + "=" * 84)
    print(f"  {'config':<30}{'pos':>5}{'train':>7}{'  P@.7':>7}{'Rc@.7':>7}"
          f"{'Rp@.7':>7}{'  P@.9':>7}{'Rc@.9':>7}{'Rp@.9':>7}")
    print("=" * 84)

    for cfg in configs:
        rng = np.random.default_rng(SEED)
        pos_pool = is_pos & has_socio if cfg["rmm"] else is_pos
        neg_pool = is_neg_elig & has_socio if cfg["rmm"] else is_neg_elig
        pos_idx = np.where(pos_pool)[0]
        neg_all = np.where(neg_pool)[0]
        # hard negatives = most built-up, sorted (largest first)
        hard_sorted = neg_all[np.argsort(-built[neg_all])]
        y = np.zeros(len(df), np.int8); y[pos_idx] = 1

        oof, n_train = run_cv(X, y, fold_of, hard_sorted, neg_all,
                              pos_idx, rng, cfg["ratio"], cfg["cw"])

        # build prob map
        pf = np.nan_to_num(oof, nan=0.0)
        psum = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
        for (r, c), p in zip(zip(rows, cols), pf):
            psum[r:r+ws, c:c+ws] += p; cnt[r:r+ws, c:c+ws] += 1
        prob_map = psum / np.maximum(cnt, 1.0)

        m = object_metrics(prob_map, rmm_mask, truth, core_geoms, peri_geoms,
                           transform, shape, px_ha)
        (n7, p7, rc7, rp7) = m[0.70]
        (n9, p9, rc9, rp9) = m[0.90]
        print(f"  {cfg['name']:<30}{len(pos_idx):>5}{n_train:>7}"
              f"{p7*100:>6.1f}%{rc7*100:>6.1f}%{rp7*100:>6.1f}%"
              f"{p9*100:>6.1f}%{rc9*100:>6.1f}%{rp9*100:>6.1f}%")

    print("=" * 84)
    print("  P=precision  Rc=recall core  Rp=recall periphery  (@ threshold)")


if __name__ == "__main__":
    main()
