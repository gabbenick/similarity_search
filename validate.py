"""
validate.py — Validate the similarity heatmap against ground-truth favela polygons.

Compares output/heatmap.tif (or any raster) against a polygon shapefile of known
favelas (e.g. IBGE "aglomerados subnormais" or a manually digitised layer):

  1. Rasterises the polygons onto the heatmap grid  → ground-truth mask
  2. Threshold-independent quality:  ROC-AUC  and  PR-AUC (average precision)
  3. Picks an operating threshold (max-F1, or --threshold)
  4. Confusion matrix → precision / recall / F1 / overall accuracy / kappa
  5. Writes an error map (TP/FP/FN) GeoTIFF + ROC/PR curves PNG for QGIS

Usage
─────
    python validate.py --truth data/favelas.shp
    python validate.py --truth data/favelas.shp --threshold 0.6
    python validate.py --truth data/favelas.shp --sample 5000   # stratified
    python validate.py --truth data/favelas.shp --raster output/svm_map_score.tif

Requires:  geopandas, scikit-learn  (pip install geopandas scikit-learn)
"""

import os
import sys
import argparse
import numpy as np
import rasterio
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

try:
    import geopandas as gpd
except ImportError:
    sys.exit("❌ geopandas not installed.  pip install geopandas")

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
    confusion_matrix, cohen_kappa_score,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_raster(path):
    if not os.path.isfile(path):
        sys.exit(f"❌ Raster not found: {path}")
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        return arr, src.profile.copy(), src.transform, src.crs, (src.height, src.width)


def rasterize_truth(shp_path, crs, transform, shape):
    """Burn polygons onto the raster grid → uint8 mask (1=favela, 0=other)."""
    if not os.path.isfile(shp_path):
        sys.exit(f"❌ Shapefile not found: {shp_path}")
    gdf = gpd.read_file(shp_path)
    if gdf.empty:
        sys.exit("❌ Shapefile has no features.")
    if gdf.crs is None:
        sys.exit("❌ Shapefile has no CRS defined — set it in QGIS and re-export.")
    if crs is not None and gdf.crs != crs:
        print(f"   Reprojecting polygons {gdf.crs} → {crs}")
        gdf = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        sys.exit("❌ No valid geometries to rasterize.")
    mask = rasterize(shapes, out_shape=shape, transform=transform,
                     fill=0, dtype="uint8", all_touched=False)
    print(f"   Ground truth: {len(shapes)} polygons → "
          f"{int(mask.sum())} favela pixels ({100*mask.mean():.2f}% of grid)")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Sampling (mitigates spatial autocorrelation from 50% window overlap)
# ─────────────────────────────────────────────────────────────────────────────

def stratified_sample(y_true, y_score, n_per_class, seed=42):
    rng = np.random.default_rng(seed)
    out_t, out_s = [], []
    for cls in (0, 1):
        idx = np.where(y_true == cls)[0]
        if len(idx) == 0:
            continue
        take = min(n_per_class, len(idx))
        pick = rng.choice(idx, size=take, replace=False)
        out_t.append(y_true[pick]); out_s.append(y_score[pick])
    return np.concatenate(out_t), np.concatenate(out_s)


def best_f1_threshold(y_true, y_score):
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[i]), float(f1[i])


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def print_report(y_true, y_pred, auc, ap, threshold):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)          # producer's accuracy
    user_acc  = precision                       # user's accuracy (favela)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    oa        = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    iou       = tp / (tp + fp + fn + 1e-8)
    kappa     = cohen_kappa_score(y_true, y_pred)

    print("\n" + "=" * 58)
    print("  VALIDATION REPORT")
    print("=" * 58)
    print(f"  Threshold-independent:")
    print(f"     ROC-AUC                 : {auc:.4f}")
    print(f"     PR-AUC (avg precision)  : {ap:.4f}   ← key for imbalanced data")
    print(f"\n  Operating threshold       : {threshold:.4f}")
    print(f"\n  Confusion matrix (pixels):")
    print(f"                  pred 0      pred 1")
    print(f"     true 0   {tn:10d}  {fp:10d}")
    print(f"     true 1   {fn:10d}  {tp:10d}")
    print(f"\n  Overall accuracy          : {oa:.4f}")
    print(f"  Precision (user's, favela): {user_acc:.4f}")
    print(f"  Recall    (producer's)    : {recall:.4f}")
    print(f"  F1-score                  : {f1:.4f}")
    print(f"  IoU / Jaccard             : {iou:.4f}")
    print(f"  Cohen's kappa             : {kappa:.4f}")
    print("=" * 58)


def save_curves(y_true, y_score, auc, ap, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate",
              title="ROC curve"); ax[0].legend()
    ax[1].plot(rec, prec, label=f"AP = {ap:.3f}")
    base = y_true.mean()
    ax[1].axhline(base, ls="--", color="k", lw=0.8, label=f"baseline = {base:.3f}")
    ax[1].set(xlabel="Recall", ylabel="Precision",
              title="Precision-Recall curve"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=150, facecolor="white"); plt.close()
    print(f"  Saved → {path}")


def save_error_map(gt, pred_map, profile, path):
    """0=TN, 1=TP, 2=FP, 3=FN — load as paletted/unique values in QGIS."""
    err = np.zeros(gt.shape, dtype=np.uint8)
    err[(gt == 1) & (pred_map == 1)] = 1   # TP
    err[(gt == 0) & (pred_map == 1)] = 2   # FP
    err[(gt == 1) & (pred_map == 0)] = 3   # FN
    p = profile.copy()
    p.update(count=1, dtype="uint8", compress="lzw", nodata=None)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(err, 1)
    print(f"  Saved → {path}   (1=TP green, 2=FP red, 3=FN orange)")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Validate heatmap vs favela polygons")
    ap.add_argument("--truth", required=True, help="ground-truth polygon shapefile (.shp)")
    ap.add_argument("--raster", default=config.HEATMAP_PATH,
                    help="raster to evaluate (default: output/heatmap.tif)")
    ap.add_argument("--threshold", type=float, default=None,
                    help="binarisation threshold (default: max-F1)")
    ap.add_argument("--sample", type=int, default=None,
                    help="stratified random pixels per class (default: use all)")
    args = ap.parse_args()

    print("=" * 58)
    print("  Heatmap validation against ground truth")
    print("=" * 58)

    print(f"\n📂 Loading raster: {args.raster}")
    heat, profile, transform, crs, shape = load_raster(args.raster)

    print(f"📐 Rasterising ground truth: {args.truth}")
    gt = rasterize_truth(args.truth, crs, transform, shape)

    # ── flatten + drop invalid heatmap pixels ─────────────────────────────
    valid = np.isfinite(heat)
    y_true_full = gt[valid].astype(np.int32)
    y_score_full = heat[valid].astype(np.float32)

    if y_true_full.sum() == 0:
        sys.exit("❌ No favela pixels overlap the raster — check CRS/extent of the shapefile.")

    # ── metrics sample (optional) ─────────────────────────────────────────
    if args.sample:
        y_true, y_score = stratified_sample(y_true_full, y_score_full, args.sample)
        print(f"   Stratified sample: {len(y_true)} pixels "
              f"({int(y_true.sum())} favela)")
    else:
        y_true, y_score = y_true_full, y_score_full

    # ── threshold-independent ─────────────────────────────────────────────
    auc = roc_auc_score(y_true, y_score)
    ap_score = average_precision_score(y_true, y_score)

    # ── threshold ─────────────────────────────────────────────────────────
    if args.threshold is None:
        threshold, f1_at = best_f1_threshold(y_true, y_score)
        print(f"\n   Auto threshold (max F1={f1_at:.3f}): {threshold:.4f}")
    else:
        threshold = args.threshold

    y_pred = (y_score >= threshold).astype(np.int32)
    print_report(y_true, y_pred, auc, ap_score, threshold)

    # ── outputs (full grid) ───────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    save_curves(y_true, y_score, auc, ap_score,
                os.path.join(config.OUTPUT_DIR, "validation_curves.png"))
    pred_map = (heat >= threshold).astype(np.uint8)
    save_error_map(gt, pred_map, profile,
                   os.path.join(config.OUTPUT_DIR, "error_map.tif"))

    print("\n  Interpretation tips:")
    print("    • PR-AUC near the favela base-rate ⇒ heatmap barely beats chance.")
    print("    • Many FP (red) over construction/rural ⇒ spectral confusion.")
    print("    • Many FN (orange) over consolidated favelas ⇒ they look 'formal'.")
    print("    • Do NOT trust metrics if the shapefile overlaps the reference zone.\n")


if __name__ == "__main__":
    main()
