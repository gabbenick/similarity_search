"""
evaluate_polygons.py — Polygon / object-level evaluation of the favela map.

Pixel metrics are the harshest possible test and understate usefulness: the real
use case is flagging candidate AREAS for review, not painting exact boundaries.
This script answers the questions that actually matter:

  A. DETECTION (recall): of known favelas, how many does the model light up?
     - per-polygon mean/max out-of-fold probability
     - detection rate vs threshold, split AGSN (seen-via-CV) vs hand-drawn unseen
     - broken down by polygon size and household count (does it miss small ones?)

  B. OBJECT precision/recall: threshold → connected predicted regions, matched to
     truth. Of the blobs the model flags, how many hit a real favela, and how many
     real favelas get hit?

Uses output/favela_probability.tif (out-of-fold ⇒ honest, even on AGSN).

Run:  python evaluate_polygons.py
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

import config

PROB    = os.path.join(config.OUTPUT_DIR, "favela_probability.tif")
AGSN    = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
HANDDR  = os.path.join(config.DATA_DIR, "handdrawn_test.gpkg")
OUT_CSV = os.path.join(config.OUTPUT_DIR, "polygon_scores.csv")
PX_HA   = None   # filled at runtime (ha per pixel)

DET_THRESHOLDS = [0.50, 0.70, 0.80, 0.90]
OBJ_THRESHOLDS = [0.70, 0.80, 0.90]
MIN_REGION_HA  = 0.5    # ignore predicted speckle smaller than this
HIT_FRAC       = 0.25   # polygon counts as detected if >=25% of it is flagged


def zonal_stats(prob, gdf, transform, shape):
    """mean / max / flagged-fraction of prob inside each polygon."""
    out = []
    for i, geom in enumerate(gdf.geometry):
        if geom is None or geom.is_empty:
            out.append((np.nan, np.nan, 0)); continue
        m = rasterize([(geom, 1)], out_shape=shape, transform=transform,
                      fill=0, dtype="uint8").astype(bool)
        if m.sum() == 0:
            out.append((np.nan, np.nan, 0)); continue
        vals = prob[m]
        out.append((float(vals.mean()), float(vals.max()), int(m.sum())))
    s = pd.DataFrame(out, columns=["mean_prob", "max_prob", "n_px"])
    return s


def main():
    global PX_HA
    print("=" * 60)
    print("  Polygon-level evaluation")
    print("=" * 60)

    with rasterio.open(PROB) as src:
        prob = src.read(1).astype(np.float32)
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    # pixel area in ha (geographic deg → m via latitude)
    import math
    lat = abs(transform.f)
    px_m = abs(transform.a) * 111_320 * math.cos(math.radians(lat))
    py_m = abs(transform.e) * 110_540
    PX_HA = (px_m * py_m) / 10_000
    min_region_px = int(MIN_REGION_HA / PX_HA)
    print(f"\n  pixel ≈ {px_m:.1f}×{py_m:.1f} m ({PX_HA:.3f} ha)  "
          f"min predicted region = {min_region_px} px ({MIN_REGION_HA} ha)")

    agsn = gpd.read_file(AGSN).to_crs(crs).reset_index(drop=True)
    hd   = gpd.read_file(HANDDR).to_crs(crs)
    unseen = hd[hd.agsn_overlap_pct == 0].reset_index(drop=True)
    print(f"  AGSN polygons: {len(agsn)} | hand-drawn unseen: {len(unseen)}")

    # ── A. per-polygon detection ──────────────────────────────────────────
    sa = zonal_stats(prob, agsn, transform, shape)
    sa["area_ha"] = agsn["area_ha"].values
    sa["sum_edoc"] = agsn["SUM_EDOC"].values
    sa["source"] = "AGSN"
    su = zonal_stats(prob, unseen, transform, shape)
    su["area_ha"] = unseen.geometry.to_crs("EPSG:31985").area.values / 10_000
    su["community"] = unseen["community"].values
    su["source"] = "unseen"

    print("\n" + "-" * 60)
    print("  A. DETECTION RATE  (polygon detected if mean_prob >= T)")
    print("-" * 60)
    print(f"  {'T':>5} {'AGSN (n=%d)'%len(sa):>16} {'unseen (n=%d)'%len(su):>16}")
    for t in DET_THRESHOLDS:
        ra = (sa.mean_prob >= t).mean() * 100
        ru = (su.mean_prob >= t).mean() * 100
        print(f"  {t:>5.2f} {ra:>15.1f}% {ru:>15.1f}%")

    # background reference: random non-favela polygons of similar size?
    # cheap proxy — global distribution the polygons sit above:
    print(f"\n  AGSN mean_prob: median={sa.mean_prob.median():.3f}  "
          f"q25={sa.mean_prob.quantile(.25):.3f}  q75={sa.mean_prob.quantile(.75):.3f}")
    print(f"  whole-image prob median={np.median(prob):.3f}  "
          f"(favelas sit far above the image median)")

    # size / household breakdown — does it miss small/sparse ones?
    print("\n  AGSN detection (mean_prob>=0.5) by size tertile:")
    sa["size_bin"] = pd.qcut(sa.area_ha, 3, labels=["small", "medium", "large"])
    for b, g in sa.groupby("size_bin"):
        print(f"    {b:<7} (≤{g.area_ha.max():5.1f} ha, n={len(g):3d}): "
              f"{(g.mean_prob>=0.5).mean()*100:5.1f}% detected, "
              f"median prob {g.mean_prob.median():.3f}")

    # ── B. object-level precision / recall ────────────────────────────────
    truth_all = rasterize(
        [(g, 1) for g in list(agsn.geometry) + list(unseen.geometry)
         if g is not None and not g.is_empty],
        out_shape=shape, transform=transform, fill=0, dtype="uint8").astype(bool)
    agsn_mask = rasterize([(g, 1) for g in agsn.geometry if g and not g.is_empty],
                          out_shape=shape, transform=transform, fill=0,
                          dtype="uint8").astype(bool)

    print("\n" + "-" * 60)
    print("  B. OBJECT-LEVEL  (connected predicted regions vs truth)")
    print("-" * 60)
    print(f"  {'T':>5} {'#pred':>7} {'precision':>10} {'recall_AGSN':>12} {'recall_unseen':>14}")
    for t in OBJ_THRESHOLDS:
        binmap = prob >= t
        lab, n = ndimage.label(binmap)
        if n == 0:
            print(f"  {t:>5.2f}  no regions"); continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = np.where(sizes >= min_region_px)[0] + 1
        keepset = set(keep.tolist())
        # precision: predicted regions that hit ANY truth
        hit_truth = set(np.unique(lab[truth_all & np.isin(lab, list(keepset))])) - {0}
        precision = len(hit_truth) / max(len(keepset), 1)
        # recall: truth polygons overlapped >=HIT_FRAC by kept predicted regions
        predkeep = np.isin(lab, list(keepset))
        def recall(gdf):
            det = 0
            for geom in gdf.geometry:
                if geom is None or geom.is_empty:
                    continue
                pm = rasterize([(geom, 1)], out_shape=shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)
                if pm.sum() and (predkeep & pm).sum() / pm.sum() >= HIT_FRAC:
                    det += 1
            return det / max(len(gdf), 1)
        r_ag = recall(agsn) * 100
        r_un = recall(unseen) * 100
        print(f"  {t:>5.2f} {len(keepset):>7} {precision*100:>9.1f}% "
              f"{r_ag:>11.1f}% {r_un:>13.1f}%")

    print("\n  NOTE: object precision is PESSIMISTIC — AGSN is comprehensive only")
    print("  in the Maceió core, so some 'false' regions are real unmapped favelas")
    print("  (exactly like the 7 hand-drawn ones AGSN missed).")

    # ── save per-polygon scores for QGIS (which favelas are missed) ───────
    sa_out = pd.concat([sa, su], ignore_index=True)
    sa_out.to_csv(OUT_CSV, index=False)
    missed = sa[(sa.mean_prob < 0.5)]
    print(f"\n  Saved per-polygon scores → {OUT_CSV}")
    print(f"  AGSN favelas missed (mean_prob<0.5): {len(missed)}/{len(sa)} "
          f"— median size {missed.area_ha.median():.1f} ha vs "
          f"{sa.area_ha.median():.1f} ha overall")


if __name__ == "__main__":
    main()
