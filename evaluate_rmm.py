"""
evaluate_rmm.py — RMM-scoped object-level evaluation (precision focus).

Scope decision: the project only cares about the RMM (Região Metropolitana de
Maceió), where the socio data is valid. This script therefore evaluates ONLY
inside the RMM, using output/favela_probability_rmm.tif (NaN outside the RMM).

It answers two questions the full-image evaluation muddied:

  A. What is the HONEST in-RMM object precision/recall? (the number to improve)
  B. How much of the "false" is label error vs model error? AGSN is comprehensive
     only in the Maceió core; the RMM periphery has unmapped favelas. We can't
     tile space by municipality (no IBGE boundary shapefile), so instead we
     EXPORT every false blob to a GeoPackage for manual review in QGIS, and we
     split AGSN DETECTION by core vs periphery (which we CAN do, since favela
     polygons carry their municipality).

Run:  python evaluate_rmm.py
"""

import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize, shapes as rio_shapes
from shapely.geometry import shape as shp_shape
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

import config

PROB_RMM = os.path.join(config.OUTPUT_DIR, "favela_probability_rmm.tif")
AGSN     = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
OUT_FALSE = os.path.join(config.OUTPUT_DIR, "false_blobs_rmm.gpkg")
OUT_CSV   = os.path.join(config.OUTPUT_DIR, "rmm_object_scores.csv")

# RMM municipalities present in AGSN_NM_MU (socio data covers these)
RMM_CORE      = {"Maceió"}
RMM_PERIPHERY = {"Marechal Deodoro", "Rio Largo", "Barra de Santo Antônio",
                 "Santa Luzia Do Norte", "Satuba", "Paripueira"}
RMM_ALL = RMM_CORE | RMM_PERIPHERY

OBJ_THRESHOLDS = [0.50, 0.70, 0.80, 0.90]
MIN_REGION_HA  = 0.5     # ignore predicted speckle smaller than this (raw)
HIT_FRAC       = 0.25    # truth polygon detected if >= this fraction is flagged
FILTER_MIN_HA       = 1.0   # post-filter: drop blobs smaller than this
FILTER_MIN_MEANPROB = 0.60  # post-filter: drop blobs whose mean prob is below this


def main():
    print("=" * 64)
    print("  RMM-scoped object evaluation  (favela_probability_rmm.tif)")
    print("=" * 64)

    with rasterio.open(PROB_RMM) as src:
        prob = src.read(1).astype(np.float32)          # NaN outside RMM
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
    rmm_valid = np.isfinite(prob)
    probz = np.where(rmm_valid, prob, 0.0)             # NaN -> 0 (never flagged)

    lat = abs(transform.f)
    px_m = abs(transform.a) * 111_320 * math.cos(math.radians(lat))
    py_m = abs(transform.e) * 110_540
    px_ha = (px_m * py_m) / 10_000
    min_region_px = max(int(MIN_REGION_HA / px_ha), 1)
    print(f"\n  pixel ≈ {px_m:.1f}×{py_m:.1f} m ({px_ha:.3f} ha) | "
          f"min region {min_region_px}px ({MIN_REGION_HA} ha) | "
          f"RMM = {rmm_valid.mean():.1%} of image")

    # ── AGSN truth, classified by municipality ────────────────────────────
    agsn = gpd.read_file(AGSN).to_crs(crs).reset_index(drop=True)
    muni = agsn["AGSN_NM_MU"].fillna("?")
    scope = np.where(muni.isin(RMM_CORE), "core",
             np.where(muni.isin(RMM_PERIPHERY), "periphery", "outside"))
    agsn["scope"] = scope
    n_core = (scope == "core").sum()
    n_peri = (scope == "periphery").sum()
    n_out  = (scope == "outside").sum()
    print(f"  AGSN: {n_core} core + {n_peri} periphery = {n_core+n_peri} in-RMM | "
          f"{n_out} outside-RMM (excluded)")

    rmm_agsn = agsn[agsn.scope != "outside"].reset_index(drop=True)
    truth = rasterize([(g, 1) for g in rmm_agsn.geometry if g and not g.is_empty],
                      out_shape=shape, transform=transform, fill=0,
                      dtype="uint8").astype(bool)

    # ── validate municipality split against socio coverage ────────────────
    def covered_frac(g):
        m = rasterize([(g, 1)], out_shape=shape, transform=transform,
                      fill=0, dtype="uint8").astype(bool)
        return float(rmm_valid[m].mean()) if m.any() else np.nan
    cov_in  = np.nanmean([covered_frac(g) for g in rmm_agsn.geometry])
    cov_out = np.nanmean([covered_frac(g) for g in
                          agsn[agsn.scope == "outside"].geometry])
    print(f"  socio-coverage check: in-RMM favelas {cov_in:.0%} covered | "
          f"outside-RMM favelas {cov_out:.0%} covered (should be ~100% / ~0%)")

    # ── A. detection by scope (per-polygon mean/max prob) ─────────────────
    def zonal(g):
        m = rasterize([(g, 1)], out_shape=shape, transform=transform,
                      fill=0, dtype="uint8").astype(bool)
        if not m.any():
            return np.nan, np.nan, np.nan
        v = prob[m]; v = v[np.isfinite(v)]
        if not v.size:
            return np.nan, np.nan, np.nan
        return v.mean(), v.max(), np.percentile(v, 90)

    stats = pd.DataFrame([zonal(g) for g in rmm_agsn.geometry],
                         columns=["mean_prob", "max_prob", "p90_prob"])
    stats["scope"] = rmm_agsn.scope.values
    # Most "missed" favelas (low whole-polygon mean) still contain a hot window:
    # report mean / p90 / max so the screening-relevant "any hot window" number
    # (max) is visible next to the conservative whole-polygon mean.
    print("\n  A. AGSN DETECTION inside RMM (% of favelas with prob >= T)")
    print(f"     {'scope':<14}{'n':>5}{'mean>=.5':>10}{'p90>=.5':>9}"
          f"{'max>=.5':>9}{'max>=.7':>9}")
    for sc in ["core", "periphery", "in-RMM (all)"]:
        g = stats if sc == "in-RMM (all)" else stats[stats.scope == sc]
        print(f"     {sc:<14}{len(g):>5}{(g.mean_prob>=.5).mean()*100:>9.1f}%"
              f"{(g.p90_prob>=.5).mean()*100:>8.1f}%"
              f"{(g.max_prob>=.5).mean()*100:>8.1f}%"
              f"{(g.max_prob>=.7).mean()*100:>8.1f}%")
    touched = (stats.mean_prob < 0.5) & (stats.max_prob >= 0.5)
    print(f"     → headline detection (any hot window, max>=0.5): "
          f"{(stats.max_prob>=.5).mean()*100:.1f}%  "
          f"({touched.sum()} of the {(stats.mean_prob<.5).sum()} low-mean favelas "
          f"are in fact touched)")

    # ── B. object precision / recall inside RMM ───────────────────────────
    # Two variants per threshold: RAW (size>=MIN_REGION_HA only) and FILTERED
    # (also require blob mean-prob >= FILTER_MIN_MEANPROB and area >= FILTER_MIN_HA).
    # The filter prunes low-confidence speckle (~half the false blobs are <1 ha),
    # raising precision at little recall cost.
    def recall_on(keepmask, gdf):
        det = 0
        for g in gdf.geometry:
            if g is None or g.is_empty:
                continue
            pm = rasterize([(g, 1)], out_shape=shape, transform=transform,
                           fill=0, dtype="uint8").astype(bool)
            if pm.any() and (keepmask & pm).sum() / pm.sum() >= HIT_FRAC:
                det += 1
        return det / max(len(gdf), 1)

    min_filt_px = max(int(FILTER_MIN_HA / px_ha), 1)
    print("\n  B. OBJECT precision/recall inside RMM   "
          f"(filtered = area>={FILTER_MIN_HA}ha & mean>={FILTER_MIN_MEANPROB})")
    print(f"     {'T':>5} | {'#pred':>6}{'prec':>7}{'Rc':>6}{'Rp':>6}"
          f"  | {'#pred*':>6}{'prec*':>7}{'Rc*':>6}{'Rp*':>6}   (* = filtered)")
    rows = []
    for t in OBJ_THRESHOLDS:
        lab, n = ndimage.label(probz >= t)
        if n == 0:
            print(f"     {t:>5.2f}  no regions"); continue
        ids = np.arange(1, n + 1)
        sizes = ndimage.sum(np.ones_like(lab), lab, ids)
        bmean = np.array(ndimage.mean(probz, lab, ids))
        keep = set((ids[sizes >= min_region_px]).tolist())
        keepf = set((ids[(sizes >= min_filt_px) &
                         (bmean >= FILTER_MIN_MEANPROB)]).tolist())

        def metrics(keepset):
            km = np.isin(lab, list(keepset))
            hit = set(np.unique(lab[truth & km]).tolist()) - {0}
            prec = len(hit) / max(len(keepset), 1)
            rc = recall_on(km, rmm_agsn[rmm_agsn.scope == "core"]) * 100
            rp = recall_on(km, rmm_agsn[rmm_agsn.scope == "periphery"]) * 100
            return len(keepset), prec, rc, rp
        n0, p0, rc0, rp0 = metrics(keep)
        n1, p1, rc1, rp1 = metrics(keepf)
        print(f"     {t:>5.2f} | {n0:>6}{p0*100:>6.1f}%{rc0:>5.0f}%{rp0:>5.0f}%"
              f"  | {n1:>6}{p1*100:>6.1f}%{rc1:>5.0f}%{rp1:>5.0f}%")
        rows.append(dict(threshold=t, n_pred=n0, precision=p0,
                         recall_core=rc0/100, recall_peri=rp0/100,
                         n_pred_filt=n1, precision_filt=p1,
                         recall_core_filt=rc1/100, recall_peri_filt=rp1/100))
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)

    # ── export FALSE blobs at T=0.7 for manual QGIS review ────────────────
    T_EXPORT = 0.70
    lab, n = ndimage.label(probz >= T_EXPORT)
    sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
    keep = set((np.where(sizes >= min_region_px)[0] + 1).tolist())
    keepmask = np.isin(lab, list(keep))
    hit = set(np.unique(lab[truth & keepmask]).tolist()) - {0}
    false_ids = keep - hit
    false_only = np.where(np.isin(lab, list(false_ids)), lab, 0).astype(np.int32)

    geoms, rec = [], []
    maxp = ndimage.maximum(prob, lab, list(false_ids)) if false_ids else []
    meanp = ndimage.mean(probz, lab, list(false_ids)) if false_ids else []
    msizes = ndimage.sum(np.ones_like(lab), lab, list(false_ids)) if false_ids else []
    for geom, val in rio_shapes(false_only, mask=false_only > 0, transform=transform):
        geoms.append(shp_shape(geom))
        rec.append(int(val))
    if geoms:
        gdf = gpd.GeoDataFrame({"blob_id": rec}, geometry=geoms, crs=crs)
        gdf = gdf.dissolve("blob_id").reset_index()
        idmap = {int(b): i for i, b in enumerate(false_ids)}
        gdf["max_prob"] = [float(maxp[idmap[b]]) for b in gdf.blob_id]
        gdf["mean_prob"] = [float(meanp[idmap[b]]) for b in gdf.blob_id]
        gdf["area_ha"] = [float(msizes[idmap[b]]) * px_ha for b in gdf.blob_id]
        gdf = gdf.sort_values("max_prob", ascending=False).reset_index(drop=True)
        gdf.to_file(OUT_FALSE, driver="GPKG")
        print(f"\n  Exported {len(gdf)} FALSE blobs (T={T_EXPORT}) → {OUT_FALSE}")
        print(f"  (review in QGIS: how many are real unmapped favelas?)")
        big = gdf[gdf.area_ha >= 2.0]
        print(f"  false blobs >= 2 ha: {len(big)} "
              f"(median max_prob {big.max_prob.median():.2f})")
    print(f"\n  Saved object metrics → {OUT_CSV}")
    print("\nDone.")


if __name__ == "__main__":
    main()
