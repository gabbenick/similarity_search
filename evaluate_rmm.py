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
MIN_REGION_HA  = 0.5     # ignore predicted speckle smaller than this
HIT_FRAC       = 0.25    # truth polygon detected if >= this fraction is flagged


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
            return np.nan, np.nan
        v = prob[m]; v = v[np.isfinite(v)]
        return (v.mean(), v.max()) if v.size else (np.nan, np.nan)

    stats = pd.DataFrame([zonal(g) for g in rmm_agsn.geometry],
                         columns=["mean_prob", "max_prob"])
    stats["scope"] = rmm_agsn.scope.values
    print("\n  A. AGSN DETECTION inside RMM (% with prob >= T)")
    print(f"     {'scope':<10}{'n':>5}{'  mean>=0.5':>11}{'mean>=0.7':>10}"
          f"{'max>=0.5':>10}{'max>=0.7':>10}")
    for sc in ["core", "periphery", "in-RMM (all)"]:
        g = stats if sc == "in-RMM (all)" else stats[stats.scope == sc]
        print(f"     {sc:<10}{len(g):>5}{(g.mean_prob>=.5).mean()*100:>10.1f}%"
              f"{(g.mean_prob>=.7).mean()*100:>9.1f}%"
              f"{(g.max_prob>=.5).mean()*100:>9.1f}%"
              f"{(g.max_prob>=.7).mean()*100:>9.1f}%")

    # ── B. object precision / recall inside RMM ───────────────────────────
    print("\n  B. OBJECT precision/recall inside RMM")
    print(f"     {'T':>5}{'#pred':>7}{'#hit':>6}{'precision':>11}"
          f"{'recall_core':>13}{'recall_peri':>13}")
    rows = []
    for t in OBJ_THRESHOLDS:
        lab, n = ndimage.label(probz >= t)
        if n == 0:
            print(f"     {t:>5.2f}  no regions"); continue
        sizes = ndimage.sum(np.ones_like(lab), lab, range(1, n + 1))
        keep = set((np.where(sizes >= min_region_px)[0] + 1).tolist())
        keepmask = np.isin(lab, list(keep))
        hit = set(np.unique(lab[truth & keepmask]).tolist()) - {0}
        precision = len(hit) / max(len(keep), 1)

        def recall(gdf):
            det = 0
            for g in gdf.geometry:
                if g is None or g.is_empty:
                    continue
                pm = rasterize([(g, 1)], out_shape=shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)
                if pm.any() and (keepmask & pm).sum() / pm.sum() >= HIT_FRAC:
                    det += 1
            return det / max(len(gdf), 1)
        r_core = recall(rmm_agsn[rmm_agsn.scope == "core"]) * 100
        r_peri = recall(rmm_agsn[rmm_agsn.scope == "periphery"]) * 100
        print(f"     {t:>5.2f}{len(keep):>7}{len(hit):>6}{precision*100:>10.1f}%"
              f"{r_core:>12.1f}%{r_peri:>12.1f}%")
        rows.append(dict(threshold=t, n_pred=len(keep), n_hit=len(hit),
                         precision=precision, recall_core=r_core/100,
                         recall_peri=r_peri/100))
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
