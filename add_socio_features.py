"""
add_socio_features.py — Augment output/features.csv with socioeconomic /
environmental features (from Eng. Bibi's layers).

Mirrors add_slope_feature.py: it reuses the existing per-window grid in
output/features.csv and appends columns WITHOUT re-running the expensive
671k-window spectral extraction.

Layers (all EPSG:31985, covering only the RMM = Maceió metro region):
  • vulnerabilidade_3_rec.tif  → vuln_mean       (AHP socio-environmental index)
  • mapa_deslizamento2.tif     → landslide_mean  (landslide susceptibility)
  • teste_ahp_3_recortado.tif  → flood_mean      (flood susceptibility; 1 bad px)
  • setores_..._baixa_renda    → lowincome_mean  (% households <= 1/2 min wage)
  • has_socio                  → 1 where RMM socio data exists, else 0

Because the layers only cover the RMM, windows in the interior get NaN means
(and has_socio=0). train_supervised.py imputes the NaNs and uses has_socio so
the model falls back to spectral+terrain outside the RMM.

Layers are smooth at 30 m, so per-window MEAN is enough (no std/roughness).

Run:  python add_socio_features.py
"""

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import geopandas as gpd

import config
from search import reference_profile


def reproject_raster_to_grid(path, ref_prof, clip_max=None):
    """Reproject a single-band raster onto the reference grid; NaN where nodata."""
    with rasterio.open(path) as src:
        dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, np.float32)
        reproject(
            source=rasterio.band(src, 1), destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
            resampling=Resampling.bilinear,
        )
        if src.nodata is not None:
            dst[dst == src.nodata] = np.nan
    # guard corrupted pixels (e.g. the single 1.1e8 px in the flood raster)
    if clip_max is not None:
        dst[dst > clip_max] = np.nan
    dst[~np.isfinite(dst)] = np.nan
    return dst


def rasterize_field_to_grid(shp_path, field, ref_prof):
    """Rasterize a continuous per-polygon attribute; NaN outside any polygon."""
    gdf = gpd.read_file(shp_path)
    if gdf.crs is not None and str(gdf.crs) != str(ref_prof["crs"]):
        gdf = gdf.to_crs(ref_prof["crs"])
    shapes = [(g, float(v)) for g, v in zip(gdf.geometry, gdf[field])
              if g is not None and not g.is_empty and pd.notna(v)]
    arr = rasterize(
        shapes, out_shape=(ref_prof["height"], ref_prof["width"]),
        transform=ref_prof["transform"], fill=np.nan, dtype="float32",
    )
    return arr


def window_means(layer, rows, cols, ws):
    """NaN-aware mean of `layer` over each [r:r+ws, c:c+ws] window, via SATs.

    Returns means (NaN where a window has no valid pixel) and the valid-pixel
    count per window.
    """
    valid = np.isfinite(layer)
    vals = np.where(valid, layer, 0.0).astype(np.float64)

    sat_v = np.zeros((layer.shape[0] + 1, layer.shape[1] + 1), np.float64)
    sat_v[1:, 1:] = vals.cumsum(0).cumsum(1)
    sat_n = np.zeros_like(sat_v)
    sat_n[1:, 1:] = valid.astype(np.float64).cumsum(0).cumsum(1)

    r0, c0 = rows, cols
    r1, c1 = rows + ws, cols + ws
    s = sat_v[r1, c1] - sat_v[r0, c1] - sat_v[r1, c0] + sat_v[r0, c0]
    n = sat_n[r1, c1] - sat_n[r0, c1] - sat_n[r1, c0] + sat_n[r0, c0]

    means = np.full(len(rows), np.nan, np.float32)
    nz = n > 0
    means[nz] = (s[nz] / n[nz]).astype(np.float32)
    return means, n


def main():
    print("Loading features.csv ...")
    df = pd.read_csv(config.FEATURES_CSV)
    print(f"  {len(df):,} windows, {len(df.columns)} columns")

    ref_prof = reference_profile()
    ws = config.WINDOW_SIZE
    rows = df["row"].to_numpy(); cols = df["col"].to_numpy()

    new_cols = {}

    # ── continuous rasters ────────────────────────────────────────────────
    for path, col, clip_max in config.SOCIO_RASTERS:
        if not os.path.isfile(path):
            print(f"  ⚠️  {col}: file not found ({path}) — skipping")
            continue
        layer = reproject_raster_to_grid(path, ref_prof, clip_max=clip_max)
        means, _ = window_means(layer, rows, cols, ws)
        new_cols[col] = means
        cov = np.isfinite(means)
        print(f"  {col:<15} {cov.sum():,} covered windows | "
              f"mean={np.nanmean(means):.3f} max={np.nanmax(means):.3f}")

    # ── income shapefile (rasterize continuous field, then window-mean) ────
    if os.path.isfile(config.SOCIO_INCOME_SHP):
        inc = rasterize_field_to_grid(
            config.SOCIO_INCOME_SHP, config.SOCIO_INCOME_FIELD, ref_prof)
        means, _ = window_means(inc, rows, cols, ws)
        new_cols[config.SOCIO_INCOME_COL] = means
        cov = np.isfinite(means)
        print(f"  {config.SOCIO_INCOME_COL:<15} {cov.sum():,} covered windows | "
              f"mean={np.nanmean(means):.3f} max={np.nanmax(means):.3f}")
    else:
        print(f"  ⚠️  income shapefile not found ({config.SOCIO_INCOME_SHP}) — skipping")

    # ── coverage indicator (RMM extent, from canonical raster) ─────────────
    cov_layer = reproject_raster_to_grid(config.SOCIO_COVERAGE_RASTER, ref_prof, clip_max=1.0)
    _, n = window_means(cov_layer, rows, cols, ws)
    has_socio = (n > 0).astype(np.int8)
    new_cols[config.SOCIO_COVERAGE_COL] = has_socio
    print(f"  {config.SOCIO_COVERAGE_COL:<15} {has_socio.sum():,} / {len(df):,} "
          f"windows inside RMM ({has_socio.mean():.1%})")

    # ── write back (drop existing socio cols first, keep CSV idempotent) ────
    socio_names = list(new_cols.keys())
    df = df.drop(columns=[c for c in socio_names if c in df.columns])
    for name, arr in new_cols.items():
        df[name] = arr

    df.to_csv(config.FEATURES_CSV, index=False)
    feat_n = len([c for c in df.columns
                  if c not in ("row", "col", "map_x", "map_y", "similarity")])
    print(f"\nSaved → {config.FEATURES_CSV}")
    print(f"  added/refreshed: {', '.join(socio_names)}")
    print(f"  feature columns now: {feat_n}")


if __name__ == "__main__":
    main()
