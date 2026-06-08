"""
make_classified_map.py — turn the probability map into a binary favela map.

Thresholds favela_probability_rmm.tif at T (default 0.70) to produce an
advisor-/QGIS-friendly "favela vs other" product:

  output/favela_classified_rmm.tif   uint8: 1 = favela, 0 = other built-up,
                                      255 = nodata (outside the RMM)
  output/favela_predicted_rmm.gpkg   the same favela class as polygons, with
                                      area_ha / mean_prob / max_prob per polygon
                                      (easy to count, measure, and click in QGIS)

By default it applies the SAME post-filter as evaluate_rmm.py (area >= 1 ha &
blob mean-prob >= 0.60), so the map matches the precision/recall numbers in the
thesis tables. Pass --no-filter for the raw threshold.

Run:  python make_classified_map.py               # T=0.70, filtered
      python make_classified_map.py --threshold 0.8
      python make_classified_map.py --no-filter
"""

import os
import math
import argparse
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape
from scipy import ndimage
import warnings
warnings.filterwarnings("ignore")

import config

PROB_RMM = os.path.join(config.OUTPUT_DIR, "favela_probability_rmm.tif")
OUT_TIF  = os.path.join(config.OUTPUT_DIR, "favela_classified_rmm.tif")
OUT_GPKG = os.path.join(config.OUTPUT_DIR, "favela_predicted_rmm.gpkg")

FILTER_MIN_HA       = 1.0    # same post-filter as evaluate_rmm.py
FILTER_MIN_MEANPROB = 0.60


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threshold", type=float, default=0.70,
                    help="probability cutoff for the favela class (default 0.70)")
    ap.add_argument("--no-filter", action="store_true",
                    help="raw threshold only (skip the area/mean post-filter)")
    args = ap.parse_args()
    T = args.threshold
    do_filter = not args.no_filter

    print("=" * 60)
    print(f"  Classified favela map  (T = {T:.2f}, "
          f"{'filtered' if do_filter else 'raw'})")
    print("=" * 60)

    with rasterio.open(PROB_RMM) as src:
        prob = src.read(1).astype(np.float32)      # NaN outside the RMM
        transform, crs, shape = src.transform, src.crs, (src.height, src.width)
        profile = src.profile.copy()
    rmm_valid = np.isfinite(prob)
    probz = np.where(rmm_valid, prob, 0.0)          # NaN -> 0 (never flagged)

    # pixel area in hectares (geographic grid → metres via latitude)
    lat = abs(transform.f)
    px_m = abs(transform.a) * 111_320 * math.cos(math.radians(lat))
    py_m = abs(transform.e) * 110_540
    px_ha = (px_m * py_m) / 10_000

    # ── threshold → connected blobs ──────────────────────────────────────────
    lab, n = ndimage.label(probz >= T)
    print(f"\n  {n} raw blobs above T={T:.2f} "
          f"({rmm_valid.mean():.0%} of image is inside the RMM)")
    if n == 0:
        print("  Nothing above threshold — nothing to write.")
        return

    ids = np.arange(1, n + 1)
    sizes = ndimage.sum(np.ones_like(lab), lab, ids)        # pixels per blob
    bmean = np.array(ndimage.mean(probz, lab, ids))
    bmax  = np.array(ndimage.maximum(prob, lab, ids))
    areas_ha = sizes * px_ha

    if do_filter:
        keep = ids[(areas_ha >= FILTER_MIN_HA) & (bmean >= FILTER_MIN_MEANPROB)]
        print(f"  post-filter (area >= {FILTER_MIN_HA} ha & mean >= "
              f"{FILTER_MIN_MEANPROB}): {len(keep)} blobs kept of {n}")
    else:
        keep = ids
    keep_set = set(keep.tolist())
    keepmask = np.isin(lab, list(keep_set))

    # ── binary raster: 1 favela / 0 other / 255 nodata (outside RMM) ──────────
    classed = np.where(keepmask, 1, 0).astype(np.uint8)
    classed[~rmm_valid] = 255
    pr = profile.copy()
    pr.update(count=1, dtype="uint8", compress="lzw", nodata=255)
    with rasterio.open(OUT_TIF, "w", **pr) as dst:
        dst.write(classed, 1)
        dst.write_colormap(1, {0: (200, 200, 200, 255),    # other = grey
                               1: (220, 20, 20, 255),       # favela = red
                               255: (0, 0, 0, 0)})          # nodata = transparent
    favela_ha = float(keepmask.sum()) * px_ha
    print(f"\n  Saved → {OUT_TIF}")
    print(f"    favela pixels: {int(keepmask.sum()):,}  (~{favela_ha:,.0f} ha)")

    # ── polygon version (one feature per favela blob) ────────────────────────
    only = np.where(keepmask, lab, 0).astype(np.int32)
    idmap = {int(b): i for i, b in enumerate(ids)}
    geoms, blob_ids = [], []
    for geom, val in rio_shapes(only, mask=only > 0, transform=transform):
        geoms.append(shp_shape(geom)); blob_ids.append(int(val))
    gdf = gpd.GeoDataFrame({"blob_id": blob_ids}, geometry=geoms, crs=crs)
    gdf = gdf.dissolve("blob_id").reset_index()
    gdf["area_ha"]   = [float(areas_ha[idmap[b]]) for b in gdf.blob_id]
    gdf["mean_prob"] = [float(bmean[idmap[b]])    for b in gdf.blob_id]
    gdf["max_prob"]  = [float(bmax[idmap[b]])     for b in gdf.blob_id]
    gdf = gdf.sort_values("area_ha", ascending=False).reset_index(drop=True)
    gdf.to_file(OUT_GPKG, driver="GPKG")
    print(f"  Saved → {OUT_GPKG}  ({len(gdf)} favela polygons, "
          f"median {gdf.area_ha.median():.1f} ha)")
    print("\nDone.")


if __name__ == "__main__":
    main()
