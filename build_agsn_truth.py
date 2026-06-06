"""
build_agsn_truth.py — Clean the raw IBGE AGSN_2019 layer into a usable
ground-truth file for the favela pipeline.

What it does
────────────
  • Reprojects AGSN (EPSG:4674 / SIRGAS 2000) onto the Sentinel-2 image CRS
  • Clips to the image extent
  • Keeps ONLY the useful fields (geometry, name, household count, municipality)
    and drops the 30 buggy "nearest health facility" columns
  • Fixes the cp1252-mislabelled-as-UTF8 settlement names
  • Writes data/agsn_truth.gpkg

Run:  python build_agsn_truth.py
"""

import os
import geopandas as gpd
import rasterio
from shapely.geometry import box
import config

RAW   = os.path.join(config.DATA_DIR, "..", "comunidades", "AGSN_2019.shp")
OUT   = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
KEEP  = ["NM_AGSN", "SUM_EDOC", "AGSN_NM_MU"]


def fix_encoding(s):
    """The .dbf is cp1252 but tagged UTF-8; repair the mojibake."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        try:
            return s.encode("cp1252", errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            return s


def main():
    print("Reading raw AGSN_2019 ...")
    agsn = gpd.read_file(RAW)
    print(f"  {len(agsn)} polygons, CRS={agsn.crs}")

    # image grid
    with rasterio.open(config.IMAGE_PATH) as src:
        img_crs = src.crs
        b = src.bounds
    img_box = gpd.GeoSeries(
        [box(b.left, b.bottom, b.right, b.top)], crs=img_crs
    ).to_crs(agsn.crs).iloc[0]

    # clip to image, reproject
    within = agsn[agsn.intersects(img_box)].copy()
    within = within.to_crs(img_crs)
    print(f"  {len(within)} polygons intersect the image extent")

    # keep only useful columns
    within = within[KEEP + ["geometry"]].copy()
    within["NM_AGSN"] = within["NM_AGSN"].map(fix_encoding)

    # add an area column (in metres, via UTM) for filtering/weighting downstream
    within["area_ha"] = within.to_crs("EPSG:31985").geometry.area / 10000

    within = within.reset_index(drop=True)
    within.to_file(OUT, driver="GPKG")
    print(f"\nSaved → {OUT}")
    print(f"  {len(within)} favela polygons | {within.area_ha.sum():.0f} ha total")
    print(f"  household counts (SUM_EDOC): "
          f"min={within.SUM_EDOC.min()} median={within.SUM_EDOC.median():.0f} "
          f"max={within.SUM_EDOC.max()}")
    print(f"  sample names: {within.NM_AGSN.dropna().head(3).tolist()}")


if __name__ == "__main__":
    main()
