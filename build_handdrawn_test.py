"""
build_handdrawn_test.py — Consolidate all hand-drawn favela community polygons
into one clean independent test layer.

These are field-known settlements digitised by hand. Many are NOT in the IBGE
AGSN layer (small / peripheral / newer), which makes the AGSN-missed subset a
genuine generalisation test for the AGSN-trained classifier.

Handles the mixed CRSs in the source files (EPSG:4326, 31984, 31985), reprojects
everything to the image grid, clips to the image extent, drops the PARIPUEIRA
duplicate of ALTO_DA_BOA_VISTA, and flags each polygon's AGSN overlap.

Run:  python build_handdrawn_test.py
"""

import os
import glob
import geopandas as gpd
import rasterio
from shapely.geometry import box
import warnings
warnings.filterwarnings("ignore")

import config

OUT = os.path.join(config.DATA_DIR, "handdrawn_test.gpkg")
AGSN = os.path.join(config.DATA_DIR, "agsn_truth.gpkg")
EXCLUDE = {"AGSN_2019", "PARIPUEIRA"}   # PARIPUEIRA == ALTO_DA_BOA_VISTA


def main():
    with rasterio.open(config.IMAGE_PATH) as src:
        img_crs = src.crs
        b = src.bounds
    img_box = box(b.left, b.bottom, b.right, b.top)
    agsn = gpd.read_file(AGSN).to_crs(img_crs)
    agsn_union = agsn.geometry.union_all()

    rows = []
    for shp in sorted(glob.glob(os.path.join("comunidades", "*.shp"))):
        name = os.path.basename(shp)[:-4]
        if name in EXCLUDE:
            continue
        g = gpd.read_file(shp)
        if g.crs is None:
            print(f"  ⚠️  {name}: no CRS, skipping")
            continue
        g = g.to_crs(img_crs)
        geom = g.geometry.union_all()
        if not geom.intersects(img_box):
            print(f"  ✗ {name}: outside image extent — skipped")
            continue
        ov = (geom.intersection(agsn_union).area / geom.area * 100) if geom.area else 0
        rows.append({"community": name, "agsn_overlap_pct": round(ov, 1),
                     "geometry": geom})
        print(f"  ✓ {name:<40} AGSN overlap {ov:5.1f}%")

    gdf = gpd.GeoDataFrame(rows, crs=img_crs)
    gdf.to_file(OUT, driver="GPKG")
    missed = gdf[gdf.agsn_overlap_pct == 0]
    print(f"\nSaved → {OUT}")
    print(f"  {len(gdf)} communities inside image | "
          f"{len(missed)} are AGSN-missed (the generalisation test):")
    print(f"   {sorted(missed.community.tolist())}")


if __name__ == "__main__":
    main()
