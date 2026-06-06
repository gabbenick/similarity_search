"""
search.py — Favela similarity search for multi-band Sentinel-2 imagery.

Loads the 6 Sentinel-2 bands (B2,B3,B4,B8,B11,B12), resamples the 20 m SWIR
bands onto the 10 m grid, extracts spectral + index features per sliding window,
and compares each window to a known favela reference zone.

Outputs (in output/):
  heatmap.tif         continuous similarity map        (QGIS)
  heatmap_norm.tif    same, scaled 0-255               (QGIS)
  svm_map.tif         binary: 1=favela-like            (QGIS)
  svm_map_score.tif   continuous SVM decision score    (QGIS)
  features.csv        per-window feature table
  overview.png        quick visual check

Run:  python search.py
"""

import os
import sys
import csv
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from extract_features import (
    extract_all_windows,
    extract_reference_patches,
    feature_names,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-band loading (with resampling to a common 10 m grid)
# ─────────────────────────────────────────────────────────────────────────────

def reference_profile():
    """Profile/grid that every band is resampled onto (from REFERENCE_BAND)."""
    if config.USE_STACKED:
        path = config.STACKED_PATH
    else:
        path = config.BANDS[config.REFERENCE_BAND]
    if not os.path.isfile(path):
        print(f"\n❌ Reference band not found: {path}")
        sys.exit(1)
    with rasterio.open(path) as src:
        return src.profile.copy()


def _read_band_on_grid(path, band_index, ref_prof):
    """Read one band and resample it onto the reference grid."""
    with rasterio.open(path) as src:
        src_arr = src.read(band_index).astype(np.float32)
        same_grid = (src.width == ref_prof["width"] and
                     src.height == ref_prof["height"] and
                     src.transform == ref_prof["transform"])
        if same_grid:
            return src_arr
        dst = np.zeros((ref_prof["height"], ref_prof["width"]), np.float32)
        reproject(
            source=src_arr, destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
            resampling=Resampling.bilinear)
        return dst


def load_multiband(ref_prof):
    """Return image (H, W, n_bands) in config.BAND_ORDER, on the 10 m grid."""
    bands = []
    for name in config.BAND_ORDER:
        if config.USE_STACKED:
            arr = _read_band_on_grid(
                config.STACKED_PATH, config.STACKED_BAND_INDEX[name], ref_prof)
        else:
            path = config.BANDS[name]
            if not os.path.isfile(path):
                print(f"\n❌ Band file not found: {path}")
                sys.exit(1)
            arr = _read_band_on_grid(path, 1, ref_prof)
        bands.append(arr)
        print(f"   {name}: loaded ({arr.shape[0]}×{arr.shape[1]})")
    return np.stack(bands, axis=-1)   # (H, W, n_bands)


def make_rgb_display(image):
    """Percentile-stretched natural-colour composite for the overview PNG."""
    chans = []
    for name in config.RGB_DISPLAY_BANDS:
        b = image[:, :, config.BAND_ORDER.index(name)].astype(np.float32)
        lo, hi = np.nanpercentile(b, 2), np.nanpercentile(b, 98)
        b = np.clip((b - lo) / (hi - lo + 1e-8), 0, 1)
        chans.append(b)
    return (np.stack(chans, axis=-1) * 255).astype(np.uint8)


def load_extra(path, label, as_int=False):
    if not os.path.isfile(path):
        print(f"   ⚠️  {label} not found ({path}) — skipping")
        return None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
    if as_int:
        arr = arr.astype(np.int32)
    print(f"   {label} loaded: {path}")
    return arr


def load_dsm(ref_prof):
    """Load DSM and reproject it onto the Sentinel-2 reference grid."""
    if not os.path.isfile(config.DSM_PATH):
        print(f"   ⚠️  DSM not found: {config.DSM_PATH} — skipping")
        return None
    with rasterio.open(config.DSM_PATH) as src:
        dst = np.full((ref_prof["height"], ref_prof["width"]), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=ref_prof["transform"], dst_crs=ref_prof["crs"],
            resampling=Resampling.bilinear,
        )
        if src.nodata is not None:
            dst[dst == src.nodata] = np.nan
    valid_px = int(np.isfinite(dst).sum())
    print(f"   DSM reprojected to image grid: {valid_px:,} valid pixels "
          f"(h={np.nanmin(dst):.0f}–{np.nanmax(dst):.0f} m)")
    return dst


def rasterize_reference(ref_prof):
    """Rasterize train.shp onto the image grid; returns bool mask or None."""
    if not config.REFERENCE_SHAPEFILE:
        return None
    if not os.path.isfile(config.REFERENCE_SHAPEFILE):
        print(f"   ⚠️  REFERENCE_SHAPEFILE not found: {config.REFERENCE_SHAPEFILE}")
        return None
    try:
        import geopandas as gpd
        from rasterio.features import rasterize as rio_rasterize
    except ImportError:
        print("   ⚠️  geopandas not installed — falling back to bounding-box reference")
        return None

    gdf = gpd.read_file(config.REFERENCE_SHAPEFILE)
    if gdf.crs is not None and str(gdf.crs) != str(ref_prof["crs"]):
        gdf = gdf.to_crs(ref_prof["crs"])
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        print("   ⚠️  No valid geometries in training shapefile")
        return None
    mask = rio_rasterize(
        shapes,
        out_shape=(ref_prof["height"], ref_prof["width"]),
        transform=ref_prof["transform"],
        fill=0, dtype=np.uint8,
    ).astype(bool)
    communities = gdf["community"].unique() if "community" in gdf.columns else ["?"]
    print(f"   Training mask: {mask.sum():,} px from {len(shapes)} polygons "
          f"({', '.join(communities)})")
    return mask


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(ref, features):
    ref_norm   = ref / (np.linalg.norm(ref) + 1e-8)
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    scores     = (features / feat_norms) @ ref_norm
    return (scores + 1.0) / 2.0


def euclidean_similarity(ref, features):
    dists = np.linalg.norm(features - ref[np.newaxis, :], axis=1)
    return 1.0 / (1.0 + dists)


def compute_similarity(ref, features):
    if config.SIMILARITY_METRIC == "cosine":
        return cosine_similarity(ref, features)
    if config.SIMILARITY_METRIC == "euclidean":
        return euclidean_similarity(ref, features)
    print(f"   ⚠️  Unknown SIMILARITY_METRIC "
          f"'{config.SIMILARITY_METRIC}' — using cosine")
    return cosine_similarity(ref, features)


# ─────────────────────────────────────────────────────────────────────────────
#  Z-score normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_features(ref_vec, ref_patches, all_features):
    mean = all_features.mean(axis=0)
    std  = all_features.std(axis=0) + 1e-8
    return ((ref_vec     - mean) / std,
            (ref_patches - mean) / std,
            (all_features - mean) / std)


# ─────────────────────────────────────────────────────────────────────────────
#  Build pixel-resolution heatmap
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap(origins, scores, image_shape):
    H, W      = image_shape[:2]
    ws        = config.WINDOW_SIZE
    score_sum = np.zeros((H, W), dtype=np.float32)
    count     = np.zeros((H, W), dtype=np.float32)
    for (r, c), score in zip(origins, scores):
        score_sum[r:r+ws, c:c+ws] += score
        count[r:r+ws,     c:c+ws] += 1.0
    return score_sum / np.maximum(count, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Save GeoTIFFs
# ─────────────────────────────────────────────────────────────────────────────

def save_tif(data, profile, path, dtype="float32"):
    p = profile.copy()
    p.update(count=1, dtype=dtype, compress="lzw", nodata=None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if dtype == "uint8":
        data = (data * 255).clip(0, 255).astype(np.uint8)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(p["dtype"]), 1)
    print(f"  Saved → {path}")


def save_features_csv(origins, features, scores, ref_prof):
    transform = ref_prof["transform"]
    names     = feature_names()
    with open(config.FEATURES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "map_x", "map_y", "similarity"] + names)
        for (r, c), score, feat in zip(origins, scores, features):
            cx = transform.c + (c + config.WINDOW_SIZE/2) * transform.a
            cy = transform.f + (r + config.WINDOW_SIZE/2) * transform.e
            writer.writerow([r, c, f"{cx:.4f}", f"{cy:.4f}",
                             f"{score:.6f}"] + [f"{v:.6f}" for v in feat])
    print(f"  Saved → {config.FEATURES_CSV}")


def save_overview(image_rgb, heatmap, binary_map=None):
    n_panels = 3 if binary_map is None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("Sentinel-2 (B4-B3-B2)", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    rect = patches.Rectangle(
        (config.REFERENCE_COL_MIN, config.REFERENCE_ROW_MIN),
        config.REFERENCE_COL_MAX - config.REFERENCE_COL_MIN,
        config.REFERENCE_ROW_MAX - config.REFERENCE_ROW_MIN,
        linewidth=2, edgecolor="yellow", facecolor="none")
    axes[1].add_patch(rect)
    axes[1].set_title("Reference zone (yellow)", fontsize=11)
    axes[1].axis("off")

    im = axes[2].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], fraction=0.04, label="Similarity")
    axes[2].set_title("Similarity heatmap\n(bright = similar to favela)",
                      fontsize=11)
    axes[2].axis("off")

    if binary_map is not None:
        axes[3].imshow(binary_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[3].set_title("One-Class SVM\n(green=favela-like, red=other)",
                          fontsize=11)
        axes[3].axis("off")

    plt.suptitle("Favela Similarity Search — Sentinel-2", fontsize=13)
    plt.tight_layout()
    plt.savefig(config.OVERVIEW_PNG, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved → {config.OVERVIEW_PNG}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Favela Similarity Search — Sentinel-2")
    print("=" * 58)

    # ── grid + bands ──────────────────────────────────────────────────────
    ref_prof = reference_profile()
    H, W = ref_prof["height"], ref_prof["width"]
    pixel_size = abs(ref_prof["transform"].a)
    print(f"\n📂 Loading {len(config.BAND_ORDER)} bands "
          f"(grid {H}×{W}, {pixel_size:.0f} m/px) ...")
    image = load_multiband(ref_prof)
    print(f"   Ground extent ≈ {H*pixel_size:.0f} m × {W*pixel_size:.0f} m")

    # ── optional extra sources ────────────────────────────────────────────
    image_dsm = load_dsm(ref_prof) if config.USE_DSM else None
    image_rf  = load_extra(config.RF_PATH, "RF map", as_int=True) if config.USE_RF else None

    # ── keep feature_names() consistent if a source is missing ────────────
    if config.USE_DSM and image_dsm is None:
        config.USE_DSM = False
    if config.USE_RF and image_rf is None:
        config.USE_RF = False

    # ── active features ───────────────────────────────────────────────────
    active = []
    if config.USE_SPECTRAL: active.append("spectral")
    if config.USE_INDICES:  active.append("indices(" + ",".join(config.INDICES) + ")")
    if config.USE_TEXTURE:  active.append("texture")
    if config.USE_EDGE:     active.append("edge")
    if config.USE_DSM:      active.append("DSM")
    if config.USE_RF:       active.append("RF")
    print(f"\n   Active features: " + " + ".join(active))
    print(f"   Total features : {len(feature_names())}")

    # ── reference zone sanity ─────────────────────────────────────────────
    zone_h = config.REFERENCE_ROW_MAX - config.REFERENCE_ROW_MIN
    zone_w = config.REFERENCE_COL_MAX - config.REFERENCE_COL_MIN
    print(f"\n📍 Reference zone: rows "
          f"{config.REFERENCE_ROW_MIN}–{config.REFERENCE_ROW_MAX}  cols "
          f"{config.REFERENCE_COL_MIN}–{config.REFERENCE_COL_MAX}  "
          f"({zone_h}×{zone_w} px = {zone_h*pixel_size:.0f}×{zone_w*pixel_size:.0f} m)")
    if (config.REFERENCE_ROW_MAX > H or config.REFERENCE_COL_MAX > W or
            config.REFERENCE_ROW_MIN < 0 or config.REFERENCE_COL_MIN < 0):
        print(f"   ⚠️  Reference zone falls outside the image ({H}×{W})! "
              f"Re-derive coords with coords.py.")

    # ── training reference mask ───────────────────────────────────────────
    print(f"\n🗂  Loading training reference ...")
    ref_mask = rasterize_reference(ref_prof)

    # ── reference features ────────────────────────────────────────────────
    print(f"\n🎯 Extracting reference features ...")
    ref_patches = extract_reference_patches(image, image_dsm, image_rf, ref_mask)
    ref_vec     = ref_patches.mean(axis=0)

    # ── all windows ───────────────────────────────────────────────────────
    print(f"\n🔲 Sliding window across full image ...")
    origins, all_features = extract_all_windows(image, image_dsm, image_rf)

    # ── normalise ─────────────────────────────────────────────────────────
    print(f"\n📐 Normalising features ...")
    ref_norm, ref_patches_norm, feat_norm = normalise_features(
        ref_vec, ref_patches, all_features)

    # ── similarity ────────────────────────────────────────────────────────
    scores = compute_similarity(ref_norm, feat_norm)
    print(f"   Scores:  min={scores.min():.4f}  "
          f"max={scores.max():.4f}  mean={scores.mean():.4f}")
    if scores.max() - scores.min() < 0.01:
        print("   ⚠️  Score range almost flat — features may be constant; "
              "check inputs / reference zone.")

    print(f"\n   Top 5 most similar windows:")
    for rank, idx in enumerate(np.argsort(scores)[::-1][:5]):
        r, c = origins[idx]
        print(f"   #{rank+1}  row={r:4d}  col={c:4d}  score={scores[idx]:.4f}")

    # ── heatmap ───────────────────────────────────────────────────────────
    print(f"\n🗺  Building heatmap ...")
    heatmap = build_heatmap(origins, scores, (H, W))
    if np.any(~np.isfinite(heatmap)):
        print("   ⚠️  Heatmap contains NaN/Inf — replacing with 0.")
        heatmap = np.nan_to_num(heatmap)
    save_tif(heatmap, ref_prof, config.HEATMAP_PATH,      "float32")
    save_tif(heatmap, ref_prof, config.HEATMAP_NORM_PATH, "uint8")

    # ── one-class SVM ─────────────────────────────────────────────────────
    binary_map = None
    if config.USE_ONE_CLASS_SVM:
        print(f"\n🤖 One-Class SVM ...")
        try:
            from oneclass import run_one_class_svm
            binary_map, _ = run_one_class_svm(
                ref_patches_norm, feat_norm, origins, (H, W), ref_prof)
        except ImportError:
            print("   scikit-learn not installed — skipping SVM "
                  "(pip install scikit-learn)")

    # ── CSV + overview ────────────────────────────────────────────────────
    print(f"\n💾 Saving feature table ...")
    save_features_csv(origins, all_features, scores, ref_prof)

    print(f"\n🖼  Saving overview ...")
    save_overview(make_rgb_display(image), heatmap, binary_map)

    print(f"\n{'='*58}\n  Done.\n{'='*58}")
    print(f"\n  Next: validate against ground-truth polygons:")
    print(f"    python validate.py --truth data/val.shp\n")


if __name__ == "__main__":
    main()
