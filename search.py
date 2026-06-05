"""
search.py — Main entry point for the favela similarity search pipeline.

Outputs
───────
  output/heatmap.tif         — continuous similarity map     (QGIS)
  output/heatmap_norm.tif    — same, scaled 0-255            (QGIS)
  output/svm_map.tif         — binary: 1=favela-like         (QGIS)
  output/svm_map_score.tif   — continuous SVM decision score (QGIS)
  output/features.csv        — per-window feature table
  output/overview.png        — quick visual check

Run
───
    python search.py
"""

import os
import sys
import csv
import numpy as np
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import config
from extract_features import (
    extract_all_windows,
    extract_reference_patches,
    extract_reference_vector,
    feature_names,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Similarity metrics
# ─────────────────────────────────────────────────────────────────────────────

def cosine_similarity(ref: np.ndarray, features: np.ndarray) -> np.ndarray:
    ref_norm   = ref / (np.linalg.norm(ref) + 1e-8)
    feat_norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
    scores     = (features / feat_norms) @ ref_norm
    return (scores + 1.0) / 2.0


def euclidean_similarity(ref: np.ndarray, features: np.ndarray) -> np.ndarray:
    dists = np.linalg.norm(features - ref[np.newaxis, :], axis=1)
    return 1.0 / (1.0 + dists)


def compute_similarity(ref, features):
    if config.SIMILARITY_METRIC == "cosine":
        return cosine_similarity(ref, features)
    return euclidean_similarity(ref, features)


# ─────────────────────────────────────────────────────────────────────────────
#  Z-score normalisation
# ─────────────────────────────────────────────────────────────────────────────

def normalise_features(ref_vec, ref_patches, all_features):
    """
    Normalise using global statistics from all windows.
    Returns normalised versions of ref_vec, ref_patches, all_features.
    """
    mean = all_features.mean(axis=0)
    std  = all_features.std(axis=0) + 1e-8
    return ((ref_vec     - mean) / std,
            (ref_patches - mean) / std,
            (all_features - mean) / std)


# ─────────────────────────────────────────────────────────────────────────────
#  Build pixel-resolution heatmap
# ─────────────────────────────────────────────────────────────────────────────

def build_heatmap(origins, scores, image_shape):
    H, W     = image_shape[:2]
    ws       = config.WINDOW_SIZE
    score_sum = np.zeros((H, W), dtype=np.float32)
    count     = np.zeros((H, W), dtype=np.float32)
    for (r, c), score in zip(origins, scores):
        score_sum[r:r+ws, c:c+ws] += score
        count[r:r+ws,     c:c+ws] += 1.0
    return score_sum / np.maximum(count, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
#  Save GeoTIFFs
# ─────────────────────────────────────────────────────────────────────────────

def save_tif(data: np.ndarray, profile: dict, path: str, dtype="float32"):
    p = profile.copy()
    p.update(count=1, dtype=dtype, compress="lzw", nodata=None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if dtype == "uint8":
        data = (data * 255).clip(0, 255).astype(np.uint8)
    with rasterio.open(path, "w", **p) as dst:
        dst.write(data.astype(p["dtype"]), 1)
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
#  Save feature CSV
# ─────────────────────────────────────────────────────────────────────────────

def save_features_csv(origins, features, scores, src_profile):
    transform = src_profile["transform"]
    names     = feature_names()
    with open(config.FEATURES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "map_x", "map_y",
                         "similarity"] + names)
        for (r, c), score, feat in zip(origins, scores, features):
            cx = transform.c + (c + config.WINDOW_SIZE/2) * transform.a
            cy = transform.f + (r + config.WINDOW_SIZE/2) * transform.e
            writer.writerow([r, c, f"{cx:.4f}", f"{cy:.4f}",
                             f"{score:.6f}"] +
                            [f"{v:.6f}" for v in feat])
    print(f"  Saved → {config.FEATURES_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
#  Overview PNG
# ─────────────────────────────────────────────────────────────────────────────

def save_overview(image_rgb, heatmap, binary_map=None):
    n_panels = 3 if binary_map is None else 4
    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 6))

    axes[0].imshow(image_rgb)
    axes[0].set_title("RGB Image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    rect = patches.Rectangle(
        (config.REFERENCE_COL_MIN, config.REFERENCE_ROW_MIN),
        config.REFERENCE_COL_MAX - config.REFERENCE_COL_MIN,
        config.REFERENCE_ROW_MAX - config.REFERENCE_ROW_MIN,
        linewidth=2, edgecolor="yellow", facecolor="none"
    )
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

    plt.suptitle("Favela Similarity Search", fontsize=13)
    plt.tight_layout()
    plt.savefig(config.OVERVIEW_PNG, dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close()
    print(f"  Saved → {config.OVERVIEW_PNG}")


# ─────────────────────────────────────────────────────────────────────────────
#  Load raster helper
# ─────────────────────────────────────────────────────────────────────────────

def load_raster(path, as_rgb=False):
    with rasterio.open(path) as src:
        data    = src.read()
        profile = src.profile.copy()
    if as_rgb:
        data = np.transpose(data, (1, 2, 0))[:, :, :3]
        if data.dtype != np.uint8:
            mn, mx = data.min(), data.max()
            data = ((data - mn) / (mx - mn + 1e-8) * 255).astype(np.uint8)
    else:
        data = data[0].astype(np.float32)
    return data, profile


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Favela Similarity Search")
    print("=" * 58)

    # ── validate files ────────────────────────────────────────────────────
    if not os.path.isfile(config.IMAGE_PATH):
        print(f"\n❌ Image not found: {config.IMAGE_PATH}")
        sys.exit(1)

    # ── load RGB image ────────────────────────────────────────────────────
    print(f"\n📂 Loading image ...")
    image_rgb, src_profile = load_raster(config.IMAGE_PATH, as_rgb=True)
    H, W = image_rgb.shape[:2]
    pixel_size = abs(src_profile["transform"].a)
    print(f"   {H} × {W} px  |  {pixel_size}m/px  |  "
          f"{H*pixel_size:.0f}m × {W*pixel_size:.0f}m")

    # ── load DSM ──────────────────────────────────────────────────────────
    image_dsm = None
    if config.USE_DSM:
        if os.path.isfile(config.DSM_PATH):
            image_dsm, _ = load_raster(config.DSM_PATH)
            print(f"   DSM loaded: {config.DSM_PATH}")
        else:
            print(f"   ⚠️  DSM not found ({config.DSM_PATH}) — skipping")

    # ── load RF map ───────────────────────────────────────────────────────
    image_rf = None
    if config.USE_RF:
        if os.path.isfile(config.RF_PATH):
            image_rf, _ = load_raster(config.RF_PATH)
            image_rf = image_rf.astype(np.int32)
            print(f"   RF map loaded: {config.RF_PATH}")
            print(f"   RF classes found: {np.unique(image_rf).tolist()}")
        else:
            print(f"   ⚠️  RF map not found ({config.RF_PATH}) — skipping RF features")

    # ── active features summary ───────────────────────────────────────────
    print(f"\n   Active features: ", end="")
    active = []
    if config.USE_SPECTRAL: active.append("spectral")
    if config.USE_TEXTURE:  active.append("texture")
    if config.USE_EDGE:     active.append("edge")
    if config.USE_DSM and image_dsm is not None: active.append("DSM")
    if config.USE_RF  and image_rf  is not None: active.append("RF-composition")
    print(" + ".join(active))
    print(f"   Total features : {len(feature_names())}")

    # ── validate reference zone ───────────────────────────────────────────
    print(f"\n📍 Reference zone:")
    print(f"   Rows {config.REFERENCE_ROW_MIN}–{config.REFERENCE_ROW_MAX}  "
          f"Cols {config.REFERENCE_COL_MIN}–{config.REFERENCE_COL_MAX}")
    zone_h = config.REFERENCE_ROW_MAX - config.REFERENCE_ROW_MIN
    zone_w = config.REFERENCE_COL_MAX - config.REFERENCE_COL_MIN
    print(f"   {zone_h}×{zone_w} px  =  "
          f"{zone_h*pixel_size:.0f}m × {zone_w*pixel_size:.0f}m on the ground")

    # ── extract reference ─────────────────────────────────────────────────
    print(f"\n🎯 Extracting reference features ...")
    ref_patches = extract_reference_patches(image_rgb, image_dsm, image_rf)
    ref_vec     = ref_patches.mean(axis=0)

    # ── extract all windows ───────────────────────────────────────────────
    print(f"\n🔲 Sliding window across full image ...")
    origins, all_features = extract_all_windows(image_rgb, image_dsm, image_rf)

    # ── normalise ─────────────────────────────────────────────────────────
    print(f"\n📐 Normalising features ...")
    ref_norm, ref_patches_norm, feat_norm = normalise_features(
        ref_vec, ref_patches, all_features)

    # ── similarity scores ─────────────────────────────────────────────────
    scores = compute_similarity(ref_norm, feat_norm)
    print(f"   Scores:  min={scores.min():.4f}  "
          f"max={scores.max():.4f}  mean={scores.mean():.4f}")

    print(f"\n   Top 5 most similar windows:")
    for rank, idx in enumerate(np.argsort(scores)[::-1][:5]):
        r, c = origins[idx]
        print(f"   #{rank+1}  row={r:4d}  col={c:4d}  "
              f"score={scores[idx]:.4f}")

    # ── build and save similarity heatmap ─────────────────────────────────
    print(f"\n🗺  Building heatmap ...")
    heatmap = build_heatmap(origins, scores, image_rgb.shape)
    save_tif(heatmap, src_profile, config.HEATMAP_PATH,      "float32")
    save_tif(heatmap, src_profile, config.HEATMAP_NORM_PATH, "uint8")

    # ── one-class SVM ─────────────────────────────────────────────────────
    binary_map = None
    if config.USE_ONE_CLASS_SVM:
        print(f"\n🤖 One-Class SVM ...")
        try:
            from oneclass import run_one_class_svm
            binary_map, _ = run_one_class_svm(
                ref_patches_norm, feat_norm,
                origins, image_rgb.shape, src_profile)
        except ImportError:
            print("   scikit-learn not installed — skipping SVM")
            print("   pip install scikit-learn")

    # ── CSV ───────────────────────────────────────────────────────────────
    print(f"\n💾 Saving feature table ...")
    save_features_csv(origins, all_features, scores, src_profile)

    # ── overview PNG ──────────────────────────────────────────────────────
    print(f"\n🖼  Saving overview ...")
    save_overview(image_rgb, heatmap, binary_map)

    # ── final instructions ────────────────────────────────────────────────
    print(f"\n{'='*58}")
    print(f"  Done.")
    print(f"{'='*58}")
    print(f"\n  Load in QGIS:")
    print(f"    heatmap.tif      → Singleband pseudocolor, Reds")
    print(f"    svm_map.tif      → Paletted/Unique values")
    print(f"                       1=favela-like  0=other")
    print(f"    features.csv     → inspect per-window values")
    print(f"\n  When RF map arrives:")
    print(f"    1. Copy it to data/rf_map.tif")
    print(f"    2. Set USE_RF = True in config.py")
    print(f"    3. Run python search.py again")
    print()


if __name__ == "__main__":
    main()
