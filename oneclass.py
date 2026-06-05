"""
oneclass.py — One-Class SVM for favela detection.

What this does
──────────────
Trains a statistical boundary around the feature distribution of the
known favela zone, then applies it to every window in the image.

Output
──────
  Binary map: 1 = inside favela distribution / 0 = outside
  Saved as output/svm_map.tif (georeferenced, load in QGIS)

How it differs from the similarity heatmap
──────────────────────────────────────────
The similarity heatmap measures distance from the mean reference vector.
It gives a smooth continuous score but can miss cases where the favela
zone has internal variety — some windows are denser, some have more
exposed soil. The one-class SVM learns the full distribution boundary,
not just the centre. A window qualifies as "favela-like" if it falls
anywhere inside that boundary, not just near the mean.

Think of it as:
  Similarity heatmap → "how close are you to the average favela?"
  One-class SVM      → "are you inside the favela family at all?"

Both are useful and complement each other.
"""

import numpy as np
import rasterio
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import config


def train_one_class_svm(ref_features: np.ndarray,
                        all_features: np.ndarray):
    """
    Train a One-Class SVM on reference zone windows and predict
    all image windows.

    Parameters
    ----------
    ref_features : (N_ref, F)  — feature matrix from reference zone only
    all_features : (N_all, F)  — feature matrix from all windows

    Returns
    -------
    predictions : (N_all,) int  — 1 = favela-like, 0 = not
    scores      : (N_all,) float — raw SVM decision scores
                                   (higher = more inside the boundary)
    scaler      : fitted StandardScaler (for inspection)
    """
    print(f"  Training One-Class SVM on {len(ref_features)} reference windows ...")
    print(f"  Settings: nu={config.SVM_NU}  kernel={config.SVM_KERNEL}")

    # scale features — SVM is sensitive to scale
    scaler = StandardScaler()
    scaler.fit(ref_features)                         # fit only on reference
    ref_scaled = scaler.transform(ref_features)
    all_scaled = scaler.transform(all_features)

    svm = OneClassSVM(
        nu=config.SVM_NU,
        kernel=config.SVM_KERNEL,
        gamma="scale",
    )
    svm.fit(ref_scaled)

    raw_preds  = svm.predict(all_scaled)             # +1 = inlier, -1 = outlier
    scores     = svm.decision_function(all_scaled)   # higher = more inside

    # convert to 0/1 (1 = favela-like)
    predictions = (raw_preds == 1).astype(np.uint8)

    n_positive = predictions.sum()
    n_total    = len(predictions)
    print(f"  SVM: {n_positive}/{n_total} windows classified as favela-like "
          f"({100*n_positive/n_total:.1f}%)")

    return predictions, scores, scaler


def build_svm_heatmap(origins: list, predictions: np.ndarray,
                      scores: np.ndarray,
                      image_shape: tuple):
    """
    Build two pixel-resolution maps:
      binary_map  — 0/1 per pixel (averaged → threshold at 0.5)
      score_map   — continuous SVM decision score (averaged)
    """
    H, W = image_shape[:2]
    ws   = config.WINDOW_SIZE

    # binary
    bin_sum   = np.zeros((H, W), dtype=np.float32)
    # score (shift to 0+ for display)
    score_min = scores.min()
    scores_pos = scores - score_min          # all positive now

    score_sum = np.zeros((H, W), dtype=np.float32)
    count     = np.zeros((H, W), dtype=np.float32)

    for (r, c), pred, score in zip(origins, predictions, scores_pos):
        bin_sum[r:r+ws,   c:c+ws] += float(pred)
        score_sum[r:r+ws, c:c+ws] += score
        count[r:r+ws,     c:c+ws] += 1.0

    count      = np.maximum(count, 1.0)
    binary_map = (bin_sum / count >= 0.5).astype(np.uint8)   # majority vote
    score_map  = (score_sum / count).astype(np.float32)

    return binary_map, score_map


def save_svm_maps(binary_map: np.ndarray, score_map: np.ndarray,
                  src_profile: dict):
    """Save binary and score maps as georeferenced GeoTIFFs."""
    os.makedirs(os.path.dirname(config.SVM_MAP_PATH), exist_ok=True)

    # binary map — uint8, values 0 or 1
    profile = src_profile.copy()
    profile.update(count=1, dtype="uint8", compress="lzw", nodata=None)
    with rasterio.open(config.SVM_MAP_PATH, "w", **profile) as dst:
        dst.write(binary_map, 1)
    print(f"  Saved binary SVM map → {config.SVM_MAP_PATH}")

    # score map — float32
    score_path = config.SVM_MAP_PATH.replace(".tif", "_score.tif")
    profile.update(dtype="float32")
    with rasterio.open(score_path, "w", **profile) as dst:
        dst.write(score_map, 1)
    print(f"  Saved SVM score map  → {score_path}")


def run_one_class_svm(ref_features: np.ndarray,
                      all_features: np.ndarray,
                      origins: list,
                      image_shape: tuple,
                      src_profile: dict):
    """
    Full one-class SVM pipeline: train → predict → build maps → save.
    Called from search.py.
    """
    predictions, scores, scaler = train_one_class_svm(ref_features,
                                                       all_features)
    binary_map, score_map = build_svm_heatmap(origins, predictions,
                                               scores, image_shape)
    save_svm_maps(binary_map, score_map, src_profile)

    # print which feature dimensions the SVM found most discriminating
    print(f"\n  In QGIS:")
    print(f"    Load {config.SVM_MAP_PATH}")
    print(f"    Style → Paletted/Unique values")
    print(f"    Value 1 = favela-like windows")
    print(f"    Value 0 = everything else")

    return binary_map, score_map
