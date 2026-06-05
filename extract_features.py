"""
extract_features.py — Feature extraction for multi-band Sentinel-2 imagery.

A "patch" is now an (H, W, n_bands) array whose band order matches
config.BAND_ORDER. Feature groups:

  • spectral  — per-band mean + std                       (2 × n_bands)
  • indices   — NDVI / NDBI / BSI / MNDWI / SWIR ratio    (2 × n_indices)
  • texture   — GLCM on one band (weak at 10 m, optional) (4)
  • edge      — Canny density on one band (optional)      (1)
  • DSM / RF  — optional extra sources                    (3 / 7)
"""

import numpy as np
import config

try:
    import cv2
    from skimage.feature import graycomatrix, graycoprops
    _HAS_CV = True
except ImportError:                       # texture/edge unavailable
    _HAS_CV = False


# ─────────────────────────────────────────────────────────────────────────────
#  Band access helpers
# ─────────────────────────────────────────────────────────────────────────────

def _band(patch: np.ndarray, name: str) -> np.ndarray:
    """Return the (H, W) float array for a named band in a patch."""
    return patch[:, :, config.BAND_ORDER.index(name)].astype(np.float32)


def _to_uint8(band: np.ndarray) -> np.ndarray:
    """Min-max scale a single band to uint8 (for OpenCV texture/edge)."""
    b = band.astype(np.float32)
    mn, mx = np.nanmin(b), np.nanmax(b)
    if not np.isfinite(mn) or not np.isfinite(mx) or (mx - mn) < 1e-8:
        return np.zeros(b.shape, dtype=np.uint8)
    return ((b - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral — per-band mean + std
# ─────────────────────────────────────────────────────────────────────────────

def spectral_features(patch: np.ndarray) -> np.ndarray:
    feats = []
    for b in range(patch.shape[2]):
        ch = patch[:, :, b].astype(np.float32)
        feats.append(float(np.nanmean(ch)))
        feats.append(float(np.nanstd(ch)))
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Spectral indices  — the core of the Sentinel-2 approach
# ─────────────────────────────────────────────────────────────────────────────

def _compute_index(patch: np.ndarray, name: str) -> np.ndarray:
    """Return the (H, W) array for a named spectral index."""
    eps = 1e-8
    b2  = _band(patch, "B2")
    b3  = _band(patch, "B3")
    b4  = _band(patch, "B4")
    b8  = _band(patch, "B8")
    b11 = _band(patch, "B11")
    b12 = _band(patch, "B12")

    if name == "NDVI":
        return (b8 - b4) / (b8 + b4 + eps)
    if name == "NDBI":
        return (b11 - b8) / (b11 + b8 + eps)
    if name == "BSI":
        return ((b11 + b4) - (b8 + b2)) / ((b11 + b4) + (b8 + b2) + eps)
    if name == "MNDWI":
        return (b3 - b11) / (b3 + b11 + eps)
    if name == "SWIR_RATIO":
        return b11 / (b12 + eps)
    raise ValueError(f"Unknown index: {name}")


def index_features(patch: np.ndarray) -> np.ndarray:
    feats = []
    for name in config.INDICES:
        idx = _compute_index(patch, name)
        feats.append(float(np.nanmean(idx)))
        feats.append(float(np.nanstd(idx)))
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Texture / edge  (single band — WEAK at 10 m, off by default)
# ─────────────────────────────────────────────────────────────────────────────

def texture_features(patch: np.ndarray) -> np.ndarray:
    grey   = _to_uint8(_band(patch, config.TEXTURE_BAND))
    grey_q = (grey // 4).astype(np.uint8)
    glcm   = graycomatrix(grey_q, distances=[1],
                          angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                          levels=64, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
    ], dtype=np.float32)


def edge_features(patch: np.ndarray) -> np.ndarray:
    grey  = _to_uint8(_band(patch, config.TEXTURE_BAND))
    edges = cv2.Canny(grey, threshold1=50, threshold2=150)
    return np.array([edges.astype(np.float32).mean() / 255.0], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  DSM / RF  (optional, aligned to the 10 m grid)
# ─────────────────────────────────────────────────────────────────────────────

def dsm_features(patch_dsm: np.ndarray) -> np.ndarray:
    valid = patch_dsm[~np.isnan(patch_dsm)]
    if len(valid) == 0:
        return np.zeros(3, dtype=np.float32)
    mean_h    = float(valid.mean())
    std_h     = float(valid.std())
    diff_r    = np.abs(np.diff(patch_dsm, axis=0))
    diff_c    = np.abs(np.diff(patch_dsm, axis=1))
    roughness = float(np.nanmean(
        np.concatenate([diff_r.ravel(), diff_c.ravel()])))
    return np.array([mean_h, std_h, roughness], dtype=np.float32)


def rf_composition_features(patch_rf: np.ndarray) -> np.ndarray:
    ids   = config.RF_CLASS_IDS
    total = patch_rf.size + 1e-8
    pct_ceramic = (patch_rf == ids["ceramic_roof"]).sum() / total
    pct_fiber   = (patch_rf == ids["fiber_cement"]).sum() / total
    pct_paved   = (patch_rf == ids["paved_road"]).sum()   / total
    pct_soil    = (patch_rf == ids["exposed_soil"]).sum() / total
    pct_veg     = ((patch_rf == ids["dense_vegetation"]) |
                   (patch_rf == ids["light_vegetation"])).sum() / total
    informality = float(pct_ceramic + pct_fiber + pct_soil)
    formality   = float(pct_paved + pct_veg)
    return np.array([pct_ceramic, pct_fiber, pct_paved, pct_soil,
                     pct_veg, informality, formality], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Combined feature vector for one window
# ─────────────────────────────────────────────────────────────────────────────

def compute_window_features(patch: np.ndarray,
                            patch_dsm: np.ndarray = None,
                            patch_rf:  np.ndarray = None) -> np.ndarray:
    parts = []
    if config.USE_SPECTRAL:
        parts.append(spectral_features(patch))
    if config.USE_INDICES:
        parts.append(index_features(patch))
    if config.USE_TEXTURE and _HAS_CV:
        parts.append(texture_features(patch))
    if config.USE_EDGE and _HAS_CV:
        parts.append(edge_features(patch))
    if config.USE_DSM and patch_dsm is not None:
        parts.append(dsm_features(patch_dsm))
    if config.USE_RF and patch_rf is not None:
        parts.append(rf_composition_features(patch_rf))
    return np.concatenate(parts).astype(np.float32)


def feature_names() -> list:
    names = []
    if config.USE_SPECTRAL:
        for b in config.BAND_ORDER:
            names += [f"{b}_mean", f"{b}_std"]
    if config.USE_INDICES:
        for ix in config.INDICES:
            names += [f"{ix}_mean", f"{ix}_std"]
    if config.USE_TEXTURE and _HAS_CV:
        names += ['GLCM_contrast', 'GLCM_homogeneity',
                  'GLCM_energy', 'GLCM_correlation']
    if config.USE_EDGE and _HAS_CV:
        names += ['edge_density']
    if config.USE_DSM:
        names += ['DSM_mean', 'DSM_std', 'DSM_roughness']
    if config.USE_RF:
        names += ['pct_ceramic', 'pct_fiber', 'pct_paved_road',
                  'pct_exposed_soil', 'pct_vegetation',
                  'informality_idx', 'formality_idx']
    return names


# ─────────────────────────────────────────────────────────────────────────────
#  Extract features for ALL windows
# ─────────────────────────────────────────────────────────────────────────────

def extract_all_windows(image: np.ndarray,
                        image_dsm: np.ndarray = None,
                        image_rf:  np.ndarray = None):
    H, W   = image.shape[:2]
    ws     = config.WINDOW_SIZE
    stride = config.STRIDE

    rows = list(range(0, H - ws + 1, stride))
    cols = list(range(0, W - ws + 1, stride))
    if not rows or not cols:
        raise ValueError(
            f"WINDOW_SIZE ({ws}) larger than image ({H}×{W}). "
            f"Lower WINDOW_SIZE in config.py.")
    if rows[-1] + ws < H: rows.append(H - ws)
    if cols[-1] + ws < W: cols.append(W - ws)

    origins  = [(r, c) for r in rows for c in cols]
    total    = len(origins)
    feat_dim = len(feature_names())
    features = np.zeros((total, feat_dim), dtype=np.float32)

    print(f"  Windows: {len(rows)} rows × {len(cols)} cols = {total} total")

    for i, (r, c) in enumerate(origins):
        patch     = image[r:r+ws, c:c+ws]
        patch_dsm = image_dsm[r:r+ws, c:c+ws] if image_dsm is not None else None
        patch_rf  = image_rf[r:r+ws,  c:c+ws] if image_rf  is not None else None
        features[i] = compute_window_features(patch, patch_dsm, patch_rf)
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  {i+1}/{total}", end="\r")

    print()
    return origins, features


# ─────────────────────────────────────────────────────────────────────────────
#  Extract reference patches from the known favela zone
# ─────────────────────────────────────────────────────────────────────────────

def extract_reference_patches(image: np.ndarray,
                               image_dsm: np.ndarray = None,
                               image_rf:  np.ndarray = None) -> np.ndarray:
    r_min, r_max = config.REFERENCE_ROW_MIN, config.REFERENCE_ROW_MAX
    c_min, c_max = config.REFERENCE_COL_MIN, config.REFERENCE_COL_MAX
    ws = config.WINDOW_SIZE
    step = max(1, ws // 2)

    ref_patches = []
    for r in range(r_min, r_max - ws + 1, step):
        for c in range(c_min, c_max - ws + 1, step):
            patch     = image[r:r+ws, c:c+ws]
            patch_dsm = image_dsm[r:r+ws, c:c+ws] if image_dsm is not None else None
            patch_rf  = image_rf[r:r+ws,  c:c+ws] if image_rf  is not None else None
            ref_patches.append(
                compute_window_features(patch, patch_dsm, patch_rf))

    if not ref_patches:
        raise ValueError(
            "No complete windows fit inside the reference zone.\n"
            f"Reference is {r_max-r_min}×{c_max-c_min} px but WINDOW_SIZE={ws}.\n"
            "Enlarge REFERENCE_* or lower WINDOW_SIZE in config.py.")

    ref_matrix = np.stack(ref_patches)
    print(f"  Reference: {len(ref_patches)} windows from the favela zone")
    return ref_matrix


def extract_reference_vector(image, image_dsm=None, image_rf=None):
    return extract_reference_patches(image, image_dsm, image_rf).mean(axis=0)
