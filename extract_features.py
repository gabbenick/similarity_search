"""
extract_features.py — Feature extraction per sliding window.

Feature groups
──────────────
SPECTRAL  : RGB mean + std                    (6 values)
TEXTURE   : GLCM contrast/homogeneity/etc.   (4 values)
EDGE      : Canny edge density                (1 value)
DSM       : height mean, std, roughness       (3 values)  ← when USE_DSM=True
RF        : class proportions + informality   (7 values)  ← when USE_RF=True

All values z-score normalised before similarity computation.
"""

import numpy as np
import cv2
from skimage.feature import graycomatrix, graycoprops
import config


# ─────────────────────────────────────────────────────────────────────────────
#  Individual feature functions
# ─────────────────────────────────────────────────────────────────────────────

def spectral_features(patch_rgb: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(3):
        ch = patch_rgb[:, :, c].astype(np.float32)
        feats.append(ch.mean())
        feats.append(ch.std())
    return np.array(feats, dtype=np.float32)


def texture_features(patch_rgb: np.ndarray) -> np.ndarray:
    grey  = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    grey_q = (grey // 4).astype(np.uint8)
    glcm  = graycomatrix(grey_q, distances=[1],
                         angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                         levels=64, symmetric=True, normed=True)
    return np.array([
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'homogeneity').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'correlation').mean(),
    ], dtype=np.float32)


def edge_features(patch_rgb: np.ndarray) -> np.ndarray:
    grey   = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)
    edges  = cv2.Canny(grey, threshold1=50, threshold2=150)
    return np.array([edges.astype(np.float32).mean() / 255.0],
                    dtype=np.float32)


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
    """
    Compute class proportion features from a Random Forest classification patch.

    patch_rf : (H, W) int  — pixel class IDs matching config.RF_CLASS_IDS
    Returns  : (7,) float

    Features
    ────────
    pct_ceramic      % ceramic roof pixels
    pct_fiber        % fiber cement pixels
    pct_paved_road   % paved road pixels
    pct_exposed_soil % exposed soil pixels
    pct_vegetation   % dense + light vegetation pixels
    informality_idx  ceramic + fiber + exposed_soil  (high = more informal)
    formality_idx    paved_road + vegetation         (high = more formal)

    Scientific rationale
    ────────────────────
    Informal settlements in Brazilian cities are characterised by high
    proportions of ceramic and fiber cement roofing, exposed soil (unpaved
    lots), and low proportions of formal paved infrastructure and managed
    vegetation. These proportions, computed at the neighbourhood window level,
    encode the urban morphology signature that pixel-level spectral features
    cannot capture alone.
    """
    ids   = config.RF_CLASS_IDS
    total = patch_rf.size + 1e-8

    pct_ceramic  = (patch_rf == ids["ceramic_roof"]).sum()    / total
    pct_fiber    = (patch_rf == ids["fiber_cement"]).sum()    / total
    pct_paved    = (patch_rf == ids["paved_road"]).sum()      / total
    pct_soil     = (patch_rf == ids["exposed_soil"]).sum()    / total
    pct_veg      = ((patch_rf == ids["dense_vegetation"]) |
                    (patch_rf == ids["light_vegetation"])).sum() / total

    informality  = float(pct_ceramic + pct_fiber + pct_soil)
    formality    = float(pct_paved   + pct_veg)

    return np.array([
        pct_ceramic, pct_fiber, pct_paved,
        pct_soil,    pct_veg,
        informality, formality,
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Combined feature vector for one window
# ─────────────────────────────────────────────────────────────────────────────

def compute_window_features(patch_rgb: np.ndarray,
                            patch_dsm: np.ndarray = None,
                            patch_rf:  np.ndarray = None) -> np.ndarray:
    parts = []
    if config.USE_SPECTRAL:
        parts.append(spectral_features(patch_rgb))
    if config.USE_TEXTURE:
        parts.append(texture_features(patch_rgb))
    if config.USE_EDGE:
        parts.append(edge_features(patch_rgb))
    if config.USE_DSM and patch_dsm is not None:
        parts.append(dsm_features(patch_dsm))
    if config.USE_RF and patch_rf is not None:
        parts.append(rf_composition_features(patch_rf))
    return np.concatenate(parts).astype(np.float32)


def feature_names() -> list:
    names = []
    if config.USE_SPECTRAL:
        for ch in ['R', 'G', 'B']:
            names += [f'{ch}_mean', f'{ch}_std']
    if config.USE_TEXTURE:
        names += ['GLCM_contrast', 'GLCM_homogeneity',
                  'GLCM_energy', 'GLCM_correlation']
    if config.USE_EDGE:
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

def extract_all_windows(image_rgb: np.ndarray,
                        image_dsm: np.ndarray = None,
                        image_rf:  np.ndarray = None):
    H, W   = image_rgb.shape[:2]
    ws     = config.WINDOW_SIZE
    stride = config.STRIDE

    rows = list(range(0, H - ws + 1, stride))
    cols = list(range(0, W - ws + 1, stride))
    if rows[-1] + ws < H: rows.append(H - ws)
    if cols[-1] + ws < W: cols.append(W - ws)

    origins  = [(r, c) for r in rows for c in cols]
    total    = len(origins)
    feat_dim = len(feature_names())
    features = np.zeros((total, feat_dim), dtype=np.float32)

    print(f"  Windows: {len(rows)} rows × {len(cols)} cols = {total} total")

    for i, (r, c) in enumerate(origins):
        patch_rgb = image_rgb[r:r+ws, c:c+ws]
        patch_dsm = image_dsm[r:r+ws, c:c+ws] if image_dsm is not None else None
        patch_rf  = image_rf[r:r+ws,  c:c+ws] if image_rf  is not None else None
        features[i] = compute_window_features(patch_rgb, patch_dsm, patch_rf)
        if (i + 1) % 100 == 0 or i == total - 1:
            print(f"  {i+1}/{total}", end="\r")

    print()
    return origins, features


# ─────────────────────────────────────────────────────────────────────────────
#  Extract reference patches from the known favela zone
# ─────────────────────────────────────────────────────────────────────────────

def extract_reference_patches(image_rgb: np.ndarray,
                               image_dsm: np.ndarray = None,
                               image_rf:  np.ndarray = None) -> np.ndarray:
    """
    Return feature matrix (N_patches, N_features) for all windows
    that fit inside the reference zone.
    Used both for computing the mean reference vector AND for training
    the one-class SVM.
    """
    r_min, r_max = config.REFERENCE_ROW_MIN, config.REFERENCE_ROW_MAX
    c_min, c_max = config.REFERENCE_COL_MIN, config.REFERENCE_COL_MAX
    ws = config.WINDOW_SIZE

    ref_patches = []
    for r in range(r_min, r_max - ws + 1, ws // 2):
        for c in range(c_min, c_max - ws + 1, ws // 2):
            patch_rgb = image_rgb[r:r+ws, c:c+ws]
            patch_dsm = image_dsm[r:r+ws, c:c+ws] if image_dsm is not None else None
            patch_rf  = image_rf[r:r+ws,  c:c+ws] if image_rf  is not None else None
            ref_patches.append(
                compute_window_features(patch_rgb, patch_dsm, patch_rf))

    if not ref_patches:
        raise ValueError(
            "No complete windows fit inside the reference zone.\n"
            "Check REFERENCE_* coordinates in config.py.")

    ref_matrix = np.stack(ref_patches)
    print(f"  Reference: {len(ref_patches)} windows from the favela zone")
    return ref_matrix


def extract_reference_vector(image_rgb, image_dsm=None, image_rf=None):
    """Return the mean reference feature vector."""
    return extract_reference_patches(image_rgb, image_dsm, image_rf).mean(axis=0)
