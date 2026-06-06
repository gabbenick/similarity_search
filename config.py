"""
config.py — Similarity Search Pipeline Configuration (Sentinel-2 edition)

Adapted for multi-band Sentinel-2 imagery (B2, B3, B4, B8, B11, B12).
Key differences from the original aerial (0.5 m) version:
  • Multi-band loading + resampling 20 m bands (B11/B12) to the 10 m grid
  • Spectral features generalised to N bands
  • Spectral INDICES (NDVI, NDBI, BSI, MNDWI, SWIR ratio) — the real strength
    of Sentinel-2 for material/land-cover discrimination
  • Texture / edge OFF by default (meaningless at 10 m/px)
  • Window size in pixels rethought for 10 m resolution
"""

import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Sentinel-2 bands
# ─────────────────────────────────────────────────────────────────────────────
# The ORDER here defines the order of spectral features and the band names the
# index formulas refer to. Keep the 6 standard names unless you adapt the index
# functions in extract_features.py.
BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]

# ── Option A: one GeoTIFF per band (default) ─────────────────────────────────
USE_STACKED = False
BANDS = {
    "B2":  os.path.join(DATA_DIR, "B2.tif"),
    "B3":  os.path.join(DATA_DIR, "B3.tif"),
    "B4":  os.path.join(DATA_DIR, "B4.tif"),
    "B8":  os.path.join(DATA_DIR, "B8.tif"),
    "B11": os.path.join(DATA_DIR, "B11.tif"),
    "B12": os.path.join(DATA_DIR, "B12.tif"),
}

# ── Option B: a single multi-band GeoTIFF (set USE_STACKED = True) ────────────
STACKED_PATH = os.path.join(DATA_DIR, "sentinel2_stack.tif")
# 1-based band index of each name inside the stacked file:
STACKED_BAND_INDEX = {"B2": 1, "B3": 2, "B4": 3, "B8": 4, "B11": 5, "B12": 6}

# Reference band that defines the output grid (10 m). All other bands are
# resampled onto this grid. coords.py also uses this as IMAGE_PATH.
REFERENCE_BAND = "B2"
IMAGE_PATH = (STACKED_PATH if USE_STACKED else BANDS[REFERENCE_BAND])

# ── Training shapefile (polygon-based reference extraction) ───────────────────
# Set to None to fall back to the REFERENCE_* bounding-box below.
REFERENCE_SHAPEFILE = os.path.join(DATA_DIR, "train.shp")

# Natural-colour composite (B4-B3-B2) for the overview PNG / reference preview.
RGB_DISPLAY_BANDS = ["B4", "B3", "B2"]

# Target pixel size after resampling (metres). Sentinel-2 native: 10 m.
PIXEL_SIZE_M = 10

# ─────────────────────────────────────────────────────────────────────────────
#  Sliding window  — re-think for 10 m/px!
# ─────────────────────────────────────────────────────────────────────────────
# At 10 m/px:  WINDOW_SIZE px  ->  WINDOW_SIZE * 10 m on the ground.
#   30 px = 300 m, 50 px = 500 m. Favelas are detected at AGGLOMERATION scale,
#   not building scale. Tune to your target.
WINDOW_SIZE = 15     # px  = ~150 m — fits the smallest training communities
STRIDE      = 10     # px  = ~67% overlap

# ─────────────────────────────────────────────────────────────────────────────
#  Reference zone — your known favela area (PIXEL coords ON THE 10 m GRID)
# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTANT: these MUST be re-derived for the Sentinel-2 image. The old aerial
#  pixel coordinates do NOT apply. Use coords.py with your QGIS map coordinates:
#       python coords.py --x <easting> --y <northing>
#  then set the bounding box below. The zone must be at least WINDOW_SIZE px.
REFERENCE_COL_MIN = 0      # ← EDIT ME
REFERENCE_COL_MAX = 60     # ← EDIT ME
REFERENCE_ROW_MIN = 0      # ← EDIT ME
REFERENCE_ROW_MAX = 60     # ← EDIT ME

# ─────────────────────────────────────────────────────────────────────────────
#  Feature groups
# ─────────────────────────────────────────────────────────────────────────────
USE_SPECTRAL = True    # per-band mean + std            (2 × n_bands values)
USE_INDICES  = True    # spectral indices               (2 × n_indices values)
USE_TEXTURE  = False   # GLCM on TEXTURE_BAND — WEAK at 10 m, off by default
USE_EDGE     = False   # Canny on TEXTURE_BAND — WEAK at 10 m, off by default
TEXTURE_BAND = "B8"    # which band to compute texture/edge on, if enabled

# Spectral indices to compute (each contributes mean + std).
# Formulas live in extract_features.py and assume the 6 standard bands.
#   NDVI  = (B8  - B4) / (B8  + B4)     vegetation      (favela: low)
#   NDBI  = (B11 - B8) / (B11 + B8)     built-up        (favela: high)
#   BSI   = ((B11+B4)-(B8+B2)) / ((B11+B4)+(B8+B2))   bare soil (favela: high)
#   MNDWI = (B3 - B11) / (B3 + B11)     water / shadow
#   SWIR_RATIO = B11 / B12              roof-material discrimination
INDICES = ["NDVI", "NDBI", "BSI", "MNDWI", "SWIR_RATIO"]

# ─────────────────────────────────────────────────────────────────────────────
#  DSM / RF — optional extra sources (must be aligned to the 10 m grid)
# ─────────────────────────────────────────────────────────────────────────────
USE_DSM  = True
DSM_PATH = os.path.join(DATA_DIR, "dsm.tif")

USE_RF   = False
RF_PATH  = os.path.join(DATA_DIR, "rf_map.tif")
RF_CLASS_IDS = {
    "exposed_soil"    : 1,
    "ceramic_roof"    : 2,
    "fiber_cement"    : 3,
    "paved_road"      : 4,
    "dense_vegetation": 5,
    "light_vegetation": 6,
    "tile_pavement"   : 7,
}

# ─────────────────────────────────────────────────────────────────────────────
#  One-class SVM
# ─────────────────────────────────────────────────────────────────────────────
USE_ONE_CLASS_SVM = True
SVM_NU            = 0.1
SVM_KERNEL        = "rbf"

# ─────────────────────────────────────────────────────────────────────────────
#  Similarity metric for heatmap
# ─────────────────────────────────────────────────────────────────────────────
SIMILARITY_METRIC = "cosine"   # "cosine" | "euclidean"

# ─────────────────────────────────────────────────────────────────────────────
#  Output paths
# ─────────────────────────────────────────────────────────────────────────────
HEATMAP_PATH      = os.path.join(OUTPUT_DIR, "heatmap.tif")
HEATMAP_NORM_PATH = os.path.join(OUTPUT_DIR, "heatmap_norm.tif")
SVM_MAP_PATH      = os.path.join(OUTPUT_DIR, "svm_map.tif")
SVM_SCORE_PATH    = os.path.join(OUTPUT_DIR, "svm_map_score.tif")
FEATURES_CSV      = os.path.join(OUTPUT_DIR, "features.csv")
OVERVIEW_PNG      = os.path.join(OUTPUT_DIR, "overview.png")
