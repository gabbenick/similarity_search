"""
config.py — Similarity Search Pipeline Configuration
"""

import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "data", "image.tif")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  Sliding window
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_SIZE = 200    # pixels  (100m × 100m at 50cm/px)
STRIDE      = 100    # pixels  (50% overlap)

# ─────────────────────────────────────────────────────────────────────────────
#  Reference zone — your known favela area (pixel coordinates)
# ─────────────────────────────────────────────────────────────────────────────
# Your QGIS map coordinates for the two corners:
#   Top-left     : X=187211.06  Y=8950614.91
#   Bottom-right : X=187340.97  Y=8950498.11
# Converted to pixels using coords.py → values below.

REFERENCE_COL_MIN = 925
REFERENCE_COL_MAX = 1125
REFERENCE_ROW_MIN = 1750
REFERENCE_ROW_MAX = 1950

# ─────────────────────────────────────────────────────────────────────────────
#  RGB + texture + edge features
# ─────────────────────────────────────────────────────────────────────────────
USE_SPECTRAL = True   # per-channel mean + std              (6 values)
USE_TEXTURE  = True   # GLCM contrast, homogeneity, etc.   (4 values)
USE_EDGE     = True   # Canny edge density                  (1 value)

# ─────────────────────────────────────────────────────────────────────────────
#  DSM / slope features
# ─────────────────────────────────────────────────────────────────────────────
USE_DSM  = True
DSM_PATH = os.path.join(BASE_DIR, "data", "dsm.tif")

# ─────────────────────────────────────────────────────────────────────────────
#  Random Forest composition features
# ─────────────────────────────────────────────────────────────────────────────
# Set USE_RF = True when you receive the RF classification map.
# Drop the file into data/rf_map.tif and flip the flag — nothing else changes.
#
# Expected RF class IDs (must match the other student's output):
#   1 = Exposed Soil
#   2 = Ceramic Roof
#   3 = Fiber Cement Roof
#   4 = Paved Road
#   5 = Dense Vegetation
#   6 = Light Vegetation
#   7 = Tile Pavement
#
# Features computed per window from the RF map:
#   pct_ceramic      — % ceramic roof pixels
#   pct_fiber        — % fiber cement pixels
#   pct_paved_road   — % paved road pixels
#   pct_exposed_soil — % exposed soil pixels
#   pct_vegetation   — % dense + light vegetation pixels
#   informality_idx  — ceramic + fiber + soil  (high = informal)
#   formality_idx    — paved road + vegetation (high = formal)

USE_RF   = True   # ← flip to True when rf_map.tif is ready
RF_PATH  = os.path.join(BASE_DIR, "data", "rf_map.tif")

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
# Trained only on windows from the reference favela zone.
# Produces a binary map: inside favela distribution / outside.
# Runs automatically alongside the similarity heatmap.
#
# nu     : upper bound on fraction of outliers (0.0–1.0)
#          lower = tighter boundary around the reference
# kernel : "rbf" works well for most cases

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
FEATURES_CSV      = os.path.join(OUTPUT_DIR, "features.csv")
OVERVIEW_PNG      = os.path.join(OUTPUT_DIR, "overview.png")
