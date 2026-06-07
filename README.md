# Favela Detection — Maceió (RMM)

Detecting informal settlements (*favelas* / *aglomerados subnormais*) in
**Sentinel-2** satellite imagery over the **Região Metropolitana de Maceió**
(Alagoas, Brazil), using a supervised machine-learning model trained on IBGE's
official favela polygons.

The output is a georeferenced **favela-probability map (0–1)** for QGIS, used as
a **screening / triage tool** to flag candidate areas for review.

> **Scope:** this model is valid **inside the RMM only** (where the
> socioeconomic data exists). Use `output/favela_probability_rmm.tif`.

---

## Explain it in one paragraph

> We take a satellite image of the Maceió metro region and slide a small window
> (~150 m) across it. For each window we measure what it looks like — colours and
> spectral indices, how *mixed* it is (favelas have jumbled rooftops), the terrain,
> and socioeconomic context (vulnerability, low income, flood/landslide risk). We
> then train a model on the government's official favela map to learn the
> difference between a favela and other built-up areas. The result is a heatmap
> where bright = "likely favela." It catches ~90% of known favelas and is meant to
> **point reviewers at candidate areas**, not to draw exact boundaries.

**Why it's useful:** the official favela map (IBGE AGSN) is incomplete outside the
city core. This model can surface settlements that were never mapped — several of
the "false alarms" it produces turn out to be real, unmapped favelas.

---

## How it works

```
Sentinel-2 bands (10 m)  +  terrain (DSM)  +  socio layers (RMM)
        ↓
  slide a 15 px (~150 m) window, stride 10 px
        ↓
  32 features / window  (spectral, indices, terrain, socio)
        ↓
  RandomForest, favela vs. other-built-up
  trained on IBGE AGSN polygons, spatial-block cross-validation
        ↓
  favela_probability_rmm.tif  (0–1)  → load in QGIS
```

- **Supervised** (favela vs. non-favela), not a similarity heatmap. The earlier
  unsupervised approach could not separate favelas from other urban areas; it is
  kept only in `search.py` / `oneclass.py` for comparison.
- **Spatial cross-validation:** the region is split into ~8 km blocks; every window
  is predicted by a model that never saw its block → no spatial leakage, honest
  scores even on training favelas.
- **RMM-scoped training** (`config.RMM_ONLY`): positives *and* negatives are sampled
  only inside the RMM, so the model learns to reject the RMM's *own* confusing
  built-up (industrial, quarries, bare soil) instead of irrelevant rural land.

---

## Results (out-of-fold, AGSN-evaluated, inside the RMM)

| Metric | Value |
|---|---|
| ROC-AUC (favela vs. rest) | **0.979** |
| Detection — any hot window (max ≥ 0.5), core / periphery | **96.8% / 91.9%** |
| Detection — object-level (≥25% lit, T 0.5) | **89.9% / 89.2%** |
| Object **precision** @ 0.7 / @ 0.9 | 18.3% / 39.0% |

**Read this correctly:**
- It's a strong **screening** tool — it finds ~90–97% of known favelas.
- **Precision is a floor, not the truth:** AGSN is complete only in the Maceió core,
  so some flagged "false" areas in the periphery are real unmapped favelas (see
  `output/false_blobs_rmm.gpkg` — review in QGIS).
- **Threshold is the dial:** T ≈ 0.7 for screening; raise it to trade recall for
  precision.

See `CLAUDE.md` for the full methodology, data inventory, and decision log.

---

## Data (`data/`, gitignored)

- **Sentinel-2:** `B2,B3,B4,B8.tif` (10 m) + `B11,B12.tif` (20 m→10 m). Reference grid:
  EPSG:4326, ~10 m/px, 8399×7988 px, covering the Alagoas coast incl. the RMM.
- **Terrain:** `dsm.tif` (UTM 25S, 30 m) — auto-reprojected.
- **Socio/env (RMM only, EPSG:31985):** vulnerability, landslide & flood
  susceptibility rasters + low-income census sectors. Cover ~42 % of the image.
- **Labels:** IBGE AGSN_2019 favela polygons (`comunidades/`) → `data/agsn_truth.gpkg`
  (273 favelas in-image; 254 inside the RMM). Hand-drawn field communities →
  `data/handdrawn_test.gpkg` (independent generalisation test).

---

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Run (current supervised workflow)

```bash
source venv/bin/activate

# 1. Build ground-truth layers (only if the shapefiles changed)
python build_agsn_truth.py        # → data/agsn_truth.gpkg
python build_handdrawn_test.py    # → data/handdrawn_test.gpkg

# 2. Extract features (slow, ~671k windows — only if bands/config change)
python search.py                  # → output/features.csv
python add_slope_feature.py       # + DSM slope (fast)
python add_socio_features.py      # + socio/env columns (~40 s)

# 3. Train + evaluate (fast — reuses features.csv)
python train_supervised.py        # → output/favela_probability_rmm.tif (+ oof_eval.npz)
python evaluate_rmm.py            # → in-RMM precision/recall + false_blobs_rmm.gpkg
python make_figures.py            # → output/fig_*.png (ROC/PR, calibration, importance, operating)
```

> Normal loop is just step 3. `evaluate_polygons.py` is the older full-image
> evaluator; `evaluate_rmm.py` is the current RMM-scoped one.

## Key config (`config.py`)

```python
WINDOW_SIZE = 15      # px ≈ 150 m
STRIDE      = 10      # px (~67% overlap)
USE_DSM     = True    # terrain features
USE_SOCIO   = True    # socio/env features
RMM_ONLY    = True    # train inside the RMM only (scope decision)
```

## Load in QGIS

```
Layer → Add Raster Layer → output/favela_probability_rmm.tif
Symbology → Singleband pseudocolor, ramp Reds, 0–1.   Triage at ≈ 0.7.
```

Also load `output/false_blobs_rmm.gpkg` to review flagged areas that don't match a
known favela — some are real unmapped settlements.

---

*Earlier docs `README_SENTINEL2.md` (Sentinel-2 migration notes) and the
unsupervised/One-Class workflow are historical and superseded by the supervised
pipeline described here.*
