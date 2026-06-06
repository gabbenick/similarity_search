# CLAUDE.md — Similarity Search Pipeline

> Read this at the start of every session to understand the current state of the project.

## What this project does

Detects informal settlements (favelas) in Sentinel-2 satellite imagery. The pipeline
slides a window across the image, extracts spectral + terrain features per window,
and compares each window to a **training reference zone** made of known favela polygons.
The output is a heatmap (cosine similarity) and a One-Class SVM binary map, both
georeferenced GeoTIFFs ready to load in QGIS.

---

## Repository layout

```
similarity_search_clone/
├── config.py            # all tuneable parameters (edit here first)
├── coords.py            # QGIS coord → pixel converter
├── extract_features.py  # feature extraction per window/patch
├── search.py            # main pipeline: load → extract → heatmap → SVM
├── oneclass.py          # One-Class SVM training & prediction
├── validate.py          # validation against ground-truth shapefile
├── comunidades/         # 9 community shapefiles (source polygons)
├── data/                # rasters + train/val shapefiles (gitignored)
└── output/              # pipeline outputs (gitignored)
```

---

## Input data (`data/`)

### Sentinel-2 bands (all identical grid)
| File | Band | Native res | In data/ |
|------|------|-----------|---------|
| B2.tif | Blue | 10 m | ✅ |
| B3.tif | Green | 10 m | ✅ |
| B4.tif | Red | 10 m | ✅ |
| B8.tif | NIR | 10 m | ✅ |
| B11.tif | SWIR-1 | 20 m (already resampled to 10 m) | ✅ |
| B12.tif | SWIR-2 | 20 m (already resampled to 10 m) | ✅ |

**CRS:** EPSG:4326 · **Resolution:** 0.000090°/px ≈ 10 m · **Size:** 8 399 × 7 988 px  
**Coverage:** lon [-36.23, -35.47] · lat [-9.86, -9.14] (region of Alagoas, Brazil)

### DSM (altitude)
| File | CRS | Resolution | Range |
|------|-----|-----------|-------|
| dsm.tif | EPSG:31985 (UTM 25S) | 30 m | 0–228 m |

The DSM is **automatically reprojected** to the B2 grid at runtime by `search.py::load_dsm()`.
No manual preprocessing needed.

### Pre-computed composites (in data/, not used by pipeline)
`RGB_432.tif`, `FCC_843.tif`, `SWIR_12114.tif`, `Urbano_1184.tif`, `Stack_Urbano.tif`,
`NDVI.tif`, `NDBI.tif`, `MNDWI.tif` — useful for QGIS visualisation only.

---

## Label sources (`comunidades/`)

Two kinds of favela labels:

**1. IBGE AGSN_2019 (primary).** Official *aglomerados subnormais* — 13,151
polygons statewide, **273 inside the image**, 11 municipalities. Cleaned by
`build_agsn_truth.py` → `data/agsn_truth.gpkg`. This is the training + CV source.

**2. Hand-drawn communities (independent test).** Field-known settlements,
digitised by hand, in MIXED CRSs (EPSG:4326 / 31984 / 31985). Consolidated by
`build_handdrawn_test.py` → `data/handdrawn_test.gpkg` (10 inside the image;
PARIPUEIRA dropped as a duplicate of ALTO_DA_BOA_VISTA; "22 de Julho" dropped —
it's in UTM zone 24S, west of the image).

| Hand-drawn community | AGSN overlap | Role |
|----------------------|-------------|------|
| BIQUINHA | 93.5% | (in AGSN) |
| NOVA_ESPERANCA | 98.7% | (in AGSN) |
| ALTO_DA_BOA_VISTA | 46.1% | (partial) |
| ANTENOR_MARINHO, BURACO_DO_JACARE, COQUEIRO_SECO, MESSIAS, RUA_DO_MATADOURO, **Conjunto Luiz Gonzaga**, **VILA DAVI** | 0% | **AGSN-missed → generalisation test** |

The 7 zero-overlap communities are the real prize: settlements AGSN never mapped,
used to test whether the AGSN-trained model generalises to unseen favelas.

> The old `data/train.shp` / `data/val.shp` (hand-drawn bbox split) are obsolete —
> superseded by AGSN training + spatial CV. Kept only for reference.

---

## Features used (25 total)

```
config.py: USE_SPECTRAL=True, USE_INDICES=True, USE_TEXTURE=False, USE_EDGE=False, USE_DSM=True
```

| Group | Features | Count |
|-------|----------|-------|
| Spectral (mean+std per band) | B2, B3, B4, B8, B11, B12 | 12 |
| Spectral indices (mean+std) | NDVI, NDBI, BSI, MNDWI, SWIR_RATIO | 10 |
| Terrain | DSM mean, std, roughness, **slope_mean, slope_std** | 5 |
| **Total** | | **27** |

Texture (GLCM) and edge (Canny) are OFF — meaningless at 10 m resolution.
Slope added via `add_slope_feature.py` (augments features.csv without re-extracting
spectral). It lands in the top-10 RF features (~7%) but did **not** move detection/
precision — these favelas are mostly low-lying, so terrain is a minor discriminator.
Spectral **heterogeneity** (`_std`) still dominates.

---

## Key config values (config.py)

```python
WINDOW_SIZE        = 15      # px ≈ 150 m — drives feature extraction (search.py)
STRIDE             = 10      # px — ~67% overlap
USE_DSM            = True    # terrain features on
# Legacy (unsupervised path only): REFERENCE_SHAPEFILE, USE_ONE_CLASS_SVM,
# SVM_NU, SIMILARITY_METRIC — not used by the supervised RF workflow.
```

---

## Current output files (`output/`)

**Supervised model (CURRENT — use these):**
| File | Description |
|------|-------------|
| `favela_probability.tif` | **RF favela probability 0–1, out-of-fold** (load in QGIS) |
| `polygon_scores.csv` | per-favela detection scores; lists the 44 missed favelas |
| `features.csv` | per-window feature table (27 features) — reused by training |

**Legacy / unsupervised (superseded — kept only for comparison):**
`heatmap.tif`, `heatmap_norm.tif`, `svm_map.tif`, `svm_map_score.tif`,
`overview.png`, `error_map.tif`, `validation_curves.png`.

---

## How to run (current supervised workflow)

```bash
source venv/bin/activate

# 1. Build / refresh ground-truth layers (only if shapefiles changed)
python build_agsn_truth.py        # → data/agsn_truth.gpkg (273 AGSN favelas)
python build_handdrawn_test.py    # → data/handdrawn_test.gpkg (10 communities)

# 2. (Re)extract features ONLY if config/bands change — slow (671k windows):
python search.py                  # regenerates output/features.csv (+ legacy maps)
python add_slope_feature.py       # adds slope cols to features.csv (fast)

# 3. Train + evaluate (fast — reuses features.csv):
python train_supervised.py        # → output/favela_probability.tif
python evaluate_polygons.py       # → polygon-level metrics + polygon_scores.csv
```

> Normal loop is just step 3. Steps 1–2 only when inputs change.
> Legacy unsupervised path: `python search.py` then `validate.py --truth ...`.

---

## Known warnings (cosmetic, not bugs)

1. `RuntimeWarning: Mean of empty slice` — a handful of edge-of-image patches have
   all-zero band values; `nan_to_num(..., nan=0.0)` in `compute_window_features`
   handles them downstream. Does not affect results.
2. `Ground extent ≈ 1 m × 1 m` — cosmetic print bug: `pixel_size` is in degrees
   (EPSG:4326), not metres. The actual resolution is ≈10 m/px. Does not affect outputs.

---

## AGSN ground truth + supervised classifier (current frontier)

`comunidades/AGSN_2019.shp` = IBGE's official *aglomerados subnormais* (13,151
polygons statewide; **273 inside our image**, across 11 municipalities). This is
the authoritative favela ground truth and replaces the 8 hand-drawn polygons as
the primary label set.

- `build_agsn_truth.py` → `data/agsn_truth.gpkg` (cleaned: reprojected to image
  grid, clipped, keeps geometry + NM_AGSN + SUM_EDOC + municipality, drops 30
  buggy "nearest-facility" columns). `SUM_EDOC` = household count per settlement.
- **Cross-check:** 7 of 10 hand-drawn communities have 0% AGSN overlap → they are
  small peripheral settlements AGSN misses. Kept as an INDEPENDENT test set
  (`data/handdrawn_test.gpkg`), not merged into training.

`train_supervised.py` — RandomForest, AGSN positives + hard (built-up) negatives,
**spatial-block cross-validation** (800 px blocks, 5 folds, no leakage). Reuses
`output/features.csv` (no re-extraction). Writes `output/favela_probability.tif`.

### Honest results (out-of-fold, full-image prevalence vs AGSN)

| Metric | Unsupervised heatmap | **Supervised RF** |
|--------|---------------------|-------------------|
| ROC-AUC | 0.899 | **0.951** |
| PR-AUC | 0.033 (~13× chance) | **0.124 (~50× chance)** |
| F1 | 0.089 | **0.194** |
| Precision | 0.056 | **0.192** |
| **favela-vs-other-urban ROC** | ~0.5 (couldn't separate) | **0.946** |

The supervised model genuinely separates favela from other built-up areas — the
capability the unsupervised approach fundamentally lacked.

### Polygon / object-level evaluation (`evaluate_polygons.py`)

The real use case is flagging candidate *areas*, so pixel metrics understate it.
Evaluated on the out-of-fold probability map:

**As a screening/recall tool — strong:**
- **84% of 273 AGSN favelas detected** (mean prob ≥ 0.5); only 43 missed (the smaller ones, median 2.0 ha).
- Favela polygons sit at median prob **0.77 vs image-wide 0.05** (~15× background).
- Detection by size: small 78% / medium 82% / large 92%.
- Generalisation: ~43% of the 7 AGSN-*unseen* hand-drawn favelas detected.

**As a precise delineator — weak (over-flags):**
- Object precision 3.5–14% — it flags ~2,400 candidate blobs for ~280 known favelas.
- Partly real over-flagging, partly an artefact: AGSN isn't comprehensive outside
  the Maceió core, so some "false" regions are likely *unmapped* favelas.
- → precision is the work item: better hard negatives, slope, socioeconomic layers.

Per-polygon scores (incl. which favelas are missed) → `output/polygon_scores.csv`.

### Key finding — what actually defines a favela at 10 m
RF feature importance: **heterogeneity (std) features dominate** — `MNDWI_std`
(0.14), `NDVI_std` (0.12), `B2_std`, `B3_std`. Favelas are spectrally *mixed* at
10 m (jumbled roofing). **DSM matters here** (`DSM_roughness`+`DSM_mean`+`DSM_std`
≈ 0.15 total) — it was DEAD in the cosine model (corr ≈ 0) but the RF uses it
non-linearly. → terrain is worth keeping; deriving SLOPE is worth trying.

## Pending / next steps

- [ ] Evaluate at POLYGON/agglomeration level (not pixel) — the real use case is
      flagging candidate areas; pixel precision (0.19) understates usefulness.
- [x] Add SLOPE from the DSM — done (`add_slope_feature.py`); top-10 feature but
      marginal effect (favelas here are low-lying). Terrain ≈ 18% of RF importance total.
- [ ] Tune RF (depth, n_estimators, negative sampling ratio); try gradient boosting.
- [ ] When socioeconomic polygons arrive: rasterize attributes → per-window zonal
      features (same mechanism as SUM_EDOC). Pipeline is already structured for it.
- [x] Independent test on 7 AGSN-missed settlements: **35 windows, ROC 0.838** —
      the AGSN-trained model generalises to favelas it never saw. (`data/handdrawn_test.gpkg`)
