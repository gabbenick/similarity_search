# Sentinel-2 edition — what changed and how to run

> **⚠️ HISTORICAL / SUPERSEDED.** These are the original migration notes from the
> aerial→Sentinel-2 move, describing the *unsupervised* similarity / One-Class SVM
> workflow. The project has since moved to a **supervised RandomForest** trained on
> IBGE AGSN polygons, scoped to the RMM. See `README.md` and `CLAUDE.md` for the
> current pipeline. Kept for reference only.

Adaptation of the favela similarity-search pipeline from **aerial 0.5 m RGB**
to **multi-band Sentinel-2** (B2, B3, B4, B8, B11, B12), plus a validation
script for ground-truth polygons.

## Files
| File | Status | Notes |
|------|--------|-------|
| `config.py`           | replaced | bands, indices, 10 m window |
| `extract_features.py` | replaced | N-band spectral + spectral indices |
| `search.py`           | replaced | multi-band loader + resampling |
| `validate.py`         | **new**  | shapefile validation (ROC/PR/confusion) |
| `oneclass.py`         | keep original | no change needed |
| `coords.py`           | keep original | reads `config.IMAGE_PATH` (=B2) |
| `requirements.txt`    | replaced | adds geopandas, scikit-learn |

## Why the approach changed
Sentinel-2 is **10 m** (B2/3/4/8) and **20 m** (B11/12). At 10 m you cannot see
individual shacks, so **texture (GLCM) and edges (Canny) are off by default** —
they were the strength of the 0.5 m version. Instead the pipeline leans on
**spectral signatures + indices**, where Sentinel-2 excels:

| Index | Formula | Favela signal |
|-------|---------|---------------|
| NDVI  | (B8−B4)/(B8+B4) | low vegetation |
| NDBI  | (B11−B8)/(B11+B8) | high built-up |
| BSI   | ((B11+B4)−(B8+B2))/((B11+B4)+(B8+B2)) | exposed soil |
| MNDWI | (B3−B11)/(B3+B11) | water / shadow |
| SWIR_RATIO | B11/B12 | roof-material discrimination |

## Setup (1-time)
1. Put your bands in `data/` as `B2.tif … B12.tif`
   *(or set `USE_STACKED=True` + `STACKED_PATH` for a single multi-band file).*
2. `pip install -r requirements.txt`
3. **Re-derive the reference zone** for the 10 m grid (the old aerial pixel
   coords are invalid). In QGIS read the favela's map coordinates, then:
   ```
   python coords.py --x <easting> --y <northing>
   ```
   Put the resulting row/col box into `REFERENCE_*` in `config.py`.
   The box must be ≥ `WINDOW_SIZE` (default 30 px = 300 m).

## Run
```
python search.py                              # produces output/heatmap.tif etc.
python validate.py --truth data/favelas.shp   # accuracy vs ground truth
```

`validate.py` options:
```
--raster output/svm_map_score.tif   # validate the SVM score instead
--threshold 0.6                     # fixed cut (default: max-F1)
--sample 5000                       # stratified sample/class (autocorrelation)
```

## Outputs of validate.py
- console report: ROC-AUC, **PR-AUC**, confusion matrix, precision/recall/F1/κ/IoU
- `output/validation_curves.png` — ROC + Precision-Recall
- `output/error_map.tif` — 1=TP (green), 2=FP (red), 3=FN (orange) for QGIS

## Things to tune
- `WINDOW_SIZE` / `STRIDE` — analysis scale (300 m default).
- `INDICES` — drop/add indices.
- `SIMILARITY_METRIC` — `cosine` | `euclidean`.
- `SVM_NU` — fraction of outliers tolerated by the one-class SVM.

## Important caveats
- **Resolution:** detection is at *agglomeration* scale, not building scale.
- **Ground truth independence:** never validate with polygons that overlap the
  reference zone — that inflates the scores.
- **Reflectance scale:** indices are ratios, so int (0–10000 L2A) or float
  reflectance both work, as long as all bands share the same scaling.
