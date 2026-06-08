# Favela Detection — Maceió (RMM)

Supervised detection of informal settlements (*favelas* / *aglomerados subnormais*)
in **Sentinel-2** satellite imagery over the **Região Metropolitana de Maceió**
(Alagoas, Brazil). Output: a georeferenced **favela-probability map (0–1)** for QGIS,
used as a **screening / triage tool**.

> **Scope:** valid **inside the RMM only** (the area with socioeconomic data). Use
> `output/favela_probability_rmm.tif`.

---

## In one paragraph

We slide a ~150 m window across the image and, for each window, measure what it looks
like — spectral colours/indices, how *visually mixed* it is (favelas have jumbled
rooftops), terrain, and socioeconomic context (vulnerability, low income, flood/landslide
risk). We train a RandomForest on the government's official favela map (IBGE AGSN) to
separate favelas from other built-up land. The result is a heatmap: **bright = likely
favela**. It catches ~96% of known favelas and is meant to **point reviewers at candidate
areas**, not to draw exact boundaries. Bonus: some of its "false alarms" are real favelas
the official map never mapped.

---

## Pipeline

```
Sentinel-2 (10 m) + DSM terrain + socio layers (RMM)
        │
        ├─ slide 15 px (~150 m) window, stride 10 px  →  671k windows
        ├─ 32 features/window: spectral, indices, terrain, socio
        ▼
RandomForest  (favela vs. other-built-up)
  • labels = IBGE AGSN polygons (overlap ≥ 50% ⇒ positive; 0% ⇒ negative)
  • hard negatives = brightest/most built-up non-favela windows (the confusers)
  • spatial-block CV (~8 km blocks) ⇒ every window scored out-of-fold (no leakage)
  • trained RMM-only (RMM_ONLY): positives & negatives sampled inside the RMM
        ▼
favela_probability_rmm.tif (0–1)  →  QGIS, triage at ≈ 0.7
```

The earlier **unsupervised** approach (cosine-similarity heatmap + One-Class SVM,
`search.py`/`oneclass.py`) could not separate favelas from other urban (ROC ~0.5) and is
**superseded** — kept only for comparison.

---

## Results (out-of-fold, AGSN-evaluated, inside the RMM)

| Metric | Value |
|---|---|
| ROC-AUC (favela vs. rest / vs. hard urban) | **0.979 / 0.978** |
| Detection — any hot window (max ≥ 0.5), core / periphery | **96.8% / 91.9%** |
| Object **precision** @0.7 (raw / filtered) | 18.3% / **26.9%** |
| Object precision @0.9 | 39–45% |
| Generalisation — unseen hand-drawn favelas (ROC) | 0.861 |

**Read correctly:** strong **screening** (finds ~96% of known favelas); **precision is a
floor**, not the truth — AGSN is incomplete in the metro periphery, so some flagged "false"
areas are real unmapped favelas (see `false_blobs_rmm.gpkg`). The blob post-filter
(≥1 ha & mean ≥ 0.6) lifts precision with no recall loss. **Threshold is the dial** (≈0.7).

---

## Data (`data/`, gitignored)

- **Sentinel-2:** `B2,B3,B4,B8.tif` (10 m) + `B11,B12.tif` (20 m→10 m). Grid: EPSG:4326,
  ~10 m/px, 8399×7988 px, Alagoas coast incl. the RMM.
- **Terrain:** `dsm.tif` (UTM 25S, 30 m) — auto-reprojected.
- **Socio/env (RMM only, EPSG:31985):** vulnerability, landslide & flood susceptibility
  rasters + low-income census sectors. Cover ~42% of the image.
- **Labels:** IBGE AGSN_2019 → `data/agsn_truth.gpkg` (273 favelas in-image, 254 in-RMM).
  Hand-drawn field communities → `data/handdrawn_test.gpkg` (independent generalisation test).

---

## Run

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Ground-truth layers (only if shapefiles change)
python build_agsn_truth.py        # → data/agsn_truth.gpkg
python build_handdrawn_test.py    # → data/handdrawn_test.gpkg

# 2. Feature extraction (slow, ~671k windows — only if bands/config change)
python search.py                  # → output/features.csv
python add_slope_feature.py       # + DSM slope
python add_socio_features.py      # + socio/env columns

# 3. Train + evaluate (fast — reuses features.csv; the normal loop)
python train_supervised.py        # → favela_probability_rmm.tif (+ oof_eval.npz)
python evaluate_rmm.py            # → rmm_object_scores.csv + false_blobs_rmm.gpkg
python make_figures.py            # → output/fig_*.png
```

**Key config (`config.py`):**
```python
WINDOW_SIZE = 15   STRIDE = 10        # ~150 m window, ~67% overlap
USE_DSM = True     USE_SOCIO = True   # terrain + socio features
RMM_ONLY = True                       # train inside the RMM only (scope decision)
```

---

## Outputs (`output/`)

| File | What |
|---|---|
| `favela_probability_rmm.tif` | **Main result** — favela probability 0–1, RMM only |
| `false_blobs_rmm.gpkg` | Flagged areas not in AGSN — candidate / discovery layer |
| `rmm_object_scores.csv` | Object precision/recall vs threshold (raw + filtered) |
| `polygon_scores.csv` | Per-favela detection scores (caught vs missed) |
| `fig_*.png` | ROC/PR, calibration, feature importance, operating curve |

Legacy/unsupervised outputs (`heatmap*.tif`, `svm_map*.tif`) are superseded.

---

## Branches

- `main` — current supervised RMM pipeline (this README).
- `drone-50cm-image` — original 0.5 m aerial / unsupervised version (preserved).

See `CLAUDE.md` for full methodology, decisions, limitations, and next steps.
