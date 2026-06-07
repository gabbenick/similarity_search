# CLAUDE.md — Similarity Search Pipeline

> Read this at the start of every session to understand the current state of the project.

---

## ▶ Next session — start here (as of 2026-06-07)

**SCOPE DECISION: the project now targets the RMM only** — outside-RMM behaviour is
explicitly out of scope (no socio data there anyway). This retires several old items.

**State:** pipeline fully working end-to-end. Current model = RandomForest
(spectral + terrain + socio), out-of-fold, spatial CV, **trained RMM-only**
(`config.RMM_ONLY = True`: positives AND negatives sampled only inside the RMM, so the
hard negatives are the RMM's own built-up confusers). Strong screening tool inside the RMM;
**object precision is still the open weakness** but materially improved this session.

**What changed this session (committed):**
- Scoped training to the RMM (`RMM_ONLY`). This was a clean **PR-curve shift up**, not just
  a tradeoff: object precision @0.7 **12.2% → 18.3%**, @0.9 **31.8% → 39.0%**; ROC-AUC
  **0.964 → 0.979**; unseen-set generalisation **0.826 → 0.861**. Recall cost is modest
  (core @0.7 81.6% → 71.9%) and recoverable by lowering the threshold (detection @0.5 still
  ~90% core & periphery).
- New `evaluate_rmm.py` — RMM-scoped object eval (core vs periphery via `AGSN_NM_MU`) and
  exports `output/false_blobs_rmm.gpkg` (the false positives, for QGIS review).
- New `tune_precision.py` — in-memory sweep of the precision levers (RMM-scope, NEG_RATIO,
  class_weight). Established RMM-scope as the win; raising NEG_RATIO/dropping class_weight
  buys precision only by killing **periphery** recall (78%→14%) → don't over-tighten.

**Agreed immediate next action: hard-negative mining (bootstrap).**
- *Why:* it's the lever that shifts the PR curve up *without* the recall cost that tightening
  the prior imposes. Precision is now an in-RMM problem — false blobs are RMM built-up
  confusers (quarries/industrial/bare soil), not outside-RMM noise (clipping didn't help).
- *How:* train → predict → take the highest-scoring non-AGSN RMM blobs as new hard negatives
  → retrain (1–2 rounds). **Caveat:** exclude the RMM *periphery* when mining, since AGSN is
  incomplete there and some "false" blobs are real unmapped favelas (don't mine them as negs).

**Two free wins added (committed):**
- **Headline detection = "any hot window" (max≥0.5): 96.1%** (vs 83.5% by whole-polygon
  mean). 32 of the 41 low-mean "missed" favelas are in fact touched — the miss was an
  averaging artifact. `evaluate_rmm.py` now reports mean / p90 / max.
- **Blob post-filter** (area≥1 ha & blob-mean≥0.6) in `evaluate_rmm.py`: at T=0.7 precision
  **18.3% → 26.9% with ZERO recall loss** (72%/68% unchanged); ~half the false blobs were
  <1 ha speckle. Use the filtered operating point for reporting.
- `make_figures.py` writes `output/fig_*.png` (ROC/PR AUC 0.979/AP 0.923, calibration,
  feature importance, object operating curve). Feature story: `vuln_mean` #1, then spectral
  heterogeneity (`MNDWI_std`, `B2_std`, `NDVI_std`) + `flood_mean`/`lowincome_mean`.

**Open human task:** review `output/false_blobs_rmm.gpkg` in QGIS over the imagery — how many
"false" blobs are real unmapped favelas? This sets the *true* precision (current is a floor;
~half the 380 false blobs are >1 ha and worth inspecting). Can't be automated (no IBGE
municipal-boundary raster to tile core vs periphery).

**Housekeeping:** `output/features.csv.bak` is a safety backup from the socio integration —
safe to delete once results are trusted.

---

## What this project does

Detects informal settlements (favelas) in Sentinel-2 satellite imagery over the Maceió
region (Alagoas, Brazil). The pipeline slides a 15 px window across the image, extracts
spectral + terrain + socioeconomic features per window, and trains a **supervised
RandomForest** (favela vs. other built-up) against IBGE's official favela polygons (AGSN),
using **spatial-block cross-validation** so every window is scored out-of-fold.

The main output is a georeferenced **favela-probability GeoTIFF** (0–1) ready for QGIS —
used as a **screening / triage tool** to flag candidate areas.

> An earlier **unsupervised** path (cosine-similarity heatmap + One-Class SVM) still exists
> in `search.py`/`oneclass.py` but is **superseded** — it could not separate favelas from
> other built-up areas. Kept only for comparison.

### How to explain it to someone (plain language)

> We take a satellite image of the Maceió metro region (RMM) and slide a small ~150 m
> window across it. For each window we measure what it looks like — colours and spectral
> indices, how *visually mixed* it is (favelas have jumbled, varied rooftops), the terrain,
> and socioeconomic context (vulnerability, low income, flood/landslide risk). We train a
> model on the **government's official favela map** (IBGE AGSN) so it learns the difference
> between a favela and other built-up land. The result is a **heatmap: bright = likely
> favela**. It catches ~90% of known favelas and is meant to **point people at candidate
> areas to review**, not to draw exact boundaries.

Three honest points to make when presenting:
1. **It's a screening tool, not a map.** High recall (~90–97% of known favelas touched),
   modest precision — it over-flags on purpose, and a reviewer confirms.
2. **Its "false alarms" are partly real.** The official map is incomplete outside the city
   core, so some flagged areas are genuinely unmapped favelas (a *feature* for discovery).
3. **It only works inside the RMM** (that's where the socioeconomic data exists).

---

## Repository layout

```
similarity_search/
├── config.py             # all tuneable parameters + paths (edit here first)
├── coords.py             # QGIS coord → pixel converter
├── extract_features.py   # per-window feature extraction (spectral/index/DSM)
├── search.py             # feature extraction over the whole image → features.csv (+ legacy maps)
├── add_slope_feature.py  # augments features.csv with DSM slope (fast, no re-extraction)
├── add_socio_features.py # augments features.csv with socio/env layers (fast)
├── build_agsn_truth.py   # comunidades/AGSN_2019.shp → data/agsn_truth.gpkg
├── build_handdrawn_test.py # hand-drawn shapefiles → data/handdrawn_test.gpkg
├── train_supervised.py   # CURRENT model: RF + spatial CV → favela_probability*.tif (RMM-scoped)
├── evaluate_polygons.py  # polygon/object-level metrics (full image) → polygon_scores.csv
├── evaluate_rmm.py       # RMM-scoped object eval (core/periphery) + exports false_blobs_rmm.gpkg
├── tune_precision.py     # in-memory precision-lever sweep (RMM-scope, NEG_RATIO, class_weight)
├── make_figures.py       # thesis figures: ROC/PR, calibration, feature importance, operating curve
├── oneclass.py           # legacy One-Class SVM
├── validate.py           # legacy validation vs ground-truth shapefile
├── comunidades/          # community + AGSN source shapefiles
├── data/                 # rasters + shapefiles/gpkg (gitignored)
└── output/               # pipeline outputs (gitignored)
```

---

## Input data (`data/`)

### Sentinel-2 bands (all identical grid — the reference grid)
| File | Band | Native res | In data/ |
|------|------|-----------|---------|
| B2.tif | Blue | 10 m | ✅ |
| B3.tif | Green | 10 m | ✅ |
| B4.tif | Red | 10 m | ✅ |
| B8.tif | NIR | 10 m | ✅ |
| B11.tif | SWIR-1 | 20 m (resampled to 10 m) | ✅ |
| B12.tif | SWIR-2 | 20 m (resampled to 10 m) | ✅ |

**CRS:** EPSG:4326 · **Resolution:** 0.000090°/px ≈ 10 m · **Size:** 8 399 × 7 988 px
**Coverage:** lon [-36.23, -35.47] · lat [-9.86, -9.14] (region of Alagoas, Brazil).
Every other layer is reprojected/rasterized onto this grid at runtime.

### DSM (altitude)
| File | CRS | Resolution | Range |
|------|-----|-----------|-------|
| dsm.tif | EPSG:31985 (UTM 25S) | 30 m | 0–228 m |

Auto-reprojected to the B2 grid by `search.py::load_dsm()`. No manual preprocessing.

### Socioeconomic / environmental layers (from Eng. Bibi — in `data/`)
All **EPSG:31985 (UTM 25S)** and cover **only the RMM (Região Metropolitana de Maceió)** —
UTM bbox ≈ 145.6k–228.4k E / 8909k–8988k N, ≈ **42% of the image**. Reprojected/rasterized
to the B2 grid by `add_socio_features.py`. Windows **outside the RMM → NaN** on these
features (see "Coverage caveat" below).

| File | Layer | Type | Valid range | Note |
|------|-------|------|-------------|------|
| `vulnerabilidade_3_rec.tif` | Socio-environmental vulnerability (AHP index) | raster 30 m | 0–0.74 (mean 0.13) | Composite (children, elderly, dist-to-highways, water/sewage); dominated by **dist-to-highways + infrastructure**; low variance ("RMM conditions generally good") |
| `mapa_deslizamento2.tif` | Landslide susceptibility | raster 30 m | 0–0.92 (mean 0.39) | |
| `teste_ahp_3_recortado.tif` | **Flood susceptibility** (despite the name) | raster 30 m | 0–1 | Has **exactly 1 corrupted pixel** (value 1.1e8) → `add_socio_features.py` clips values > 1; rest are clean |
| `setores_AL_baixa_renda_RMM.shp` | Low income (census sectors) | vector | `porc_bx_re` 0–1.045 | Fraction of households per sector earning **≤ ½ minimum wage** (continuous). Key = `CD_GEOCODI`; 1429 sectors (2 are >1.0 = 1.045, 32 are NaN → tell Bibi; harmless to RF) |

### Pre-computed composites (in data/, not used by pipeline)
`RGB_432.tif`, `FCC_843.tif`, `SWIR_12114.tif`, `Urbano_1184.tif`, `Stack_Urbano.tif`,
`NDVI.tif`, `NDBI.tif`, `MNDWI.tif` — useful for QGIS visualisation only.

---

## Coverage caveat — where the model is valid (READ THIS)

The socio/env layers only exist for the **RMM (Região Metropolitana de Maceió)** — the
whole metropolitan region, i.e. Maceió **and its peripheral metro municipalities**, not
just the Maceió urban core. Two distinct things follow, often conflated:

1. **Missing socio data = anything OUTSIDE the RMM.** The 20 AGSN favelas in the deeper
   interior of Alagoas (inside the satellite image but outside the metro region) get the
   imputation sentinel → the socio model is effectively **blind** there (0% detection).
   Communities in the RMM's *peripheral municipalities* **do** have data — they are covered.
2. **Inside the RMM but outside the Maceió core**, the *socio data exists*, but the **AGSN
   label** is less complete (AGSN is comprehensive only in the Maceió core). So some
   "false positives" there are likely **real unmapped favelas**, which is why object
   precision is a *floor*, not the true precision.

→ If the project scope is the RMM, this is fine: use `favela_probability_rmm.tif` (clipped
to coverage). To work outside the RMM you'd need either socio data for those areas or a
socio-free model.

---

## Label sources (`comunidades/`)

**1. IBGE AGSN_2019 (primary, training + CV).** Official *aglomerados subnormais* — 13,151
polygons statewide, **273 inside the image**, 11 municipalities. `build_agsn_truth.py` →
`data/agsn_truth.gpkg` (reprojected to grid, clipped; keeps geometry + NM_AGSN + `SUM_EDOC`
[household count] + municipality; drops 30 buggy "nearest-facility" columns). The
authoritative favela ground truth — "built-up but NOT in AGSN" is a trustworthy negative.

**2. Hand-drawn communities (independent generalisation test).** Field-known settlements,
hand-digitised, MIXED CRSs (EPSG:4326 / 31984 / 31985). `build_handdrawn_test.py` →
`data/handdrawn_test.gpkg` (10 inside the image; PARIPUEIRA dropped = duplicate of
ALTO_DA_BOA_VISTA; "22 de Julho" dropped = UTM zone 24S, west of image). **Not merged
into training.**

| Hand-drawn community | AGSN overlap | Role |
|----------------------|-------------|------|
| BIQUINHA | 93.5% | (in AGSN) |
| NOVA_ESPERANCA | 98.7% | (in AGSN) |
| ALTO_DA_BOA_VISTA | 46.1% | (partial) |
| ANTENOR_MARINHO, BURACO_DO_JACARE, COQUEIRO_SECO, MESSIAS, RUA_DO_MATADOURO, **Conjunto Luiz Gonzaga**, **VILA DAVI** | 0% | **AGSN-missed → generalisation test** |

The 7 zero-overlap communities are the honest check: settlements AGSN never mapped, used to
test whether the AGSN-trained model generalises to unseen favelas.

> Old `data/train.shp` / `data/val.shp` (hand-drawn bbox split) are obsolete — superseded
> by AGSN training + spatial CV. Kept only for reference.

---

## Features used (32 total)

```
config.py: USE_SPECTRAL=True, USE_INDICES=True, USE_TEXTURE=False, USE_EDGE=False,
           USE_DSM=True, USE_SOCIO=True
```

| Group | Features | Count |
|-------|----------|-------|
| Spectral (mean+std per band) | B2, B3, B4, B8, B11, B12 | 12 |
| Spectral indices (mean+std) | NDVI, NDBI, BSI, MNDWI, SWIR_RATIO | 10 |
| Terrain | DSM mean, std, roughness, slope_mean, slope_std | 5 |
| **Socio/env (per-window mean + coverage flag)** | vuln_mean, landslide_mean, flood_mean, lowincome_mean, has_socio | 5 |
| **Total** | | **32** |

- Texture (GLCM) and edge (Canny) are **OFF** — meaningless at 10 m resolution.
- **Slope** (`add_slope_feature.py`): top-10 feature but marginal — these favelas are
  low-lying; terrain ≈ 18% of RF importance, mostly DSM roughness.
- **Socio/env** (`add_socio_features.py`): smooth 30 m layers → **mean only** (no std).
  `vuln_mean` is now the **#1 RF feature**; socio ≈ 31% of total importance. RMM-only
  coverage → `has_socio` flags it; NaN imputed in training (see below).
- What defines a favela at 10 m: **spectral heterogeneity (`_std`) co-dominates** with
  `vuln_mean` — favelas are spectrally *mixed* (jumbled roofing): `MNDWI_std`, `NDVI_std`,
  `B2_std`, `B3_std`.

---

## Key config values (config.py)

```python
WINDOW_SIZE = 15      # px ≈ 150 m — drives feature extraction (search.py)
STRIDE      = 10      # px — ~67% overlap
USE_DSM     = True    # terrain features on
USE_SOCIO   = True    # socio/env features on (add_socio_features.py)
RMM_ONLY    = True    # train on in-RMM windows only (positives + negatives); scope decision
SOCIO_RASTERS, SOCIO_INCOME_SHP, SOCIO_COVERAGE_RASTER, ...  # socio paths/fields
# Legacy (unsupervised path only): REFERENCE_SHAPEFILE, USE_ONE_CLASS_SVM,
# SVM_NU, SIMILARITY_METRIC — not used by the supervised RF workflow.
```

---

## Output files (`output/`)

**Supervised model (CURRENT — use these):**
| File | Description |
|------|-------------|
| `favela_probability.tif` | RF favela probability 0–1, out-of-fold (full image) |
| `favela_probability_rmm.tif` | **Same, clipped to RMM coverage** — prefer this in QGIS when scope = RMM (the socio model is only valid there) |
| `polygon_scores.csv` | Per-favela detection scores; lists the missed favelas |
| `features.csv` | Per-window feature table (32 features) — reused by training |

**Legacy / unsupervised (superseded — comparison only):**
`heatmap.tif`, `heatmap_norm.tif`, `svm_map.tif`, `svm_map_score.tif`, `overview.png`,
`error_map.tif`, `validation_curves.png`.

---

## How to run (current supervised workflow)

```bash
source venv/bin/activate

# 1. Build / refresh ground-truth layers (only if shapefiles changed)
python build_agsn_truth.py        # → data/agsn_truth.gpkg (273 AGSN favelas)
python build_handdrawn_test.py    # → data/handdrawn_test.gpkg (10 communities)

# 2. (Re)extract features ONLY if config/bands change — slow (671k windows):
python search.py                  # regenerates output/features.csv (+ legacy maps)
python add_slope_feature.py       # adds DSM slope cols (fast)
python add_socio_features.py      # adds socio/env cols (vuln/flood/landslide/income, ~40s)

# 3. Train + evaluate (fast — reuses features.csv):
python train_supervised.py        # → favela_probability.tif (+ _rmm clipped)
python evaluate_polygons.py       # → polygon-level metrics + polygon_scores.csv
```

> Normal loop is just step 3. Steps 1–2 only when inputs change.
> `add_socio_features.py` is idempotent (drops & re-adds its own columns).
> Legacy unsupervised path: `python search.py` then `validate.py --truth ...`.

---

## Results (out-of-fold, AGSN-evaluated)

### Pixel / discrimination metrics
| Metric | Unsupervised heatmap | Supervised RF (spectral+terrain) | **+ socio (current)** |
|--------|---------------------|----------------------------------|-----------------------|
| ROC-AUC (full image) | 0.899 | 0.951 | ~0.96 |
| favela-vs-other-urban ROC | ~0.5 (couldn't separate) | 0.946 | **0.962** |
| Top RF feature | — | MNDWI_std (0.14) | **vuln_mean (0.13)** |

The supervised model genuinely separates favela from other built-up — the capability the
unsupervised approach fundamentally lacked. Socio features push the favela-vs-urban
separation further and `vuln_mean` becomes the single most important feature.

### Polygon / object-level (`evaluate_polygons.py`) — the real use case

> **NOTE (2026-06-07):** the table below is the *pre-RMM-scoping* model. The current
> RMM-scoped model (`RMM_ONLY=True`, see "Next session") improves object precision to
> **18.3% @0.7 / 39.0% @0.9** with ROC-AUC 0.979. Use `evaluate_rmm.py` for current
> in-RMM numbers (core vs periphery). The table is kept for the before/after comparison.

The real use case is flagging candidate *areas*, so pixel precision understates usefulness.
**Inside vs outside the RMM** (the split that matters, pre-RMM-scoping model):

| | **Inside RMM** (n=253 AGSN) | Outside RMM (n=20) |
|---|---|---|
| Detection @0.5 | **91.3%** | 0.0% |
| Detection @0.7 | 70.8% | 0.0% |
| median mean_prob | **0.787** (vs ~0.04 background, ~20×) | 0.091 |
| Object precision @0.7 | 12.2% | — |
| Object precision @0.9 | 31.8% | — |

**Verdict:** as a **screening/triage tool inside the RMM, it works well** — 91% of known
AGSN favelas caught at prob ≥ 0.5, sitting ~20× above background. Detection scales with
size (small ~77% / medium ~82% / large ~95%).

**Open weakness = precision**, even inside the RMM (~12% @0.7 ≈ 6× over-flag). But that's a
*floor*: AGSN is incomplete outside the Maceió core, so some "false" blobs are real unmapped
favelas. Use a higher threshold to trade recall for precision (≈32% precision @0.9, but
detection drops to ~47%).

**Operating guidance:** triage at **T ≈ 0.7**; load **`favela_probability_rmm.tif`** when
the area of interest is the RMM (avoids misleading noise outside coverage).

### Honest cost of the socio features
- Generalisation to the **7 AGSN-unseen** hand-drawn communities **dropped** (43% → 14% at
  T0.5; per-window ROC ~flat 0.838 → 0.826, n=35, noisy). This is the **circularity** we
  flagged: IBGE partly *defines* AGSN by deprivation, so the model leans on vuln/income and
  does worse on favelas outside the labelled distribution / outside RMM coverage (sentinel).
- The socio model is **blind outside the RMM** (those 20 favelas fall to ~0.09; the
  spectral-only model caught some). Acceptable **if scope = RMM**.
- → Keep the hand-drawn set as the honest check. For interior / unmapped-favela *discovery*,
  consider a socio-free model or down-weighting socio.

---

## How the socio integration works (`add_socio_features.py` + `train_supervised.py`)

1. **Align**: reproject the 3 rasters to the B2 grid (NaN-safe; clips flood's 1 bad px),
   rasterize `porc_bx_re` per sector. Compute per-window **means** via NaN-aware
   summed-area tables. Append 5 cols to `features.csv`: `vuln_mean`, `landslide_mean`,
   `flood_mean`, `lowincome_mean`, `has_socio`. No spectral re-extraction (~40 s).
2. **Coverage**: socio covers ~43% of windows (the RMM). `train_supervised.py` imputes the
   NaNs with sentinel **−1.0** (outside the valid [0,1] range) and keeps `has_socio` so the
   RF routes around the sentinel and falls back to spectral+terrain outside the RMM.
   RandomForest kept (HistGradientBoosting native-NaN is a candidate A/B).
3. **Clipped map**: `train_supervised.py` also writes `favela_probability_rmm.tif`, masking
   everything outside the RMM coverage to nodata.

### What arrived vs originally requested
We asked for **continuous, non-binarized, per-sector** variables (per-capita income, % w/o
water/sewage/garbage separately, household/pop counts, census year). What came:
- **Low income** (`porc_bx_re`): % households ≤ **½** min wage, continuous, per sector. ✅
- **Vulnerability**: a **pre-combined AHP index** (not raw components — those "had little
  data"). Less ideal (RF can't reweight a composite) but it's what's available.
- **Landslide** + **flood** susceptibility — bonus environmental risk layers.

Still pending from Bibi (nice-to-have, not blocking): raw deprivation components,
continuous per-capita income, census-year confirmation, cleaner flood raster if any.

---

## Per-municipality / regional heterogeneity — DECISION

Collaborators (Bibi, Ju) flagged "a favela in Maceió ≠ a favela in Murici" and asked whether
to split by municipality. **Agreed: do NOT train one model per municipality** (~273 favelas,
concentrated in the Maceió core → sample starvation elsewhere). Instead:
1. **One global model + region/context as a feature** (municipality and/or continuous urban
   context). Model learns regional differences via interactions without fragmenting data.
2. **Test empirically** with **leave-one-municipality-out (LORO) CV**. NB: since socio only
   covers the RMM, in practice also report metrics **inside vs outside RMM** (done above).
3. If LORO shows real degradation, group into **2–3 strata** (Maceió metro vs interior),
   not 11 models, and report per stratum. (Assigning municipality to every window needs the
   IBGE municipal-boundary shapefile; `agsn_truth.gpkg` already carries `municipality` for
   positives.)

---

## Pending / next steps

- [x] Polygon/object-level evaluation (`evaluate_polygons.py`).
- [x] DSM slope (`add_slope_feature.py`) — top-10 feature, marginal (low-lying favelas).
- [x] Independent test on 7 AGSN-missed settlements (n=35, ROC 0.838 pre-socio).
- [x] Integrate socio/env layers (`add_socio_features.py`) — big precision gain inside RMM.
- [x] Report metrics inside vs outside RMM (table above).
- [x] RMM-clipped probability map (`favela_probability_rmm.tif`).
- [x] **RMM-scoped training** (`RMM_ONLY`) — precision @0.7 12.2%→18.3%, ROC 0.964→0.979.
- [x] RMM-scoped object eval + false-blob export (`evaluate_rmm.py`).
- [x] Precision-lever sweep (`tune_precision.py`) — RMM-scope wins; don't over-tighten prior.
- [ ] **Hard-negative mining (bootstrap)** — next lever; shifts PR curve up w/o recall cost.
      Exclude RMM periphery when mining (AGSN incomplete there).
- [ ] **Human: review `false_blobs_rmm.gpkg` in QGIS** — how many false blobs are real
      unmapped favelas? Sets the true precision (current numbers are a floor).
- [ ] Optional precision/recall dial: expose `NEG_RATIO` in config (sweep showed r8 → P@0.7
      ~20% but periphery recall 78%→57%).
- [ ] Add urban-context feature (GHSL/WSF building density) to reject non-residential bright
      land — the most durable precision fix.
- [ ] Ask Bibi for raw deprivation components, continuous per-capita income, census year.

**Retired (out of scope — RMM-only decision):** HGB-vs-RF A/B (moot: no NaN inside RMM),
extending socio outside the RMM, LORO across non-RMM municipalities.

---

## Known warnings (cosmetic, not bugs)

1. `RuntimeWarning: Mean of empty slice` — a few edge-of-image patches are all-zero;
   `nan_to_num(..., nan=0.0)` handles them downstream. Does not affect results.
2. `Ground extent ≈ 1 m × 1 m` — cosmetic print bug: `pixel_size` is in degrees
   (EPSG:4326), not metres. Actual resolution ≈ 10 m/px. Does not affect outputs.
3. `Geometry is in a geographic CRS ... centroid` — from ad-hoc analysis scripts using
   centroids in EPSG:4326; negligible at this scale.
