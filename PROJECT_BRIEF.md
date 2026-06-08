# Favela Detection — Project Brief (paste-into-chat context)

> Self-contained summary of the whole project: what it is, the data, the method,
> the results, the decisions, the limitations, and the outputs. Written so it can
> be dropped into a fresh chatbot conversation as context. Last updated 2026-06-08.

---

## 1. One-paragraph summary

Supervised detection of **favelas** (informal settlements) in **Sentinel-2** satellite
imagery over the **Região Metropolitana de Maceió (RMM)**, Alagoas, Brazil. The image is
cut into ~150 m windows; each window is described by 32 features (spectral + terrain +
socioeconomic) and scored 0–1 by a **RandomForest** trained on IBGE's official favela
polygons (AGSN). Spatial-block cross-validation makes every score honest (no leakage).
Output = a favela-probability map for QGIS, used as a **screening / triage tool** (not a
precise delineator). It detects ~96% of known favelas inside the RMM and also surfaces
**unmapped** settlements the official map misses.

---

## 2. The problem & framing

- **Goal:** automatically flag likely favelas across the metro region from satellite data.
- **Framing (important):** this is **supervised classification**, *not* "similarity search".
  The repo name `similarity_search` is **legacy** — it refers to an abandoned first approach
  (cosine similarity to one reference favela + One-Class SVM) that failed (ROC ≈ 0.5, could
  not separate favela from other built-up). Describe the project as supervised classification.
- **Use case:** a triage layer for planners — "look here first" — that also helps find
  settlements missing from the official registry.

---

## 3. Data

Reference grid = Sentinel-2 B2: **EPSG:4326, ~10 m/px, 8399×7988 px** (Alagoas coast).
Everything else is reprojected onto it.

- **Sentinel-2 bands:** B2, B3, B4, B8 (10 m) + B11, B12 (20 m→10 m) = blue, green, red,
  NIR, SWIR-1, SWIR-2.
- **Terrain:** `dsm.tif` (digital surface model) → slope + roughness features.
- **Socioeconomic / environmental (RMM only, ~42% of image):** socio-environmental
  vulnerability index (AHP), landslide susceptibility, flood susceptibility, and % of
  low-income households per census sector. **These exist only inside the RMM** — which is
  why the whole project is scoped to the RMM.
- **Labels:**
  - **IBGE AGSN_2019** (official favelas) — training + cross-validation. 254 inside the RMM
    (217 Maceió core + 37 periphery). "Built-up but not in AGSN" = a trustworthy negative.
  - **Hand-drawn communities** — a small independent test set of field-known settlements
    AGSN never mapped; used only to measure generalisation, never for training.

---

## 4. Method

1. **Feature extraction** — slide a 15 px (~150 m) window with stride 10 over the image;
   compute per window: spectral mean/std per band, spectral indices (NDVI, NDBI, BSI, MNDWI,
   SWIR ratio), terrain (slope/roughness), and socio/env means + a `has_socio` coverage flag.
   → 32 features × ~671k windows, saved once to `features.csv` and reused.
2. **Labelling** — a window is **positive** if ≥50% inside an AGSN favela, **negative** if 0%
   (0–50% is ambiguous → discarded).
3. **Negatives** — 4 per positive, half of them "hard" (the brightest/most built-up
   non-favela windows), so the model learns favela-vs-other-urban, not built-vs-vegetation.
4. **RMM-only training** — both positives and negatives are sampled *inside* the RMM, so the
   hard negatives are the metro's own confusers. This materially improved precision.
5. **Model** — RandomForest (300 trees, balanced class weights).
6. **Spatial cross-validation** — tile the map into ~8 km blocks → 5 folds; predict every
   window from a model that never saw its block. No spatial leakage → honest scores even on
   training favelas.
7. **Outputs** — a full-image probability GeoTIFF + an RMM-clipped version (valid only where
   socio data exists).

**Why RandomForest, not SVM (benchmarked, not asserted):** features are heterogeneous
(spectral + terrain + socio) with **structured missing data** (socio is NaN outside the RMM,
imputed with a −1 sentinel). RF handles mixed scales, splits around the sentinel via the
`has_socio` flag, gives calibrated 0–1 probabilities, is fast on 671k windows, and is
**interpretable** (feature importances are central to the story). A supervised RBF-SVM was
benchmarked on the *same* satellite features, labels, and spatial-CV (`compare_models.py`):

| Model | favela vs negatives | favela vs hard urban |
|---|---|---|
| **RandomForest** | ROC 0.979 / PR 0.923 | **ROC 0.979 / PR 0.960** |
| Supervised RBF-SVM | ROC 0.966 / PR 0.895 | ROC 0.966 / PR 0.937 |

RF wins on every metric, *and* the SVM can't cleanly handle the missing socio data or give
native probabilities/importances. Note: the old "SVM ROC ≈ 0.5" refers to the **unsupervised
One-Class** SVM and traces to the **aerial** work — it was never scored on Sentinel-2.

**Framing for the paper:** the headline contrast is the **paradigm shift** — unsupervised
similarity / One-Class (ROC ≈ 0.5, can't separate favela from other built-up) → **supervised
classification with socioeconomic fusion** (ROC ≈ 0.98). RF-vs-SVM is a model-selection
footnote *inside* the supervised arm, not the contribution. The other novel angle is the
**label-incompleteness finding** (the model surfaces unmapped favelas the registry misses).

**Top features:** socio-environmental vulnerability (#1), flood susceptibility, then spectral
**heterogeneity** (MNDWI_std, B2_std, NDVI_std) and low-income share. Interpretation: a favela
at 10 m = **socioeconomically deprived AND spectrally jumbled** (mixed, irregular roofing).

---

## 5. Results (out-of-fold, inside the RMM)

- **Discrimination:** ROC-AUC **0.979** (favela vs sampled negatives), **0.978** vs hard urban
  negatives, PR-AP **0.923**. (Failed unsupervised predecessor: ~0.5.)
- **Detection** (% of in-RMM favelas with a hot window, max ≥ 0.5): **96.1%** overall
  (96.8% core, 91.9% periphery). This is the **headline** number.
- **Object precision/recall** at T=0.7, **post-filtered** (area ≥1 ha & mean ≥0.6):
  precision **~27%**, recall ~72% core / ~68% periphery. At T=0.9: precision ~45%.
- **Generalisation:** on unseen hand-drawn favelas, ROC **0.861** (vs 0.979 in-distribution) —
  the honest gap (see circularity in §7).
- **Classified map** at T=0.7 filtered: **268 predicted favela patches**, ~4,982 ha total,
  median 3.4 ha each (vs 254 AGSN favelas in the RMM — same ballpark).

**Precision is a FLOOR, not a true measure** — see §7.

---

## 6. Key decisions

- **RMM only.** Outside the RMM there's no socio data and the model is blind → out of scope.
- **RandomForest, not SVM or HGB.** (HGB's native-NaN edge is moot inside the RMM; SVM as above.)
- **One global model + region as context**, not one model per municipality (favelas are
  concentrated in the Maceió core → per-municipality models would starve). Report core vs
  periphery instead.
- **Screening tool, not a delineator.** Operate at T≈0.7 on the RMM-clipped map; report the
  "any hot window" (max) detection alongside the conservative whole-polygon mean.

---

## 7. Limitations (the honest part — state these, don't hide them)

- **Precision is a floor, not measurable yet.** AGSN is comprehensive only in the Maceió core,
  so many "false" flags in the periphery are likely **real unmapped favelas** → that's exactly
  what `false_blobs_rmm.gpkg` is for. True precision needs field/QGIS verification.
- **Spatial concentration.** ~53% of positive windows are in 3 CV blocks; only 16/70 RMM blocks
  contain a favela. It's effectively a **Maceió-core model**; the periphery is thin (n=37,
  recall CI ≈ ±15 pp).
- **Resolution.** 10 m floors sub-hectare favelas (tiny-favela recall ~73% by mean / ~92% by max).
- **Circularity.** AGSN is partly *defined* by deprivation, and vulnerability is the top feature
  → inflates in-distribution metrics; lower on truly-unseen favelas (hence the 0.979 → 0.861 gap).
- It is **not** a raw-data-volume problem (1,419 positive windows already give ROC 0.98). The
  real levers are **label completeness/diversity** and **resolution**.

---

## 8. Outputs / deliverables (in `output/`, curated copy in `output_share/`)

| File | What it is |
|---|---|
| `favela_probability_rmm.tif` | **Main result** — favela probability 0–1, RMM only. Triage at ~0.7. |
| `favela_classified_rmm.tif` | Binary map: 1=favela (red), 0=other, 255=nodata. Advisor/slide friendly. |
| `favela_predicted_rmm.gpkg` | The favela class as 268 polygons (area_ha, mean_prob, max_prob). |
| `false_blobs_rmm.gpkg` | Flags ≥0.7 NOT in AGSN — candidate **unmapped** favelas to verify in QGIS. |
| `rmm_object_scores.csv` | Object precision/recall vs threshold (raw + filtered). |
| `fig_comparison.png` | 3-panel slide: RGB \| probability \| binary. |
| `fig_roc_pr.png`, `fig_calibration.png`, `fig_feature_importance.png`, `fig_object_operating.png` | Method/result figures. |

**Pipeline (how to reproduce):** edit `config.py` → `train_supervised.py` →
`evaluate_rmm.py` → `make_classified_map.py` → `make_comparison_figure.py` (all fast, reuse
`features.csv`). Re-extract (`search.py` + `add_slope_feature.py` + `add_socio_features.py`)
only when bands or window size change (~671k windows, slow).

---

## 9. Next steps

- **Hard-negative mining:** train → predict → add top-scoring non-AGSN *core* blobs as hard
  negatives → retrain. The lever that lifts precision without losing recall. (Exclude the
  periphery when mining — its "false" blobs may be real favelas.)
- **Human: verify `false_blobs_rmm.gpkg` in QGIS** → turns the precision *floor* into a true
  number and yields new positive labels.
- **0.5 m aerial validation** (on the `drone-50cm-image` branch) for small-favela recall and a
  multi-resolution story.
- **Expand the hand-drawn test set** (cheap, strengthens generalisation claims).
- **Get raw socio components / continuous income / census year** from the collaborator (Bibi)
  to reduce circularity.

---

## 10. How to talk about it (one-liner)

"We score every 150 m window of Sentinel-2 imagery 0–1 for being a favela, using a
RandomForest trained on the official IBGE favela map with spatial cross-validation so the
scores are honest. It detects ~96% of known favelas in the Maceió metro region and also
flags settlements the official map misses. Precision is reported as a floor because the
reference map is incomplete in the periphery — some 'false' flags are real unmapped favelas."
