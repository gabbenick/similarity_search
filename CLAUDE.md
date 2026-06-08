# CLAUDE.md — Favela Detection Pipeline (Maceió / RMM)

> Technical state, methodology, decisions, and caveats. Read with `README.md`
> (project overview + how to run). Concise on purpose.

---

## ▶ Next session — start here (as of 2026-06-08)

**State:** complete and pushed to `main` (commit `b7fbcc9`). RMM-scoped RandomForest works
end-to-end; results are thesis-ready — ROC **0.979**, **~96%** detection (any hot window),
object precision **18–27% @0.7** (a *floor*, see §8). Docs (this + `README.md`) are
self-contained; the curated bundle for the QGIS collaborator is in `output_share/`.

**#1 next action — hard-negative mining** (§10): the one lever that lifts precision *without*
the recall cost of tightening the prior. Train → predict → add the top-scoring non-AGSN
**core** blobs as hard negatives → retrain. **Exclude the periphery** when mining (its
"false" blobs may be real unmapped favelas).

**Blocked on a human:** QGIS verification of `false_blobs_rmm.gpkg` → yields the *true*
precision and new positive labels. Until then, all precision numbers are a floor.

**Thesis framing:** this is **supervised classification for favela detection**, *not*
similarity search (the repo name is legacy — see §11). Lead with §7 results + §8 limitations;
the honest data caveats (concentration, label incompleteness, resolution, circularity) are
the part to state explicitly, not hide.

---

## 1. What this is

Supervised detection of favelas (informal settlements) in **Sentinel-2** imagery over the
**Região Metropolitana de Maceió** (RMM), Alagoas, Brazil. A RandomForest scores every
~150 m window 0–1 (favela vs. other built-up), trained on IBGE's official AGSN favela
polygons with **spatial-block cross-validation** (every window scored out-of-fold).
Output = a favela-probability GeoTIFF for QGIS, used as a **screening/triage tool**.

**Status:** working end-to-end; RMM-scoped RF is the current model. Strong screening inside
the RMM; **object precision** is the open weakness (and partly a label artifact, see §8).

**Scope decision: RMM only.** Outside the RMM there is no socio data and the model is blind
— explicitly out of scope. This retired the HGB-vs-RF A/B (no NaN inside the RMM), socio
expansion beyond the RMM, and non-RMM LORO CV.

---

## 2. Repository / pipeline

```
config.py              all params + paths (edit here first)
search.py              feature extraction over the image → output/features.csv (+ legacy maps)
add_slope_feature.py   + DSM slope columns (fast, no re-extraction)
add_socio_features.py  + socio/env columns (vuln/landslide/flood/income + has_socio)
build_agsn_truth.py    AGSN_2019.shp → data/agsn_truth.gpkg (labels)
build_handdrawn_test.py hand-drawn shapefiles → data/handdrawn_test.gpkg (independent test)
train_supervised.py    CURRENT model: RF + spatial CV → favela_probability(_rmm).tif, oof_eval.npz
evaluate_rmm.py        RMM-scoped object eval (core/periphery) → rmm_object_scores.csv + false_blobs_rmm.gpkg
evaluate_polygons.py   older full-image object eval → polygon_scores.csv
tune_precision.py      in-memory sweep of precision levers (RMM-scope, NEG_RATIO, class_weight)
make_figures.py        figures → output/fig_*.png
make_classified_map.py threshold prob map → favela_classified_rmm.tif (binary) + favela_predicted_rmm.gpkg (polygons). Default T=0.7, post-filtered to match the reported precision; --no-filter / --threshold to vary.
make_comparison_figure.py 3-panel slide figure (RGB | probability | binary) → output/fig_comparison.png
search.py/oneclass.py  LEGACY unsupervised path (cosine heatmap + One-Class SVM) — superseded
```

**Normal loop:** edit `config.py` → `train_supervised.py` → `evaluate_rmm.py` →
`make_figures.py` (all fast, reuse `features.csv`). Re-extract (`search.py` + the two
`add_*`) only when bands/window change (~671k windows, slow).

---

## 3. Data (`data/`, gitignored)

Reference grid = the Sentinel-2 B2 grid: **EPSG:4326, ~10 m/px, 8399×7988 px**, Alagoas
coast. Everything else is reprojected/rasterized onto it at runtime.

- **Sentinel-2 bands:** B2/B3/B4/B8 (10 m), B11/B12 (20 m→10 m). Blue/Green/Red/NIR/SWIR1/SWIR2.
- **DSM:** `dsm.tif` (UTM 25S, 30 m), reprojected by `search.py`.
- **Socio/env (EPSG:31985, RMM only ≈ 42% of image):** `vulnerabilidade_3_rec.tif` (AHP
  socio-env vulnerability index), `mapa_deslizamento2.tif` (landslide), `teste_ahp_3_recortado.tif`
  (flood, despite the name; has 1 corrupt pixel → clipped >1), `setores_..._baixa_renda_RMM.shp`
  (% households ≤ ½ min wage, continuous, per census sector).
- **Composites (viz only, not used by model):** RGB_432, FCC_843, NDVI/NDBI/MNDWI, etc.

---

## 4. Labels

- **IBGE AGSN_2019 (training + CV).** Official *aglomerados subnormais*. `agsn_truth.gpkg` =
  **273 in-image / 254 in-RMM** (217 Maceió core + 37 periphery), with `AGSN_NM_MU`
  (municipality), `SUM_EDOC` (household count), `area_ha`. "Built-up but not in AGSN" is a
  trustworthy negative.
- **Hand-drawn communities (independent generalisation test).** Field-known settlements;
  `handdrawn_test.gpkg` (10 in-image). The 7 with 0% AGSN overlap are the honest test of
  generalisation to favelas AGSN never mapped. **Not in training.**

---

## 5. Features (32)

| Group | Features | n |
|---|---|---|
| Spectral (mean+std/band) | B2,B3,B4,B8,B11,B12 | 12 |
| Indices (mean+std) | NDVI, NDBI, BSI, MNDWI, SWIR_RATIO | 10 |
| Terrain | DSM mean/std/roughness, slope mean/std | 5 |
| Socio/env (per-window mean + flag) | vuln_mean, landslide_mean, flood_mean, lowincome_mean, has_socio | 5 |

Texture (GLCM) and edges (Canny) are **off** — meaningless at 10 m. Socio layers are smooth
30 m → mean only (no std). **Top features:** `vuln_mean` (#1), `flood_mean`, then spectral
**heterogeneity** (`MNDWI_std`, `B2_std`, `NDVI_std`) + `lowincome_mean`. Story: a favela at
10 m = **socioeconomically deprived AND spectrally jumbled** (mixed roofing).

---

## 6. Methodology

- **Labelling:** window is positive if ≥50% inside AGSN, negative if 0% (0<x<50% excluded
  as ambiguous). Negatives = 4× positives, half "hard" (brightest/most built-up non-favela).
- **Spatial CV:** ~8 km blocks → 5 folds; predict every window from a model that never saw
  its block (no spatial leakage; honest scores even on training favelas).
- **RMM-only training (`RMM_ONLY=True`):** positives AND negatives sampled inside the RMM
  (`has_socio==1`), so hard negatives are the RMM's *own* built-up confusers. This was a
  PR-curve shift up (see §7), not just a tradeoff.
- **Socio NaN handling:** outside-RMM windows have no socio → imputed with sentinel −1.0
  (kept for the full-image map; irrelevant to RMM-scoped training since there's no NaN there).
- **RMM-clipped map:** `favela_probability_rmm.tif` masks everything outside socio coverage.
- **Post-filter (eval):** candidate blobs filtered by area ≥1 ha & blob-mean ≥0.6 to prune
  speckle (~half the false blobs are <1 ha).

---

## 7. Results (out-of-fold, AGSN, inside RMM)

**Discrimination:** ROC-AUC **0.979** (favela vs sampled neg), **0.978** vs hard urban,
PR-AP **0.923**. (Unsupervised predecessor: ~0.5 on the hard task.)

**Detection** (% of in-RMM favelas):
| definition | core (217) | periphery (37) | all (254) |
|---|---|---|---|
| any hot window (max ≥ 0.5) — **headline** | 96.8% | 91.9% | **96.1%** |
| object-level (≥25% lit, T 0.5) | 89.9% | 89.2% | — |
| whole-polygon mean ≥ 0.5 | 84.8% | 75.7% | 83.5% |

(83.5% mean is an averaging artifact — 32 of 41 "missed" favelas contain a hot window.)

**Object precision/recall** (raw / **filtered**):
| T | precision | recall core | recall peri |
|---|---|---|---|
| 0.70 | 18.3% / **26.9%** | 72% | 68% |
| 0.90 | 39.0% / **44.7%** | 41% | 14% |

Filter at T0.7 lifts precision with **zero recall loss**. Precision is a **floor** (§8).

**Generalisation:** unseen hand-drawn favelas ROC **0.861** (vs 0.979 in-distribution) — the
honest gap (circularity, §8). RMM-scoping improved this from 0.826.

---

## 8. Limitations (the real bottleneck = data, not the model)

- **Spatial concentration:** 53% of positive windows in 3 CV blocks; only 16/70 RMM blocks
  contain a favela. It's effectively a **Maceió-core model**; periphery is thin.
- **Precision is a floor, not measurable yet:** AGSN is comprehensive only in the Maceió
  core, so many "false" blobs in the periphery are **real unmapped favelas** → see
  `false_blobs_rmm.gpkg`. True precision needs field/QGIS verification.
- **Periphery statistical power:** n=37 → recall 95% CI ≈ ±15 pp. Periphery claims are weak.
- **Resolution:** 10 m floors sub-hectare favelas (tiny-favela recall ~73% by mean / 92% by max).
- **Circularity:** AGSN is partly *defined* by deprivation and `vuln_mean` is the top feature
  → inflates in-distribution metrics; lower on truly-unseen favelas.

Not a raw-volume problem (1,419 positive windows → ROC 0.98). The levers are **label
completeness/diversity and resolution.**

---

## 9. Key decisions

- **RF, not HGB.** HGB's native-NaN advantage is moot inside the RMM (no NaN there).
- **RF, not SVM — now benchmarked on satellite** (`compare_models.py`, same labels/CV):
  RF **ROC 0.979 / PR 0.960** vs supervised RBF-SVM **0.966 / 0.937** on favela-vs-hard-urban.
  RF edges a fair RBF-SVM and adds native missing-data handling, probabilities, and
  interpretability.
- **Paradigm comparison, quantified on equal footing** (`compare_paradigms.py`, same labeled
  set + spatial CV) — the paper's real contrast. On **favela-vs-hard-urban**:
  | approach | ROC | PR | note |
  |---|---|---|---|
  | Cosine similarity (1 reference, no negatives) | **0.496** | 0.321 | coin-flip — *confirms* the old "~0.5" claim, on satellite |
  | One-Class SVM (positives only, no negatives) | 0.855 | 0.748 | better than expected, but no negatives = ceiling |
  | RandomForest (supervised, +negatives) | **0.979** | 0.960 | the production model |
  **Correction:** the long-cited "~0.5" was the *cosine-similarity* approach, **not** the
  One-Class SVM (which is 0.855). The monotonic story = one reference → full positive
  distribution → +negatives; **the negatives are the jump** (0.855 → 0.979 on the hard task).
- **One global model + region as context, NOT one model per municipality** (273 favelas
  concentrated in the core → sample starvation). Report core vs periphery instead.
- **Screening tool, not a delineator.** Operate at T≈0.7 on the RMM-clipped map; report
  max/“any hot window” detection alongside the conservative mean.

---

## 10. Next steps

- [ ] **Hard-negative mining (bootstrap):** train→predict→add top non-AGSN RMM blobs as hard
      negatives→retrain. The lever that lifts precision *without* the recall cost of tightening
      the prior. **Exclude the periphery when mining** (its "false" blobs may be real favelas).
- [ ] **Human: verify `false_blobs_rmm.gpkg` in QGIS** → true precision + new positive labels.
- [ ] Use the **0.5 m aerial** (on `drone-50cm-image` branch) for small-favela validation /
      a multi-resolution story.
- [ ] Expand the hand-drawn test set (cheap labels, strengthens generalisation claims).
- [ ] Ask collaborator (Bibi) for raw socio components / continuous per-capita income / census
      year (reduces circularity, breaks the pre-combined-index limitation).
- [ ] Optional precision/recall dial: expose `NEG_RATIO` (sweep showed r8 → P@0.7 ~20% but
      periphery recall 78%→57%; don't over-tighten — kills periphery first).

---

## 11. Branches & gotchas

- **Branches:** `main` = current supervised RMM pipeline; `drone-50cm-image` = original 0.5 m
  aerial/unsupervised version (preserved); `sentinel2-support` = working branch (= main).
- **Repo name is legacy:** `similarity_search` reflects the *original* approach (cosine
  similarity to one reference favela + One-Class SVM), which was superseded because it could
  not separate favela from other built-up (ROC ~0.5). The current pipeline is **supervised
  classification**, not similarity search. Describe it as such in the thesis.
- **Cosmetic warnings (not bugs):** `Mean of empty slice` (all-zero edge patches, handled by
  `nan_to_num`); `Ground extent ≈ 1 m` (print bug — `pixel_size` in degrees, real ≈10 m);
  geographic-CRS centroid warnings (negligible at this scale).
- **Housekeeping:** `output/features.csv.bak` (socio-integration backup) — safe to delete.
- **Sharing:** `output_share/` holds the curated result + input rasters for collaborators.
