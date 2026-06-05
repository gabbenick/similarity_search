# Favela Similarity Search
### Feature-based similarity search for informal settlement detection
### Maceió, Brazil — 50cm/px aerial imagery

---

## How it works

You define one known favela zone in the image. The pipeline characterises
that zone using texture, colour and edge features, then slides a window
across the full image computing how similar every location is to the
reference. The output is a heatmap where bright areas = most similar
to your favela zone.

No model training required. No large labeled dataset required.
Just one known example and the rest follows from feature similarity.

```
Known favela zone
      ↓
Feature vector (texture + colour + edge density)
      ↓
Compare to every window in the image
      ↓
Similarity heatmap → load in QGIS
```

---

## Setup

```bash
pip install -r requirements.txt
```

Place your raster in `data/image.tif`.

---

## Step 1 — Find your reference coordinates

Open `image.tif` in QGIS. Hover over the corners of your known
favela zone. The bottom bar shows X, Y in map coordinates.

Convert to pixel coordinates:
```bash
python coords.py --info           # see image extent
python coords.py --x 652300 --y 8840500   # convert a point
python coords.py                  # interactive mode
```

---

## Step 2 — Set reference zone in config.py

```python
REFERENCE_COL_MIN = 800   # left edge of favela (pixels)
REFERENCE_COL_MAX = 1200  # right edge
REFERENCE_ROW_MIN = 600   # top edge
REFERENCE_ROW_MAX = 1000  # bottom edge
```

---

## Step 3 — Run the search

```bash
python search.py
```

Takes ~1-2 minutes for a 2000×2000 image.

---

## Step 4 — Load results in QGIS

```
Layer → Add Layer → Add Raster Layer → output/heatmap.tif
```

Style: Properties → Symbology → Singleband pseudocolor
Color ramp: Reds or Spectral reversed
Higher values = more similar to your reference favela zone.

Also load `output/heatmap_norm.tif` — same data scaled 0-255,
sometimes easier to style in QGIS.

---

## Step 5 — Add slope/DSM later

When you have altitude data, place it in `data/dsm.tif` and set:

```python
USE_DSM = True   # in config.py
```

Rerun `search.py`. The altitude features (mean height, height
variation, surface roughness) will be added to the feature vector,
improving discrimination between informal and formal areas.

---

## Output files

| File | Description |
|---|---|
| `output/heatmap.tif` | Float32 similarity map — load in QGIS |
| `output/heatmap_norm.tif` | Same, scaled 0-255 (uint8) |
| `output/features.csv` | Per-window feature table — inspect values |
| `output/overview.png` | Quick visual check without opening QGIS |

---

## Tuning

**Window size** — controls the scale of analysis:
- `WINDOW_SIZE = 100` → 50m × 50m  (fine, sees individual blocks)
- `WINDOW_SIZE = 200` → 100m × 100m  (recommended, neighbourhood scale)
- `WINDOW_SIZE = 400` → 200m × 200m  (coarse, district scale)

**Stride** — controls overlap and heatmap smoothness:
- `STRIDE = WINDOW_SIZE` → no overlap, faster
- `STRIDE = WINDOW_SIZE // 2` → 50% overlap, smoother (recommended)

**Similarity metric**:
- `"cosine"` → recommended, scale-invariant
- `"euclidean"` → try if cosine gives flat results
