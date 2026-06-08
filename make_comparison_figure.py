"""
make_comparison_figure.py — slide-ready 3-panel comparison.

Crops to the RMM extent and renders, side by side:
  1. Sentinel-2 true colour (RGB_432)  — what the ground looks like
  2. Favela probability (0-1, Reds)    — the model's continuous score
  3. Favela classified (>= T)          — the binary decision

Output: output/fig_comparison.png

Run:  python make_comparison_figure.py
"""

import os
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import config

RGB    = os.path.join(config.DATA_DIR, "RGB_432.tif")
PROB   = os.path.join(config.OUTPUT_DIR, "favela_probability_rmm.tif")
CLASS  = os.path.join(config.OUTPUT_DIR, "favela_classified_rmm.tif")
OUT    = os.path.join(config.OUTPUT_DIR, "fig_comparison.png")
MAX_DIM = 1600   # longest side of each panel, in pixels


def valid_bbox(path, pad=20):
    """Row/col bounds of finite (in-RMM) pixels, with a small pad."""
    with rasterio.open(path) as src:
        a = src.read(1)
        H, W = src.height, src.width
    m = np.isfinite(a) if a.dtype.kind == "f" else (a != src.nodata)
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad, H)
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad, W)
    return r0, r1, c0, c1


def read_window(path, bbox, bands=1):
    """Read a window, decimated so its longest side <= MAX_DIM."""
    r0, r1, c0, c1 = bbox
    h, w = r1 - r0, c1 - c0
    scale = max(h, w) / MAX_DIM
    oh, ow = int(h / scale), int(w / scale)
    win = Window(c0, r0, w, h)
    with rasterio.open(path) as src:
        idx = list(range(1, bands + 1))
        arr = src.read(idx, window=win, out_shape=(bands, oh, ow),
                       resampling=rasterio.enums.Resampling.bilinear)
    return arr


def stretch(rgb):
    """Percentile stretch a (3,H,W) uint8/float array → (H,W,3) 0-1."""
    out = []
    for b in rgb:
        b = b.astype(np.float32)
        lo, hi = np.nanpercentile(b, 2), np.nanpercentile(b, 98)
        out.append(np.clip((b - lo) / (hi - lo + 1e-8), 0, 1))
    return np.stack(out, -1)


def main():
    bbox = valid_bbox(PROB)
    print(f"  RMM crop rows {bbox[0]}–{bbox[1]}, cols {bbox[2]}–{bbox[3]}")

    rgb  = stretch(read_window(RGB, bbox, bands=3))
    prob = read_window(PROB, bbox, bands=1)[0].astype(np.float32)
    cls  = read_window(CLASS, bbox, bands=1)[0]

    prob_masked = np.ma.masked_invalid(prob)
    cls_masked  = np.ma.masked_where(cls == 255, cls)        # outside RMM
    binary_cmap = ListedColormap([(0.82, 0.82, 0.82), (0.86, 0.08, 0.08)])

    fig, axes = plt.subplots(1, 3, figsize=(21, 8))

    axes[0].imshow(rgb)
    axes[0].set_title("Sentinel-2 true colour (RGB 432)", fontsize=14)

    im = axes[1].imshow(prob_masked, cmap="Reds", vmin=0, vmax=1)
    axes[1].set_title("Favela probability (RandomForest, 0–1)", fontsize=14)
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02, label="P(favela)")

    axes[2].imshow(cls_masked, cmap=binary_cmap, vmin=0, vmax=1)
    axes[2].set_title("Classified  (red = favela ≥ 0.7)", fontsize=14)

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    plt.suptitle("Favela detection — Região Metropolitana de Maceió",
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved → {OUT}")


if __name__ == "__main__":
    main()
