"""
add_slope_feature.py — Augment output/features.csv with DSM slope features.

Adds DSM_slope_mean / DSM_slope_std (and refreshes the other DSM columns) using
the updated extract_features.dsm_features(), WITHOUT re-running the expensive
671k-window spectral extraction. The spectral/index columns are untouched.

Run:  python add_slope_feature.py
"""

import os
import numpy as np
import pandas as pd

import config
from search import reference_profile, load_dsm
from extract_features import dsm_features

DSM_COLS = ["DSM_mean", "DSM_std", "DSM_roughness",
            "DSM_slope_mean", "DSM_slope_std"]


def main():
    print("Loading features.csv ...")
    df = pd.read_csv(config.FEATURES_CSV)
    print(f"  {len(df):,} windows, {len(df.columns)} columns")

    ref_prof = reference_profile()
    image_dsm = load_dsm(ref_prof)
    if image_dsm is None:
        raise SystemExit("DSM not available — cannot add slope.")

    ws = config.WINDOW_SIZE
    rows = df["row"].to_numpy()
    cols = df["col"].to_numpy()
    out = np.zeros((len(df), len(DSM_COLS)), dtype=np.float32)

    print("Computing DSM + slope per window ...")
    for i, (r, c) in enumerate(zip(rows, cols)):
        out[i] = dsm_features(image_dsm[r:r+ws, c:c+ws])
        if (i + 1) % 50000 == 0:
            print(f"  {i+1:,}/{len(df):,}", end="\r")
    print()

    # drop any existing DSM columns, then re-insert the full refreshed set
    df = df.drop(columns=[c for c in DSM_COLS if c in df.columns])
    for j, name in enumerate(DSM_COLS):
        df[name] = out[:, j]

    df.to_csv(config.FEATURES_CSV, index=False)
    print(f"Saved → {config.FEATURES_CSV}")
    print(f"  slope: mean={out[:,3].mean():.2f}°  "
          f"max-window-mean={out[:,3].max():.2f}°  "
          f"std-range=[{out[:,4].min():.2f},{out[:,4].max():.2f}]")
    print(f"  feature columns now: "
          f"{[c for c in df.columns if c not in ('row','col','map_x','map_y','similarity')].__len__()}")


if __name__ == "__main__":
    main()
