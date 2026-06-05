"""
coords.py — Helper to convert QGIS map coordinates to pixel coordinates.

The REFERENCE_* values in config.py need to be in PIXEL coordinates.
QGIS shows coordinates in map units (metres for EPSG:31985).
This script converts between them.

Usage
─────
    # Find pixel coordinates of a map point:
    python coords.py --x 652300 --y 8840500

    # Find map coordinates of a pixel:
    python coords.py --row 600 --col 800

    # Print image extent and suggest reference zone:
    python coords.py --info

    # Interactive mode — enter coordinates one by one:
    python coords.py
"""

import sys
import argparse
import rasterio
import numpy as np
import config


def load_image_meta():
    with rasterio.open(config.IMAGE_PATH) as src:
        return {
            "width"    : src.width,
            "height"   : src.height,
            "transform": src.transform,
            "crs"      : src.crs,
            "bounds"   : src.bounds,
            "res"      : src.res,
        }


def map_to_pixel(x_map, y_map, transform):
    """Convert map coordinates (X, Y) to pixel (row, col)."""
    col = (x_map - transform.c) / transform.a
    row = (y_map - transform.f) / transform.e
    return int(round(row)), int(round(col))


def pixel_to_map(row, col, transform):
    """Convert pixel (row, col) to map coordinates (X, Y)."""
    x = transform.c + col * transform.a
    y = transform.f + row * transform.e
    return x, y


def print_info(meta):
    t = meta["transform"]
    b = meta["bounds"]
    print(f"\n{'─'*50}")
    print(f"  Image info")
    print(f"{'─'*50}")
    print(f"  Size       : {meta['width']} × {meta['height']} px")
    print(f"  Resolution : {meta['res'][0]:.4f} m/px")
    print(f"  CRS        : {meta['crs']}")
    print(f"\n  Map extent (metres):")
    print(f"    Left   : {b.left:.2f}")
    print(f"    Right  : {b.right:.2f}")
    print(f"    Bottom : {b.bottom:.2f}")
    print(f"    Top    : {b.top:.2f}")
    print(f"\n  Pixel origin (top-left = row 0, col 0):")
    print(f"    Map X  : {t.c:.2f}")
    print(f"    Map Y  : {t.f:.2f}")
    print(f"\n  Tip: In QGIS, hover over your favela zone corners")
    print(f"  and note the X, Y coordinates shown in the bottom bar.")
    print(f"  Then run:  python coords.py --x <X> --y <Y>")


def main():
    parser = argparse.ArgumentParser(
        description="Convert between QGIS map coordinates and pixel coordinates."
    )
    parser.add_argument("--x",    type=float, help="Map X coordinate (easting)")
    parser.add_argument("--y",    type=float, help="Map Y coordinate (northing)")
    parser.add_argument("--row",  type=int,   help="Pixel row")
    parser.add_argument("--col",  type=int,   help="Pixel column")
    parser.add_argument("--info", action="store_true",
                        help="Print image extent information")
    args = parser.parse_args()

    try:
        meta = load_image_meta()
    except Exception as e:
        print(f"❌ Could not open image: {e}")
        print(f"   Check IMAGE_PATH in config.py: {config.IMAGE_PATH}")
        sys.exit(1)

    transform = meta["transform"]

    if args.info:
        print_info(meta)
        return

    if args.x is not None and args.y is not None:
        row, col = map_to_pixel(args.x, args.y, transform)
        print(f"\n  Map ({args.x:.2f}, {args.y:.2f})  →  pixel row={row}, col={col}")
        if 0 <= row < meta["height"] and 0 <= col < meta["width"]:
            print(f"  ✅ Within image bounds")
        else:
            print(f"  ⚠️  Outside image bounds "
                  f"(image is {meta['height']}×{meta['width']})")
        return

    if args.row is not None and args.col is not None:
        x, y = pixel_to_map(args.row, args.col, transform)
        print(f"\n  Pixel row={args.row}, col={args.col}  →  map ({x:.2f}, {y:.2f})")
        return

    # interactive mode
    print_info(meta)
    print(f"\n{'─'*50}")
    print("  Interactive coordinate converter")
    print("  Enter map coordinates from QGIS to get pixel row/col")
    print("  Press Ctrl+C to exit")
    print(f"{'─'*50}\n")

    while True:
        try:
            x_str = input("  Map X (easting,  or 'q' to quit): ").strip()
            if x_str.lower() == 'q':
                break
            y_str = input("  Map Y (northing): ").strip()
            x_map = float(x_str)
            y_map = float(y_str)
            row, col = map_to_pixel(x_map, y_map, transform)
            print(f"  → pixel row={row}, col={col}\n")
        except ValueError:
            print("  Enter a valid number.\n")
        except KeyboardInterrupt:
            print("\n  Bye.")
            break


if __name__ == "__main__":
    main()
