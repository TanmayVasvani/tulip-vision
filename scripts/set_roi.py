"""Set the brush-head region of interest (ROI) for a SKU.

    python scripts/set_roi.py --sku zecta_classic                 # drag a box with the mouse
    python scripts/set_roi.py --sku zecta_classic --roi 0.09,0.32,0.43,0.68
    python scripts/set_roi.py --sku zecta_classic --image shot.jpg

Tufting and trimming defects live on the head, so the model should look at
the head at full detail instead of the whole frame shrunk down. Draw the box
around the head with a few millimetres of margin (placement in the jig
varies slightly). Saved as frame fractions to data/<sku>/roi.json, plus a
preview image. Retrain after changing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, data_dir, load_config, run_cli  # noqa: E402
from qc.data import list_images, load_image, parse_roi, roi_box_px, write_roi_file  # noqa: E402
from qc.train import choose_input_size  # noqa: E402
from qc.vis import roi_preview  # noqa: E402


def pick_with_mouse(img):
    try:
        import cv2
        import numpy as np
    except ImportError:
        raise SystemExit("OpenCV is not installed — pass --roi x0,y0,x1,y1 instead")
    scale = min(1.0, 1280 / img.width, 800 / img.height)
    shown = img.resize((round(img.width * scale), round(img.height * scale)))
    bgr = cv2.cvtColor(np.asarray(shown), cv2.COLOR_RGB2BGR)
    title = "Drag a box around the brush head, then ENTER (c = cancel)"
    try:
        x, y, w, h = cv2.selectROI(title, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()
    except cv2.error:
        raise SystemExit("this OpenCV build has no GUI (headless) — pass --roi x0,y0,x1,y1 instead")
    if w == 0 or h == 0:
        raise SystemExit("cancelled — nothing saved")
    return (x / shown.width, y / shown.height, (x + w) / shown.width, (y + h) / shown.height)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None, help="reference frame (default: first image in good/)")
    ap.add_argument("--roi", default=None, help="x0,y0,x1,y1 as fractions of the frame")
    args = ap.parse_args()

    sku_dir = data_dir(load_config(args.config), check_sku(args.sku))
    ref = Path(args.image) if args.image else next(iter(list_images(sku_dir / "good")), None)
    if ref is None or not ref.exists():
        raise SystemExit(f"no reference image: capture a few goods into {sku_dir / 'good'} or pass --image")
    img = load_image(ref)
    roi = parse_roi(args.roi) if args.roi else pick_with_mouse(img)
    p = write_roi_file(sku_dir, roi, img.size)
    preview = sku_dir / "roi_preview.jpg"
    roi_preview(img, roi).save(preview, quality=90)
    l, t, r, b = roi_box_px(img.size, roi)
    w, h = choose_input_size(r - l, b - t)
    print(f"ROI {tuple(round(v, 4) for v in roi)} -> {r - l}x{b - t} px of a {img.width}x{img.height} frame")
    print(f"model input will be {w}x{h} (aspect kept)")
    print(f"saved {p} and {preview} — retrain to apply")


if __name__ == "__main__":
    run_cli(main)
