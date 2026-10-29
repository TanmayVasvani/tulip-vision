"""Heatmaps an operator can read at a glance.

Patch distances are divided by the decision threshold, so colour means the
same thing on every SKU: clear below 75% of the threshold, half-strength red
at the threshold, full red from 125%. A rejected brush always shows a red
region plus a box around its worst spot; a clean pass shows almost nothing,
and a near-miss shows a faint blush exactly where it is borderline.
"""

from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .data import roi_box_px

_RED = np.array([226.0, 24.0, 36.0], dtype=np.float32)


def _resize_long_side(img: Image.Image, long_side: int | None) -> Image.Image:
    if not long_side:
        return img
    w, h = img.size
    s = long_side / max(w, h)
    if abs(s - 1.0) < 1e-3:
        return img
    return img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.Resampling.LANCZOS)


def heat_mask(dists, grid, size, threshold) -> np.ndarray:
    """Patch distances -> smooth [H, W] mask in 0..1 at image ``size`` (w, h)."""
    h, w = grid
    m = np.asarray(dists, dtype=np.float32).reshape(h, w) / max(float(threshold), 1e-9)
    m = np.clip((m - 0.75) / 0.5, 0.0, 1.0)
    im = Image.fromarray((m * 255.0).astype(np.uint8), mode="L")
    im = im.resize(size, Image.Resampling.BILINEAR)
    cell = size[0] / float(w)
    im = im.filter(ImageFilter.GaussianBlur(radius=max(cell * 0.5, 1.0)))
    return np.asarray(im, dtype=np.float32) / 255.0


def anomaly_overlay(img: Image.Image, dists, grid, threshold, *, long_side: int | None = 560,
                    alpha_max: float = 0.6, mark_peak: bool = True) -> Image.Image:
    img = _resize_long_side(img.convert("RGB"), long_side)
    base = np.asarray(img, dtype=np.float32)
    a = heat_mask(dists, grid, img.size, threshold)[..., None] * alpha_max
    out = Image.fromarray(np.clip(base * (1.0 - a) + _RED * a, 0, 255).astype(np.uint8))
    d = np.asarray(dists)
    if mark_peak and float(d.max()) >= float(threshold):
        h, w = grid
        r, c = divmod(int(np.argmax(d)), w)
        cw, ch = out.width / w, out.height / h
        cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
        half = max(cw, ch) * 2.2
        box = [cx - half, cy - half, cx + half, cy + half]
        draw = ImageDraw.Draw(out)
        draw.rectangle(box, outline=(18, 18, 18), width=5)
        draw.rectangle(box, outline=(255, 210, 0), width=3)
    return out


def roi_preview(img: Image.Image, roi, dim: float = 0.45) -> Image.Image:
    """Full frame with everything outside the ROI darkened and the ROI outlined."""
    img = img.convert("RGB")
    if roi is None:
        return img.copy()
    left, top, right, bottom = roi_box_px(img.size, roi)
    arr = np.asarray(img, dtype=np.float32) * dim
    arr[top:bottom, left:right] = np.asarray(img, dtype=np.float32)[top:bottom, left:right]
    out = Image.fromarray(arr.astype(np.uint8))
    d = ImageDraw.Draw(out)
    d.rectangle([left, top, right - 1, bottom - 1], outline=(255, 210, 0), width=3)
    return out


def to_b64_jpeg(img: Image.Image, max_side: int | None = None, quality: int = 85) -> str:
    if max_side and max(img.size) > max_side:
        img = _resize_long_side(img, max_side)
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
