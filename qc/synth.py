"""Synthetic toothbrush images: a demo dataset before the rig exists, and test data.

A top-down camera view of a brush lying in a jig: matte grey background,
soft shadow, blue handle, 38 tufts of white and blue bristles. Good images
vary the way production does — placement in the jig, a fraction of a degree
of rotation, tuft position and size tolerance, bristle texture, lighting,
focus, sensor noise. Defects are planted with known boxes, so heatmaps can
be checked against the truth.

This proves the pipeline end to end. It says nothing about accuracy on real
brushes; that number only comes from the real rig.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

DEFECT_TYPES = ("missing_tuft", "sparse_tuft", "splayed_tuft", "stray_bristle", "contamination")
_SS = 2  # supersampling factor for anti-aliased edges


@dataclass(frozen=True)
class BrushStyle:
    frame: tuple = (960, 540)
    px_per_mm: float = 9.0
    head_center: tuple = (250.0, 270.0)
    head_len_mm: float = 27.0
    head_w_mm: float = 12.6
    neck_len_mm: float = 26.0
    neck_w_mm: float = 5.6
    handle_w_mm: float = 12.0
    rows: tuple = (9, 10, 10, 9)
    pitch_mm: float = 2.4
    tuft_d_mm: float = 2.15
    plastic: tuple = (58, 128, 196)
    bristle: tuple = (236, 236, 230)
    bristle_accent: tuple = (70, 165, 222)
    background: tuple = (124, 127, 122)

    def tufts(self):
        """[(row, col, x_mm, y_mm, accent)] relative to the head centre."""
        out = []
        nr = len(self.rows)
        for r, n in enumerate(self.rows):
            y = (r - (nr - 1) / 2.0) * self.pitch_mm
            for c in range(n):
                x = (c - (n - 1) / 2.0) * self.pitch_mm
                out.append((r, c, x, y, c in (0, n - 1)))
        return out


def default_roi(style: BrushStyle = BrushStyle(), margin_mm: float = 4.5):
    """Head plus margin, as frame fractions — what you'd draw with set_roi.py."""
    w, h = style.frame
    cx, cy = style.head_center
    hx = (style.head_len_mm / 2 + margin_mm) * style.px_per_mm
    hy = (style.head_w_mm / 2 + margin_mm) * style.px_per_mm
    return (round((cx - hx) / w, 4), round((cy - hy) / h, 4), round((cx + hx) / w, 4), round((cy + hy) / h, 4))


# ------------------------------------------------------------- primitives
def _smooth_noise(shape, rng, radius: float) -> np.ndarray:
    n = rng.standard_normal(shape)
    im = Image.fromarray(np.clip(128 + 42 * n, 0, 255).astype(np.uint8), mode="L")
    im = im.filter(ImageFilter.GaussianBlur(radius=radius))
    a = np.asarray(im, dtype=np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)


def _blend(rgb, alpha, x0, y0, col, a):
    """Porter-Duff 'over' (straight alpha) of a colour patch with coverage ``a`` at (x0, y0)."""
    hh, ww = a.shape
    H, W = alpha.shape
    xs, ys = max(0, x0), max(0, y0)
    xe, ye = min(W, x0 + ww), min(H, y0 + hh)
    if xs >= xe or ys >= ye:
        return
    sa = a[ys - y0 : ye - y0, xs - x0 : xe - x0]
    sc = col[ys - y0 : ye - y0, xs - x0 : xe - x0] if col.ndim == 3 else col
    ao = alpha[ys:ye, xs:xe]
    na = sa + ao * (1 - sa)
    rgb[ys:ye, xs:xe] = (sc * sa[..., None] + rgb[ys:ye, xs:xe] * (ao * (1 - sa))[..., None]) / np.maximum(
        na, 1e-6
    )[..., None]
    alpha[ys:ye, xs:xe] = na


def _disk(rgb, alpha, cx, cy, r, color, soft=1.2):
    x0, y0 = int(cx - r - 3), int(cy - r - 3)
    yy, xx = np.mgrid[y0 : int(cy + r + 4), x0 : int(cx + r + 4)].astype(np.float32) + 0.5
    a = np.clip((r - np.hypot(xx - cx, yy - cy)) / soft, 0, 1)
    _blend(rgb, alpha, x0, y0, np.asarray(color, np.float32), a)


def _tuft(rgb, alpha, cx, cy, r, color, rng, *, irregular=0.04, gaps=0.0):
    x0, y0 = int(cx - r * 1.2 - 4), int(cy - r * 1.2 - 4)
    x1, y1 = int(cx + r * 1.2 + 5), int(cy + r * 1.2 + 5)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32) + 0.5
    dx, dy = xx - cx, yy - cy
    dist = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    k = int(rng.integers(5, 9))
    ph = rng.uniform(0, 2 * np.pi, 2)
    reff = r * (1 + irregular * (0.6 * np.sin(k * th + ph[0]) + 0.4 * np.sin((k + 3) * th + ph[1])))
    a = np.clip((reff - dist) / (1.1 * _SS), 0, 1)
    grain = _smooth_noise(dist.shape, rng, 0.9)          # filament ends
    tex = 0.84 + 0.085 * grain
    radial = 0.88 + 0.14 * np.clip(1 - (dist / r) ** 2, 0, 1)
    rim = 0.78 + 0.22 * np.clip((reff - dist) / (0.35 * r), 0, 1)
    col = np.asarray(color, np.float32) * (tex * radial * rim)[..., None]
    if gaps > 0:
        holes = _smooth_noise(dist.shape, rng, 1.4) < np.quantile(grain, gaps)
        a = a * np.where(holes, 0.12, 1.0)
    _blend(rgb, alpha, x0, y0, col, a.astype(np.float32))


def _stamp_mask(rgb, alpha, mask_img: Image.Image, color, blur=0.7, strength=1.0):
    m = np.asarray(mask_img.filter(ImageFilter.GaussianBlur(radius=blur)), dtype=np.float32) / 255.0
    _blend(rgb, alpha, 0, 0, np.asarray(color, np.float32), m * strength)


# ------------------------------------------------------------------ render
def _background(style: BrushStyle, rng) -> np.ndarray:
    """Matte grey card with a lighting gradient and low-frequency blotches."""
    W, H = style.frame
    yyf, xxf = np.mgrid[0:H, 0:W].astype(np.float32)
    g = rng.uniform(-8, 8) * (xxf / W - 0.5) + rng.uniform(-8, 8) * (yyf / H - 0.5)
    low = Image.fromarray(np.clip(128 + 40 * rng.standard_normal((6, 10)), 0, 255).astype(np.uint8), "L")
    low = (np.asarray(low.resize((W, H), Image.Resampling.BICUBIC), np.float32) - 128) / 40 * 2.5
    return np.asarray(style.background, np.float32) + (g + low)[..., None]


def _camera(frame: np.ndarray, rng) -> Image.Image:
    """Exposure, white balance, focus and sensor noise."""
    H, W = frame.shape[:2]
    c, bri, tint = rng.uniform(0.96, 1.04), rng.uniform(0.96, 1.04), rng.uniform(-0.015, 0.015)
    frame = ((frame - 128.0) * c + 128.0) * bri * np.array([1 + tint, 1.0, 1 - tint], np.float32)
    img = Image.fromarray(np.clip(frame, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 0.75)))
    arr = np.asarray(img, np.float32) + rng.normal(0, 1.8, (H, W, 1)).astype(np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def render(style: BrushStyle = BrushStyle(), rng=None, defect: str | None = None, empty: bool = False):
    """-> (PIL RGB frame, info{'defect', 'bbox' [x0,y0,x1,y1] frame px or None}).

    ``empty=True`` renders the jig with no brush in it.
    """
    rng = rng if rng is not None else np.random.default_rng()
    if defect is not None and defect not in DEFECT_TYPES:
        raise ValueError(f"unknown defect {defect!r}; options: {DEFECT_TYPES}")
    if empty:
        return _camera(_background(style, rng), rng), {"defect": None, "bbox": None, "empty": True}
    W, H = style.frame
    ppm = style.px_per_mm * _SS
    left_mm = style.head_len_mm / 2 + 7
    right_mm = (W - style.head_center[0] + 40) / style.px_per_mm
    half_mm = style.head_w_mm / 2 + 8
    cw, ch = int((left_mm + right_mm) * ppm), int(2 * half_mm * ppm)
    ox, oy = left_mm * ppm, half_mm * ppm

    def P(x_mm, y_mm):
        return (ox + x_mm * ppm, oy + y_mm * ppm)

    # ---- plastic body: head, neck, handle
    L, Wd = style.head_len_mm, style.head_w_mm
    mask = Image.new("L", (cw, ch), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([P(-L / 2, -Wd / 2), P(L / 2, Wd / 2)], radius=Wd / 2 * 0.92 * ppm, fill=255)
    nw, hw, nl = style.neck_w_mm / 2, style.handle_w_mm / 2, style.neck_len_mm
    top = [(L / 2 - 6, -Wd * 0.34), (L / 2 + 3, -nw * 1.18), (L / 2 + nl * 0.55, -nw),
           (L / 2 + nl, -nw * 1.12), (L / 2 + nl + 16, -hw), (right_mm + 5, -hw)]
    d.polygon([P(x, y) for x, y in top] + [P(x, -y) for x, y in reversed(top)], fill=255)
    m = np.asarray(mask, dtype=np.float32) / 255.0
    b = np.asarray(mask.filter(ImageFilter.GaussianBlur(radius=1.7 * ppm)), dtype=np.float32) / 255.0
    yy = (np.arange(ch, dtype=np.float32)[:, None] - oy) / (half_mm * ppm)
    shade = (0.66 + 0.42 * np.clip((b - 0.3) / 0.7, 0, 1)) * (1.0 - 0.05 * yy)
    rgb = np.asarray(style.plastic, np.float32) * shade[..., None]
    alpha = m.copy()

    # ---- tuft holes, then tufts
    tufts = style.tufts()
    jit = rng.normal(0, 0.05, (len(tufts), 2))
    rad = style.tuft_d_mm / 2 * ppm * (1 + rng.normal(0, 0.025, len(tufts)))
    centres = [P(x + j[0], y + j[1]) for (_, _, x, y, _), j in zip(tufts, jit)]
    for cx, cy in centres:
        _disk(rgb, alpha, cx, cy, style.tuft_d_mm * 0.40 * ppm, np.asarray(style.plastic) * 0.32)

    target = None
    if defect in ("missing_tuft", "sparse_tuft", "splayed_tuft"):
        target = int(rng.integers(len(tufts)))
    bbox_mm = None
    for i, ((_, _, x, y, accent), (cx, cy)) in enumerate(zip(tufts, centres)):
        color = style.bristle_accent if accent else style.bristle
        r = rad[i]
        if i == target and defect == "missing_tuft":
            bbox_mm = (x - 1.3, y - 1.3, x + 1.3, y + 1.3)
            continue
        if i == target and defect == "sparse_tuft":
            r *= rng.uniform(0.62, 0.80)
            _tuft(rgb, alpha, cx, cy, r, color, rng, gaps=rng.uniform(0.25, 0.42))
            bbox_mm = (x - 1.3, y - 1.3, x + 1.3, y + 1.3)
            continue
        if i == target and defect == "splayed_tuft":
            continue  # drawn last, on top of its neighbours
        _tuft(rgb, alpha, cx, cy, r, color, rng)
    if defect == "splayed_tuft":
        _, _, x, y, accent = tufts[target]
        cx, cy = centres[target]
        f = rng.uniform(1.22, 1.42)
        _tuft(rgb, alpha, cx, cy, rad[target] * f, style.bristle_accent if accent else style.bristle,
              rng, irregular=rng.uniform(0.10, 0.16))
        e = style.tuft_d_mm / 2 * f * 1.2
        bbox_mm = (x - e, y - e, x + e, y + e)

    if defect == "stray_bristle":
        edge = [i for i, (r_, c_, _, _, _) in enumerate(tufts)
                if r_ in (0, len(style.rows) - 1) or c_ in (0, style.rows[r_] - 1)]
        i = int(rng.choice(edge))
        _, _, x, y, _ = tufts[i]
        ang = math.atan2(y, x * 0.55) + rng.uniform(-0.6, 0.6)  # roughly outward
        length = rng.uniform(2.2, 5.0)
        x0m, y0m = x + 0.5 * math.cos(ang), y + 0.5 * math.sin(ang)
        x1m, y1m = x + (0.5 + length) * math.cos(ang), y + (0.5 + length) * math.sin(ang)
        lm = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(lm).line([P(x0m, y0m), P(x1m, y1m)], fill=255,
                                width=max(2, int(round(rng.uniform(1.2, 1.7) * _SS))))
        _stamp_mask(rgb, alpha, lm, np.asarray(style.bristle, np.float32) * 0.97, blur=0.6)
        bbox_mm = (min(x0m, x1m) - 0.4, min(y0m, y1m) - 0.4, max(x0m, x1m) + 0.4, max(y0m, y1m) + 0.4)

    if defect == "contamination":
        _, _, x, y, _ = tufts[int(rng.integers(len(tufts)))]
        bx, by = x + rng.uniform(-0.8, 0.8), y + rng.uniform(-0.8, 0.8)
        bm = Image.new("L", (cw, ch), 0)
        bd = ImageDraw.Draw(bm)
        ext = 0.0
        for _ in range(int(rng.integers(2, 5))):
            ex, ey = bx + rng.uniform(-0.35, 0.35), by + rng.uniform(-0.35, 0.35)
            rx, ry = rng.uniform(0.18, 0.45), rng.uniform(0.18, 0.45)
            ext = max(ext, abs(ex - bx) + rx, abs(ey - by) + ry)
            bd.ellipse([P(ex - rx, ey - ry), P(ex + rx, ey + ry)], fill=255)
        dirt = np.asarray((52, 40, 30), np.float32) * rng.uniform(0.85, 1.15)
        _stamp_mask(rgb, alpha, bm, dirt, blur=0.8, strength=0.93)
        bbox_mm = (bx - ext - 0.2, by - ext - 0.2, bx + ext + 0.2, by + ext + 0.2)

    # ---- pose: rotate about the head centre, downsample, place in frame
    angle = rng.uniform(-1.2, 1.2)
    hx = style.head_center[0] + rng.uniform(-6, 6)
    hy = style.head_center[1] + rng.uniform(-6, 6)
    # premultiplied colour + alpha go through identical transforms, so edges
    # (a stray bristle over the background) never pick up colour fringes
    size_ds = (cw // _SS, ch // _SS)

    def _xf(im):
        im = im.rotate(angle, resample=Image.Resampling.BICUBIC, center=(ox, oy))
        return im.resize(size_ds, Image.Resampling.LANCZOS)

    rgbp = _xf(Image.fromarray(np.clip(rgb * alpha[..., None], 0, 255).astype(np.uint8), "RGB"))
    alay = _xf(Image.fromarray((alpha * 255.0).astype(np.uint8), "L"))
    lx, ly = int(round(hx - ox / _SS)), int(round(hy - oy / _SS))

    # ---- background with a soft shadow under the brush
    frame = _background(style, rng)
    a_full = Image.new("L", (W, H), 0)
    a_full.paste(alay, (lx, ly))
    shadow = Image.new("L", (W, H), 0)
    shadow.paste(a_full, (4, 6))
    sh = np.asarray(shadow.filter(ImageFilter.GaussianBlur(radius=6)), np.float32) / 255.0
    frame *= (1 - 0.28 * sh)[..., None]
    rgbp_full = Image.new("RGB", (W, H), 0)
    rgbp_full.paste(rgbp, (lx, ly))
    a1 = np.asarray(a_full, np.float32)[..., None] / 255.0
    frame = frame * (1 - a1) + np.asarray(rgbp_full, np.float32)

    img = _camera(frame, rng)

    bbox = None
    if bbox_mm is not None:
        t = math.radians(angle)
        pts = []
        for xm, ym in ((bbox_mm[0], bbox_mm[1]), (bbox_mm[2], bbox_mm[1]),
                       (bbox_mm[0], bbox_mm[3]), (bbox_mm[2], bbox_mm[3])):
            dx, dy = xm * style.px_per_mm, ym * style.px_per_mm
            pts.append((hx + dx * math.cos(t) + dy * math.sin(t), hy - dx * math.sin(t) + dy * math.cos(t)))
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        bbox = [round(min(xs), 1), round(min(ys), 1), round(max(xs), 1), round(max(ys), 1)]
    return img, {"defect": defect, "bbox": bbox, "angle_deg": round(angle, 3),
                 "head_px": [round(hx, 1), round(hy, 1)]}


# ----------------------------------------------------------------- dataset
def make_dataset(out_dir, n_good: int = 320, defects_per_type: int = 10, class_per_type: int = 30,
                 stream_good: int = 36, stream_defects_per_type: int = 3, n_empty: int = 8, seed: int = 0,
                 style: BrushStyle = BrushStyle(), quality: int = 92, log=print) -> dict:
    """Write a complete demo SKU in the same layout the real workflow uses."""
    out = Path(out_dir)
    for sub in ("good", "defect", "classes", "stream", "empty"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    def save(img, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path, quality=quality)

    def rng_for(kind: int, idx: int, k2: int = 0):
        return np.random.default_rng([seed, kind, idx, k2])

    for i in range(n_good):
        save(render(style, rng_for(1, i))[0], out / "good" / f"good_{i:04d}.jpg")
        if (i + 1) % 80 == 0:
            log(f"[synth] good {i + 1}/{n_good}")
    ann = {}
    for t_i, t in enumerate(DEFECT_TYPES):
        for j in range(defects_per_type):
            img, info = render(style, rng_for(2, j, t_i), t)
            rel = f"{t}/{t}_{j:03d}.jpg"
            save(img, out / "defect" / rel)
            ann[rel] = info["bbox"]
        for j in range(class_per_type):
            save(render(style, rng_for(3, j, t_i), t)[0], out / "classes" / t / f"{t}_c{j:03d}.jpg")
        log(f"[synth] {t}: {defects_per_type} eval + {class_per_type} classifier examples")
    for i in range(n_empty):
        save(render(style, rng_for(6, i), empty=True)[0], out / "empty" / f"empty_{i:02d}.jpg")
    for i in range(stream_good):
        save(render(style, rng_for(4, i))[0], out / "stream" / f"s_good_{i:03d}.jpg")
    if stream_good:
        for i in range(2):
            save(render(style, rng_for(7, i), empty=True)[0], out / "stream" / f"s_empty_{i}.jpg")
    for t_i, t in enumerate(DEFECT_TYPES):
        for j in range(stream_defects_per_type):
            save(render(style, rng_for(5, j, t_i), t)[0], out / "stream" / f"s_{t}_{j}.jpg")
    (out / "defect_annotations.json").write_text(json.dumps(ann, indent=1), encoding="utf-8")
    roi = default_roi(style)
    (out / "roi.json").write_text(json.dumps({"roi": list(roi), "reference_frame_size": list(style.frame)},
                                             indent=2), encoding="utf-8")
    manifest = {
        "synthetic": True,
        "seed": seed,
        "style": asdict(style),
        "counts": {"good": n_good, "defect_per_type": defects_per_type, "class_per_type": class_per_type,
                   "empty": n_empty,
                   "stream": stream_good + stream_defects_per_type * len(DEFECT_TYPES) + (2 if stream_good else 0)},
        "defect_types": list(DEFECT_TYPES),
        "roi": list(roi),
    }
    (out / "synth_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
