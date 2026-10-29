"""Images in, consistent pixels out.

* EXIF-safe loading: phone JPEGs store rotation as a tag; PIL ignores it
  unless told, which would hand the model a sideways brush.
* ROI crops: the rig frames the brush identically every time, so the brush
  head (where tufting and trimming defects live) sits in a fixed box. The
  box is stored as fractions of the frame, so it survives a change of
  camera resolution as long as the framing is unchanged.
* Deterministic splits: a file's split comes from a hash of its name, so
  re-running training never reshuffles. Once a SKU has enough goods that the
  hash alone fills the minimum holdouts (about 60), adding images never moves
  existing ones between train / calibration / test.
"""

from __future__ import annotations

import io
import json
import os
import time
import zlib
from pathlib import Path

from PIL import Image, ImageOps

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------------------------------------------- loading
def list_images(folder, recursive: bool = False) -> list[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        return []
    it = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        p
        for p in it
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def load_image(src) -> Image.Image:
    """Path, bytes or file-like -> upright RGB image, fully read into memory."""
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def atomic_replace(tmp, dst, attempts: int = 40, delay: float = 0.05) -> None:
    """os.replace with a short retry. On Windows, a reader that happens to have
    the target open (the station reading CURRENT, say) blocks the swap briefly."""
    for i in range(attempts):
        try:
            os.replace(tmp, dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(delay)


def atomic_write_bytes(path, data: bytes) -> None:
    """Write via a temp file, so a power cut never leaves a half-written image."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    tmp.write_bytes(data)
    atomic_replace(tmp, path)


def read_text_any(path) -> str:
    """Text from any editor: UTF-8 with or without BOM, or UTF-16 (PowerShell 5's default)."""
    raw = Path(path).read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw[3:].decode("utf-8")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")
    return text.replace("\x00", "")


def image_size(path) -> tuple[int, int]:
    """Upright (w, h) from the file header, without decoding the pixels."""
    with Image.open(path) as im:
        w, h = im.size
        orientation = im.getexif().get(0x0112, 1)
    return (h, w) if orientation in (5, 6, 7, 8) else (w, h)


def jpeg_bytes(img: Image.Image, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return buf.getvalue()


# --------------------------------------------------------------------- ROI
class ROIError(ValueError):
    pass


def check_roi(roi):
    """Validate an ROI given as fractions [x0, y0, x1, y1]; None means full frame."""
    if roi is None:
        return None
    try:
        vals = [float(v) for v in roi]
    except (TypeError, ValueError):
        raise ROIError(f"ROI {roi!r} must be four numbers x0,y0,x1,y1 (fractions of the frame)")
    if len(vals) != 4:
        raise ROIError("ROI needs four numbers: x0,y0,x1,y1 (fractions of the frame)")
    x0, y0, x1, y1 = vals
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ROIError(f"ROI {vals} must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    if (x1 - x0) < 0.02 or (y1 - y0) < 0.02:
        raise ROIError("ROI is too small (under 2% of the frame on a side)")
    return (x0, y0, x1, y1)


def parse_roi(text: str):
    parts = [p for p in str(text).replace(" ", "").split(",") if p]
    return check_roi(parts)


def roi_box_px(size, roi) -> tuple[int, int, int, int]:
    w, h = size
    if roi is None:
        return (0, 0, w, h)
    x0, y0, x1, y1 = check_roi(roi)
    left = min(max(int(round(x0 * w)), 0), w - 1)
    top = min(max(int(round(y0 * h)), 0), h - 1)
    right = max(min(int(round(x1 * w)), w), left + 1)
    bottom = max(min(int(round(y1 * h)), h), top + 1)
    return (left, top, right, bottom)


def crop_roi(img: Image.Image, roi) -> Image.Image:
    return img if roi is None else img.crop(roi_box_px(img.size, roi))


def read_roi_file(sku_data_dir):
    p = Path(sku_data_dir) / "roi.json"
    if not p.exists():
        return None
    return check_roi(json.loads(read_text_any(p))["roi"])


def write_roi_file(sku_data_dir, roi, frame_size=None) -> Path:
    p = Path(sku_data_dir) / "roi.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {"roi": [round(v, 4) for v in check_roi(roi)]}
    if frame_size:
        payload["reference_frame_size"] = list(frame_size)
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p


# ------------------------------------------------------------------ splits
def split_bucket(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) % 1000


def split_goods(paths, cal_frac: float = 0.15, test_frac: float = 0.15, min_holdout: int = 8) -> dict:
    """Good images -> {'train', 'cal', 'test'}, disjoint and deterministic.

    train  builds the memory bank
    cal    sets the threshold
    test   is never touched until the report: a false-reject rate
    """
    test_cut = int(round(test_frac * 1000))
    cal_cut = test_cut + int(round(cal_frac * 1000))
    parts: dict[str, list[Path]] = {"train": [], "cal": [], "test": []}
    for p in sorted((Path(x) for x in paths), key=lambda q: (split_bucket(q.name), q.name)):
        b = split_bucket(p.name)
        parts["test" if b < test_cut else "cal" if b < cal_cut else "train"].append(p)
    # small datasets: guarantee minimum holdouts by taking the train items
    # whose hash sits closest to the boundary (still deterministic)
    for part in ("test", "cal"):
        while len(parts[part]) < min_holdout and len(parts["train"]) > 1:
            parts[part].append(parts["train"].pop(0))
    return {k: sorted(v) for k, v in parts.items()}


# ----------------------------------------------------------------- defects
def defect_groups(defect_dir) -> dict[str, list[Path]]:
    """defect/<type>/*.jpg -> {type: [...]}; loose files in defect/ -> 'unlabelled'."""
    defect_dir = Path(defect_dir)
    groups: dict[str, list[Path]] = {}
    loose = list_images(defect_dir)
    if loose:
        groups["unlabelled"] = loose
    if defect_dir.is_dir():
        for sub in sorted(p for p in defect_dir.iterdir() if p.is_dir()):
            imgs = list_images(sub, recursive=True)
            if imgs:
                groups[sub.name] = imgs
    return groups


def read_annotations(sku_data_dir) -> dict:
    """Optional defect boxes: data/<sku>/defect_annotations.json

    {"<path relative to defect/>": [x0, y0, x1, y1], ...} in original-image
    pixels. Used only to check that heatmaps point at the real defect.
    """
    p = Path(sku_data_dir) / "defect_annotations.json"
    if not p.exists():
        return {}
    raw = json.loads(read_text_any(p))
    return {k.replace("\\", "/"): [float(v) for v in box] for k, box in raw.items()}


def class_folders(classes_dir, min_per_class: int = 5) -> tuple[dict, dict]:
    """classes/<label>/*.jpg -> ({label: paths} usable, {label: count} skipped)."""
    classes_dir = Path(classes_dir)
    usable, skipped = {}, {}
    if classes_dir.is_dir():
        for sub in sorted(p for p in classes_dir.iterdir() if p.is_dir()):
            imgs = list_images(sub, recursive=True)
            if len(imgs) >= min_per_class:
                usable[sub.name] = imgs
            elif imgs:
                skipped[sub.name] = len(imgs)
    return usable, skipped
