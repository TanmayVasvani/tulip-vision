"""Versioned, offline model bundles.

    models/<sku>/
        CURRENT              name of the live version (one line of text)
        20261030-051200/     immutable version folder
            meta.json        thresholds, ROI, frame and input size, training report
            memory.npy       coreset of good-brush patch features
            global.npy       one summary vector per training brush (recognition check)
            extractor.pt     backbone weights (no internet needed on site)
            classifier.npz   optional defect-type classifier
            splits.json      which images trained, calibrated and tested it
            report.txt       the training report

Training writes a new version folder, then flips CURRENT. The inspection
server notices the flip and hot-swaps the model without a restart, and
every verdict in the audit log records the version that produced it.
Roll back with scripts/set_version.py.

Two checks decide a verdict:
    recognition   does the frame show a brush the way training did at all?
                  An empty jig or a badly changed camera picture is
                  'unrecognised' — never a pass.
    patch         is any small region of the head unlike good production?
                  A misplaced brush or another handle colour fails here.
Neither check reads the SKU: a different model with the same colours and
similar tufts can look normal, so the operator picks the SKU on screen.
"""

from __future__ import annotations

import datetime as dt
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .classifier import DefectClassifier
from .core import MemoryBank
from .data import atomic_replace, crop_roi, read_text_any, roi_box_px
from .features import FeatureExtractor

SCHEMA_VERSION = 3
NO_BRUSH_FACTOR = 2.0  # this far past the recognition limit: nothing brush-like at all


class ModelError(RuntimeError):
    pass


class FrameMismatch(ValueError):
    """A frame whose size differs from the frames the model was trained on."""

    def __init__(self, got, expected):
        self.got, self.expected = tuple(got), tuple(expected)
        super().__init__(f"frame is {self.got[0]}x{self.got[1]} but the model was trained on "
                         f"{self.expected[0]}x{self.expected[1]} frames")


# ---------------------------------------------------------------- versions
def resolve_version_dir(path) -> Path:
    """A version folder, or a SKU folder whose CURRENT names one."""
    p = Path(path)
    if (p / "meta.json").exists():
        return p
    cur = p / "CURRENT"
    if cur.exists():
        name = read_text_any(cur).strip()
        if name and (p / name / "meta.json").exists():
            return p / name
        raise ModelError(f"{cur} points to a missing version {name!r} — fix it with scripts/set_version.py")
    raise ModelError(f"no trained model at {p} — run scripts/train_anomaly.py first")


def model_stamp(path):
    """Cheap change-detector for hot reload: (version dir, meta mtime)."""
    v = resolve_version_dir(path)
    return (str(v), (v / "meta.json").stat().st_mtime_ns)


def list_versions(sku_models_dir) -> list[str]:
    d = Path(sku_models_dir)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and (p / "meta.json").exists())


def new_version_dir(sku_models_dir) -> Path:
    d = Path(sku_models_dir)
    d.mkdir(parents=True, exist_ok=True)
    base = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name, i = base, 1
    while (d / name).exists():
        name, i = f"{base}-{i}", i + 1
    (d / name).mkdir()
    return d / name


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    atomic_replace(tmp, path)


def set_current(sku_models_dir, version_name: str) -> None:
    d = Path(sku_models_dir)
    if not version_name or not (d / version_name / "meta.json").exists():
        raise ModelError(f"cannot activate {version_name!r}: no meta.json in {d / version_name}")
    _atomic_write_text(d / "CURRENT", version_name + "\n")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def write_meta(version_dir, meta: dict) -> None:
    _atomic_write_text(Path(version_dir) / "meta.json", json.dumps(meta, indent=2, default=_json_default))


def _save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, np.ascontiguousarray(arr, dtype=np.float32))
    atomic_replace(tmp, path)


def save_bundle(version_dir, extractor: FeatureExtractor, bank: MemoryBank, global_bank: np.ndarray,
                meta: dict, classifier: DefectClassifier | None = None) -> None:
    """Write all artefacts; meta.json goes last so a half-written folder never loads."""
    v = Path(version_dir)
    v.mkdir(parents=True, exist_ok=True)
    _save_npy(v / "memory.npy", bank.memory)
    _save_npy(v / "global.npy", global_bank)
    extractor.save(v / "extractor.pt")
    if classifier is not None:
        classifier.save(v / "classifier.npz")
    write_meta(v, dict(meta, schema=SCHEMA_VERSION, has_classifier=classifier is not None))


def copy_anomaly_files(src_version_dir, dst_version_dir) -> None:
    for name in ("memory.npy", "global.npy", "extractor.pt", "splits.json"):
        src = Path(src_version_dir) / name
        if src.exists():
            shutil.copy2(src, Path(dst_version_dir) / name)


# --------------------------------------------------------------- geometry
def peak_point(dists, grid, image_size, roi) -> tuple[float, float]:
    """Centre of the most anomalous patch, in original-image pixels."""
    h, w = grid
    r, c = divmod(int(np.argmax(dists)), w)
    left, top, right, bottom = roi_box_px(image_size, roi)
    return (left + (c + 0.5) / w * (right - left), top + (r + 0.5) / h * (bottom - top))


def nearest_distance(vec: np.ndarray, bank: np.ndarray) -> float:
    d2 = ((bank - vec[None, :].astype(np.float32)) ** 2).sum(axis=1)
    return float(np.sqrt(max(float(d2.min()), 0.0)))


def check_frame(img: Image.Image, frame_size) -> Image.Image:
    """Refuse frames whose size differs from the training frames.

    Resizing would not make them comparable: a different resolution changes
    the sharpness and noise the model learned as normal, and a different
    shape moves the ROI. Any camera or resolution change is a new dataset.
    """
    if frame_size and tuple(img.size) != tuple(frame_size):
        raise FrameMismatch(img.size, frame_size)
    return img


# ------------------------------------------------------------------ model
class InspectionModel:
    def __init__(self, extractor: FeatureExtractor, bank: MemoryBank, global_bank: np.ndarray, meta: dict,
                 classifier: DefectClassifier | None = None, version_dir=None):
        self.extractor = extractor
        self.bank = bank
        self.global_bank = np.ascontiguousarray(global_bank, dtype=np.float32)
        self.meta = meta
        self.classifier = classifier
        self.version_dir = Path(version_dir) if version_dir else None
        self._lock = threading.Lock()

    @property
    def threshold(self) -> float:
        return float(self.meta["threshold"]["value"])

    @property
    def recognition_threshold(self) -> float:
        return float(self.meta["recognition"]["value"])

    @property
    def roi(self):
        roi = self.meta.get("roi")
        return tuple(roi) if roi else None

    @property
    def frame_size(self):
        fs = self.meta.get("frame_size")
        return tuple(fs) if fs else None

    @property
    def version(self) -> str:
        return str(self.meta.get("version") or (self.version_dir.name if self.version_dir else "unsaved"))

    @classmethod
    def load(cls, path, device: str | None = None) -> "InspectionModel":
        v = resolve_version_dir(path)
        meta = json.loads((v / "meta.json").read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA_VERSION:
            raise ModelError(f"{v}: model format {meta.get('schema')} is from another version — retrain")
        extractor = FeatureExtractor.load(v / "extractor.pt", device=device)
        bank = MemoryBank(np.load(v / "memory.npy", allow_pickle=False))
        gbank = np.load(v / "global.npy", allow_pickle=False)
        clf = DefectClassifier.load(v / "classifier.npz") if (v / "classifier.npz").exists() else None
        return cls(extractor, bank, gbank, meta, clf, v)

    def analyse(self, img: Image.Image):
        """Full frame -> (ROI image, patch features, patch distances, grid, recognition distance)."""
        img = check_frame(img, self.frame_size)
        roi_img = crop_roi(img, self.roi)
        with self._lock:
            feats, grid = next(self.extractor.extract([roi_img], batch_size=1))
            dists = self.bank.nn_distances(feats)
        return roi_img, feats, dists, grid, nearest_distance(feats.mean(axis=0), self.global_bank)

    def inspect(self, img: Image.Image) -> dict:
        t0 = time.perf_counter()
        roi_img, feats, dists, grid, gdist = self.analyse(img)
        score = float(dists.max())
        thr, gthr = self.threshold, self.recognition_threshold
        reason = None
        if gdist >= gthr:
            verdict = "unrecognised"  # empty jig or a changed camera picture: never a pass
            nb = self.meta["recognition"].get("no_brush_at") or NO_BRUSH_FACTOR * gthr
            reason = "no_brush" if gdist >= nb else "differs"  # nothing brush-like vs a changed picture
        else:
            verdict = "reject" if score >= thr else "pass"
        guess = None
        if self.classifier is not None and verdict == "reject":
            guess = self.classifier.predict(feats, dists)
        return {
            "score": score,
            "threshold": thr,
            "ratio": score / thr if thr > 0 else float("inf"),
            "recognition_ratio": gdist / gthr if gthr > 0 else float("inf"),
            "verdict": verdict,
            "unrecognised_reason": reason,
            "patch_dists": dists,
            "grid": grid,
            "roi_image": roi_img,
            "defect_guess": guess,
            "model_version": self.version,
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }
