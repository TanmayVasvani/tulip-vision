"""Training and evaluation."""

Good images are split three ways by a hash of their file name:

    train   builds the memory bank
    cal     sets the decision thresholds
    test    held back until the report, so the false-reject rate is measured

With the default threshold policy ("goods") defect images are used for
nothing except evaluation, so the catch rate is out-of-sample too. Every
number in the report is something the model has not seen while training.

Optional evaluation folders (read only, never trained on):
    data/<sku>/defect/<type>/   real defects
    data/<sku>/empty/           photos of the empty jig: the report proves
                                the station never passes an empty jig
"""

from __future__ import annotations

import datetime as dt
import io
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from . import __version__
from .classifier import descriptor, train_defect_classifier
from .core import MemoryBank, auroc, score_summary, threshold_from_gap, threshold_from_goods
from .data import (
    check_roi,
    class_folders,
    crop_roi,
    defect_groups,
    image_size,
    list_images,
    load_image,
    read_annotations,
    read_roi_file,
    roi_box_px,
    split_goods,
)
from .features import FeatureExtractor, WeightsError
from .model import (
    FrameMismatch,
    InspectionModel,
    check_frame,
    copy_anomaly_files,
    nearest_distance,
    new_version_dir,
    peak_point,
    resolve_version_dir,
    save_bundle,
    set_current,
    write_meta,
)


class TrainError(RuntimeError):
    pass


@dataclass
class TrainParams:
    backbone: str = "resnet18"
    weights: str = "imagenet"
    input_long_side: int = 448
    input_size: tuple | None = None   # (w, h); default follows the ROI's aspect ratio
    roi: tuple | None = None          # None -> data/<sku>/roi.json if present, else full frame
    use_roi_file: bool = True
    coreset_size: int = 4000
    pool_cap: int = 120_000
    quantile: float = 0.995
    margin: float = 1.08
    policy: str = "goods"             # goods | gap
    recog_quantile: float = 0.995
    recog_margin: float = 1.25
    cal_frac: float = 0.15
    test_frac: float = 0.15
    min_holdout: int = 8
    min_good: int = 40
    include_feedback: bool = False
    classifier: str = "auto"          # auto | off
    min_per_class: int = 5
    batch_size: int = 8
    seed: int = 0
    device: str | None = None


def choose_input_size(roi_w: int, roi_h: int, long_side: int = 448, multiple: int = 16) -> tuple[int, int]:
    """Model input (w, h) that keeps the ROI's aspect ratio (no squashing)."""
    aspect = roi_w / float(roi_h)
    if aspect >= 1.0:
        w, h = long_side, long_side / aspect
    else:
        w, h = long_side * aspect, long_side
    snap = lambda v: max(multiple * 4, int(round(v / multiple)) * multiple)  # noqa: E731
    return snap(w), snap(h)


def _iter_feats(extractor: FeatureExtractor, paths, roi, batch_size: int, frame_size=None, bad=None):
    """Stream (path, features, grid, original size), holding few images at a time.

    Unreadable files, and frames whose shape differs from the training
    frames, are skipped and listed in ``bad`` as (path, reason).
    """
    paths = list(paths)
    for i in range(0, len(paths), batch_size):
        ok, sizes, crops = [], [], []
        for p in paths[i : i + batch_size]:
            try:
                img = load_image(p)
                size = img.size
                crops.append(crop_roi(check_frame(img, frame_size), roi))
            except FrameMismatch as exc:
                if bad is not None:
                    bad.append((Path(p), f"{exc.got[0]}x{exc.got[1]} frame, model uses {exc.expected[0]}x{exc.expected[1]}"))
                continue
            except (OSError, ValueError, SyntaxError) as exc:  # truncated or not an image
                if bad is not None:
                    bad.append((Path(p), f"unreadable: {exc}"))
                continue
            ok.append(p)
            sizes.append(size)
        if not ok:
            continue
        for p, size, (f, grid) in zip(ok, sizes, extractor.extract(crops, batch_size)):
            yield p, f, grid, size


def _score(extractor, bank, gbank, paths, roi, batch_size, frame_size, bad, keep=False):
    out = []
    for p, f, grid, size in _iter_feats(extractor, paths, roi, batch_size, frame_size, bad):
        d = bank.nn_distances(f)
        rec = {"path": Path(p), "score": float(d.max()), "g": nearest_distance(f.mean(axis=0), gbank),
               "grid": grid, "size": size}
        if keep:
            rec["dists"] = d
        out.append(rec)
    return out


def _class_descriptors(extractor, bank, roi, usable: dict, batch_size: int, frame_size=None, bad=None) -> dict:
    out = {}
    for label, paths in usable.items():
        descs = [descriptor(f, bank.nn_distances(f))
                 for _, f, _, _ in _iter_feats(extractor, paths, roi, batch_size, frame_size, bad)]
        if descs:
            out[label] = descs
    return out


def _quality_variants(img: Image.Image, rng) -> list[Image.Image]:
    """Picture-quality drift a working rig can see: compression, sensor noise,
    focus, sharpening, exposure. A good brush must stay recognised under all."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=65)
    arr = np.asarray(img, dtype=np.float32)
    noisy = np.clip(arr + rng.normal(0.0, 3.5, arr.shape), 0, 255).astype(np.uint8)
    return [
        load_image(buf.getvalue()),
        Image.fromarray(noisy),
        img.filter(ImageFilter.GaussianBlur(radius=1.3)),
        img.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=0)),
        ImageEnhance.Brightness(img).enhance(0.85),
        ImageEnhance.Brightness(img).enhance(1.15),
    ]


def _variant_distances(extractor, gbank, paths, roi, frame_size, batch_size, seed=0) -> list[float]:
    """Recognition distances of quality-varied copies of the given good images."""
    rng = np.random.default_rng(seed)
    out, crops = [], []

    def flush():
        for f, _ in extractor.extract(crops, batch_size):
            out.append(nearest_distance(f.mean(axis=0), gbank))
        crops.clear()

    for p in paths:
        try:
            img = check_frame(load_image(p), frame_size)
        except (FrameMismatch, OSError, ValueError, SyntaxError):
            continue
        crops.extend(crop_roi(v, roi) for v in _quality_variants(img, rng))
        if len(crops) >= 4 * batch_size:
            flush()
    if crops:
        flush()
    return out


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _verdict(rec: dict, thr: float, gthr: float) -> str:
    if rec["g"] >= gthr:
        return "unrecognised"
    return "reject" if rec["score"] >= thr else "pass"


def _frame_size(paths, bad) -> tuple[int, int]:
    sizes = Counter()
    for p in paths:
        try:
            sizes[image_size(p)] += 1
        except (OSError, ValueError, SyntaxError) as exc:
            bad.append((Path(p), f"unreadable: {exc}"))
    if not sizes:
        raise TrainError("none of the good images could be read")
    return sizes.most_common(1)[0][0]


# --------------------------------------------------------------- anomaly
def train_anomaly(data_dir, sku_models_dir, params: TrainParams | None = None, log=print):
    """Train, evaluate, save a new model version, and make it CURRENT.

    Returns (version_dir, report).
    """
    p = params or TrainParams()
    data_dir = Path(data_dir)
    sku = data_dir.name
    t_start = time.perf_counter()
    bad: list[tuple[Path, str]] = []

    goods = list_images(data_dir / "good")
    if p.include_feedback:
        goods += list_images(data_dir / "feedback" / "good")
    if len(goods) < p.min_good:
        raise TrainError(
            f"only {len(goods)} good images in {data_dir / 'good'} — need at least {p.min_good} "
            "(300+ distinct brushes for a trustworthy threshold)"
        )
    roi = p.roi if p.roi is not None else (read_roi_file(data_dir) if p.use_roi_file else None)
    roi = check_roi(roi)
    frame_size = _frame_size(goods, bad)
    split = split_goods(goods, p.cal_frac, p.test_frac, p.min_holdout)
    groups = defect_groups(data_dir / "defect")
    defect_items = [(t, path) for t, paths in groups.items() for path in paths]
    empties = list_images(data_dir / "empty", recursive=True)
    log(f"[train] {sku}: good {len(split['train'])} train / {len(split['cal'])} cal / "
        f"{len(split['test'])} test | defects {len(defect_items)} "
        f"({', '.join(f'{t}:{len(v)}' for t, v in groups.items()) or 'none'}) | empty-jig frames {len(empties)}")

    box = roi_box_px(frame_size, roi)
    input_size = tuple(p.input_size) if p.input_size else choose_input_size(
        box[2] - box[0], box[3] - box[1], p.input_long_side
    )
    try:
        extractor = FeatureExtractor(p.backbone, input_size, p.weights, p.device)
    except (WeightsError, ValueError, FileNotFoundError) as exc:
        raise TrainError(str(exc)) from exc
    log(f"[train] backbone {p.backbone} ({extractor.weights_desc}) | frames {frame_size[0]}x{frame_size[1]} | "
        f"ROI {roi or 'full frame'} -> input {input_size[0]}x{input_size[1]} on {extractor.device}")

    # ---- memory bank (patches) + recognition bank (one vector per brush)
    rng = np.random.default_rng(p.seed)
    n_train = len(split["train"])
    pool, gvecs, grid, per_image = [], [], None, None
    t0 = time.perf_counter()
    for i, (_, f, grid, _) in enumerate(
        _iter_feats(extractor, split["train"], roi, p.batch_size, frame_size, bad), 1
    ):
        gvecs.append(f.mean(axis=0))
        if per_image is None:
            per_image = int(min(f.shape[0], max(64, p.pool_cap // n_train)))
        take = rng.choice(f.shape[0], per_image, replace=False) if per_image < f.shape[0] else slice(None)
        pool.append(f[take])
        if i % 50 == 0:
            log(f"[train] features {i}/{n_train}")
    if len(gvecs) < max(10, p.min_good // 2):
        raise TrainError(f"only {len(gvecs)} training images were usable — see the unreadable files")
    pool = np.concatenate(pool, axis=0)
    gbank = np.stack(gvecs).astype(np.float32)
    t_feat = time.perf_counter() - t0
    t0 = time.perf_counter()
    bank = MemoryBank.fit(pool, coreset_size=p.coreset_size, seed=p.seed)
    t_core = time.perf_counter() - t0
    log(f"[train] patch pool {pool.shape[0]:,} x {pool.shape[1]} -> coreset {bank.size:,} "
        f"({t_feat:.0f}s features, {t_core:.0f}s coreset)")

    # ---- score held-out goods, defects, empty-jig frames
    args = (roi, p.batch_size, frame_size, bad)
    cal = _score(extractor, bank, gbank, split["cal"], *args)
    test = _score(extractor, bank, gbank, split["test"], *args)
    dtypes = {path: t for t, path in defect_items}
    dfx = _score(extractor, bank, gbank, [path for _, path in defect_items], *args, keep=True)
    for rec in dfx:
        rec["type"] = dtypes[rec["path"]]
    emp = _score(extractor, bank, gbank, empties, *args)
    if len(cal) < 2:
        raise TrainError("too few readable calibration images")
    cal_s, test_s, def_s = ([r["score"] for r in rs] for rs in (cal, test, dfx))

    # ---- thresholds
    warnings: list[str] = []
    thr = threshold_from_goods(cal_s, p.quantile, p.margin)
    policy_used = "goods"
    note = f"{p.quantile:g} quantile of calibration goods x {p.margin:g}; defects unused (catch rate is out-of-sample)"
    if p.policy == "gap":
        gap = threshold_from_gap(cal_s, def_s)
        if gap is None:
            warnings.append("gap policy requested but defects overlap calibration goods — kept the goods policy")
        else:
            thr, policy_used = gap, "gap"
            note = "midpoint between worst calibration good and mildest defect — catch rate below is IN-SAMPLE"
    # recognition limit: clean calibration goods AND quality-varied copies of them must
    # stay recognised; the empty jig (if photographed) must stay far outside
    log("[train] calibrating the recognition check on quality-varied goods")
    g_cal = [r["g"] for r in cal]
    g_cal_var = _variant_distances(extractor, gbank, [r["path"] for r in cal], roi, frame_size, p.batch_size, p.seed)
    gthr = threshold_from_goods(g_cal + g_cal_var, p.recog_quantile, p.recog_margin)
    no_brush_at = None
    if emp:
        g_empty = min(r["g"] for r in emp)
        cap = g_empty / p.recog_margin  # the empty jig must stay clearly outside the limit
        if gthr > cap:
            gthr = max(cap, math.sqrt(max(g_cal) * g_empty))
            warnings.append("the recognition limit was tightened to keep refusing the empty jig, so focus, light "
                            "or JPEG changes may be refused as 'picture differs' — keep the rig stable")
        no_brush_at = max(gthr, 0.8 * g_empty)  # this close to the empty jig reads as "no brush"
    g_test_var = _variant_distances(extractor, gbank, [r["path"] for r in test], roi, frame_size, p.batch_size,
                                    p.seed + 1)

    # ---- metrics on combined verdicts
    for rs in (test, dfx, emp):
        for r in rs:
            r["verdict"] = _verdict(r, thr, gthr)
    false_alarms = [r for r in test if r["verdict"] != "pass"]
    caught = [r for r in dfx if r["verdict"] != "pass"]
    metrics = {
        "auroc": auroc(test_s, def_s),
        "good": len(test),
        "false_rejects": len(false_alarms),
        "false_reject_rate": (len(false_alarms) / len(test)) if test else None,
        "defects": len(dfx),
        "caught": len(caught),
        "catch_rate": (len(caught) / len(dfx)) if dfx else None,
    }
    per_type = {}
    for t in groups:
        rs = [r for r in dfx if r["type"] == t]
        if not rs:
            continue
        ss = [r["score"] for r in rs]
        n_caught = sum(r["verdict"] != "pass" for r in rs)
        per_type[t] = {"n": len(rs), "caught": n_caught, "catch_rate": n_caught / len(rs),
                       "min_ratio": min(ss) / thr, "median_ratio": float(np.median(ss)) / thr}
    missed = [{"type": r["type"], "file": r["path"].name, "ratio": r["score"] / thr}
              for r in sorted(dfx, key=lambda r: r["score"]) if r["verdict"] == "pass"]
    false_rejects = [{"file": r["path"].name, "ratio": r["score"] / thr, "verdict": r["verdict"]}
                     for r in false_alarms]
    recognition = {
        "threshold": gthr,
        "test_goods_unrecognised": sum(r["verdict"] == "unrecognised" for r in test),
        "max_good_ratio": max((r["g"] for r in test), default=0.0) / gthr,
        "quality_variants": len(g_test_var),
        "quality_variants_refused": sum(g >= gthr for g in g_test_var),
        "defects_unrecognised": sum(r["verdict"] == "unrecognised" for r in dfx),
        "empty_frames": len(emp),
        "empty_unrecognised": sum(r["verdict"] == "unrecognised" for r in emp),
        "empty_passed": sum(r["verdict"] == "pass" for r in emp),
        "min_empty_ratio": min((r["g"] for r in emp), default=0.0) / gthr if emp else None,
    }

    localization = None
    ann = read_annotations(data_dir)
    if ann:
        hits = n = 0
        for r in dfx:
            key = r["path"].relative_to(data_dir / "defect").as_posix()
            if key not in ann:
                continue
            n += 1
            x, y = peak_point(r["dists"], r["grid"], r["size"], roi)
            bx = roi_box_px(r["size"], roi)
            tol = 0.75 * (bx[2] - bx[0]) / r["grid"][1]
            x0, y0, x1, y1 = ann[key]
            hits += (x0 - tol) <= x <= (x1 + tol) and (y0 - tol) <= y <= (y1 + tol)
        if n:
            localization = {"annotated": n, "hits": int(hits), "hit_rate": hits / n}

    meta_core = {"threshold": {"value": thr}, "recognition": {"value": gthr}, "roi": list(roi) if roi else None,
                 "frame_size": list(frame_size), "version": "eval"}
    model = InspectionModel(extractor, bank, gbank, meta_core)
    lat = []
    for r in test[:12]:
        img = load_image(r["path"])
        t0 = time.perf_counter()
        model.inspect(img)
        lat.append((time.perf_counter() - t0) * 1000.0)

    # ---- warnings and tips
    tips: list[str] = []
    if bad:
        warnings.append(f"skipped {len(bad)} image(s): " + "; ".join(f"{b.name} ({why})" for b, why in bad[:5])
                        + (" ..." if len(bad) > 5 else ""))
    if len(goods) < 300:
        warnings.append(f"{len(goods)} good images: fine for a trial, 300+ distinct brushes recommended")
    if not def_s:
        warnings.append("no defect images: catch rate and AUROC are unmeasured — shoot the reject bin into defect/<type>/")
    if metrics["auroc"] is not None and metrics["auroc"] < 0.9:
        warnings.append(f"AUROC {metrics['auroc']:.3f} is weak — check rig consistency and whether the camera "
                        "can see the defects")
    if metrics["false_reject_rate"] is not None and metrics["false_reject_rate"] > 0.02:
        warnings.append("held-out false-reject rate above 2% — operators will lose trust; review the rig and the good set")
    if missed:
        kinds = sorted({m["type"] for m in missed})
        warnings.append(f"{len(missed)} defect image(s) passed ({', '.join(kinds)}) — "
                        "can the camera actually see these at this angle and lighting?")
    if recognition["quality_variants_refused"]:
        warnings.append(f"{recognition['quality_variants_refused']}/{recognition['quality_variants']} quality-varied "
                        "held-out goods (JPEG, noise, blur, sharpening, exposure) were 'not recognised'")
    if recognition["test_goods_unrecognised"]:
        warnings.append(f"{recognition['test_goods_unrecognised']} held-out good brush(es) were 'not recognised' — "
                        "framing or lighting varies more than the training set covers")
    if recognition["empty_passed"]:
        warnings.append(f"{recognition['empty_passed']} empty-jig frame(s) PASSED — do not gate production with "
                        "this model; check the ROI and lighting")
    if not emp:
        tips.append("put 5-10 photos of the empty jig in data/<sku>/empty/ — the report will then prove the "
                    "station never passes an empty jig")

    # ---- Phase-2 classifier (optional)
    clf = clf_report = None
    if p.classifier != "off":
        usable, skipped = class_folders(data_dir / "classes", p.min_per_class)
        if len(usable) >= 2:
            log(f"[train] defect-type classifier on {', '.join(f'{k}:{len(v)}' for k, v in usable.items())}")
            descs = _class_descriptors(extractor, bank, roi, usable, p.batch_size, frame_size, bad)
            if len(descs) >= 2:
                clf, clf_report = train_defect_classifier(descs, seed=p.seed)
                if skipped:
                    clf_report["skipped_small_classes"] = skipped
        elif usable or skipped:
            clf_report = {"skipped": f"need >= 2 classes with >= {p.min_per_class} images each in classes/"}

    # ---- save new version and activate it
    vdir = new_version_dir(sku_models_dir)
    report = {
        "sku": sku,
        "version": vdir.name,
        "created": _now(),
        "data": {
            "good_total": len(goods),
            "train_good": len(split["train"]),
            "cal_good": len(split["cal"]),
            "test_good": len(split["test"]),
            "defects": len(def_s),
            "defect_types": {t: len(v) for t, v in groups.items()},
            "empty_frames": len(emp),
            "frame_size": list(frame_size),
            "skipped": [{"file": b.name, "reason": why} for b, why in bad],
        },
        "threshold": {"value": thr, "policy": policy_used, "quantile": p.quantile, "margin": p.margin, "note": note},
        "recognition": recognition,
        "metrics": {
            **metrics,
            "per_type": per_type,
            "missed": missed,
            "false_rejected_test_goods": false_rejects,
            "localization": localization,
            "scores": {
                "cal_good": score_summary(cal_s),
                "test_good": score_summary(test_s),
                "defect": score_summary(def_s),
            },
            "latency_ms_median": float(np.median(lat)) if lat else None,
        },
        "classifier": clf_report,
        "timings_s": {"features": round(t_feat, 1), "coreset": round(t_core, 1),
                      "total": round(time.perf_counter() - t_start, 1)},
        "warnings": warnings,
        "tips": tips,
    }
    meta = {
        "version": vdir.name,
        "sku": sku,
        "created": report["created"],
        "tulip_vision": __version__,
        "backbone": p.backbone,
        "weights": extractor.weights_desc,
        "frame_size": list(frame_size),
        "input_size": list(input_size),
        "roi": list(roi) if roi else None,
        "grid": list(grid),
        "feature_dim": bank.dim,
        "coreset_size": bank.size,
        "pool_size": int(pool.shape[0]),
        "threshold": report["threshold"],
        "recognition": {"value": gthr, "no_brush_at": no_brush_at, "quantile": p.recog_quantile,
                        "margin": p.recog_margin,
                        "note": "one summary vector per brush vs training brushes, calibrated on calibration goods "
                                "and quality-varied copies; empty jig / changed camera picture -> "
                                "unrecognised, never a pass"},
        "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(p).items()},
        "report": report,
    }
    (vdir / "splits.json").write_text(json.dumps({
        k: [str(x.relative_to(data_dir)) if x.is_relative_to(data_dir) else str(x) for x in v]
        for k, v in split.items()
    }, indent=1), encoding="utf-8")
    (vdir / "report.txt").write_text(format_report(report), encoding="utf-8")
    save_bundle(vdir, extractor, bank, gbank, meta, clf)
    set_current(sku_models_dir, vdir.name)
    return vdir, report


# ------------------------------------------------------------ classifier
def retrain_classifier(data_dir, sku_models_dir, min_per_class: int = 5, seed: int = 0,
                       device: str | None = None, batch_size: int = 8, log=print):
    """Fit the defect-type classifier on the current anomaly model; new version."""
    data_dir = Path(data_dir)
    cur = resolve_version_dir(sku_models_dir)
    model = InspectionModel.load(cur, device=device)
    usable, skipped = class_folders(data_dir / "classes", min_per_class)
    if len(usable) < 2:
        raise TrainError(
            f"need >= 2 classes with >= {min_per_class} images in {data_dir / 'classes'} "
            f"(found {', '.join(f'{k}:{len(v)}' for k, v in usable.items()) or 'none'}"
            f"{'; too small: ' + str(skipped) if skipped else ''})"
        )
    log(f"[clf] classes: {', '.join(f'{k}:{len(v)}' for k, v in usable.items())} on model {cur.name}")
    bad: list = []
    descs = _class_descriptors(model.extractor, model.bank, model.roi, usable, batch_size, model.frame_size, bad)
    if len(descs) < 2:
        raise TrainError("fewer than two classes had readable images")
    clf, rep = train_defect_classifier(descs, seed=seed)
    if skipped:
        rep["skipped_small_classes"] = skipped
    if bad:
        rep["skipped_images"] = [f"{b.name} ({why})" for b, why in bad]
    vdir = new_version_dir(sku_models_dir)
    copy_anomaly_files(cur, vdir)
    clf.save(vdir / "classifier.npz")
    meta = dict(model.meta)
    report = dict(meta.get("report") or {}, classifier=rep, version=vdir.name, created=_now())
    meta.update(version=vdir.name, parent_version=cur.name, created=report["created"], has_classifier=True,
                report=report)
    (vdir / "report.txt").write_text(format_report(report), encoding="utf-8")
    write_meta(vdir, meta)
    set_current(sku_models_dir, vdir.name)
    return vdir, rep


# ---------------------------------------------------------------- report
def _pct(x) -> str:
    return "n/a" if x is None else f"{100.0 * x:.1f}%"


def format_report(r: dict) -> str:
    m, d, t, g = r["metrics"], r["data"], r["threshold"], r.get("recognition") or {}
    lines = [
        "=" * 66,
        f" TULIP VISION — training report   {r['sku']}  v{r['version']}",
        "=" * 66,
        f" data        good {d['good_total']} (train {d['train_good']} / cal {d['cal_good']} / "
        f"test {d['test_good']})   defects {d['defects']}   empty jig {d.get('empty_frames', 0)}",
        f" threshold   {t['value']:.4f}   [{t['policy']}] {t['note']}",
        "",
        " HELD-OUT RESULTS (images the model never trained on)",
        f"   AUROC                  {('%.3f' % m['auroc']) if m['auroc'] is not None else 'n/a (no defects)'}",
        f"   false rejects (goods)  {m['false_rejects']}/{m['good']}  = {_pct(m['false_reject_rate'])}",
        f"   caught (defects)       {m['caught']}/{m['defects']}  = {_pct(m['catch_rate'])}",
    ]
    for typ, s in m["per_type"].items():
        lines.append(
            f"     {typ:<18} {s['caught']}/{s['n']}  weakest {s['min_ratio']:.2f}x limit, "
            f"median {s['median_ratio']:.2f}x"
        )
    if g:
        emp = (f"empty jig {g['empty_unrecognised']}/{g['empty_frames']} flagged"
               + (f" (weakest {g['min_empty_ratio']:.1f}x limit)" if g.get("min_empty_ratio") else "")
               if g.get("empty_frames") else "no empty-jig photos")
        lines.append(f"   not recognised         goods {g['test_goods_unrecognised']}/{m['good']} "
                     f"(worst {g['max_good_ratio']:.2f}x limit) · {emp}")
        if g.get("quality_variants"):
            lines.append(f"                          quality-varied goods refused "
                         f"{g['quality_variants_refused']}/{g['quality_variants']} (JPEG, noise, blur, exposure)")
    if m.get("localization"):
        loc = m["localization"]
        lines.append(f"   heatmap on defect      {loc['hits']}/{loc['annotated']}  = {_pct(loc['hit_rate'])}")
    if m.get("latency_ms_median") is not None:
        lines.append(f"   model time (median)    {m['latency_ms_median']:.0f} ms per brush on this machine")
    for miss in m["missed"][:10]:
        lines.append(f"   MISSED  {miss['type']:<16} {miss['file']}  ({miss['ratio']:.2f}x limit)")
    for fr in m["false_rejected_test_goods"][:10]:
        lines.append(f"   FALSE ALARM  {fr['file']}  ({fr['verdict']}, {fr['ratio']:.2f}x limit)")
    c = r.get("classifier")
    if c and c.get("cv"):
        lines += ["", f" DEFECT-TYPE CLASSIFIER  {', '.join(f'{k}:{v}' for k, v in c['counts'].items())}",
                  f"   cross-validated accuracy {_pct(c['cv']['accuracy'])} ({c['cv']['note']})"]
        for k, v in c["cv"]["per_class_recall"].items():
            lines.append(f"     {k:<18} recall {_pct(v)}")
    elif c and c.get("skipped"):
        lines += ["", f" defect-type classifier skipped: {c['skipped']}"]
    if r["warnings"]:
        lines += ["", " WARNINGS"] + [f"   ! {w}" for w in r["warnings"]]
    if r.get("tips"):
        lines += ["", " TIPS"] + [f"   - {w}" for w in r["tips"]]
    lines += ["", f" time {r['timings_s']['total']:.0f}s", "=" * 66]
    return "\n".join(lines)
