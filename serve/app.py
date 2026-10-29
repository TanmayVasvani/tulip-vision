"""Tulip Vision inspection server — Flask + Waitress, the same stack as Tulip Ops.

    python serve/app.py                                          # camera from config.yaml
    python serve/app.py --camera folder:data/demo_synthetic/stream  # demo, no hardware
    python serve/app.py --camera none                            # photo upload only

Operator station: http://<inspection-pc>:8600 from any phone, tablet or PC on
the LAN. The camera is attached to the inspection PC and the *server* grabs
the frame, so training and inspection images come through identical code.

API (every POST needs an X-Station-Pin header: the station PIN when
server.pin is set in config.yaml, any value otherwise)
    GET  /api/status                  SKUs, live model versions, camera state
    GET  /api/camera/frame.jpg        latest camera frame (live preview)
    POST /api/inspect                 form {sku, [image]} -> verdict, score, heatmap, defect guess
    POST /api/feedback                json {id, label: good|defect, [defect_type]}
    POST /api/capture                 form {sku, label: good|defect|empty, [defect_type], [image]} -> dataset
    GET  /api/capture/counts?sku=     dataset image counts
    GET  /api/stats?sku=              today's inspections, rejects, operator agreement
    GET  /api/recent?sku=&limit=      most recent inspections
    GET  /api/inspections/<id>/image  stored evidence image

Every inspection is written to SQLite with its image and the model version
that judged it. Operator confirmations are copied into the dataset folders
the training scripts read, so the station builds its own training data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hmac
import json
import logging
import re
import shutil
import socket
import sqlite3
import sys
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qc import __version__  # noqa: E402
from qc.camera import CameraError, camera_configured, open_camera  # noqa: E402
from qc.config import (  # noqa: E402
    ConfigError,
    check_sku,
    data_dir,
    defect_label,
    load_config,
    model_dir,
    serve_data_dir,
    sku_cfg,
)
from qc.data import atomic_write_bytes, jpeg_bytes, list_images, load_image  # noqa: E402
from qc.model import FrameMismatch, InspectionModel, ModelError, model_stamp  # noqa: E402
from qc.vis import anomaly_overlay, to_b64_jpeg  # noqa: E402

log = logging.getLogger("tulip.serve")
ID_RE = re.compile(r"^[0-9a-f]{12}$")
_DEFAULT = object()
JUDGED = ("pass", "reject")

SCHEMA = [
    ("id", "TEXT PRIMARY KEY"),
    ("ts", "TEXT NOT NULL"),
    ("sku", "TEXT NOT NULL"),
    ("model_version", "TEXT"),
    ("source", "TEXT"),
    ("score", "REAL NOT NULL"),
    ("threshold", "REAL NOT NULL"),
    ("verdict", "TEXT NOT NULL"),
    ("defect_guess", "TEXT"),
    ("defect_guess_conf", "REAL"),
    ("operator_label", "TEXT"),
    ("operator_defect_type", "TEXT"),
    ("labelled_at", "TEXT"),
    ("image_path", "TEXT NOT NULL"),
    ("latency_ms", "REAL"),
]


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


# ------------------------------------------------------------------ storage
class AuditLog:
    """SQLite audit trail. Migrates a v1 table in place by adding columns."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.tx() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS inspections (" + ", ".join(f"{n} {t}" for n, t in SCHEMA) + ")")
            have = {r[1] for r in c.execute("PRAGMA table_info(inspections)")}
            for name, typ in SCHEMA:
                if name not in have:
                    c.execute(f"ALTER TABLE inspections ADD COLUMN {name} {typ.replace(' NOT NULL', '')}")
            c.execute("CREATE INDEX IF NOT EXISTS idx_insp_ts ON inspections(ts)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_insp_sku_ts ON inspections(sku, ts)")

    @contextmanager
    def tx(self):
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()


class ModelRegistry:
    """Loads each SKU's CURRENT model once and reloads when training flips it.

    A new version that fails to load (a half-copied bundle, a version mix-up)
    never takes the station down: the model already in memory keeps serving,
    the error is shown in /api/status, and the load is retried every 30 s, so
    finishing the copy is enough — no restart needed.
    """

    RETRY_S = 30.0

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._cache: dict[str, tuple] = {}
        self._failed: dict[str, tuple] = {}
        self._errors: dict[str, str] = {}

    def error(self, sku: str) -> str | None:
        return self._errors.get(sku)

    def get(self, sku: str) -> InspectionModel:
        path = model_dir(self.cfg, sku)
        with self._lock:
            hit = self._cache.get(sku)
            try:
                stamp = model_stamp(path)
            except (ModelError, OSError) as exc:
                if hit:
                    self._errors[sku] = f"{exc} — still serving {hit[1].version}"
                    return hit[1]
                raise ModelError(str(exc))
            if hit and hit[0] == stamp:
                self._errors.pop(sku, None)
                return hit[1]
            failed = self._failed.get(sku)
            if failed and failed[0] == stamp and time.monotonic() - failed[2] < self.RETRY_S:
                if hit:
                    self._errors[sku] = f"{failed[1]} — still serving {hit[1].version}"
                    return hit[1]
                raise ModelError(failed[1])
            try:
                model = InspectionModel.load(path)
            except Exception as exc:
                msg = f"version {Path(stamp[0]).name} failed to load: {exc}"
                log.exception("model load failed for %s", sku)
                self._failed[sku] = (stamp, msg, time.monotonic())
                if hit:
                    self._errors[sku] = f"{msg} — still serving {hit[1].version}"
                    return hit[1]
                raise ModelError(msg)
            self._cache[sku] = (model_stamp(model.version_dir), model)
            self._failed.pop(sku, None)
            self._errors.pop(sku, None)
            log.info("model %s %s loaded (limit %.4f%s)", sku, model.version, model.threshold,
                     ", + defect types" if model.classifier else "")
            return model


class CameraManager:
    """Opens the configured camera and keeps retrying in the background if it
    is not ready yet (the PC often boots before the phone's webcam app)."""

    def __init__(self, spec, cam_cfg: dict, root: Path):
        self.spec, self.cam_cfg, self.root = spec, cam_cfg, root
        self.configured = camera_configured(spec)
        self.camera = None
        self.error: str | None = None
        self._stop = threading.Event()
        if self.configured and not self._try_open():
            threading.Thread(target=self._retry, name="camera-retry", daemon=True).start()

    def _try_open(self) -> bool:
        try:
            cam = open_camera(self.spec, self.cam_cfg, self.root)
        except CameraError as exc:
            self.error = str(exc)
            log.warning("camera not available (retrying every 5 s): %s", exc)
            return False
        except ImportError:
            self.error = "OpenCV is not installed (pip install opencv-python)"
            log.warning(self.error)
            self._stop.set()
            return False
        self.camera, self.error = cam, None
        log.info("camera ready: %s", cam.label)
        return True

    def _retry(self) -> None:
        while not self._stop.wait(5.0):
            if self._try_open():
                return

    def status(self) -> dict:
        if self.camera is not None:
            return {**self.camera.status(), "configured": True, "error": None}
        return {"configured": self.configured, "ok": False, "error": self.error,
                "retrying": self.configured and not self._stop.is_set()}

    def close(self) -> None:
        self._stop.set()
        if self.camera is not None:
            self.camera.close()


class State:
    def __init__(self, cfg: dict, cams: CameraManager):
        self.cfg = cfg
        self.cams = cams
        self.registry = ModelRegistry(cfg)
        self.serve_dir = serve_data_dir(cfg)
        self.audit = AuditLog(self.serve_dir / "inspections.db")
        self.quality = int(cfg["camera"].get("jpeg_quality", 92))
        self.feedback_lock = threading.Lock()
        self._cleanup_day = None
        self._cleanup_lock = threading.Lock()

    @property
    def camera(self):
        return self.cams.camera

    def maybe_cleanup(self) -> None:
        """Once a day, delete evidence folders older than storage.keep_days."""
        today = dt.date.today()
        with self._cleanup_lock:
            if self._cleanup_day == today:
                return
            self._cleanup_day = today
        keep = int(self.cfg.get("storage", {}).get("keep_days", 60))
        threading.Thread(target=_cleanup, args=(self.serve_dir / "inspections", keep), daemon=True).start()


def _cleanup(root: Path, keep_days: int) -> None:
    if not root.is_dir() or keep_days <= 0:
        return
    cutoff = (dt.date.today() - dt.timedelta(days=keep_days)).strftime("%Y%m%d")
    for sku_dir in root.iterdir():
        if not sku_dir.is_dir():
            continue
        for day in sku_dir.iterdir():
            if day.is_dir() and day.name.isdigit() and len(day.name) == 8 and day.name < cutoff:
                shutil.rmtree(day, ignore_errors=True)
                log.info("evidence cleanup: removed %s", day)


def _feedback_targets(ddir: Path, label, dtype, verdict) -> list[Path]:
    """Where an operator label files the image (and where an earlier label filed it)."""
    if label == "good":
        targets = [ddir / "feedback" / "good"]
        if verdict != "pass":
            targets.append(ddir / "classes" / "good")  # false alarm: teaches the classifier
        return targets
    if label == "defect":
        targets = [ddir / "defect" / (dtype or "unspecified")]
        if dtype and dtype != "other":
            targets.append(ddir / "classes" / dtype)
        return targets
    return []


# ---------------------------------------------------------------- the app
def create_app(config_path=None, camera=_DEFAULT) -> Flask:
    cfg = load_config(config_path)
    spec = cfg["camera"].get("source") if camera is _DEFAULT else camera
    state = State(cfg, CameraManager(spec, cfg["camera"], Path(cfg["_root"])))
    pin = str((cfg.get("server") or {}).get("pin") or "")

    app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
    app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
    app.json.ensure_ascii = False  # Gujarati labels stay readable in API responses
    app.extensions["tulip"] = state

    # -------------------------------------------------------------- helpers
    def sku_param(value) -> str:
        if not value or value not in (cfg.get("skus") or {}):
            raise ApiError(400, "unknown_sku", f"unknown SKU {value!r} — add it under skus: in config.yaml")
        return check_sku(value)

    def defect_types(sku: str) -> list[str]:
        return list(sku_cfg(cfg, sku)["defect_types"])

    def acquire():
        """-> (upright RGB image, its stored JPEG bytes, source). Inspects exactly what it stores."""
        f = request.files.get("image")
        if f is not None and f.filename:
            try:
                data = jpeg_bytes(load_image(f.read()), state.quality)
            except Exception:
                raise ApiError(400, "bad_image", "the uploaded file is not a readable image")
            return load_image(data), data, "upload"
        cam = state.camera
        if cam is None:
            if state.cams.configured:
                raise ApiError(503, "camera_unavailable",
                               f"the camera is not connected ({state.cams.error}) — retrying; or attach a photo")
            raise ApiError(400, "no_image", "no camera on this station — attach a photo")
        frozen_msg = "the camera keeps sending the same picture — restart the webcam app or replug the cable"
        if cam.is_frozen():
            raise ApiError(503, "camera_frozen", frozen_msg)
        settle = float(cfg["camera"].get("settle_ms", 250)) / 1000.0
        after = time.time() + settle
        # only a frame taken after the press whose pixels changed then: a feed that froze
        # a moment ago must never pass off the previous brush's picture as this one
        frame, ts = cam.fresh_frame(after, timeout=settle + 1.5,
                                    require_change=bool(cfg["camera"].get("freeze_guard", True)))
        if frame is None:
            st = cam.status()
            if st["frame_age_s"] is not None and st["frame_age_s"] < 1.0:
                raise ApiError(503, "camera_frozen", frozen_msg)
            raise ApiError(503, "camera_stale", "the camera is not delivering frames — check the cable")
        data = jpeg_bytes(frame, state.quality)
        return load_image(data), data, "camera"

    def mismatch_message(exc: FrameMismatch, source: str) -> str:
        (w, h), (fw, fh) = exc.got, exc.expected
        if source == "upload":
            return (f"this photo is {w}x{h}, but the model was trained on {fw}x{fh} frames from the rig camera — "
                    "photos must come from the rig camera at that resolution")
        return (f"the camera now delivers {w}x{h}, but the model was trained on {fw}x{fh} — set camera "
                f"width/height back to {fw}x{fh} in config.yaml, or retrain on frames from this setting")

    def label_for(key: str) -> dict:
        return {"key": key, **defect_label(cfg, key)}

    @app.before_request
    def guard_writes():
        """POSTs need X-Station-Pin: blocks cross-site form posts, and checks the PIN when one is set."""
        if request.method != "POST":
            return None
        got = request.headers.get("X-Station-Pin", "")
        if not got:
            return jsonify({"error": "missing X-Station-Pin header", "code": "missing_header"}), 403
        if pin and not hmac.compare_digest(got.encode(), pin.encode()):
            return jsonify({"error": "wrong station PIN", "code": "bad_pin"}), 401
        return None

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        skus = [{"id": s, "name": sku_cfg(cfg, s)["display_name"]} for s in (cfg.get("skus") or {})]
        return render_template("index.html", skus=skus, version=__version__)

    @app.get("/api/status")
    def api_status():
        out = []
        for sku in cfg.get("skus") or {}:
            sc = sku_cfg(cfg, sku)
            info, err = None, None
            try:
                m = state.registry.get(sku)
                met = (m.meta.get("report") or {}).get("metrics") or {}
                info = {
                    "version": m.version,
                    "threshold": m.threshold,
                    "roi": list(m.roi) if m.roi else None,
                    "frame_size": list(m.frame_size) if m.frame_size else None,
                    "input_size": m.meta.get("input_size"),
                    "has_classifier": m.classifier is not None,
                    "created": m.meta.get("created"),
                    "auroc": met.get("auroc"),
                    "false_reject_rate": met.get("false_reject_rate"),
                    "catch_rate": met.get("catch_rate"),
                }
                err = state.registry.error(sku)
            except ModelError as exc:
                err = str(exc)
            except Exception as exc:  # never let one SKU take the page down
                log.exception("status failed for %s", sku)
                err = f"model failed to load: {exc}"
            out.append({"id": sku, "name": sc["display_name"], "model": info, "model_error": err,
                        "defect_types": [label_for(t) for t in sc["defect_types"]]})
        return jsonify({
            "version": __version__,
            "camera": state.cams.status(),
            "pin_required": bool(pin),
            "skus": out,
            "server_time": dt.datetime.now().isoformat(timespec="seconds"),
        })

    @app.get("/api/camera/frame.jpg")
    def camera_frame():
        cam = state.camera
        if cam is None:
            raise ApiError(404 if not state.cams.configured else 503, "no_camera",
                           state.cams.error or "no camera configured")
        data = cam.preview_jpeg(max_side=960)
        if data is None:
            raise ApiError(503, "camera_frozen" if cam.is_frozen() else "camera_stale",
                           "no live frame from the camera")
        resp = Response(data, mimetype="image/jpeg")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    # -------------------------------------------------------------- inspect
    @app.post("/api/inspect")
    def api_inspect():
        sku = sku_param(request.form.get("sku"))
        try:
            model = state.registry.get(sku)
        except ModelError as exc:
            raise ApiError(503, "no_model", str(exc))
        img, data, source = acquire()
        try:
            r = model.inspect(img)
        except FrameMismatch as exc:
            raise ApiError(422, "frame_mismatch", mismatch_message(exc, source))
        insp_id = uuid.uuid4().hex[:12]
        now = dt.datetime.now()
        rel = Path("inspections") / sku / now.strftime("%Y%m%d") / f"{insp_id}.jpg"
        path = state.serve_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(path, data)
        guess = r["defect_guess"]
        with state.audit.tx() as c:
            c.execute(
                "INSERT INTO inspections (id, ts, sku, model_version, source, score, threshold, verdict,"
                " defect_guess, defect_guess_conf, image_path, latency_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (insp_id, now.isoformat(timespec="seconds"), sku, r["model_version"], source, r["score"],
                 r["threshold"], r["verdict"], guess["label"] if guess else None,
                 guess["confidence"] if guess else None, rel.as_posix(), r["latency_ms"]),
            )
        if source == "camera":  # demo replay: next brush after a moment; no-op on real cameras
            state.camera.advance(float(cfg["camera"].get("replay_advance_s", 2.0)))
        state.maybe_cleanup()
        if r["verdict"] == "unrecognised":
            picture = r["roi_image"]  # nothing to localise: show what the camera saw
        else:
            picture = anomaly_overlay(r["roi_image"], r["patch_dists"], r["grid"], r["threshold"])
        return jsonify({
            "id": insp_id,
            "ts": now.isoformat(timespec="seconds"),
            "sku": sku,
            "verdict": r["verdict"],
            "unrecognised_reason": r["unrecognised_reason"],
            "score": round(r["score"], 4),
            "threshold": round(r["threshold"], 4),
            "ratio": round(r["ratio"], 3),
            "recognition_ratio": round(r["recognition_ratio"], 3),
            "latency_ms": round(r["latency_ms"]),
            "model_version": r["model_version"],
            "source": source,
            "heatmap": to_b64_jpeg(picture, max_side=560, quality=86),
            "defect_guess": {**label_for(guess["label"]), "confidence": round(guess["confidence"], 3)}
            if guess else None,
        })

    # ------------------------------------------------------------- feedback
    @app.post("/api/feedback")
    def api_feedback():
        p = request.get_json(silent=True)
        if not isinstance(p, dict):
            raise ApiError(400, "bad_body", "send a JSON object: {id, label, defect_type}")
        insp_id, label, dtype = p.get("id"), p.get("label"), p.get("defect_type") or None
        if not isinstance(insp_id, str) or not ID_RE.match(insp_id):
            raise ApiError(400, "bad_id", "missing or malformed inspection id")
        if label not in ("good", "defect"):
            raise ApiError(400, "bad_label", "label must be 'good' or 'defect'")
        with state.feedback_lock:  # one relabel at a time: copies always match the database
            with state.audit.tx() as c:
                row = c.execute("SELECT * FROM inspections WHERE id=?", (insp_id,)).fetchone()
            if row is None:
                raise ApiError(404, "unknown_id", "no inspection with that id")
            if row["verdict"] not in JUDGED:  # an empty jig must never reach the training folders
                raise ApiError(409, "not_labelable", "this frame was not recognised as a brush — nothing to label")
            sku = row["sku"]
            if label == "defect" and dtype is not None and dtype not in defect_types(sku):
                raise ApiError(400, "bad_defect_type", f"defect_type must be one of {defect_types(sku)}")
            if label == "good":
                dtype = None
            ddir = data_dir(cfg, sku)
            name = f"{insp_id}.jpg"
            old = _feedback_targets(ddir, row["operator_label"], row["operator_defect_type"], row["verdict"])
            src = state.serve_dir / row["image_path"]
            data = src.read_bytes() if src.exists() else None
            if data is None:  # evidence already cleaned up: reuse an earlier copy
                for folder in old:
                    if (folder / name).exists():
                        data = (folder / name).read_bytes()
                        break
            with state.audit.tx() as c:
                c.execute("UPDATE inspections SET operator_label=?, operator_defect_type=?, labelled_at=? WHERE id=?",
                          (label, dtype, dt.datetime.now().isoformat(timespec="seconds"), insp_id))
            saved = []
            if data is not None:
                for folder in old:
                    (folder / name).unlink(missing_ok=True)
                for folder in _feedback_targets(ddir, label, dtype, row["verdict"]):
                    folder.mkdir(parents=True, exist_ok=True)
                    atomic_write_bytes(folder / name, data)
                    saved.append((folder / name).relative_to(ddir).as_posix())
        agreed = (label == "good") == (row["verdict"] == "pass")
        out = {"ok": True, "model_agreed": agreed, "saved_to": saved}
        if data is None:
            out["note"] = "label recorded; the image is no longer stored, so nothing was filed"
        return jsonify(out)

    # -------------------------------------------------------------- capture
    @app.post("/api/capture")
    def api_capture():
        sku = sku_param(request.form.get("sku"))
        label = request.form.get("label")
        dtype = request.form.get("defect_type") or None
        if label not in ("good", "defect", "empty"):
            raise ApiError(400, "bad_label", "label must be 'good', 'defect' or 'empty'")
        if label == "defect" and dtype is not None and dtype not in defect_types(sku):
            raise ApiError(400, "bad_defect_type", f"defect_type must be one of {defect_types(sku)}")
        _, data, source = acquire()
        ddir = data_dir(cfg, sku)
        if label == "defect":
            folder = ddir / "defect" / dtype if dtype else ddir / "defect"
        else:
            folder = ddir / label  # good/ or empty/ (photos of the empty jig, for the report's safety check)
        folder.mkdir(parents=True, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        out = folder / f"{ts}.jpg"
        n = 1
        while out.exists():
            out, n = folder / f"{ts}_{n}.jpg", n + 1
        atomic_write_bytes(out, data)
        with (ddir / "capture_log.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"file": out.relative_to(ddir).as_posix(), "label": label,
                                 "defect_type": dtype if label == "defect" else None, "ts": ts,
                                 "via": f"server/{source}"}) + "\n")
        if source == "camera":
            state.camera.advance(0.0)
        return jsonify({"ok": True, "file": out.relative_to(ddir).as_posix(), "counts": _counts(ddir)})

    @app.get("/api/capture/counts")
    def api_capture_counts():
        sku = sku_param(request.args.get("sku"))
        return jsonify(_counts(data_dir(cfg, sku)))

    # ---------------------------------------------------------------- stats
    @app.get("/api/stats")
    def api_stats():
        sku = request.args.get("sku") or None
        if sku:
            sku_param(sku)
        day = dt.date.today().isoformat()
        q = "SELECT verdict, operator_label, operator_defect_type FROM inspections WHERE ts >= ?"
        args: list = [day]
        if sku:
            q += " AND sku = ?"
            args.append(sku)
        with state.audit.tx() as c:
            rows = c.execute(q, args).fetchall()
        judged = [r for r in rows if r["verdict"] in JUDGED]
        total = len(judged)
        rejects = sum(r["verdict"] == "reject" for r in judged)
        labelled = [r for r in judged if r["operator_label"]]
        agreed = sum((r["operator_label"] == "good") == (r["verdict"] == "pass") for r in labelled)
        return jsonify({
            "date": day,
            "sku": sku or "all",
            "inspections": total,
            "rejects": rejects,
            "reject_rate": round(rejects / total, 4) if total else 0.0,
            "unrecognised": sum(r["verdict"] == "unrecognised" for r in rows),
            "operator_labels": len(labelled),
            "model_operator_agreement": round(agreed / len(labelled), 4) if labelled else None,
            "false_rejects": sum(r["verdict"] == "reject" and r["operator_label"] == "good" for r in labelled),
            "misses": sum(r["verdict"] == "pass" and r["operator_label"] == "defect" for r in labelled),
            "defect_types": dict(Counter(r["operator_defect_type"] or "unspecified"
                                         for r in labelled if r["operator_label"] == "defect")),
        })

    @app.get("/api/recent")
    def api_recent():
        sku = request.args.get("sku") or None
        if sku:
            sku_param(sku)
        try:
            limit = max(1, min(200, int(request.args.get("limit", 20))))
        except ValueError:
            raise ApiError(400, "bad_limit", "limit must be a number")
        q = ("SELECT id, ts, sku, verdict, score, threshold, defect_guess, operator_label, operator_defect_type,"
             " model_version FROM inspections")
        args: list = []
        if sku:
            q += " WHERE sku = ?"
            args.append(sku)
        q += " ORDER BY ts DESC, rowid DESC LIMIT ?"
        args.append(limit)
        with state.audit.tx() as c:
            rows = [dict(r) for r in c.execute(q, args).fetchall()]
        return jsonify(rows)

    @app.get("/api/inspections/<insp_id>/image")
    def api_image(insp_id):
        if not ID_RE.match(insp_id):
            raise ApiError(400, "bad_id", "malformed inspection id")
        with state.audit.tx() as c:
            row = c.execute("SELECT image_path FROM inspections WHERE id=?", (insp_id,)).fetchone()
        path = (state.serve_dir / row["image_path"]) if row else None
        if path is None or not path.exists():
            raise ApiError(404, "not_found", "image not found (it may have been cleaned up)")
        return send_file(path, mimetype="image/jpeg")

    # --------------------------------------------------------------- errors
    @app.errorhandler(ApiError)
    def on_api_error(e: ApiError):
        return jsonify({"error": e.message, "code": e.code}), e.status

    @app.errorhandler(413)
    def on_too_large(_e):
        return jsonify({"error": "image larger than 25 MB", "code": "too_large"}), 413

    @app.errorhandler(Exception)
    def on_error(e):
        code = getattr(e, "code", None)
        if isinstance(code, int) and 400 <= code < 500:  # werkzeug HTTP errors (404 route etc.)
            return jsonify({"error": str(e), "code": "http_error"}), code
        log.exception("unhandled error")
        return jsonify({"error": f"server error: {e.__class__.__name__}", "code": "server_error"}), 500

    state.maybe_cleanup()
    return app


def _counts(ddir: Path) -> dict:
    return {"good": len(list_images(ddir / "good")), "defect": len(list_images(ddir / "defect", recursive=True)),
            "empty": len(list_images(ddir / "empty", recursive=True))}


def _lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packet is sent; picks the LAN interface
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


def _port_busy(port: int) -> bool:
    """Something already answers on this port (Windows would let a second server bind it too)."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            return True
    except OSError:
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", default=None, help="override camera.source: 0, 1, folder:<path>, none")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        raise SystemExit(f"config error: {exc}")
    host = args.host or cfg["server"].get("host", "0.0.0.0")
    port = int(args.port or cfg["server"].get("port", 8600))
    if _port_busy(port):
        raise SystemExit(f"port {port} is already in use — is the station already running? "
                         "Stop it first, or pass --port.")
    app = create_app(args.config, camera=args.camera if args.camera is not None else _DEFAULT)
    st = app.extensions["tulip"]
    cam = st.camera
    print(f"[serve] Tulip Vision {__version__}  ->  http://localhost:{port}   LAN: http://{_lan_ip()}:{port}")
    print(f"[serve] camera: {cam.label if cam else (st.cams.error or 'none (photo upload)')}"
          f" | skus: {', '.join(cfg['skus'])}")
    try:
        from waitress import serve
    except ImportError:
        app.run(host=host, port=port, threaded=True)
    else:
        serve(app, host=host, port=port, threads=int(cfg["server"].get("threads", 8)))


if __name__ == "__main__":
    main()
