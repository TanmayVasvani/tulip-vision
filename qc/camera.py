"""Camera sources for the capture station and the inspection server.

The camera hangs off the inspection PC, and *the server* grabs frames from
it. Training images (captured through the same path) and inspection images
therefore come from identical code, resolution and JPEG settings, and any
phone or tablet on the LAN can trigger an inspection without needing
HTTPS camera access in its browser.

Sources
    0, 1, ...              a USB webcam or a phone running a webcam app
    "folder:<path>"        replays images from a folder — demos and tests;
                           moves to the next image after each inspection
    None / "none" / off    no camera: the UI falls back to photo upload

Safety: a real sensor never delivers two identical frames (noise alone
changes every pixel), so a feed whose pixels stop changing is treated as
frozen — a stalled webcam app re-sending its last frame must never be
judged over and over as if it were each new brush.
"""

from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from pathlib import Path

from PIL import Image

from .data import jpeg_bytes, list_images, load_image

log = logging.getLogger("tulip.camera")

FROZEN_AFTER_S = 2.0   # identical pixels for this long = frozen feed
STALE_AFTER_S = 2.0    # no new frame for this long = camera gone
REOPEN_FROZEN_S = 5.0  # a frozen feed is reopened after this long, then every 10 s


class CameraError(RuntimeError):
    pass


class DeviceSource:
    """cv2.VideoCapture with best-effort locked exposure / focus / white balance."""

    replay = False

    def __init__(self, index: int, width: int = 1280, height: int = 720,
                 backend: str = "auto", lock: dict | None = None):
        import cv2  # imported lazily: folder sources and tests do not need OpenCV

        self._cv2 = cv2
        self._apis = {
            "auto": cv2.CAP_ANY,
            "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
            "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
            "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
            "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
        }
        if backend not in self._apis:
            raise CameraError(f"unknown camera backend {backend!r}; options: {sorted(self._apis)}")
        self.index, self.width, self.height = int(index), int(width), int(height)
        self.backend, self.lock = backend, dict(lock or {})
        self.applied: dict = {}
        self.actual = (0, 0)
        self.cap = None
        self.open()

    @property
    def label(self) -> str:
        w, h = self.actual
        return f"camera {self.index} ({self.backend}, {w}x{h})"

    def open(self) -> None:
        cv2 = self._cv2
        cap = cv2.VideoCapture(self.index, self._apis[self.backend])
        if not cap.isOpened():
            raise CameraError(f"camera {self.index} did not open — check the index and cable, and that no "
                              "other program (Zoom, the camera app, capture.py) is using it")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        props = {
            "autofocus": cv2.CAP_PROP_AUTOFOCUS,
            "focus": cv2.CAP_PROP_FOCUS,
            "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
            "exposure": cv2.CAP_PROP_EXPOSURE,
            "auto_wb": cv2.CAP_PROP_AUTO_WB,
            "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
            "gain": cv2.CAP_PROP_GAIN,
        }
        self.applied = {}
        for key, val in self.lock.items():
            if key in props:
                ok = cap.set(props[key], float(val))
                self.applied[key] = {"requested": val, "accepted": bool(ok), "now": cap.get(props[key])}
        self.actual = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.actual != (self.width, self.height):
            log.warning("camera %d delivers %dx%d, not the %dx%d in config.yaml",
                        self.index, *self.actual, self.width, self.height)
        self.cap = cap

    def read(self) -> Image.Image | None:
        ok, frame = self.cap.read() if self.cap is not None else (False, None)
        if not ok or frame is None:
            return None
        return Image.fromarray(self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB))

    def reopen(self) -> None:
        self.close()
        self.open()

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class FolderSource:
    """Replays the images in a folder as if they were camera frames."""

    replay = True

    def __init__(self, folder, shuffle_seed: int | None = 7):
        self.folder = Path(folder)
        self.paths = list_images(self.folder, recursive=True)
        if not self.paths:
            raise CameraError(f"folder camera: no images in {self.folder}")
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.paths)
        self.label = f"folder replay: {self.folder.name} ({len(self.paths)} images)"
        self.applied: dict = {}
        self._i = 0
        self._img: Image.Image | None = None
        self._lock = threading.Lock()

    @property
    def current_path(self) -> Path:
        return self.paths[self._i]

    def read(self) -> Image.Image | None:
        with self._lock:
            if self._img is None:
                self._img = load_image(self.paths[self._i])
            return self._img

    def advance(self) -> None:
        with self._lock:
            self._i = (self._i + 1) % len(self.paths)
            self._img = None

    def reopen(self) -> None:
        pass

    def close(self) -> None:
        pass


class Camera:
    """Background grabber: always holds the freshest frame from a source.

    Continuous reading matters for USB cameras, which buffer frames — a
    frame read on demand can be several frames stale.
    """

    def __init__(self, source, fps: float = 15.0, jpeg_quality: int = 92):
        self.source = source
        self.jpeg_quality = int(jpeg_quality)
        self._interval = 1.0 / max(1.0, float(fps))
        self._lock = threading.Lock()
        self._frame: Image.Image | None = None
        self._ts = 0.0
        self._seq = 0
        self._sig: bytes | None = None
        self._changed_ts = 0.0
        self._preview_cache: tuple[int, int, bytes] | None = None
        self._failures = 0
        self._last_reopen = 0.0
        self._advance_timer: threading.Timer | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="camera-grabber", daemon=True)
        self._thread.start()

    @property
    def label(self) -> str:
        return self.source.label

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                frame = self.source.read()
            except Exception as exc:  # a flaky USB link must not kill the thread
                log.warning("camera read failed: %s", exc)
                frame = None
            if frame is None:
                self._failures += 1
                if self._failures == 20 or self._failures % 50 == 0:  # ~2 s, then every ~5 s
                    log.warning("%s: %d failed reads, reopening", self.label, self._failures)
                    try:
                        self.source.reopen()
                    except Exception as exc:
                        log.warning("reopen failed: %s", exc)
                self._stop.wait(0.1)
                continue
            self._failures = 0
            now = time.time()
            sig = None if self.source.replay else hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
            with self._lock:
                if frame is not self._frame:
                    self._seq += 1
                if sig is None or sig != self._sig:
                    self._sig, self._changed_ts = sig, now
                self._frame, self._ts = frame, now
                frozen_for = now - self._changed_ts
            if (not self.source.replay and frozen_for > REOPEN_FROZEN_S
                    and now - self._last_reopen > 10.0):  # a stalled webcam app: try a fresh connection
                self._last_reopen = now
                log.warning("%s: feed frozen for %.0f s, reopening", self.label, frozen_for)
                try:
                    self.source.reopen()
                except Exception as exc:
                    log.warning("reopen failed: %s", exc)
            if self.source.replay:  # device reads already block at the camera's frame rate
                self._stop.wait(max(0.0, self._interval - (time.monotonic() - t0)))

    def is_frozen(self) -> bool:
        """True when a real camera keeps delivering pixel-identical frames."""
        if self.source.replay:
            return False
        with self._lock:
            return self._frame is not None and (self._ts - self._changed_ts) > FROZEN_AFTER_S

    def fresh_frame(self, after: float, timeout: float, require_change: bool = True):
        """The first frame taken at or after ``after`` whose pixels also changed then.

        A live sensor changes every frame (noise alone), so this costs nothing
        normally. A feed that froze — even a second ago — never satisfies it,
        so a stale picture of the previous brush can never be judged as the
        new one. Returns (None, None) on timeout.
        """
        deadline = time.time() + timeout
        while True:
            with self._lock:
                frame, ts, changed = self._frame, self._ts, self._changed_ts
            if frame is not None and ts >= after and (
                not require_change or self.source.replay or changed >= after
            ):
                return frame, ts
            if time.time() >= deadline:
                return None, None
            time.sleep(0.01)

    def latest(self, newer_than: float | None = None, timeout: float = 2.0):
        """(frame, timestamp); optionally wait for a frame captured after ``newer_than``."""
        deadline = time.time() + timeout
        while True:
            with self._lock:
                frame, ts = self._frame, self._ts
            if frame is not None and (newer_than is None or ts >= newer_than):
                return frame, ts
            if time.time() >= deadline:
                return (frame, ts) if frame is not None else (None, None)
            time.sleep(0.02)

    def preview_jpeg(self, max_side: int = 960, quality: int = 80) -> bytes | None:
        """Latest frame for the live view; None when the feed is stale or frozen."""
        with self._lock:
            frame, seq, ts = self._frame, self._seq, self._ts
            cached = self._preview_cache
        if frame is None or (time.time() - ts) > STALE_AFTER_S or self.is_frozen():
            return None
        if cached and cached[0] == seq and cached[1] == max_side:
            return cached[2]
        img = frame
        if max(img.size) > max_side:
            img = img.copy()
            img.thumbnail((max_side, max_side))
        data = jpeg_bytes(img, quality)
        with self._lock:
            self._preview_cache = (seq, max_side, data)
        return data

    def advance(self, delay: float = 0.0) -> None:
        """Folder replay: move on to the next 'brush' (no-op for real cameras).

        A delay keeps the inspected brush on screen for a moment, the way a
        real brush stays in the jig until the operator swaps it.
        """
        if not self.source.replay:
            return
        with self._lock:
            if self._advance_timer is not None:
                self._advance_timer.cancel()
                self._advance_timer = None
            if delay > 0:
                timer = threading.Timer(delay, self._do_advance, args=(None,))
                timer.args = (timer,)
                timer.daemon = True
                self._advance_timer = timer
                timer.start()
                return
        self._do_advance(None)

    def _do_advance(self, timer) -> None:
        self.source.advance()
        frame = self.source.read()
        with self._lock:
            self._frame, self._ts, self._seq = frame, time.time(), self._seq + 1
            if timer is not None and self._advance_timer is timer:
                self._advance_timer = None

    def status(self) -> dict:
        with self._lock:
            ts, size = self._ts, (self._frame.size if self._frame is not None else None)
        age = (time.time() - ts) if ts else None
        frozen = self.is_frozen()
        return {
            "label": self.label,
            "ok": age is not None and age < STALE_AFTER_S and not frozen,
            "frozen": frozen,
            "frame_age_s": round(age, 2) if age is not None else None,
            "frame_size": list(size) if size else None,
            "failures": self._failures,
            "settings": getattr(self.source, "applied", {}),
            "replay": bool(self.source.replay),
        }

    def close(self) -> None:
        with self._lock:
            if self._advance_timer is not None:
                self._advance_timer.cancel()
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.source.close()


def camera_configured(spec) -> bool:
    if spec is None or isinstance(spec, bool):  # YAML turns off/no/false into False
        return False
    return not (isinstance(spec, str) and spec.strip().lower() in ("", "none", "null", "off", "no", "false"))


def open_camera(spec, cam_cfg: dict | None = None, root: Path | None = None) -> Camera | None:
    """Config value -> Camera (or None when no camera is configured)."""
    cam_cfg = cam_cfg or {}
    if not camera_configured(spec):
        return None
    if isinstance(spec, str) and spec.startswith("folder:"):
        folder = Path(spec[len("folder:"):])
        if not folder.is_absolute() and root is not None:
            folder = root / folder
        source = FolderSource(folder)
    else:
        try:
            index = int(str(spec).strip())
        except ValueError:
            raise CameraError(f"camera source {spec!r}: use a device index (0, 1, ...), 'folder:<path>' or null")
        source = DeviceSource(
            index,
            cam_cfg.get("width", 1280),
            cam_cfg.get("height", 720),
            cam_cfg.get("backend", "auto"),
            cam_cfg.get("lock") or {},
        )
    return Camera(source, fps=cam_cfg.get("fps", 15), jpeg_quality=cam_cfg.get("jpeg_quality", 92))
