"""Capture station — build the dataset with the fixed rig, keyboard-driven.

    python scripts/capture.py --sku zecta_classic
    python scripts/capture.py --sku zecta_classic --camera 1

Keys
    SPACE   save the current frame
    g       label = good            d   label = defect
    e       label = empty jig (5-10 photos let the report prove an empty jig never passes)
    1..9    defect type (shown in the bar) — saves into defect/<type>/
    r       show / hide the ROI box q   quit

Frames are saved at full resolution with the same JPEG quality the
inspection server uses, into data/<sku>/good/ or data/<sku>/defect/<type>/.
Every capture is appended to data/<sku>/capture_log.jsonl. Nothing is ever
overwritten.

The operator UI's Capture tab does the same job from a browser; use one or
the other (only one program can hold the camera).

Rig rules (they matter more than the model): camera clamped, brush in a
jig, diffuse light, matte grey background, hands out of frame when shooting.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.camera import CameraError, DeviceSource  # noqa: E402
from qc.config import check_sku, data_dir, load_config, run_cli, sku_cfg  # noqa: E402
from qc.data import atomic_write_bytes, jpeg_bytes, list_images, read_roi_file, roi_box_px  # noqa: E402

GREEN, RED, AMBER = (60, 170, 60), (50, 50, 220), (0, 200, 255)  # BGR
WINDOW = "Tulip Vision capture"
MIN_GAP_S = 0.7     # never two photos closer than this
REPEAT_GAP_S = 0.15  # SPACE events closer than this are key-repeat from a held key


def window_closed(cv2) -> bool:
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--camera", type=int, default=None, help="device index (default: config camera.source)")
    args = ap.parse_args()

    import cv2
    from PIL import Image

    cfg = load_config(args.config)
    sku = check_sku(args.sku)
    cam_cfg = cfg["camera"]
    index = args.camera if args.camera is not None else cam_cfg.get("source")
    try:
        index = int(index)
    except (TypeError, ValueError):
        raise SystemExit("capture.py needs a real camera index (--camera 0)")
    root = data_dir(cfg, sku)
    types = [t for t in sku_cfg(cfg, sku)["defect_types"]][:9]
    (root / "good").mkdir(parents=True, exist_ok=True)
    roi = read_roi_file(root)
    quality = int(cam_cfg.get("jpeg_quality", 92))
    try:
        cam = DeviceSource(index, cam_cfg.get("width", 1280), cam_cfg.get("height", 720),
                           cam_cfg.get("backend", "auto"), cam_cfg.get("lock") or {})
    except CameraError as exc:
        raise SystemExit(str(exc))
    if cam.applied:
        print(f"[capture] camera settings: {cam.applied}")

    label, dtype, show_roi = "good", types[0] if types else "other", True
    counts = {"good": len(list_images(root / "good")),
              "defect": len(list_images(root / "defect", recursive=True))}
    print(f"[capture] {sku}: {counts['good']} good, {counts['defect']} defect already saved")
    print("[capture] SPACE save | g good | d defect | 1-9 defect type | e empty jig | r ROI box | q quit")

    import numpy as np

    last_save, last_space, last_sig, fails, shown = 0.0, 0.0, None, 0, False
    while True:
        ok, frame = cam.cap.read() if cam.cap is not None else (False, None)
        if not ok or frame is None:
            fails += 1
            if fails % 60 == 0:  # about every 2 s: try to get the camera back
                print("[capture] no frames from the camera — reconnecting")
                try:
                    cam.reopen()
                except CameraError as exc:
                    print(f"[capture] {exc}")
            blank = np.zeros((360, 640, 3), np.uint8)
            cv2.putText(blank, "NO CAMERA FRAMES - check the cable (q = quit)", (16, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
            cv2.imshow(WINDOW, blank)
            shown = True
            if (cv2.waitKey(30) & 0xFF) == ord("q") or window_closed(cv2):
                break
            continue
        fails = 0
        hud = frame.copy()
        if roi and show_roi:
            l, t, r, b = roi_box_px((hud.shape[1], hud.shape[0]), roi)
            cv2.rectangle(hud, (l, t), (r, b), AMBER, 2)
        cv2.rectangle(hud, (0, 0), (hud.shape[1], 44), (25, 25, 25), -1)
        what = {"good": "GOOD", "empty": "EMPTY JIG"}.get(label, f"DEFECT: {dtype}")
        cv2.putText(hud, f"{sku} | {what} | good {counts['good']}  defect {counts['defect']} | SPACE = save",
                    (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.65, {"good": GREEN, "empty": AMBER}.get(label, RED), 2)
        cv2.imshow(WINDOW, hud)
        shown = True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or (shown and window_closed(cv2)):
            break
        if key == ord("g"):
            label = "good"
        elif key == ord("e"):
            label = "empty"
        elif key == ord("d"):
            label = "defect"
        elif ord("1") <= key <= ord("9") and key - ord("1") < len(types):
            label, dtype = "defect", types[key - ord("1")]
        elif key == ord("r"):
            show_roi = not show_roi
        elif key == ord(" "):
            now = time.monotonic()
            repeat = now - last_space < REPEAT_GAP_S
            last_space = now
            if repeat or now - last_save < MIN_GAP_S:  # a held key is one press: one brush, one photo
                continue
            sig = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
            if sig == last_sig:
                print("[capture] the camera sent the exact same picture again — is the feed frozen? not saved")
                continue
            last_save, last_sig = now, sig
            ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            folder = root / "defect" / dtype if label == "defect" else root / label
            folder.mkdir(parents=True, exist_ok=True)
            out = folder / f"{ts}.jpg"
            # raw frame (no overlay), encoded exactly like the inspection server does
            rgb = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            atomic_write_bytes(out, jpeg_bytes(rgb, quality))
            counts[label] = counts.get(label, 0) + 1
            with (root / "capture_log.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"file": str(out.relative_to(root)), "label": label,
                                    "defect_type": dtype if label == "defect" else None,
                                    "ts": ts, "camera": index, "via": "capture.py"}) + "\n")
            print(f"[capture] saved {out.relative_to(root)}")
    cam.close()
    cv2.destroyAllWindows()
    print(f"[capture] totals: {counts}")


if __name__ == "__main__":
    run_cli(main)
