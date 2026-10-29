"""Bench inference: run the current model on an image or a folder.

    python scripts/infer.py --sku zecta_classic --input shot.jpg
    python scripts/infer.py --sku zecta_classic --input data/zecta_classic/defect --heatmaps out/
    python scripts/infer.py --model models/zecta_classic/20261030-051200 --input folder/ --csv results.csv

Heatmaps show the brush-head crop the model saw: red where it deviates from
good production, a yellow box on the worst spot of every reject.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, load_config, model_dir, run_cli  # noqa: E402
from qc.data import list_images, load_image  # noqa: E402
from qc.model import FrameMismatch, InspectionModel, ModelError  # noqa: E402
from qc.vis import anomaly_overlay  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sku", help="use models/<sku>/CURRENT")
    src.add_argument("--model", help="a SKU model folder or one version folder")
    ap.add_argument("--config", default=None)
    ap.add_argument("--input", required=True, help="image file or folder (searched recursively)")
    ap.add_argument("--heatmaps", default=None, help="folder for overlay images")
    ap.add_argument("--csv", default=None, help="write one row per image")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    path = model_dir(load_config(args.config), check_sku(args.sku)) if args.sku else Path(args.model)
    try:
        model = InspectionModel.load(path, device=args.device)
    except ModelError as exc:
        raise SystemExit(str(exc))

    src_path = Path(args.input)
    paths = list_images(src_path, recursive=True) if src_path.is_dir() else [src_path]
    if not paths or not paths[0].exists():
        raise SystemExit(f"no images found at {src_path}")
    heat_dir = Path(args.heatmaps) if args.heatmaps else None
    if heat_dir:
        heat_dir.mkdir(parents=True, exist_ok=True)

    print(f"model {model.version}  threshold {model.threshold:.4f}  ROI {model.roi or 'full frame'}"
          f"{'  + defect-type classifier' if model.classifier else ''}\n")
    rows, lat, skipped = [], [], 0
    for p in paths:
        rel = p.relative_to(src_path) if src_path.is_dir() else Path(p.name)
        try:
            r = model.inspect(load_image(p))
        except FrameMismatch as exc:
            skipped += 1
            print(f"SKIPPED {rel}: {exc}")
            continue
        except OSError as exc:
            skipped += 1
            print(f"SKIPPED {rel}: unreadable ({exc})")
            continue
        lat.append(r["latency_ms"])
        guess = r["defect_guess"]
        gtxt = f"  likely {guess['label']} ({guess['confidence']:.0%})" if guess else ""
        print(f"{r['verdict'].upper():<12} {r['ratio']:5.2f}x limit  {r['latency_ms']:5.0f} ms  {rel}{gtxt}")
        rows.append({"file": str(rel), "verdict": r["verdict"], "score": round(r["score"], 5),
                     "threshold": round(r["threshold"], 5), "ratio": round(r["ratio"], 4),
                     "defect_guess": guess["label"] if guess else "",
                     "guess_confidence": round(guess["confidence"], 4) if guess else "",
                     "model_version": r["model_version"]})
        if heat_dir:
            out = heat_dir / rel.with_name(rel.stem + "_heat.jpg")
            out.parent.mkdir(parents=True, exist_ok=True)
            pic = r["roi_image"] if r["verdict"] == "unrecognised" else anomaly_overlay(
                r["roi_image"], r["patch_dists"], r["grid"], r["threshold"])
            pic.save(out, quality=92)

    if not rows:
        raise SystemExit("no image could be inspected")
    n = {v: sum(r["verdict"] == v for r in rows) for v in ("pass", "reject", "unrecognised")}
    print(f"\n{len(rows)} images | {n['reject']} rejected | {n['pass']} passed | {n['unrecognised']} not recognised"
          f"{f' | {skipped} skipped' if skipped else ''} | median {statistics.median(lat):.0f} ms per image")
    if heat_dir:
        print(f"heatmaps -> {heat_dir}")
    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"csv -> {args.csv}")


if __name__ == "__main__":
    run_cli(main)
