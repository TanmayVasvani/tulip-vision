"""Train the anomaly model for one SKU and print the report."""

    python scripts/train_anomaly.py --sku zecta_classic

Reads
    data/<sku>/good/            good brushes from the rig (300+ distinct units)
    data/<sku>/defect/<type>/   real defects, for evaluation only
    data/<sku>/roi.json         brush-head box (scripts/set_roi.py); optional
    data/<sku>/classes/<type>/  labelled defects -> defect-type classifier (auto)

Writes a new version under models/<sku>/ and makes it CURRENT; a running
inspection server picks it up without a restart. Every number in the
report comes from images the model never trained on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, data_dir, load_config, model_dir, run_cli  # noqa: E402
from qc.data import parse_roi  # noqa: E402
from qc.train import TrainError, TrainParams, format_report, train_anomaly  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", default="resnet18", choices=["resnet18", "wide_resnet50_2"])
    ap.add_argument("--weights", default="imagenet",
                    help="'imagenet' (torchvision download, once) or a local .pth ResNet state dict")
    ap.add_argument("--roi", default=None, help="x0,y0,x1,y1 as frame fractions (default: data/<sku>/roi.json)")
    ap.add_argument("--full-frame", action="store_true", help="ignore roi.json and use the whole frame")
    ap.add_argument("--input-long-side", type=int, default=448,
                    help="model input size along the long side; 576 catches subtler defects, ~1.7x slower")
    ap.add_argument("--coreset", type=int, default=4000)
    ap.add_argument("--quantile", type=float, default=0.995)
    ap.add_argument("--margin", type=float, default=1.08)
    ap.add_argument("--policy", choices=["goods", "gap"], default="goods",
                    help="goods: threshold from good images only; gap: also use defects")
    ap.add_argument("--include-feedback", action="store_true",
                    help="also train on operator-confirmed goods in data/<sku>/feedback/good/ (review them first)")
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--min-good", type=int, default=40)
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sku = check_sku(args.sku)
    params = TrainParams(
        backbone=args.backbone,
        weights=args.weights,
        input_long_side=args.input_long_side,
        roi=parse_roi(args.roi) if args.roi else None,
        use_roi_file=not args.full_frame,
        coreset_size=args.coreset,
        quantile=args.quantile,
        margin=args.margin,
        policy=args.policy,
        include_feedback=args.include_feedback,
        classifier="off" if args.no_classifier else "auto",
        min_good=args.min_good,
        device=args.device,
        seed=args.seed,
    )
    try:
        vdir, report = train_anomaly(data_dir(cfg, sku), model_dir(cfg, sku), params)
    except TrainError as exc:
        raise SystemExit(f"[train] {exc}")
    print()
    print(format_report(report))
    print(f" saved   {vdir}  (now CURRENT)")
    print(f" next    python scripts/infer.py --sku {sku} --input data/{sku}/defect --heatmaps out/{sku}")


if __name__ == "__main__":
    run_cli(main)
