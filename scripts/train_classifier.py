"""Retrain only the defect-type classifier on the current anomaly model."""

    python scripts/train_classifier.py --sku zecta_classic

Reads labelled defects from data/<sku>/classes/<type>/ (the operator
station fills these when an operator marks a brush bad and picks the
defect type). A 'good' folder there holds false alarms the operator
overruled. Needs at least two classes with 5+ images each; 30+ per class
is where it starts to be dependable. Writes a new model version (anomaly
model unchanged) and makes it CURRENT.

train_anomaly.py already does this automatically when classes/ is usable;
use this script to refresh the classifier as feedback accumulates.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, data_dir, load_config, model_dir, run_cli  # noqa: E402
from qc.model import ModelError  # noqa: E402
from qc.train import TrainError, retrain_classifier  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--min-per-class", type=int, default=5)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    sku = check_sku(args.sku)
    try:
        vdir, rep = retrain_classifier(data_dir(cfg, sku), model_dir(cfg, sku),
                                       min_per_class=args.min_per_class, device=args.device)
    except (TrainError, ModelError) as exc:
        raise SystemExit(f"[clf] {exc}")
    cv = rep.get("cv")
    print(f"[clf] classes {rep['counts']}")
    if cv:
        print(f"[clf] cross-validated accuracy {cv['accuracy']:.1%} ({cv['folds']}-fold)")
        for k, v in cv["per_class_recall"].items():
            print(f"        {k:<18} recall {v:.1%}")
    print(f"[clf] saved {vdir} (now CURRENT)")


if __name__ == "__main__":
    run_cli(main)
