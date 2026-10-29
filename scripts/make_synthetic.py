"""Generate a synthetic demo SKU so the whole system can run before the rig exists.

    python scripts/make_synthetic.py                    # -> data/demo_synthetic/
    python scripts/make_synthetic.py --good 120 --sku demo_small

Writes the same layout the real workflow produces:
    good/                 good brushes (train / calibrate / test)
    defect/<type>/        planted defects for evaluation, with true boxes in
                          defect_annotations.json
    empty/                the jig with no brush (the station must never pass it)
    classes/<type>/       extra labelled defects for the Phase-2 classifier
    stream/               a shuffled mix the demo camera replays
    roi.json              the brush-head ROI

Synthetic results prove the pipeline, not accuracy on real brushes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, data_dir, load_config, run_cli  # noqa: E402
from qc.synth import make_dataset  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", default="demo_synthetic")
    ap.add_argument("--config", default=None, help="config.yaml (default: repo root)")
    ap.add_argument("--good", type=int, default=320)
    ap.add_argument("--defects-per-type", type=int, default=10)
    ap.add_argument("--class-per-type", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="delete an existing folder first")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = data_dir(cfg, check_sku(args.sku))
    if out.exists() and any(out.iterdir()):
        if not args.force:
            raise SystemExit(f"{out} already has files — pass --force to regenerate it")
        if not (out / "synth_manifest.json").exists():
            raise SystemExit(f"refusing to delete {out}: it is not a synthetic dataset (no synth_manifest.json)")
        shutil.rmtree(out)
    t0 = time.perf_counter()
    m = make_dataset(out, n_good=args.good, defects_per_type=args.defects_per_type,
                     class_per_type=args.class_per_type, seed=args.seed)
    print(f"[synth] wrote {out} in {time.perf_counter() - t0:.0f}s: {m['counts']}")
    print(f"[synth] next: python scripts/train_anomaly.py --sku {args.sku}")


if __name__ == "__main__":
    run_cli(main)
