"""List a SKU's model versions, or switch the live one (rollback / roll forward).

    python scripts/set_version.py --sku zecta_classic                    # list
    python scripts/set_version.py --sku zecta_classic --use 20261030-000947

Use this instead of editing models/<sku>/CURRENT by hand. A running station
picks the change up on its next inspection — no restart needed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qc.config import check_sku, load_config, model_dir, run_cli  # noqa: E402
from qc.model import InspectionModel, ModelError, list_versions, resolve_version_dir, set_current  # noqa: E402


def _pct(x) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sku", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--use", default=None, help="version name to make live")
    args = ap.parse_args()

    sku_models = model_dir(load_config(args.config), check_sku(args.sku))
    versions = list_versions(sku_models)
    if not versions:
        raise SystemExit(f"no versions in {sku_models} — train first")
    if args.use:
        name = args.use.strip()
        if name not in versions:
            raise SystemExit(f"no version {name!r}\navailable: {', '.join(versions)}")
        try:
            InspectionModel.load(sku_models / name, device="cpu")  # never switch to a version that cannot load
        except Exception as exc:
            raise SystemExit(f"version {name} does not load ({exc}) — not switched")
        set_current(sku_models, name)
        print(f"live version for {args.sku} is now {name}")
        return

    try:
        live = resolve_version_dir(sku_models).name
    except ModelError as exc:
        live = None
        print(f"warning: {exc}")
    print(f"{'':2}{'version':<20}{'created':<21}{'AUROC':>7}{'false rej':>11}{'caught':>9}  types")
    for v in versions:
        try:
            meta = json.loads((sku_models / v / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            print(f"  {v:<20}(unreadable meta.json)")
            continue
        m = (meta.get("report") or {}).get("metrics") or {}
        auc = m.get("auroc")
        print(f"{'*' if v == live else ' '} {v:<20}{meta.get('created', '?'):<21}"
              f"{(f'{auc:.3f}' if auc is not None else 'n/a'):>7}{_pct(m.get('false_reject_rate')):>11}"
              f"{_pct(m.get('catch_rate')):>9}  {'yes' if meta.get('has_classifier') else '-'}")
    print("\n* = live. Switch with: --use <version>")


if __name__ == "__main__":
    run_cli(main)
