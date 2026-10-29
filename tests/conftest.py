"""Shared fixtures: a tiny synthetic SKU and a quickly trained model.

The backbone runs with random weights here (no download, deterministic),
so these tests check mechanics — splits, bundles, versions, the server —
not accuracy. Accuracy is measured by scripts/train_anomaly.py on real
(or synthetic demo) data with ImageNet weights.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SKU = "demo"


def small_style():
    from qc.synth import BrushStyle

    return BrushStyle(frame=(480, 270), px_per_mm=4.5, head_center=(125.0, 135.0))


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """tmp/{config.yaml, data/demo/..., models/, serve/data/} with a tiny dataset."""
    from qc.synth import make_dataset

    root = tmp_path_factory.mktemp("tv")
    make_dataset(root / "data" / SKU, n_good=48, defects_per_type=2, class_per_type=6,
                 stream_good=6, stream_defects_per_type=1, seed=3, style=small_style(), log=lambda *a: None)
    (root / "config.yaml").write_text(
        "data_root: data\nmodels_root: models\nserve_data: serve/data\n"
        "camera:\n  source: null\n  settle_ms: 0\n  replay_advance_s: 0\n"
        f"skus:\n  {SKU}:\n    display_name: Demo\n"
        "    defect_types: [missing_tuft, sparse_tuft, splayed_tuft, stray_bristle, contamination, other]\n"
        "  untrained:\n    display_name: Not trained\n",
        encoding="utf-8",
    )
    return root


@pytest.fixture(scope="session")
def trained(workspace):
    """Train once per session (random-weight backbone) -> (version_dir, report)."""
    import torch

    from qc.train import TrainParams, train_anomaly

    torch.manual_seed(0)
    params = TrainParams(weights=None, input_long_side=192, coreset_size=500, min_good=20, device="cpu")
    return train_anomaly(workspace / "data" / SKU, workspace / "models" / SKU, params, log=lambda *a: None)
