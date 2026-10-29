import json

import numpy as np
import pytest

from conftest import SKU
from qc.data import list_images, load_image
from qc.model import InspectionModel, ModelError, list_versions, resolve_version_dir, set_current
from qc.vis import anomaly_overlay


def test_version_folder_is_complete(trained, workspace):
    vdir, report = trained
    sku_models = workspace / "models" / SKU
    assert (sku_models / "CURRENT").read_text().strip() == vdir.name
    for name in ("meta.json", "memory.npy", "global.npy", "extractor.pt", "classifier.npz", "splits.json",
                 "report.txt"):
        assert (vdir / name).exists(), name
    meta = json.loads((vdir / "meta.json").read_text())
    assert meta["schema"] == 3 and meta["has_classifier"] is True
    assert meta["roi"] is not None and len(meta["input_size"]) == 2 and meta["frame_size"] == [480, 270]
    rec = report["recognition"]
    assert rec["empty_frames"] == 8 and rec["empty_unrecognised"] == 8 and rec["empty_passed"] == 0
    assert rec["test_goods_unrecognised"] == 0
    splits = json.loads((vdir / "splits.json").read_text())
    parts = [set(splits[k]) for k in ("train", "cal", "test")]
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    m = report["metrics"]
    assert m["good"] == len(splits["test"]) and m["defects"] == 10
    assert set(m["per_type"]) == {"missing_tuft", "sparse_tuft", "splayed_tuft", "stray_bristle", "contamination"}
    assert report["classifier"]["cv"]["folds"] == 5


def test_loaded_model_inspects_deterministically(trained, workspace):
    vdir, _ = trained
    model = InspectionModel.load(workspace / "models" / SKU, device="cpu")
    assert model.version == vdir.name
    img = load_image(list_images(workspace / "data" / SKU / "good")[0])
    a, b = model.inspect(img), model.inspect(img)
    assert a["score"] == pytest.approx(b["score"])
    assert a["verdict"] in ("pass", "reject")
    h, w = a["grid"]
    assert a["patch_dists"].shape == (h * w,)
    assert tuple(model.meta["grid"]) == (h, w)
    assert a["roi_image"].size[0] < img.size[0]  # ROI applied


def test_heatmap_marks_rejects(trained, workspace):
    model = InspectionModel.load(workspace / "models" / SKU, device="cpu")
    img = load_image(list_images(workspace / "data" / SKU / "good")[0])
    r = model.inspect(img)
    over = anomaly_overlay(r["roi_image"], r["patch_dists"], r["grid"], threshold=r["score"] * 0.5)
    px = np.asarray(over)
    yellow = (px[..., 0] > 230) & (px[..., 1] > 180) & (px[..., 2] < 60)
    assert yellow.sum() > 20, "rejects get a box around the worst spot"
    calm = anomaly_overlay(r["roi_image"], r["patch_dists"], r["grid"], threshold=r["score"] * 10)
    assert np.asarray(calm).shape == px.shape


def test_resolve_and_rollback(trained, workspace, tmp_path):
    sku_models = workspace / "models" / SKU
    assert resolve_version_dir(sku_models) == trained[0]
    with pytest.raises(ModelError):
        resolve_version_dir(tmp_path)
    with pytest.raises(ModelError):
        set_current(sku_models, "does-not-exist")
    assert trained[0].name in list_versions(sku_models)


def test_weight_problems_become_clear_errors(monkeypatch, tmp_path):
    import qc.features as qf
    from qc.train import TrainError, TrainParams, train_anomaly

    def offline(*a, **k):  # what torchvision raises when the download is blocked
        raise OSError("Tunnel connection failed")

    monkeypatch.setitem(qf.BACKBONES, "resnet18", (offline, "ResNet18_Weights"))
    with pytest.raises(qf.WeightsError, match="--weights"):
        qf.FeatureExtractor("resnet18", (64, 64), "imagenet", "cpu")
    monkeypatch.undo()

    from conftest import small_style
    from qc.synth import make_dataset

    make_dataset(tmp_path / "data" / "x", n_good=24, defects_per_type=0, class_per_type=0, stream_good=0,
                 stream_defects_per_type=0, style=small_style(), log=lambda *a: None)
    with pytest.raises(TrainError, match="nope.pth"):
        train_anomaly(tmp_path / "data" / "x", tmp_path / "models" / "x",
                      TrainParams(weights=str(tmp_path / "nope.pth"), min_good=10), log=lambda *a: None)
