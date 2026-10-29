"""Regression tests for earlier fixes."""

Each test names the failure it guards against, so a future change that
brings one back fails loudly.
"""

import io
import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image

from conftest import ROOT, SKU
from qc.data import list_images, load_image


@pytest.fixture(scope="module")
def app(trained, workspace):
    from serve.app import create_app

    a = create_app(workspace / "config.yaml", camera=f"folder:{workspace / 'data' / SKU / 'stream'}")
    yield a
    a.extensions["tulip"].cams.close()


@pytest.fixture()
def client(app):
    c = app.test_client()
    c.environ_base["HTTP_X_STATION_PIN"] = "-"
    return c


def _upload(path_or_img):
    buf = io.BytesIO()
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img.convert("RGB").save(buf, format="JPEG", quality=92)
    buf.seek(0)
    return (buf, "shot.jpg")


# ------------------------------------------------- critical: empty jig, frozen feed
def test_empty_jig_is_never_a_pass(trained, workspace):
    from qc.model import InspectionModel

    model = InspectionModel.load(workspace / "models" / SKU, device="cpu")
    for p in list_images(workspace / "data" / SKU / "empty"):
        r = model.inspect(load_image(p))
        assert r["verdict"] == "unrecognised" and r["unrecognised_reason"] == "no_brush"


def test_empty_jig_upload_is_flagged_and_kept_out_of_the_counts(client, workspace):
    before = client.get(f"/api/stats?sku={SKU}").get_json()
    empty = list_images(workspace / "data" / SKU / "empty")[0]
    j = client.post("/api/inspect", data={"sku": SKU, "image": _upload(empty)},
                    content_type="multipart/form-data").get_json()
    assert j["verdict"] == "unrecognised" and j["defect_guess"] is None
    after = client.get(f"/api/stats?sku={SKU}").get_json()
    assert after["inspections"] == before["inspections"]
    assert after["unrecognised"] == before["unrecognised"] + 1
    assert j["unrecognised_reason"] == "no_brush"
    r = client.post("/api/feedback", json={"id": j["id"], "label": "good"})  # an empty jig never reaches training
    assert r.status_code == 409 and r.get_json()["code"] == "not_labelable"


class _FrozenSource:
    """A webcam app that stalled and keeps re-sending its last frame."""

    replay = False
    label = "frozen test camera"
    applied: dict = {}

    def __init__(self, img):
        self.img = img

    def read(self):
        time.sleep(0.02)
        return self.img.copy()  # new object, identical pixels

    def reopen(self):
        pass

    def close(self):
        pass


def test_frozen_feed_is_detected(monkeypatch, workspace):
    from qc import camera as cammod

    monkeypatch.setattr(cammod, "FROZEN_AFTER_S", 0.3)
    img = load_image(list_images(workspace / "data" / SKU / "good")[0])
    cam = cammod.Camera(_FrozenSource(img), fps=30)
    try:
        time.sleep(0.15)
        assert not cam.is_frozen()
        time.sleep(0.5)
        assert cam.is_frozen() and cam.preview_jpeg() is None
        assert cam.status()["ok"] is False and cam.status()["frozen"] is True
    finally:
        cam.close()


def test_server_refuses_to_judge_a_frozen_feed(monkeypatch, client, app, workspace):
    from qc import camera as cammod

    monkeypatch.setattr(cammod, "FROZEN_AFTER_S", 0.3)
    mgr = app.extensions["tulip"].cams
    original = mgr.camera
    img = load_image(list_images(workspace / "data" / SKU / "good")[0])
    mgr.camera = cammod.Camera(_FrozenSource(img), fps=30)
    try:
        time.sleep(0.6)
        r = client.post("/api/inspect", data={"sku": SKU})
        assert r.status_code == 503 and r.get_json()["code"] == "camera_frozen"
        assert client.get("/api/camera/frame.jpg").status_code == 503
    finally:
        mgr.camera.close()
        mgr.camera = original


# ------------------------------------------------------------------- major
def test_frames_of_another_size_are_refused_with_a_reason(client, workspace):
    good = Image.open(list_images(workspace / "data" / SKU / "good")[0])
    for size in ((400, 300), (960, 540), (240, 135)):  # other shape, and same shape at other resolutions
        r = client.post("/api/inspect", data={"sku": SKU, "image": _upload(good.resize(size))},
                        content_type="multipart/form-data")
        j = r.get_json()
        assert r.status_code == 422 and j["code"] == "frame_mismatch" and "rig camera" in j["error"]


def test_model_refuses_other_frame_sizes(trained, workspace):
    from qc.model import FrameMismatch, InspectionModel

    model = InspectionModel.load(workspace / "models" / SKU, device="cpu")
    img = load_image(list_images(workspace / "data" / SKU / "good")[3])
    model.inspect(img)
    with pytest.raises(FrameMismatch) as exc:
        model.inspect(img.resize((img.width * 2, img.height * 2)))
    assert exc.value.expected == img.size


def test_broken_new_version_does_not_take_the_station_down(client, workspace):
    from qc.model import resolve_version_dir, set_current

    sku_models = workspace / "models" / SKU
    good = resolve_version_dir(sku_models)
    assert client.post("/api/inspect", data={"sku": SKU}).status_code == 200
    broken = sku_models / "broken-version"
    shutil.copytree(good, broken)
    (broken / "memory.npy").write_bytes(b"truncated")
    set_current(sku_models, broken.name)
    try:
        r = client.post("/api/inspect", data={"sku": SKU})
        assert r.status_code == 200 and r.get_json()["model_version"] == good.name
        info = {s["id"]: s for s in client.get("/api/status").get_json()["skus"]}[SKU]
        assert "still serving" in info["model_error"]
    finally:
        set_current(sku_models, good.name)
        shutil.rmtree(broken)
    info = {s["id"]: s for s in client.get("/api/status").get_json()["skus"]}[SKU]
    assert info["model_error"] is None


def test_current_written_by_windows_tools(trained, tmp_path):
    from qc.model import resolve_version_dir

    sku_models = tmp_path / "m"
    shutil.copytree(trained[0], sku_models / trained[0].name)
    for raw in (trained[0].name.encode("utf-16"), b"\xef\xbb\xbf" + trained[0].name.encode() + b"\r\n"):
        (sku_models / "CURRENT").write_bytes(raw)
        assert resolve_version_dir(sku_models).name == trained[0].name


def test_camera_retries_until_it_appears(tmp_path, workspace):
    from serve.app import CameraManager

    folder = tmp_path / "late_camera"
    mgr = CameraManager(f"folder:{folder}", {"fps": 30}, tmp_path)  # not there yet
    try:
        assert mgr.camera is None and mgr.status()["retrying"] is True and mgr.status()["error"]
        folder.mkdir()
        shutil.copy(list_images(workspace / "data" / SKU / "good")[0], folder / "a.jpg")
        deadline = time.time() + 8
        while mgr.camera is None and time.time() < deadline:
            time.sleep(0.2)
        assert mgr.camera is not None and mgr.status()["configured"] is True
    finally:
        mgr.close()


def test_yaml_off_means_no_camera():
    from qc.camera import camera_configured, open_camera

    for spec in (None, False, "off", "none", "", "no"):
        assert camera_configured(spec) is False and open_camera(spec) is None
    assert camera_configured(0) is True and camera_configured("1") is True


# ------------------------------------------------------------------- minor
def test_posts_need_the_station_header(app):
    bare = app.test_client()
    r = bare.post("/api/inspect", data={"sku": SKU})
    assert r.status_code == 403 and r.get_json()["code"] == "missing_header"
    assert bare.get("/api/status").status_code == 200  # reads stay open


def test_station_pin(trained, workspace, tmp_path):
    from serve.app import create_app

    (tmp_path / "config.yaml").write_text(
        f"data_root: {workspace / 'data'}\nmodels_root: {workspace / 'models'}\n"
        f"serve_data: {tmp_path / 'serve'}\nserver:\n  pin: '4321'\ncamera:\n  source: null\n"
        f"skus:\n  {SKU}:\n    display_name: Demo\n", encoding="utf-8")
    c = create_app(tmp_path / "config.yaml", camera=None).test_client()
    assert c.get("/api/status").get_json()["pin_required"] is True
    img = list_images(workspace / "data" / SKU / "good")[0]
    r = c.post("/api/inspect", data={"sku": SKU, "image": _upload(img)}, headers={"X-Station-Pin": "1111"})
    assert r.status_code == 401
    r = c.post("/api/inspect", data={"sku": SKU, "image": _upload(img)}, headers={"X-Station-Pin": "4321"})
    assert r.status_code == 200


def test_feedback_rejects_non_object_bodies(client):
    for body in ([1, 2], "good", 7):
        r = client.post("/api/feedback", json=body)
        assert r.status_code == 400 and r.get_json()["code"] == "bad_body"


def test_relabel_after_evidence_cleanup_keeps_the_copy(client, app, workspace):
    ddir = workspace / "data" / SKU
    good = list_images(ddir / "good")[5]
    iid = client.post("/api/inspect", data={"sku": SKU, "image": _upload(good)},
                      content_type="multipart/form-data").get_json()["id"]
    client.post("/api/feedback", json={"id": iid, "label": "defect", "defect_type": "contamination"})
    st = app.extensions["tulip"]
    with st.audit.tx() as c:
        rel = c.execute("SELECT image_path FROM inspections WHERE id=?", (iid,)).fetchone()[0]
    (st.serve_dir / rel).unlink()
    j = client.post("/api/feedback", json={"id": iid, "label": "defect", "defect_type": "sparse_tuft"}).get_json()
    assert j["saved_to"] and (ddir / "classes" / "sparse_tuft" / f"{iid}.jpg").exists()
    assert not (ddir / "classes" / "contamination" / f"{iid}.jpg").exists()


def test_concurrent_relabels_leave_exactly_one_copy(client, workspace):
    import threading

    ddir = workspace / "data" / SKU
    iid = client.post("/api/inspect", data={"sku": SKU}).get_json()["id"]
    a, b = ("stray_bristle", "missing_tuft")
    threads = [threading.Thread(target=lambda t=t: client.post(
        "/api/feedback", json={"id": iid, "label": "defect", "defect_type": t})) for t in (a, b) * 4]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    copies = [p for p in (ddir / "classes").rglob(f"{iid}.jpg")]
    assert len(copies) == 1


def test_classifier_only_versions_are_complete(trained, workspace):
    from qc.train import retrain_classifier

    vdir, _ = retrain_classifier(workspace / "data" / SKU, workspace / "models" / SKU, device="cpu",
                                 log=lambda *a: None)
    for name in ("splits.json", "report.txt", "global.npy", "classifier.npz"):
        assert (vdir / name).exists(), name


def test_unreadable_images_are_skipped_not_fatal(tmp_path):
    import torch

    from conftest import small_style
    from qc.synth import make_dataset
    from qc.train import TrainParams, train_anomaly

    ddir = tmp_path / "data" / "x"
    make_dataset(ddir, n_good=30, defects_per_type=1, class_per_type=0, stream_good=0, stream_defects_per_type=0,
                 n_empty=0, style=small_style(), log=lambda *a: None)
    (ddir / "good" / "zz_truncated.jpg").write_bytes((ddir / "good" / "good_0000.jpg").read_bytes()[:900])
    torch.manual_seed(0)
    _, rep = train_anomaly(ddir, tmp_path / "models" / "x",
                           TrainParams(weights=None, input_long_side=128, coreset_size=200, min_good=10, device="cpu"),
                           log=lambda *a: None)
    assert any(s["file"] == "zz_truncated.jpg" for s in rep["data"]["skipped"])
    assert any("skipped" in w for w in rep["warnings"])


def test_make_synthetic_force_never_deletes_real_data(tmp_path):
    real = tmp_path / "data" / "zecta"
    (real / "good").mkdir(parents=True)
    (real / "good" / "precious.jpg").write_bytes(b"x")
    (tmp_path / "config.yaml").write_text("data_root: data\nskus: {}\n", encoding="utf-8")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_synthetic.py"), "--sku", "zecta",
                        "--config", str(tmp_path / "config.yaml"), "--force"], capture_output=True, text=True)
    assert r.returncode != 0 and "not a synthetic dataset" in (r.stderr + r.stdout)
    assert (real / "good" / "precious.jpg").exists()


def test_set_version_lists_and_switches(trained, workspace):
    from qc.model import list_versions, resolve_version_dir

    script = [sys.executable, str(ROOT / "scripts" / "set_version.py"), "--sku", SKU,
              "--config", str(workspace / "config.yaml")]
    live = resolve_version_dir(workspace / "models" / SKU).name
    out = subprocess.run(script, capture_output=True, text=True)
    assert out.returncode == 0 and f"* {live}" in out.stdout
    other = [v for v in list_versions(workspace / "models" / SKU) if v != live][0]
    assert subprocess.run(script + ["--use", other], capture_output=True, text=True).returncode == 0
    assert resolve_version_dir(workspace / "models" / SKU).name == other
    bad = subprocess.run(script + ["--use", "nope"], capture_output=True, text=True)
    assert bad.returncode != 0 and "available" in bad.stderr
    subprocess.run(script + ["--use", live], check=True, capture_output=True)


def test_upload_is_inspected_exactly_as_stored(client, app, workspace):
    """A PNG upload is re-encoded first, so the stored JPEG reproduces the score."""
    from qc.model import InspectionModel

    png = io.BytesIO()
    Image.open(list_images(workspace / "data" / SKU / "good")[7]).save(png, format="PNG")
    png.seek(0)
    j = client.post("/api/inspect", data={"sku": SKU, "image": (png, "shot.png")},
                    content_type="multipart/form-data").get_json()
    stored = client.get(f"/api/inspections/{j['id']}/image").data
    model = InspectionModel.load(workspace / "models" / SKU, device="cpu")
    assert model.inspect(load_image(stored))["score"] == pytest.approx(j["score"], abs=2e-4)
    assert np.isfinite(j["score"])


def test_capture_empty_jig_photos(client, workspace):
    before = client.get(f"/api/capture/counts?sku={SKU}").get_json()["empty"]
    j = client.post("/api/capture", data={"sku": SKU, "label": "empty"}).get_json()
    assert j["file"].startswith("empty/") and j["counts"]["empty"] == before + 1
    (workspace / "data" / SKU / j["file"]).unlink()  # keep the shared dataset as generated



class _FreezableSource:
    """A live noisy camera that can stall and repeat its last frame on command."""

    replay = False
    label = "freezable test camera"
    applied: dict = {}

    def __init__(self, img):
        self.base = np.asarray(img, dtype=np.int16)
        self.frozen = None
        self.rng = np.random.default_rng(0)

    def read(self):
        time.sleep(0.03)
        if self.frozen is not None:
            return self.frozen.copy()
        noisy = np.clip(self.base + self.rng.integers(-2, 3, self.base.shape), 0, 255).astype(np.uint8)
        self.last = Image.fromarray(noisy)
        return self.last

    def freeze(self):
        self.frozen = self.last

    def reopen(self):
        pass

    def close(self):
        pass


def test_a_feed_that_froze_a_moment_ago_is_never_judged(client, app, workspace):
    from qc.camera import Camera

    mgr = app.extensions["tulip"].cams
    original = mgr.camera
    src = _FreezableSource(load_image(list_images(workspace / "data" / SKU / "good")[0]))
    mgr.camera = Camera(src, fps=30)
    try:
        time.sleep(0.3)
        assert client.post("/api/inspect", data={"sku": SKU}).status_code == 200  # live feed: fine
        src.freeze()  # the webcam app stalls; the next brush is placed
        time.sleep(0.15)  # let a read already in flight land
        r = client.post("/api/inspect", data={"sku": SKU})  # well inside the 2 s frozen window
        assert r.status_code == 503 and r.get_json()["code"] == "camera_frozen"
    finally:
        mgr.camera.close()
        mgr.camera = original


def test_a_failed_version_is_retried_without_a_restart(client, workspace, monkeypatch):
    from qc.model import resolve_version_dir, set_current
    from serve.app import ModelRegistry

    monkeypatch.setattr(ModelRegistry, "RETRY_S", 0.5)
    sku_models = workspace / "models" / SKU
    good = resolve_version_dir(sku_models)
    assert client.post("/api/inspect", data={"sku": SKU}).status_code == 200
    copy = sku_models / "late-copy"
    shutil.copytree(good, copy)
    meta = json.loads((copy / "meta.json").read_text())
    (copy / "meta.json").write_text(json.dumps(dict(meta, version="late-copy")))
    (copy / "extractor.pt").rename(copy / "extractor.pt.part")  # copy not finished yet
    set_current(sku_models, copy.name)
    try:
        assert client.post("/api/inspect", data={"sku": SKU}).get_json()["model_version"] == good.name
        info = {s["id"]: s for s in client.get("/api/status").get_json()["skus"]}[SKU]
        assert "still serving" in info["model_error"]
        (copy / "extractor.pt.part").rename(copy / "extractor.pt")  # copy finishes
        time.sleep(0.6)
        assert client.post("/api/inspect", data={"sku": SKU}).get_json()["model_version"] == "late-copy"
        info = {s["id"]: s for s in client.get("/api/status").get_json()["skus"]}[SKU]
        assert info["model_error"] is None
    finally:
        set_current(sku_models, good.name)
        client.get("/api/status")
        shutil.rmtree(copy)


def test_unquoted_numeric_pin_is_refused(tmp_path):
    from qc.config import ConfigError, load_config

    (tmp_path / "config.yaml").write_text("server:\n  pin: 0123\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="quoted"):
        load_config(tmp_path / "config.yaml")
    (tmp_path / "config.yaml").write_text("server:\n  pin: '0123'\n", encoding="utf-8")
    assert load_config(tmp_path / "config.yaml")["server"]["pin"] == "0123"


def test_recognition_robustness_is_measured_on_held_out_goods(trained):
    # with random test weights only the bookkeeping is meaningful; the real numbers come
    # from training with ImageNet weights (see the README's demo results)
    rec = trained[1]["recognition"]
    assert rec["quality_variants"] == 6 * trained[1]["data"]["test_good"]
    assert 0 <= rec["quality_variants_refused"] <= rec["quality_variants"] and rec["empty_passed"] == 0
