import base64
import datetime as dt
import io
import json
import sqlite3
import time

import pytest
from PIL import Image

from conftest import SKU
from qc.data import list_images


@pytest.fixture(scope="module")
def app(trained, workspace):
    from serve.app import create_app

    a = create_app(workspace / "config.yaml", camera=f"folder:{workspace / 'data' / SKU / 'stream'}")
    yield a
    cam = a.extensions["tulip"].camera
    if cam:
        cam.close()


@pytest.fixture()
def client(app):
    c = app.test_client()
    c.environ_base["HTTP_X_STATION_PIN"] = "-"
    return c


def _jpeg(path):
    return (io.BytesIO(path.read_bytes()), "shot.jpg")


def test_status_lists_models_and_camera(client, trained):
    from qc.model import resolve_version_dir

    j = client.get("/api/status").get_json()
    by_id = {s["id"]: s for s in j["skus"]}
    live = resolve_version_dir(trained[0].parent).name  # other test modules may have retrained
    assert by_id[SKU]["model"]["version"] == live
    assert by_id[SKU]["model"]["has_classifier"] is True
    assert by_id["untrained"]["model"] is None and "train" in by_id["untrained"]["model_error"]
    assert j["camera"]["replay"] is True and j["camera"]["ok"] is True
    assert any(t["key"] == "missing_tuft" and t["gu"] for t in by_id[SKU]["defect_types"])


def test_index_and_preview(client):
    html = client.get("/").get_data(as_text=True)
    assert "TULIP VISION" in html and f'value="{SKU}"' in html
    r = client.get("/api/camera/frame.jpg")
    assert r.status_code == 200 and r.mimetype == "image/jpeg"


def test_inspect_from_camera_and_upload(client, workspace):
    r = client.post("/api/inspect", data={"sku": SKU})
    assert r.status_code == 200, r.get_json()
    j = r.get_json()
    assert j["source"] == "camera" and j["verdict"] in ("pass", "reject")
    Image.open(io.BytesIO(base64.b64decode(j["heatmap"]))).verify()
    defect = list_images(workspace / "data" / SKU / "defect", recursive=True)[0]
    j2 = client.post("/api/inspect", data={"sku": SKU, "image": _jpeg(defect)},
                     content_type="multipart/form-data").get_json()
    assert j2["source"] == "upload" and j2["id"] != j["id"]
    row = client.get(f"/api/recent?sku={SKU}&limit=1").get_json()[0]
    assert row["id"] == j2["id"] and row["model_version"] == j2["model_version"]
    assert client.get(f"/api/inspections/{j2['id']}/image").status_code == 200


@pytest.mark.parametrize("data,status,code", [
    ({"sku": "nope"}, 400, "unknown_sku"),
    ({"sku": "untrained"}, 503, "no_model"),
    ({"sku": SKU, "image": (io.BytesIO(b"not an image"), "x.jpg")}, 400, "bad_image"),
])
def test_inspect_errors(client, data, status, code):
    r = client.post("/api/inspect", data=data, content_type="multipart/form-data")
    assert r.status_code == status and r.get_json()["code"] == code


def test_feedback_routes_images_into_the_dataset(client, workspace):
    ddir = workspace / "data" / SKU
    insp = client.post("/api/inspect", data={"sku": SKU}).get_json()
    iid = insp["id"]

    r = client.post("/api/feedback", json={"id": iid, "label": "defect", "defect_type": "stray_bristle"})
    assert r.status_code == 200
    assert (ddir / "defect" / "stray_bristle" / f"{iid}.jpg").exists()
    assert (ddir / "classes" / "stray_bristle" / f"{iid}.jpg").exists()

    # the operator changes their mind: old copies go, new ones appear
    r = client.post("/api/feedback", json={"id": iid, "label": "good"}).get_json()
    assert not (ddir / "classes" / "stray_bristle" / f"{iid}.jpg").exists()
    assert (ddir / "feedback" / "good" / f"{iid}.jpg").exists()
    assert ((ddir / "classes" / "good" / f"{iid}.jpg").exists()) == (insp["verdict"] == "reject")
    assert r["model_agreed"] == (insp["verdict"] == "pass")


@pytest.mark.parametrize("payload,status", [
    ({"id": "0123456789ab", "label": "good"}, 404),
    ({"id": "../../etc/pass", "label": "good"}, 400),
    ({"label": "good"}, 400),
])
def test_feedback_rejects_bad_ids(client, payload, status):
    assert client.post("/api/feedback", json=payload).status_code == status


def test_feedback_rejects_bad_label_and_type(client):
    iid = client.post("/api/inspect", data={"sku": SKU}).get_json()["id"]
    assert client.post("/api/feedback", json={"id": iid, "label": "meh"}).status_code == 400
    r = client.post("/api/feedback", json={"id": iid, "label": "defect", "defect_type": "uneven_trim"})
    assert r.status_code == 400 and r.get_json()["code"] == "bad_defect_type"


def test_capture_saves_into_dataset(client, workspace):
    ddir = workspace / "data" / SKU
    before = client.get(f"/api/capture/counts?sku={SKU}").get_json()
    j = client.post("/api/capture", data={"sku": SKU, "label": "good"}).get_json()
    assert j["ok"] and (ddir / j["file"]).exists() and j["counts"]["good"] == before["good"] + 1
    j = client.post("/api/capture", data={"sku": SKU, "label": "defect", "defect_type": "contamination"}).get_json()
    assert j["file"].startswith("defect/contamination/")
    log = (ddir / "capture_log.jsonl").read_text().strip().splitlines()
    assert json.loads(log[-1])["defect_type"] == "contamination"


def test_stats_count_today(client):
    j = client.get(f"/api/stats?sku={SKU}").get_json()
    assert j["inspections"] >= 4 and j["operator_labels"] >= 1
    assert 0.0 <= j["reject_rate"] <= 1.0
    assert client.get("/api/stats?sku=nope").status_code == 400


def test_hot_reload_after_retraining(client, workspace, trained):
    from qc.train import retrain_classifier

    new_dir, _ = retrain_classifier(workspace / "data" / SKU, workspace / "models" / SKU, device="cpu",
                                    log=lambda *a: None)
    j = client.get("/api/status").get_json()
    live = {s["id"]: s for s in j["skus"]}[SKU]["model"]["version"]
    assert live == new_dir.name != trained[0].name
    j = client.post("/api/inspect", data={"sku": SKU}).get_json()
    assert j["model_version"] == new_dir.name


def test_upload_only_station(trained, workspace):
    from serve.app import create_app

    c = create_app(workspace / "config.yaml", camera=None).test_client()
    c.environ_base["HTTP_X_STATION_PIN"] = "-"
    assert c.get("/api/status").get_json()["camera"]["configured"] is False
    r = c.post("/api/inspect", data={"sku": SKU})
    assert r.status_code == 400 and r.get_json()["code"] == "no_image"
    assert c.get("/api/camera/frame.jpg").status_code == 404


def test_v1_database_is_migrated(tmp_path, trained, workspace):
    from serve.app import AuditLog

    db = tmp_path / "inspections.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE inspections (id TEXT PRIMARY KEY, ts TEXT NOT NULL, sku TEXT NOT NULL,"
                " score REAL NOT NULL, threshold REAL NOT NULL, verdict TEXT NOT NULL, operator_label TEXT,"
                " image_path TEXT NOT NULL, latency_ms REAL)")
    con.execute("INSERT INTO inspections VALUES ('abc', '2026-07-17T10:00:00', 'x', 1, 2, 'pass', NULL, 'p', 5)")
    con.commit()
    con.close()
    AuditLog(db)
    con = sqlite3.connect(db)
    cols = {r[1] for r in con.execute("PRAGMA table_info(inspections)")}
    assert {"model_version", "defect_guess", "operator_defect_type", "labelled_at"} <= cols
    assert con.execute("SELECT verdict FROM inspections WHERE id='abc'").fetchone()[0] == "pass"
    con.close()


def test_evidence_cleanup(tmp_path):
    from serve.app import _cleanup

    old = tmp_path / "demo" / "20200101"
    new = tmp_path / "demo" / dt.date.today().strftime("%Y%m%d")
    for d in (old, new):
        d.mkdir(parents=True)
        (d / "a.jpg").write_bytes(b"x")
    _cleanup(tmp_path, keep_days=30)
    assert not old.exists() and new.exists()


def test_replay_camera_advances(workspace):
    from qc.camera import open_camera

    cam = open_camera(f"folder:{workspace / 'data' / SKU / 'stream'}", {"fps": 30})
    try:
        first = cam.source.current_path
        cam.advance(0.0)
        assert cam.source.current_path != first
        cam.advance(0.15)
        second = cam.source.current_path
        time.sleep(0.4)
        assert cam.source.current_path != second
        frame, ts = cam.latest(newer_than=time.time(), timeout=1.0)
        assert frame is not None and ts is not None
    finally:
        cam.close()
