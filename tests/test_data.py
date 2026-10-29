from pathlib import Path

import pytest
from PIL import Image

from qc.data import (
    check_roi,
    crop_roi,
    defect_groups,
    load_image,
    parse_roi,
    read_roi_file,
    roi_box_px,
    split_goods,
    write_roi_file,
)


def _names(n, prefix="20261030_0900"):
    return [Path(f"/data/good/{prefix}{i:05d}.jpg") for i in range(n)]


def test_split_is_deterministic_disjoint_and_complete():
    paths = _names(300)
    a, b = split_goods(paths), split_goods(list(reversed(paths)))
    assert a == b
    sets = [set(a[k]) for k in ("train", "cal", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set().union(*sets) == set(paths)
    assert 25 <= len(a["test"]) <= 65 and 25 <= len(a["cal"]) <= 65


def test_split_is_stable_when_images_are_added():
    old = split_goods(_names(300))
    new = split_goods(_names(300) + _names(80, prefix="20261031_1000"))
    for part in ("train", "cal", "test"):
        assert set(old[part]) <= set(new[part]), f"{part}: adding images moved existing ones"


def test_split_guarantees_minimum_holdouts():
    s = split_goods(_names(20), min_holdout=5)
    assert len(s["cal"]) >= 5 and len(s["test"]) >= 5 and len(s["train"]) >= 1


def test_roi_validation_and_crop():
    img = Image.new("RGB", (100, 50))
    roi = (0.1, 0.2, 0.5, 0.8)
    assert roi_box_px(img.size, roi) == (10, 10, 50, 40)
    assert crop_roi(img, roi).size == (40, 30)
    assert crop_roi(img, None).size == (100, 50)
    assert parse_roi("0.1, 0.2,0.5,0.8") == roi
    for bad in [(0.5, 0.2, 0.1, 0.8), (0, 0, 1.2, 1), (0.1, 0.1, 0.11, 0.9), (0.1, 0.2, 0.3)]:
        with pytest.raises(ValueError):
            check_roi(bad)


def test_roi_file_roundtrip(tmp_path):
    write_roi_file(tmp_path, (0.1, 0.2, 0.5, 0.8), (1280, 720))
    assert read_roi_file(tmp_path) == (0.1, 0.2, 0.5, 0.8)
    assert read_roi_file(tmp_path / "nothing") is None


def test_exif_orientation_is_applied(tmp_path):
    img = Image.new("RGB", (40, 20), (200, 10, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # 'rotate 90 CW to display' — what phones write when held upright
    p = tmp_path / "phone.jpg"
    img.save(p, exif=exif.tobytes())
    assert Image.open(p).size == (40, 20)
    assert load_image(p).size == (20, 40)
    assert load_image(p.read_bytes()).size == (20, 40)


def test_defect_groups(tmp_path):
    d = tmp_path / "defect"
    (d / "missing_tuft").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(d / "loose.jpg")
    Image.new("RGB", (8, 8)).save(d / "missing_tuft" / "a.jpg")
    (d / "empty_type").mkdir()
    g = defect_groups(d)
    assert set(g) == {"unlabelled", "missing_tuft"}
    assert defect_groups(tmp_path / "absent") == {}


def test_atomic_replace_retries_while_target_is_busy(tmp_path, monkeypatch):
    import os

    from qc import data

    src, dst = tmp_path / "a.tmp", tmp_path / "a.json"
    src.write_text("new")
    dst.write_text("old")
    real, calls = os.replace, {"n": 0}

    def flaky(a, b):  # Windows: a reader holds the file open for a moment
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("in use")
        real(a, b)

    monkeypatch.setattr(data.os, "replace", flaky)
    data.atomic_replace(src, dst, delay=0.001)
    assert dst.read_text() == "new" and calls["n"] == 3
