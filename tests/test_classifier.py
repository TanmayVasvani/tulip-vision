import numpy as np

from qc.classifier import DefectClassifier, descriptor, train_defect_classifier


def test_descriptor_focuses_on_anomalous_patches():
    feats = np.zeros((100, 4), np.float32)
    feats[7] = 10.0
    dists = np.zeros(100, np.float32)
    dists[7] = 5.0
    d = descriptor(feats, dists, topk=1)
    assert d.shape == (8,)
    assert np.allclose(d[:4], 10.0) and np.allclose(d[4:], 0.1)


def test_classifier_learns_separable_classes_and_roundtrips(tmp_path):
    rng = np.random.default_rng(0)
    centres = rng.normal(0, 3, (3, 16))
    data = {f"type_{c}": [centres[c] + rng.normal(0, 1, 16) for _ in range(20)] for c in range(3)}
    clf, rep = train_defect_classifier(data)
    assert rep["cv"]["accuracy"] >= 0.95
    assert sorted(rep["cv"]["per_class_recall"]) == sorted(data)
    probe = np.stack([centres[1]])
    p = clf.predict_proba(probe)
    assert p.shape == (1, 3) and np.isclose(p.sum(), 1.0)
    assert clf.classes[int(p.argmax())] == "type_1"
    path = tmp_path / "clf.npz"
    clf.save(path)
    again = DefectClassifier.load(path)
    np.testing.assert_allclose(again.predict_proba(probe), p, rtol=1e-6)
    assert again.report["cv"]["folds"] == 5


def test_classifier_needs_two_classes():
    import pytest

    with pytest.raises(ValueError):
        train_defect_classifier({"only": [np.zeros(4)] * 6})
