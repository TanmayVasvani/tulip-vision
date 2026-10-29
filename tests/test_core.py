import itertools

import numpy as np
import pytest

from qc.core import (
    MemoryBank,
    auroc,
    kcenter_greedy,
    rates_at,
    threshold_from_gap,
    threshold_from_goods,
)


def test_kcenter_unique_and_reaches_outlier():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 64)).astype(np.float32)
    x[1234] += 40.0  # one far-away patch
    idx = kcenter_greedy(x, 50, seed=1)
    assert len(set(idx.tolist())) == 50
    assert 1234 in idx[:3], "farthest-point selection must pick the outlier almost immediately"


def test_kcenter_handles_duplicates_and_small_inputs():
    x = np.zeros((10, 4), dtype=np.float32)  # all identical
    idx = kcenter_greedy(x, 6, proj_dim=None)
    assert len(set(idx.tolist())) == 6
    assert kcenter_greedy(x, 50).tolist() == list(range(10))


def test_nn_distances_match_brute_force():
    rng = np.random.default_rng(2)
    mem = rng.normal(size=(300, 32)).astype(np.float32)
    q = rng.normal(size=(500, 32)).astype(np.float32)
    bank = MemoryBank(mem)
    got = bank.nn_distances(q, chunk=128)
    want = np.sqrt(((q[:, None, :] - mem[None, :, :]) ** 2).sum(-1)).min(1)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


def test_bank_rejects_wrong_dim():
    bank = MemoryBank(np.ones((5, 8), np.float32))
    with pytest.raises(ValueError):
        bank.nn_distances(np.ones((3, 7), np.float32))


def test_planted_anomalies_separate_and_threshold_holds():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(400, 24, 64)).astype(np.float32)
    bank = MemoryBank.fit(good[:300].reshape(-1, 64), coreset_size=600)
    cal = [bank.score(im) for im in good[300:350]]
    test = [bank.score(im) for im in good[350:]]
    bad = good[:20].copy()
    bad[:, :3] += 4.0
    defects = [bank.score(im) for im in bad]
    thr = threshold_from_goods(cal)
    r = rates_at(thr, test, defects)
    assert r["catch_rate"] == 1.0
    assert r["false_reject_rate"] <= 0.04
    assert auroc(test, defects) == 1.0


def test_threshold_policies():
    assert threshold_from_goods([1.0] * 10, quantile=0.995, margin=1.1) == pytest.approx(1.1)
    assert threshold_from_gap([1, 2, 3], [5, 6]) == pytest.approx(4.0)
    assert threshold_from_gap([1, 2, 6], [5, 6]) is None
    with pytest.raises(ValueError):
        threshold_from_goods([])


def test_auroc_matches_pairwise_definition_with_ties():
    rng = np.random.default_rng(5)
    g = rng.integers(0, 6, 40).astype(float)
    d = rng.integers(2, 9, 30).astype(float)
    pairs = [(1.0 if b > a else 0.5 if b == a else 0.0) for a, b in itertools.product(g, d)]
    assert auroc(g, d) == pytest.approx(np.mean(pairs))
    assert auroc([1, 2], [3, 4]) == 1.0
    assert auroc([3, 4], [1, 2]) == 0.0
    assert auroc([], [1]) is None


def test_rates_at():
    r = rates_at(2.0, [1, 2, 3], [1, 5])
    assert r["false_rejects"] == 2 and r["caught"] == 1
    assert r["false_reject_rate"] == pytest.approx(2 / 3) and r["catch_rate"] == 0.5
