"""Anomaly math, numpy only: coreset memory bank, scoring, thresholds, metrics.

The method is PatchCore (Roth et al., CVPR 2022), reduced to what a single
inspection station needs:

1. Every *good* brush image is turned into a grid of patch features by a
   frozen ImageNet backbone (qc/features.py). Each patch vector describes
   the local texture and shape around one small region of the brush head.
2. Those patches are compressed into a memory bank with greedy k-center
   selection, which keeps the patches that *cover* the variety of good
   production instead of oversampling its most common look.
3. A new brush is scored patch by patch: distance to the nearest memory
   patch. A brush that looks like good production is close everywhere; a
   missing tuft or a stray bristle produces at least one patch that is far
   from anything good. The image score is the worst patch.

Nothing here imports torch, so the scoring math is unit-testable on its own.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "kcenter_greedy",
    "MemoryBank",
    "threshold_from_goods",
    "threshold_from_gap",
    "auroc",
    "rates_at",
    "score_summary",
]


def _sqnorm(x: np.ndarray) -> np.ndarray:
    return np.einsum("ij,ij->i", x, x)


# --------------------------------------------------------------------- coreset
def kcenter_greedy(
    x: np.ndarray,
    n_select: int,
    *,
    proj_dim: int | None = 128,
    seed: int = 0,
) -> np.ndarray:
    """Greedy k-center coreset: indices of ``n_select`` rows of ``x`` [N, D].

    Each step adds the point farthest from everything already selected, so
    the selection covers the whole manifold of good patches. As in the
    PatchCore reference implementation, selection runs on a random
    projection to ``proj_dim`` dimensions (distances are approximately
    preserved, Johnson-Lindenstrauss) which makes it several times faster.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    n, d = x.shape
    if n_select >= n:
        return np.arange(n)
    if n_select < 1:
        raise ValueError("n_select must be >= 1")
    rng = np.random.default_rng(seed)
    if proj_dim and d > proj_dim:
        proj = rng.standard_normal((d, proj_dim)).astype(np.float32)
        proj /= np.sqrt(proj_dim, dtype=np.float32)
        x = x @ proj
    sq = _sqnorm(x)
    selected = np.empty(n_select, dtype=np.int64)
    start = int(rng.integers(n))
    selected[0] = start
    # squared distance from every point to its nearest selected point
    mind = sq + sq[start] - 2.0 * (x @ x[start])
    mind[start] = -np.inf
    for k in range(1, n_select):
        i = int(np.argmax(mind))
        selected[k] = i
        np.minimum(mind, sq + sq[i] - 2.0 * (x @ x[i]), out=mind)
        mind[i] = -np.inf  # never pick the same row twice, even with duplicates
    return selected


# ---------------------------------------------------------------- memory bank
class MemoryBank:
    """Coreset of good-patch features with nearest-neighbour scoring."""

    def __init__(self, memory: np.ndarray):
        mem = np.ascontiguousarray(memory, dtype=np.float32)
        if mem.ndim != 2 or mem.shape[0] == 0:
            raise ValueError("memory must be a non-empty [M, D] array")
        self.memory = mem
        self._sq = _sqnorm(mem)

    @property
    def size(self) -> int:
        return int(self.memory.shape[0])

    @property
    def dim(self) -> int:
        return int(self.memory.shape[1])

    @classmethod
    def fit(cls, patches: np.ndarray, *, coreset_size: int = 4000, seed: int = 0) -> "MemoryBank":
        """Select the coreset from pooled good patches [N, D]."""
        patches = np.ascontiguousarray(patches, dtype=np.float32)
        idx = kcenter_greedy(patches, min(coreset_size, patches.shape[0]), seed=seed)
        return cls(patches[idx])

    def nn_distances(self, feats: np.ndarray, chunk: int = 2048) -> np.ndarray:
        """Euclidean distance from each patch [P, D] to its nearest memory patch."""
        f = np.ascontiguousarray(feats, dtype=np.float32)
        if f.ndim != 2 or f.shape[1] != self.dim:
            raise ValueError(f"expected [P, {self.dim}] features, got {f.shape}")
        out = np.empty(f.shape[0], dtype=np.float32)
        for s in range(0, f.shape[0], chunk):
            blk = f[s : s + chunk]
            d2 = _sqnorm(blk)[:, None] + self._sq[None, :] - 2.0 * (blk @ self.memory.T)
            out[s : s + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
        return out

    def score(self, feats: np.ndarray) -> float:
        """Image anomaly score: the single worst patch."""
        return float(self.nn_distances(feats).max())


# ----------------------------------------------------------------- thresholds
def threshold_from_goods(cal_good_scores, quantile: float = 0.995, margin: float = 1.08) -> float:
    """Decision threshold from calibration goods only.

    A high quantile of good scores, widened by ``margin``. Defects are not
    used, which keeps the catch rate measured on them an
    out-of-sample number.
    """
    s = np.asarray(list(cal_good_scores), dtype=np.float64)
    if s.size == 0:
        raise ValueError("need at least one calibration score")
    if not (0.5 <= quantile <= 1.0):
        raise ValueError("quantile must be in [0.5, 1.0]")
    return float(np.quantile(s, quantile) * margin)


def threshold_from_gap(good_scores, defect_scores) -> float | None:
    """Midpoint of the gap between the worst good and the mildest defect.

    Returns None when the two overlap (no clean gap exists). Using this
    policy spends the defect examples on choosing the threshold, so any
    catch rate reported on those same defects is in-sample.
    """
    g = np.asarray(list(good_scores), dtype=np.float64)
    d = np.asarray(list(defect_scores), dtype=np.float64)
    if g.size == 0 or d.size == 0:
        return None
    hi, lo = float(g.max()), float(d.min())
    return (hi + lo) / 2.0 if lo > hi else None


# -------------------------------------------------------------------- metrics
def _rankdata(a: np.ndarray) -> np.ndarray:
    """1-based ranks with ties averaged (scipy.stats.rankdata, 'average')."""
    a = np.asarray(a, dtype=np.float64)
    n = a.size
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    first = np.r_[True, sa[1:] != sa[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], n]
    avg = (starts + ends - 1) / 2.0 + 1.0
    group = np.cumsum(first) - 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg[group]
    return ranks


def auroc(good_scores, defect_scores) -> float | None:
    """Probability that a random defect scores above a random good brush.

    1.0 = perfect separation, 0.5 = coin flip. Threshold-free, so it
    measures the model itself rather than the threshold choice.
    """
    g = np.asarray(list(good_scores), dtype=np.float64)
    d = np.asarray(list(defect_scores), dtype=np.float64)
    if g.size == 0 or d.size == 0:
        return None
    ranks = _rankdata(np.concatenate([g, d]))
    u = ranks[g.size :].sum() - d.size * (d.size + 1) / 2.0
    return float(u / (g.size * d.size))


def rates_at(threshold: float, good_scores, defect_scores) -> dict:
    """False-reject rate on goods and catch rate on defects at ``threshold``."""
    g = np.asarray(list(good_scores), dtype=np.float64)
    d = np.asarray(list(defect_scores), dtype=np.float64)
    fr = int((g >= threshold).sum())
    caught = int((d >= threshold).sum())
    return {
        "good": int(g.size),
        "false_rejects": fr,
        "false_reject_rate": (fr / g.size) if g.size else None,
        "defects": int(d.size),
        "caught": caught,
        "catch_rate": (caught / d.size) if d.size else None,
    }


def score_summary(scores) -> dict | None:
    s = np.asarray(list(scores), dtype=np.float64)
    if s.size == 0:
        return None
    return {
        "n": int(s.size),
        "min": float(s.min()),
        "median": float(np.median(s)),
        "p95": float(np.quantile(s, 0.95)),
        "max": float(s.max()),
    }
