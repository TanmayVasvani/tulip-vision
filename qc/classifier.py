"""Name the defect, not just flag it."""

The anomaly model answers "is this brush wrong?". A rework station needs
"what is wrong". Labelled examples arrive slowly (operator feedback), tens
per defect type at first, so fine-tuning a network would overfit. Instead
this is a linear classifier on the frozen features the anomaly model
already computes:

    descriptor = [mean of the k most anomalous patches,  mean of all patches]

The first half describes the defect region itself (wherever the heatmap
peaks); the second half gives context. Multinomial logistic regression
with L2, trained by accelerated gradient descent. Accuracy is measured by
stratified k-fold cross-validation, so the reported number is out-of-sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import atomic_replace

TOPK = 8
L2_GRID = (0.01, 0.1, 1.0)


def descriptor(feats: np.ndarray, dists: np.ndarray, topk: int = TOPK) -> np.ndarray:
    feats = np.asarray(feats, dtype=np.float32)
    dists = np.asarray(dists, dtype=np.float32)
    k = int(min(topk, dists.shape[0]))
    idx = np.argpartition(dists, -k)[-k:]
    return np.concatenate([feats[idx].mean(axis=0), feats.mean(axis=0)]).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _standardizer(x: np.ndarray):
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.maximum(sd, 0.01 * float(np.median(sd)) + 1e-12)
    return mu, sd


def fit_softmax(x: np.ndarray, y: np.ndarray, n_classes: int, l2: float = 0.1, iters: int = 800):
    """Multinomial logistic regression on standardised x. Returns (W [D, C], b [C])."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    xa = np.hstack([x, np.ones((n, 1))])
    yy = np.eye(n_classes)[y]
    # step size from the Lipschitz constant: 0.5 * lambda_max(Xa'Xa / n) + l2
    v = np.ones(d + 1) / np.sqrt(d + 1)
    for _ in range(60):
        v = xa.T @ (xa @ v) / n
        v /= np.linalg.norm(v) + 1e-12
    lam = float(v @ (xa.T @ (xa @ v)) / n)
    step = 1.0 / (0.5 * lam + l2)
    reg = np.ones((d + 1, 1))
    reg[-1] = 0.0  # bias is not regularised
    w = np.zeros((d + 1, n_classes))
    z = w.copy()
    t = 1.0
    for _ in range(iters):
        grad = xa.T @ (_softmax(xa @ z) - yy) / n + l2 * reg * z
        w_next = z - step * grad
        t_next = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_next + ((t - 1.0) / t_next) * (w_next - w)
        w, t = w_next, t_next
    return w[:-1].astype(np.float32), w[-1].astype(np.float32)


def stratified_folds(y: np.ndarray, k: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    offset = 0
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        for i, j in enumerate(idx):
            folds[(i + offset) % k].append(int(j))
        offset += len(idx)
    return [np.asarray(sorted(f), dtype=np.int64) for f in folds]


def _cv_predict(x, y, n_classes, l2, k, seed):
    pred = np.empty_like(y)
    for test in stratified_folds(y, k, seed):
        train = np.setdiff1d(np.arange(len(y)), test)
        mu, sd = _standardizer(x[train])
        w, b = fit_softmax((x[train] - mu) / sd, y[train], n_classes, l2=l2)
        pred[test] = np.argmax(((x[test] - mu) / sd) @ w + b, axis=1)
    return pred


class DefectClassifier:
    def __init__(self, classes, w, b, mu, sd, topk: int = TOPK, report: dict | None = None):
        self.classes = [str(c) for c in classes]
        self.w = np.asarray(w, dtype=np.float32)
        self.b = np.asarray(b, dtype=np.float32)
        self.mu = np.asarray(mu, dtype=np.float32)
        self.sd = np.asarray(sd, dtype=np.float32)
        self.topk = int(topk)
        self.report = report or {}

    def predict_proba(self, desc: np.ndarray) -> np.ndarray:
        d = np.atleast_2d(np.asarray(desc, dtype=np.float32))
        return _softmax(((d - self.mu) / self.sd) @ self.w + self.b)

    def predict(self, feats: np.ndarray, dists: np.ndarray) -> dict:
        p = self.predict_proba(descriptor(feats, dists, self.topk))[0]
        order = np.argsort(p)[::-1]
        return {
            "label": self.classes[int(order[0])],
            "confidence": float(p[order[0]]),
            "ranked": [(self.classes[int(i)], float(p[i])) for i in order[:3]],
        }

    # ------------------------------------------------------------ persist
    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            classes=np.array(self.classes),
            w=self.w,
            b=self.b,
            mu=self.mu,
            sd=self.sd,
            topk=np.array(self.topk),
            report=np.array(json.dumps(self.report)),
        )
        atomic_replace(tmp, path)

    @classmethod
    def load(cls, path) -> "DefectClassifier":
        with np.load(path, allow_pickle=False) as z:
            return cls(
                [str(c) for c in z["classes"]],
                z["w"],
                z["b"],
                z["mu"],
                z["sd"],
                int(z["topk"]),
                json.loads(str(z["report"])),
            )


def _select_l2(x, y, n_classes, k, seed) -> float:
    """L2 strength with the best k-fold accuracy (ties go to stronger regularisation)."""
    best = None
    for l2 in L2_GRID:
        acc = float((_cv_predict(x, y, n_classes, l2, k, seed) == y).mean())
        if best is None or (acc, l2) > best:
            best = (acc, l2)
    return best[1]


def train_defect_classifier(descs_by_label: dict, seed: int = 0):
    """{label: [descriptor, ...]} -> (DefectClassifier, report with CV metrics).

    Accuracy comes from nested cross-validation: the regularisation strength
    is chosen inside each outer training fold, so the reported number is
    out-of-sample.
    """
    labels = sorted(descs_by_label)
    if len(labels) < 2:
        raise ValueError("need at least two classes to train a classifier")
    x = np.concatenate([np.stack(descs_by_label[c]) for c in labels]).astype(np.float64)
    y = np.concatenate([np.full(len(descs_by_label[c]), i) for i, c in enumerate(labels)])
    n_classes = len(labels)
    k = int(min(5, min(len(descs_by_label[c]) for c in labels)))

    cv = None
    if k >= 2:
        pred = np.empty_like(y)
        for test in stratified_folds(y, k, seed):
            train = np.setdiff1d(np.arange(len(y)), test)
            k_in = int(min(k - 1, np.bincount(y[train], minlength=n_classes).min()))
            l2 = _select_l2(x[train], y[train], n_classes, k_in, seed + 1) if k_in >= 2 else L2_GRID[1]
            mu, sd = _standardizer(x[train])
            w, b = fit_softmax((x[train] - mu) / sd, y[train], n_classes, l2=l2)
            pred[test] = np.argmax(((x[test] - mu) / sd) @ w + b, axis=1)
        conf = np.zeros((n_classes, n_classes), dtype=int)
        np.add.at(conf, (y, pred), 1)
        cv = {
            "folds": k,
            "accuracy": float((pred == y).mean()),
            "per_class_recall": {labels[i]: float(conf[i, i] / conf[i].sum()) for i in range(n_classes)},
            "confusion": conf.tolist(),
            "note": f"nested stratified {k}-fold CV",
        }
        best_l2 = _select_l2(x, y, n_classes, k, seed)
    else:
        best_l2 = L2_GRID[1]

    mu, sd = _standardizer(x)
    w, b = fit_softmax((x - mu) / sd, y, n_classes, l2=best_l2)
    report = {
        "classes": labels,
        "counts": {c: len(descs_by_label[c]) for c in labels},
        "l2": best_l2,
        "cv": cv,
    }
    return DefectClassifier(labels, w, b, mu, sd, TOPK, report), report
