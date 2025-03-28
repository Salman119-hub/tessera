"""Binary classification metrics in numpy, with exact tie handling.

Fraud labels are heavily imbalanced, so ROC AUC alone flatters models.
Average precision and recall at a fixed false-positive rate describe what an
operations team actually experiences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _arrays(y_true: ArrayLike, scores: ArrayLike) -> tuple[FloatArray, FloatArray]:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y_true and scores must be 1-D arrays of equal length")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true must be binary 0/1")
    return y, s


def rankdata(x: FloatArray) -> FloatArray:
    """Average ranks (1-based), matching ``scipy.stats.rankdata(method='average')``."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, weights=ranks)
    return (sums / counts)[inverse]


def roc_auc(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Mann-Whitney formulation; tied scores count as half."""
    y, s = _arrays(y_true, scores)
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("ROC AUC needs both classes")
    ranks = rankdata(s)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _curve(y: FloatArray, s: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cumulative (tp, fp, threshold) at each distinct score, highest first."""
    order = np.argsort(-s, kind="mergesort")
    y_sorted, s_sorted = y[order], s[order]
    distinct = np.where(np.diff(s_sorted))[0]
    last = np.r_[distinct, len(y_sorted) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    return tp, fp, s_sorted[last]


def average_precision(y_true: ArrayLike, scores: ArrayLike) -> float:
    """Area under the precision-recall curve (step-wise, as in scikit-learn)."""
    y, s = _arrays(y_true, scores)
    n_pos = y.sum()
    if n_pos == 0:
        raise ValueError("average precision needs at least one positive")
    tp, fp, _ = _curve(y, s)
    precision = tp / (tp + fp)
    recall = tp / n_pos
    previous = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - previous) * precision))


def recall_at_fpr(y_true: ArrayLike, scores: ArrayLike, max_fpr: float) -> float:
    y, s = _arrays(y_true, scores)
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    tp, fp, _ = _curve(y, s)
    ok = fp / n_neg <= max_fpr
    return float(np.max(tp[ok] / n_pos)) if ok.any() else 0.0


def log_loss(y_true: ArrayLike, probs: ArrayLike, eps: float = 1e-12) -> float:
    y, p = _arrays(y_true, probs)
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(y_true: ArrayLike, probs: ArrayLike) -> float:
    y, p = _arrays(y_true, probs)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y_true: ArrayLike, probs: ArrayLike, bins: int = 10) -> float:
    """Weighted mean |observed rate - mean predicted| over equal-width probability bins."""
    y, p = _arrays(y_true, probs)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(y[mask].mean() - p[mask].mean())
    return float(total / len(y))


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    expected_cost: float
    precision: float
    recall: float
    flagged_rate: float


def choose_threshold(
    y_true: ArrayLike, probs: ArrayLike, *, cost_fp: float, cost_fn: float
) -> ThresholdChoice:
    """Threshold minimising ``cost_fp * FP + cost_fn * FN``; flag when ``score >= threshold``."""
    y, p = _arrays(y_true, probs)
    n_pos = y.sum()
    tp, fp, thresholds = _curve(y, p)
    fn = n_pos - tp
    costs = cost_fp * fp + cost_fn * fn
    # "flag nothing" is also a candidate
    none_cost = cost_fn * n_pos
    best = int(np.argmin(costs))
    if none_cost < costs[best]:
        return ThresholdChoice(
            float(np.nextafter(float(np.max(p)), np.inf)), float(none_cost), 0.0, 0.0, 0.0
        )
    flagged = tp[best] + fp[best]
    return ThresholdChoice(
        threshold=float(thresholds[best]),
        expected_cost=float(costs[best]),
        precision=float(tp[best] / flagged) if flagged else 0.0,
        recall=float(tp[best] / n_pos) if n_pos else 0.0,
        flagged_rate=float(flagged / len(y)),
    )


def classification_report(y_true: ArrayLike, probs: ArrayLike) -> dict[str, float]:
    y, p = _arrays(y_true, probs)
    return {
        "rows": float(len(y)),
        "positive_rate": float(y.mean()),
        "roc_auc": roc_auc(y, p),
        "average_precision": average_precision(y, p),
        "recall_at_1pct_fpr": recall_at_fpr(y, p, 0.01),
        "log_loss": log_loss(y, p),
        "brier": brier(y, p),
        "ece": expected_calibration_error(y, p),
    }
