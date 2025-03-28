"""A transparent baseline: preprocessing plus L2-regularised logistic regression.

The model is fit with Newton's method (IRLS), which converges in a handful of
iterations and is deterministic. Everything (weights, imputation values,
scaling and the drift reference) serialises to JSON. Nothing is pickled
(ADR-0005).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

LOG_THRESHOLD = 50.0
DRIFT_BINS = 10


def to_float(values: Sequence[Any] | pd.Series) -> FloatArray:
    out = np.empty(len(values), dtype=np.float64)
    for i, v in enumerate(values):
        if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
            out[i] = np.nan
        elif isinstance(v, (bool, np.bool_)):
            out[i] = 1.0 if v else 0.0
        elif isinstance(v, str):
            raise TypeError(
                f"string feature value {v!r}: encode categorical features numerically "
                "with an on-demand view"
            )
        else:
            out[i] = float(v)
    return out


def _signed_log(x: FloatArray) -> FloatArray:
    result: FloatArray = np.sign(x) * np.log1p(np.abs(x))
    return result


@dataclass
class ColumnSpec:
    name: str
    median: float
    log: bool
    mean: float
    std: float
    missing_indicator: bool
    drift_edges: list[float]
    drift_reference: list[float]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Preprocessor:
    """Imputes, log-compresses heavy-tailed columns and standardises."""

    columns: list[ColumnSpec] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def output_names(self) -> list[str]:
        names: list[str] = []
        for c in self.columns:
            names.append(c.name)
            if c.missing_indicator:
                names.append(f"{c.name}__missing")
        return names

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Sequence[str]) -> Preprocessor:
        specs: list[ColumnSpec] = []
        for name in columns:
            raw = to_float(frame[name].tolist())
            present = raw[~np.isnan(raw)]
            median = float(np.median(present)) if len(present) else 0.0
            filled = np.where(np.isnan(raw), median, raw)
            p99 = float(np.percentile(np.abs(filled), 99)) if len(filled) else 0.0
            use_log = p99 > LOG_THRESHOLD
            transformed = _signed_log(filled) if use_log else filled
            std = float(transformed.std())
            edges, reference = drift_reference(present)
            specs.append(
                ColumnSpec(
                    name=name,
                    median=median,
                    log=use_log,
                    mean=float(transformed.mean()),
                    std=std if std > 1e-12 else 1.0,
                    missing_indicator=bool(np.isnan(raw).any()),
                    drift_edges=edges,
                    drift_reference=reference,
                )
            )
        return cls(specs)

    def transform_columns(self, data: dict[str, Sequence[Any]]) -> FloatArray:
        n = len(next(iter(data.values()))) if data else 0
        parts: list[FloatArray] = []
        for spec in self.columns:
            raw = to_float(data[spec.name]) if spec.name in data else np.full(n, np.nan)
            missing = np.isnan(raw)
            filled = np.where(missing, spec.median, raw)
            if spec.log:
                filled = _signed_log(filled)
            parts.append((filled - spec.mean) / spec.std)
            if spec.missing_indicator:
                parts.append(missing.astype(np.float64))
        return np.column_stack(parts) if parts else np.empty((n, 0))

    def transform(self, frame: pd.DataFrame) -> FloatArray:
        return self.transform_columns(
            {c.name: frame[c.name].tolist() for c in self.columns if c.name in frame}
        )

    def transform_rows(self, rows: Sequence[dict[str, Any]]) -> FloatArray:
        return self.transform_columns({c.name: [r.get(c.name) for r in rows] for c in self.columns})

    def to_dict(self) -> dict[str, Any]:
        return {"columns": [c.to_dict() for c in self.columns]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Preprocessor:
        return cls([ColumnSpec(**c) for c in data["columns"]])


def drift_reference(values: FloatArray, bins: int = DRIFT_BINS) -> tuple[list[float], list[float]]:
    """Quantile bin edges and the training proportion in each bin (for PSI)."""
    if len(values) == 0:
        return [], []
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return [float(e) for e in edges], [float(c) for c in counts / counts.sum()]


def _sigmoid(z: FloatArray) -> FloatArray:
    out: FloatArray = 0.5 * (1.0 + np.tanh(0.5 * z))  # overflow-free logistic
    return out


@dataclass
class LogisticModel:
    weights: list[float] = field(default_factory=list)
    intercept: float = 0.0
    l2: float = 1.0
    iterations: int = 0
    converged: bool = False

    def fit(
        self,
        x: FloatArray,
        y: FloatArray,
        *,
        sample_weight: FloatArray | None = None,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> LogisticModel:
        n, d = x.shape
        design = np.hstack([np.ones((n, 1)), x])
        s = np.ones(n) if sample_weight is None else sample_weight
        reg = np.full(d + 1, self.l2)
        reg[0] = 0.0  # intercept is not penalised
        base_rate = float(np.clip(np.average(y, weights=s), 1e-6, 1 - 1e-6))
        w = np.zeros(d + 1)
        w[0] = math.log(base_rate / (1 - base_rate))

        def objective(weights: FloatArray) -> float:
            z = design @ weights
            # log(1 + e^z) - y*z is the per-row negative log-likelihood, overflow-free.
            nll = float(np.sum(s * (np.logaddexp(0.0, z) - y * z)))
            return nll + 0.5 * float(np.sum(reg * weights * weights))

        self.converged = False
        current = objective(w)
        for iteration in range(1, max_iter + 1):
            p = _sigmoid(design @ w)
            grad = design.T @ (s * (p - y)) + reg * w
            hess = (design * (s * p * (1 - p))[:, None]).T @ design + np.diag(reg)
            hess[np.diag_indices_from(hess)] += 1e-9
            step = np.linalg.solve(hess, grad)
            # Armijo backtracking: pure Newton overshoots on near-separable data.
            slope = float(grad @ step)
            t = 1.0
            while t > 1e-8:
                candidate = objective(w - t * step)
                if candidate <= current - 1e-4 * t * slope:
                    break
                t *= 0.5
            w = w - t * step
            improvement = current - candidate
            current = candidate
            self.iterations = iteration
            if float(np.max(np.abs(t * step))) < tol or improvement < tol * max(1.0, abs(current)):
                self.converged = True
                break
        self.intercept = float(w[0])
        self.weights = [float(v) for v in w[1:]]
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        z = x @ np.asarray(self.weights) + self.intercept
        return _sigmoid(z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights,
            "intercept": self.intercept,
            "l2": self.l2,
            "iterations": self.iterations,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticModel:
        return cls(**data)
