"""Training/serving skew audits and feature drift.

Skew audit
    Replays logged online requests through the offline engine at the same
    timestamps and compares the values feature by feature. Parity is the
    design goal, so any mismatch has an operational cause worth finding:
    events dropped as late, a stream outage, a snapshot not materialised,
    or a definition deployed without a backfill.

Drift
    Population Stability Index of served values against the training
    distribution saved in the model artifact.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tessera.definitions import FeatureService
from tessera.offline import get_historical_features, records
from tessera.training.artifact import ModelArtifact
from tessera.training.model import to_float

PSI_WARN = 0.1
PSI_ALERT = 0.25


def values_match(online: Any, offline: Any, *, rtol: float = 1e-6, atol: float = 1e-9) -> bool:
    if online is None or offline is None:
        return online is None and offline is None
    if isinstance(online, bool) or isinstance(offline, bool):
        return bool(online) == bool(offline)
    if isinstance(online, (int, float)) and isinstance(offline, (int, float)):
        return math.isclose(float(online), float(offline), rel_tol=rtol, abs_tol=atol)
    return bool(online == offline)


@dataclass
class FeatureSkew:
    feature: str
    compared: int
    mismatches: int
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def mismatch_rate(self) -> float:
        return self.mismatches / self.compared if self.compared else 0.0


@dataclass
class SkewReport:
    service: str
    rows: int
    skipped_fingerprint_mismatch: int
    features: list[FeatureSkew]

    @property
    def mismatch_rate(self) -> float:
        compared = sum(f.compared for f in self.features)
        bad = sum(f.mismatches for f in self.features)
        return bad / compared if compared else 0.0

    def worst(self, n: int = 5) -> list[FeatureSkew]:
        return sorted(self.features, key=lambda f: -f.mismatch_rate)[:n]

    def to_dict(self) -> dict[str, Any]:
        return {
            "service": self.service,
            "rows": self.rows,
            "skipped_fingerprint_mismatch": self.skipped_fingerprint_mismatch,
            "mismatch_rate": round(self.mismatch_rate, 6),
            "features": [
                {
                    "feature": f.feature,
                    "compared": f.compared,
                    "mismatches": f.mismatches,
                    "mismatch_rate": round(f.mismatch_rate, 6),
                    "examples": f.examples,
                }
                for f in self.features
            ],
        }


def skew_audit(
    service: FeatureService,
    logged: Sequence[Mapping[str, Any]],
    *,
    sources: Mapping[str, pd.DataFrame] | None = None,
    max_examples: int = 3,
) -> SkewReport:
    """Compare logged online vectors with offline recomputation at the same instants."""
    usable = [r for r in logged if r["service"] == service.name]
    current = [r for r in usable if r["service_fingerprint"] == service.fingerprint]
    skipped = len(usable) - len(current)
    columns = service.output_columns
    if not current:
        return SkewReport(service.name, 0, skipped, [FeatureSkew(c, 0, 0) for c in columns])
    spine = pd.DataFrame(
        [{**r["entities"], **r["request"], "timestamp": int(r["request_time"])} for r in current]
    )
    offline = records(get_historical_features(service, spine, sources=sources, run_quality=False))
    report: list[FeatureSkew] = []
    for column in columns:
        skew = FeatureSkew(column, 0, 0)
        for served, recomputed in zip(current, offline):
            online_value = served["features"].get(column)
            offline_value = recomputed.get(column)
            skew.compared += 1
            if not values_match(online_value, offline_value):
                skew.mismatches += 1
                if len(skew.examples) < max_examples:
                    skew.examples.append(
                        {
                            "entities": served["entities"],
                            "request_time": served["request_time"],
                            "online": online_value,
                            "offline": offline_value,
                        }
                    )
        report.append(skew)
    return SkewReport(service.name, len(current), skipped, report)


def psi(expected: Sequence[float], actual: Sequence[float], eps: float = 1e-4) -> float:
    """Population Stability Index between two binned distributions."""
    e = np.clip(np.asarray(expected, dtype=np.float64), eps, None)
    a = np.clip(np.asarray(actual, dtype=np.float64), eps, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


@dataclass(frozen=True)
class FeatureDrift:
    feature: str
    psi: float
    observed: int
    missing_rate: float

    @property
    def status(self) -> str:
        if self.psi >= PSI_ALERT:
            return "alert"
        if self.psi >= PSI_WARN:
            return "warn"
        return "ok"


def feature_drift(
    artifact: ModelArtifact, rows: Sequence[Mapping[str, Any]], *, min_rows: int = 100
) -> list[FeatureDrift]:
    """PSI per model input against the training reference. Needs ``min_rows`` rows."""
    if len(rows) < min_rows:
        return []
    out: list[FeatureDrift] = []
    for spec in artifact.preprocessor.columns:
        if not spec.drift_reference:
            continue
        values = to_float([r.get(spec.name) for r in rows])
        present = values[~np.isnan(values)]
        missing_rate = 1 - len(present) / len(values)
        if len(present) == 0:
            out.append(FeatureDrift(spec.name, float("inf"), 0, missing_rate))
            continue
        edges = np.asarray(spec.drift_edges)
        counts = np.bincount(
            np.searchsorted(edges, present, side="right"), minlength=len(edges) + 1
        )
        out.append(
            FeatureDrift(
                spec.name,
                round(psi(spec.drift_reference, (counts / counts.sum()).tolist()), 6),
                len(present),
                round(missing_rate, 6),
            )
        )
    return sorted(out, key=lambda d: -d.psi)
