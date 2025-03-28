"""Train, evaluate and gate a model on a point-in-time training set."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tessera.definitions import FeatureService
from tessera.timeutil import iso, parse_duration, series_to_epoch_seconds
from tessera.training.artifact import ModelArtifact
from tessera.training.metrics import choose_threshold, classification_report
from tessera.training.model import LogisticModel, Preprocessor


@dataclass(frozen=True)
class TrainConfig:
    label: str = "label"
    timestamp_column: str = "timestamp"
    test_fraction: float = 0.2
    gap: str = "1d"
    l2: float = 1.0
    cost_fp: float = 1.0
    cost_fn: float = 25.0
    max_iter: int = 50
    exclude: tuple[str, ...] = ()


def time_split(
    frame: pd.DataFrame, timestamp_column: str, test_fraction: float, gap_seconds: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Split by time: the last ``test_fraction`` of the span is the test set.

    Training rows within ``gap_seconds`` before the cutoff are dropped. Labels
    such as chargebacks mature days after the event, and windowed features
    computed just before the cutoff overlap the test period, so a gap stops
    that overlap from leaking into evaluation.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    ts = series_to_epoch_seconds(frame[timestamp_column])
    lo, hi = int(ts.min()), int(ts.max())
    cutoff = int(hi - (hi - lo) * test_fraction)
    train = frame[ts < cutoff - gap_seconds]
    test = frame[ts >= cutoff]
    if train.empty or test.empty:
        raise ValueError("time split produced an empty train or test set")
    return train, test, cutoff


@dataclass
class TrainResult:
    artifact: ModelArtifact
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    cutoff: int
    extra: dict[str, Any] = field(default_factory=dict)


def train_model(
    name: str, service: FeatureService, dataset: pd.DataFrame, config: TrainConfig
) -> TrainResult:
    """Fit the baseline model on the service's columns and evaluate out of time."""
    if config.label not in dataset.columns:
        raise ValueError(f"dataset has no label column {config.label!r}")
    features = [c for c in service.output_columns if c not in config.exclude]
    missing = [c for c in features if c not in dataset.columns]
    if missing:
        raise ValueError(f"dataset is missing feature columns {missing}")
    train, test, cutoff = time_split(
        dataset, config.timestamp_column, config.test_fraction, parse_duration(config.gap)
    )
    y_train = train[config.label].astype(float).to_numpy()
    y_test = test[config.label].astype(float).to_numpy()
    if y_train.sum() == 0 or y_test.sum() == 0:
        raise ValueError("both train and test sets need positive labels")
    preprocessor = Preprocessor.fit(train, features)
    x_train = preprocessor.transform(train)
    x_test = preprocessor.transform(test)
    model = LogisticModel(l2=config.l2).fit(x_train, y_train, max_iter=config.max_iter)
    p_train = model.predict_proba(x_train)
    p_test = model.predict_proba(x_test)
    # The threshold is chosen on training data so the test metrics stay unbiased.
    choice = choose_threshold(y_train, p_train, cost_fp=config.cost_fp, cost_fn=config.cost_fn)
    test_metrics = classification_report(y_test, p_test)
    flagged = p_test >= choice.threshold
    tp = float((flagged & (y_test == 1)).sum())
    test_metrics.update(
        {
            "threshold": choice.threshold,
            "precision_at_threshold": tp / float(flagged.sum()) if flagged.any() else 0.0,
            "recall_at_threshold": tp / float(y_test.sum()),
            "flagged_rate": float(flagged.mean()),
        }
    )
    artifact = ModelArtifact(
        name=name,
        service=service.name,
        service_fingerprint=service.fingerprint,
        features=features,
        preprocessor=preprocessor,
        model=model,
        threshold=choice.threshold,
        metrics={k: round(float(v), 6) for k, v in test_metrics.items()},
        training={
            "rows_train": len(train),
            "rows_test": len(test),
            "cutoff": iso(cutoff),
            "gap": config.gap,
            "l2": config.l2,
            "cost_fp": config.cost_fp,
            "cost_fn": config.cost_fn,
            "converged": model.converged,
            "iterations": model.iterations,
        },
    )
    return TrainResult(
        artifact=artifact,
        train_metrics=classification_report(y_train, p_train),
        test_metrics=test_metrics,
        cutoff=cutoff,
    )


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PromotionGate:
    """Promote only if the candidate clears absolute floors and does not
    regress against production beyond the tolerances."""

    min_roc_auc: float = 0.75
    min_average_precision: float = 0.2
    max_ece: float = 0.05
    max_auc_drop: float = 0.005
    max_ap_drop: float = 0.01

    def evaluate(
        self, candidate: dict[str, float], production: dict[str, float] | None
    ) -> GateDecision:
        reasons: list[str] = []
        if candidate["roc_auc"] < self.min_roc_auc:
            reasons.append(f"roc_auc {candidate['roc_auc']:.4f} < floor {self.min_roc_auc}")
        if candidate["average_precision"] < self.min_average_precision:
            reasons.append(
                f"average_precision {candidate['average_precision']:.4f} < floor "
                f"{self.min_average_precision}"
            )
        if candidate["ece"] > self.max_ece:
            reasons.append(f"ece {candidate['ece']:.4f} > max {self.max_ece}")
        if production:
            auc_drop = production["roc_auc"] - candidate["roc_auc"]
            ap_drop = production["average_precision"] - candidate["average_precision"]
            if auc_drop > self.max_auc_drop:
                reasons.append(f"roc_auc regressed by {auc_drop:.4f} vs production")
            if ap_drop > self.max_ap_drop:
                reasons.append(f"average_precision regressed by {ap_drop:.4f} vs production")
        return GateDecision(promote=not reasons, reasons=tuple(reasons))


def feature_importance(artifact: ModelArtifact) -> list[tuple[str, float]]:
    """Absolute standardised coefficients, largest first."""
    weights = np.abs(np.asarray(artifact.model.weights))
    names = artifact.preprocessor.output_names
    order = np.argsort(-weights)
    return [(names[i], float(weights[i])) for i in order]
