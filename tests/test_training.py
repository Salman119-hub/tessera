from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tessera import FeatureService, Field, on_demand
from tessera.errors import SkewGuardError
from tessera.training import (
    ModelArtifact,
    ModelRegistry,
    Preprocessor,
    PromotionGate,
    TrainConfig,
    train_model,
)
from tessera.training.metrics import (
    average_precision,
    brier,
    choose_threshold,
    classification_report,
    expected_calibration_error,
    log_loss,
    rankdata,
    recall_at_fpr,
    roc_auc,
)
from tessera.training.model import LogisticModel
from tessera.training.train import feature_importance, time_split

from .conftest import service


class TestMetrics:
    def test_rankdata_ties(self) -> None:
        assert rankdata(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [3.5, 1.0, 3.5, 2.0]

    def test_auc_known_values(self) -> None:
        assert roc_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(0.75)
        assert roc_auc([0, 1], [0.5, 0.5]) == pytest.approx(0.5)  # ties count half
        assert roc_auc([0, 1, 1], [0.1, 0.9, 0.8]) == 1.0
        with pytest.raises(ValueError, match="both classes"):
            roc_auc([1, 1], [0.2, 0.3])

    def test_average_precision_known_value(self) -> None:
        # ranks: 0.8(+) 0.4(-) 0.35(+) 0.1(-): precision at recall .5 = 1, at 1.0 = 2/3
        assert average_precision([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]) == pytest.approx(
            (1 + 2 / 3) / 2
        )

    def test_recall_at_fpr_and_losses(self) -> None:
        y = [0, 0, 0, 0, 1, 1]
        s = [0.1, 0.2, 0.3, 0.9, 0.8, 0.95]
        assert recall_at_fpr(y, s, 0.0) == 0.5
        assert recall_at_fpr(y, s, 0.25) == 1.0
        assert log_loss([1, 0], [0.9, 0.1]) == pytest.approx(-np.log(0.9))
        assert brier([1, 0], [0.9, 0.1]) == pytest.approx(0.01)
        assert expected_calibration_error([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.0)
        with pytest.raises(ValueError, match="binary"):
            brier([2, 0], [0.1, 0.1])
        with pytest.raises(ValueError, match="equal length"):
            brier([1, 0], [0.1])

    def test_choose_threshold(self) -> None:
        y = [0, 1, 0, 1, 1]
        p = [0.1, 0.2, 0.6, 0.7, 0.9]
        # False positives expensive: accept one missed positive rather than any FP.
        strict = choose_threshold(y, p, cost_fp=10, cost_fn=1)
        assert strict.threshold == 0.7 and strict.expected_cost == 1.0
        assert strict.precision == 1.0 and strict.recall == pytest.approx(2 / 3)
        # Misses expensive: flag down to 0.2 and pay for one false positive.
        lenient = choose_threshold(y, p, cost_fp=1, cost_fn=20)
        assert lenient.threshold == 0.2 and lenient.recall == 1.0 and lenient.expected_cost == 1.0
        nothing = choose_threshold([0, 0, 1], [0.9, 0.8, 0.1], cost_fp=100, cost_fn=1)
        assert nothing.flagged_rate == 0.0

    def test_report_keys(self) -> None:
        report = classification_report([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.9])
        assert {"roc_auc", "average_precision", "ece", "recall_at_1pct_fpr"} <= set(report)


def synthetic(n: int = 4000, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    true_w = np.array([2.0, -1.0, 0.0])
    p = 1 / (1 + np.exp(-(x @ true_w - 1.0)))
    y = (rng.random(n) < p).astype(float)
    return x, y, true_w


class TestModel:
    def test_recovers_coefficients(self) -> None:
        x, y, true_w = synthetic()
        model = LogisticModel(l2=1e-3).fit(x, y)
        assert model.converged
        assert np.allclose(model.weights, true_w, atol=0.15)
        assert model.intercept == pytest.approx(-1.0, abs=0.15)

    def test_separable_data_does_not_diverge(self) -> None:
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]] * 50)
        y = np.array([0.0, 0.0, 1.0, 1.0] * 50)
        model = LogisticModel(l2=1.0).fit(x, y)
        assert np.isfinite(model.weights).all()
        assert roc_auc(y, model.predict_proba(x)) == 1.0

    def test_preprocessor(self) -> None:
        frame = pd.DataFrame(
            {
                "a": [1.0, None, 3.0, 5.0],
                "b": [10.0, 2000.0, 30.0, 40.0],
                "c": [True, False, True, None],
            }
        )
        pre = Preprocessor.fit(frame, ["a", "b", "c"])
        specs = {c.name: c for c in pre.columns}
        assert specs["a"].missing_indicator and specs["a"].median == 3.0
        assert specs["b"].log  # heavy tail is log-compressed
        assert pre.output_names == ["a", "a__missing", "b", "c", "c__missing"]
        x = pre.transform(frame)
        assert x.shape == (4, 5) and np.isfinite(x).all()
        rows = frame.to_dict("records")
        assert np.allclose(pre.transform_rows(rows), x)
        assert np.allclose(Preprocessor.from_dict(pre.to_dict()).transform(frame), x)
        with pytest.raises(TypeError, match="string feature"):
            Preprocessor.fit(pd.DataFrame({"s": ["x"]}), ["s"])


def dataset(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    value_sum = rng.gamma(2, 20, n)
    label = (rng.random(n) < 1 / (1 + np.exp(-(value_sum / 20 - 4)))).astype(int)
    cols = {c: rng.normal(size=n) for c in service.output_columns}
    cols["user_clicks__value_sum_1h"] = value_sum
    cols["click_context__has_history"] = rng.random(n) < 0.5
    cols["user_clicks__item_last_1d"] = None
    frame = pd.DataFrame(cols)
    frame["label"] = label
    frame["timestamp"] = 1_700_000_000 + np.arange(n) * 600
    return frame


class TestTraining:
    config = TrainConfig(label="label", exclude=("user_clicks__item_last_1d",), gap="1h")

    def test_train_model(self) -> None:
        result = train_model("m", service, dataset(), self.config)
        artifact = result.artifact
        assert artifact.service_fingerprint == service.fingerprint
        assert result.test_metrics["roc_auc"] > 0.8
        assert "user_clicks__item_last_1d" not in artifact.features
        top_feature, _ = feature_importance(artifact)[0]
        assert top_feature == "user_clicks__value_sum_1h"
        assert (
            artifact.training["rows_train"] + artifact.training["rows_test"] < 3000
        )  # gap dropped rows

    def test_time_split(self) -> None:
        frame = dataset(100)
        train, test, cutoff = time_split(frame, "timestamp", 0.2, 3600)
        assert train["timestamp"].max() < cutoff - 3600
        assert test["timestamp"].min() >= cutoff
        with pytest.raises(ValueError, match="test_fraction"):
            time_split(frame, "timestamp", 1.5, 0)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="label"):
            train_model("m", service, dataset().drop(columns=["label"]), self.config)
        with pytest.raises(ValueError, match="missing feature"):
            train_model(
                "m", service, dataset().drop(columns=["user_clicks__value_sum_1h"]), self.config
            )

    def test_artifact_round_trip_and_registry(self, tmp_path: Path) -> None:
        artifact = train_model("fraud", service, dataset(), self.config).artifact
        registry = ModelRegistry(tmp_path)
        assert registry.register(artifact) == 1
        assert registry.register(artifact) == 2
        registry.set_alias("fraud", "production", 1)
        loaded = registry.load("fraud", "production")
        assert loaded.version == 1
        rows = dataset(20).to_dict("records")
        assert np.allclose(loaded.predict_proba_rows(rows), artifact.predict_proba_rows(rows))
        assert registry.load("fraud", "latest").version == 2
        assert registry.load("fraud", 2).version == 2
        assert [o["decision"] for o in loaded.score(rows)] == [
            o["decision"] for o in artifact.score(rows)
        ]
        assert len(loaded.explain(rows[0], top=3)) == 3
        with pytest.raises(KeyError):
            registry.load("fraud", "staging")
        with pytest.raises(KeyError):
            registry.set_alias("fraud", "production", 9)
        with pytest.raises(ValueError, match="invalid model name"):
            registry.versions("../escape")
        with pytest.raises(KeyError):
            ModelRegistry(tmp_path / "empty").resolve("x", "latest")
        assert ModelArtifact.load(tmp_path / "fraud" / "v1" / "model.json").name == "fraud"

    def test_skew_guard(self) -> None:
        artifact = train_model("m", service, dataset(), self.config).artifact
        artifact.check_compatible(service)

        @on_demand(request=(Field("price", "float"),), outputs=(Field("x", "float"),))
        def other(row: dict) -> dict:  # type: ignore[type-arg]
            return {"x": 1.0}

        changed = FeatureService("recs_v1", features=(other,))
        with pytest.raises(SkewGuardError, match="Retrain"):
            artifact.check_compatible(changed)
        with pytest.raises(SkewGuardError, match="not 'other_service'"):
            artifact.check_compatible(FeatureService("other_service", features=(other,)))


class TestGate:
    good = {"roc_auc": 0.95, "average_precision": 0.6, "ece": 0.01}

    def test_floors(self) -> None:
        decision = PromotionGate().evaluate(
            {"roc_auc": 0.7, "average_precision": 0.1, "ece": 0.2}, None
        )
        assert not decision.promote and len(decision.reasons) == 3

    def test_regression_vs_production(self) -> None:
        gate = PromotionGate()
        assert gate.evaluate(self.good, self.good).promote
        worse = {**self.good, "roc_auc": 0.93}
        decision = gate.evaluate(worse, self.good)
        assert not decision.promote and "regressed" in decision.reasons[0]
        within_tolerance = {**self.good, "roc_auc": 0.948}
        assert gate.evaluate(within_tolerance, self.good).promote
