from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from tessera.monitoring import feature_drift, psi, skew_audit, values_match
from tessera.online.server import FeatureServer, MemoryFeatureLogger
from tessera.online.store import InMemoryStore
from tessera.online.writer import OnlineWriter
from tessera.repository import Repository
from tessera.schema import coerce_frame
from tessera.training import TrainConfig, train_model

from .conftest import T0, clicks, profiles, service
from .test_training import dataset


def served_log(
    repo: Repository, events: list[dict[str, Any]], *, skip: set[str] = frozenset()
) -> list[dict[str, Any]]:  # type: ignore[assignment]
    store = InMemoryStore()
    writer = OnlineWriter(repo, store)
    logger = MemoryFeatureLogger()
    server = FeatureServer(repo, store, logger=logger)
    for i, event in enumerate(events):
        if i % 5 == 0:
            server.get_features(
                service, [{"user_id": event["user_id"], "price": 5.0}], request_time=event["ts"]
            )
        if event["event_id"] not in skip:
            writer.ingest("clicks", [event])
    return [r.to_dict() for r in logger.rows]


def frames(events: list[dict[str, Any]], profile_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "clicks": coerce_frame(pd.DataFrame(events), clicks),
        "profiles": coerce_frame(profile_rows, profiles),
    }


def test_values_match() -> None:
    assert values_match(None, None)
    assert not values_match(None, 0)
    assert values_match(1.0, 1.0 + 1e-12)
    assert not values_match(1.0, 1.01)
    assert values_match(True, 1)
    assert values_match("a", "a") and not values_match("a", "b")


def test_skew_audit_clean_run(
    repo: Repository, click_events: list[dict[str, Any]], profile_rows: pd.DataFrame
) -> None:
    empty_profiles = profile_rows.iloc[0:0]
    log = served_log(repo, click_events)
    report = skew_audit(service, log, sources=frames(click_events, empty_profiles))
    assert report.rows == len(log)
    assert report.mismatch_rate == 0.0
    assert report.to_dict()["mismatch_rate"] == 0.0


def test_skew_audit_finds_dropped_events(
    repo: Repository, click_events: list[dict[str, Any]], profile_rows: pd.DataFrame
) -> None:
    dropped = {e["event_id"] for e in click_events[::10]}  # a stream outage lost 10% of events
    log = served_log(repo, click_events, skip=dropped)
    report = skew_audit(service, log, sources=frames(click_events, profile_rows.iloc[0:0]))
    assert report.mismatch_rate > 0
    worst = {f.feature for f in report.worst(3)}
    assert worst & {
        "user_clicks__value_mean_1d",
        "user_clicks__value_std_1d",
        "user_clicks__value_sum_1h",
    }
    example = report.worst(1)[0].examples[0]
    assert example["online"] != example["offline"]


def test_skew_audit_skips_other_fingerprints(
    repo: Repository, click_events: list[dict[str, Any]], profile_rows: pd.DataFrame
) -> None:
    log = served_log(repo, click_events[:20])
    for row in log:
        row["service_fingerprint"] = "000000000000"
    report = skew_audit(service, log, sources=frames(click_events, profile_rows))
    assert report.rows == 0 and report.skipped_fingerprint_mismatch == len(log)


def test_psi() -> None:
    assert psi([0.25] * 4, [0.25] * 4) == pytest.approx(0.0)
    assert psi([0.25] * 4, [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_feature_drift() -> None:
    artifact = train_model(
        "m",
        service,
        dataset(),
        TrainConfig(label="label", exclude=("user_clicks__item_last_1d",), gap="1h"),
    ).artifact
    same = dataset(800, seed=5).to_dict("records")
    assert all(d.status == "ok" for d in feature_drift(artifact, same))
    shifted = dataset(800, seed=6)
    shifted["user_clicks__value_sum_1h"] *= 4
    drift = feature_drift(artifact, shifted.to_dict("records"))
    assert drift[0].feature == "user_clicks__value_sum_1h" and drift[0].status == "alert"
    assert feature_drift(artifact, same[:10]) == []
    missing = [{**r, "user_clicks__value_sum_1h": None} for r in same]
    by_name = {d.feature: d for d in feature_drift(artifact, missing)}
    assert by_name["user_clicks__value_sum_1h"].missing_rate == 1.0
    assert np.isinf(by_name["user_clicks__value_sum_1h"].psi)


def test_t0_is_midnight() -> None:
    assert T0 % 86_400 == 0
