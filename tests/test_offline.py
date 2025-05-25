from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from tessera.errors import QualityCheckError, SchemaError
from tessera.offline import get_historical_features, records
from tessera.schema import coerce_frame

from .conftest import T0, click, clicks, profiles, service


def sources(events: list[dict[str, Any]], profile_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "clicks": coerce_frame(pd.DataFrame(events), clicks),
        "profiles": coerce_frame(profile_rows, profiles),
    }


def spine(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    return pd.DataFrame([{"user_id": u, "timestamp": t, "price": p} for u, t, p in rows])


def test_no_future_leakage(profile_rows: pd.DataFrame) -> None:
    events = [
        click(1, "u1", T0 + 60, 10.0),
        click(2, "u1", T0 + 120, 30.0),
        click(3, "u1", T0 + 180, 50.0),
    ]
    frame = get_historical_features(
        service,
        spine([("u1", T0 + 120, 1.0), ("u1", T0 + 121, 1.0), ("u1", T0 + 60, 1.0)]),
        sources=sources(events, profile_rows),
    )
    # At exactly T0+120 the event at T0+120 is excluded; one second later it is in.
    assert frame["user_clicks__events_count_10m"].tolist() == [1, 2, 0]
    assert frame["user_clicks__value_sum_1h"].tolist() == [10.0, 40.0, 0.0]
    assert frame["user_clicks__item_last_1d"].iloc[2] is None or pd.isna(
        frame["user_clicks__item_last_1d"].iloc[2]
    )


def test_preserves_spine_order_index_and_columns(
    profile_rows: pd.DataFrame, click_events: list[dict[str, Any]]
) -> None:
    rows = spine([("u2", T0 + 50_000, 2.0), ("u1", T0 + 10_000, 3.0), ("u2", T0 + 10, 4.0)])
    rows.index = [7, 7, 3]  # duplicate labels must survive
    frame = get_historical_features(service, rows, sources=sources(click_events, profile_rows))
    assert list(frame.index) == [7, 7, 3]
    assert frame["price"].tolist() == [2.0, 3.0, 4.0]
    assert list(frame.columns) == ["user_id", "timestamp", "price", *service.output_columns]


def test_snapshot_as_of_join_strict_and_ttl(profile_rows: pd.DataFrame) -> None:
    frame = get_historical_features(
        service,
        spine(
            [
                ("u1", T0 - 3600, 1.0),  # exactly at first update: not yet visible
                ("u1", T0, 1.0),  # first row
                ("u1", T0 + 86_400 + 1, 1.0),  # second row
                ("u2", T0, 1.0),  # u2 row is 5 days old, TTL is 2 days
                ("u3", T0, 1.0),  # unknown entity
            ]
        ),
        sources=sources([click(1, "u1", T0, 1.0)], profile_rows),
    )
    scores = records(frame[["user_profile__score"]])
    assert [r["user_profile__score"] for r in scores] == [None, 0.9, 0.7, None, None]


def test_on_demand_sees_none_for_missing_history(profile_rows: pd.DataFrame) -> None:
    events = [click(1, "u1", T0, 10.0), click(2, "u1", T0 + 10, 30.0)]
    frame = get_historical_features(
        service,
        spine([("u1", T0 + 100, 40.0), ("u2", T0 + 100, 40.0)]),
        sources=sources(events, profile_rows),
    )
    assert frame["click_context__price_vs_mean"].tolist() == [2.0, 0.0]
    assert frame["click_context__has_history"].tolist() == [True, False]


def test_duplicate_events_in_archive_count_once(profile_rows: pd.DataFrame) -> None:
    events = [click(1, "u1", T0, 10.0), click(1, "u1", T0, 10.0)]
    frame = get_historical_features(
        service, spine([("u1", T0 + 1, 1.0)]), sources=sources(events, profile_rows)
    )
    assert frame["user_clicks__events_count_10m"].tolist() == [1]


def test_spine_validation(profile_rows: pd.DataFrame) -> None:
    src = sources([click(1, "u1", T0, 1.0)], profile_rows)
    with pytest.raises(SchemaError, match="timestamp"):
        get_historical_features(
            service, pd.DataFrame({"user_id": ["u1"], "price": [1.0]}), sources=src
        )
    with pytest.raises(SchemaError, match="price"):
        get_historical_features(
            service, pd.DataFrame({"user_id": ["u1"], "timestamp": [T0]}), sources=src
        )


def test_quality_checks_block_bad_sources(profile_rows: pd.DataFrame) -> None:
    events = [click(1, "u1", T0, 5000.0)]  # value outside [0, 1000]
    with pytest.raises(QualityCheckError):
        get_historical_features(
            service, spine([("u1", T0 + 1, 1.0)]), sources=sources(events, profile_rows)
        )
    frame = get_historical_features(
        service,
        spine([("u1", T0 + 1, 1.0)]),
        sources=sources(events, profile_rows),
        run_quality=False,
    )
    assert frame["user_clicks__value_max_1h"].tolist() == [5000.0]


def test_accepts_uncoerced_source_frames(profile_rows: pd.DataFrame) -> None:
    raw = {"clicks": pd.DataFrame([click(1, "u1", T0, 3.0)]), "profiles": profile_rows}
    frame = get_historical_features(service, spine([("u1", T0 + 5, 1.0)]), sources=raw)
    assert frame["user_clicks__value_sum_1h"].tolist() == [3.0]
