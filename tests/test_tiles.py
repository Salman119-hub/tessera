"""The tile engine against a brute-force reference of the documented window semantics."""

from __future__ import annotations

import random
from typing import Any

import pytest

from tessera.aggregation import get_aggregator
from tessera.tiles import ApplyStatus, EntityState, TileEngine
from tessera.timeutil import align

from .conftest import T0, user_clicks


def brute_force(events: list[tuple[int, list[Any]]], t: int) -> dict[str, Any]:
    """Direct implementation of ``[align(t, hop) - window, t)`` over raw events."""
    engine = TileEngine(user_clicks)
    out = {}
    for slot in engine.slots:
        lo = align(t, slot.hop) - slot.window
        agg = get_aggregator(slot.aggregator.name)
        state = agg.init()
        for ts, values in sorted(events, key=lambda e: e[0]):
            if lo <= ts < t:
                value = values[slot.column_index] if slot.column_index is not None else None
                agg.update(state, value, ts)
        out[slot.name] = agg.finalize(state)
    return out


def assert_values_equal(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=1e-9, abs=1e-9), key
        else:
            assert actual[key] == value, key


def random_events(seed: int, n: int, span: int) -> list[tuple[int, list[Any]]]:
    rng = random.Random(seed)
    return sorted(
        (T0 + rng.randint(0, span), [round(rng.uniform(0, 100), 2), rng.choice("abcdefghij")])
        for _ in range(n)
    )


def test_engine_layout() -> None:
    engine = TileEngine(user_clicks)
    assert engine.resolutions == [60, 3600, 86_400]
    assert engine.input_columns == ["value", "item"]
    hops = {s.name: s.hop for s in engine.slots}
    assert hops["events_count_10m"] == 60
    assert hops["value_mean_1d"] == 3600
    assert hops["value_min_3d"] == 86_400
    # 1-minute tiles must span the hourly filler; daily tiles the 3-day window.
    assert engine.tile_retention[60] >= 3600
    assert engine.tile_retention[86_400] == 3 * 86_400


@pytest.mark.parametrize("seed", range(6))
def test_reads_match_brute_force_at_random_times(seed: int) -> None:
    events = random_events(seed, 300, 4 * 86_400)
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    rng = random.Random(seed + 100)
    queries = sorted(rng.randint(T0, T0 + 4 * 86_400 + 3600) for _ in range(60))
    pointer = 0
    for t in queries:
        while pointer < len(events) and events[pointer][0] < t:
            engine.apply(state, events[pointer][0], events[pointer][1])
            pointer += 1
        assert_values_equal(engine.read(state, t), brute_force(events, t))


def test_window_excludes_event_at_exactly_t() -> None:
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    engine.apply(state, T0 + 100, [10.0, "a"])
    engine.apply(state, T0 + 200, [20.0, "b"])
    assert engine.read(state, T0 + 200)["events_count_10m"] == 1
    assert engine.read(state, T0 + 201)["events_count_10m"] == 2


def test_online_read_with_newer_events_ingested_matches_reference() -> None:
    """Online, the store may already hold events at or after t (the head tile);
    the raw buffer filters them."""
    events = random_events(9, 120, 3 * 3600)
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    for ts, values in events:
        engine.apply(state, ts, values)
    t = events[-1][0] - 30  # inside the head minute, before the final event
    assert_values_equal(engine.read(state, t), brute_force(events[:-1], t))


def test_out_of_order_within_lateness_equals_in_order() -> None:
    events = random_events(4, 200, 6 * 3600)
    shuffled = list(events)
    rng = random.Random(5)
    # Swap neighbours that are less than the 5-minute lateness apart.
    for i in range(0, len(shuffled) - 1, 2):
        if shuffled[i + 1][0] - shuffled[i][0] < 300 and rng.random() < 0.7:
            shuffled[i], shuffled[i + 1] = shuffled[i + 1], shuffled[i]
    engine = TileEngine(user_clicks)
    a, b = engine.new_state(), engine.new_state()
    for ts, values in events:
        assert engine.apply(a, ts, values) is ApplyStatus.APPLIED
    for ts, values in shuffled:
        assert engine.apply(b, ts, values) is ApplyStatus.APPLIED
    t = events[-1][0] + 1
    assert_values_equal(engine.read(b, t), engine.read(a, t))


def test_deduplicates_by_event_id() -> None:
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    assert engine.apply(state, T0, [5.0, "a"], "e1") is ApplyStatus.APPLIED
    assert engine.apply(state, T0, [5.0, "a"], "e1") is ApplyStatus.DUPLICATE
    assert engine.read(state, T0 + 1)["value_sum_1h"] == 5.0


def test_rejects_events_beyond_allowed_lateness() -> None:
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    engine.apply(state, T0 + 3600, [1.0, "a"])
    assert engine.apply(state, T0 + 3600 - 299, [1.0, "a"]) is ApplyStatus.APPLIED
    assert engine.apply(state, T0 + 3600 - 301, [1.0, "a"]) is ApplyStatus.LATE


def test_state_is_pruned_to_bounded_size() -> None:
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    for i in range(20_000):
        engine.apply(state, T0 + i * 60, [1.0, "a"])  # two weeks, one event per minute
    assert len(state.raw) <= engine.raw_retention // 60 + 1
    assert len(state.tiles[60]) <= engine.tile_retention[60] // 60 + 2
    assert len(state.tiles[86_400]) <= 4
    t = T0 + 20_000 * 60
    assert engine.read(state, t)["events_count_10m"] == 10


def test_serialisation_round_trip() -> None:
    engine = TileEngine(user_clicks)
    state = engine.new_state()
    for ts, values in random_events(8, 50, 86_400):
        engine.apply(state, ts, values, f"id-{ts}")
    restored = EntityState.from_json(state.to_json())
    t = T0 + 86_400 + 5
    assert engine.read(restored, t) == engine.read(state, t)
    with pytest.raises(ValueError, match="version"):
        EntityState.from_json(b'{"v": 99}')


def test_empty_state_values() -> None:
    values = TileEngine(user_clicks).empty_values()
    assert values["events_count_10m"] == 0
    assert values["value_sum_1h"] == 0.0
    assert values["value_mean_1d"] is None
    assert values["item_count_distinct_1h"] == 0
