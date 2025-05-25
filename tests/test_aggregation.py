from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

from tessera.aggregation import AGGREGATORS, CountDistinct, get_aggregator

NUMERIC = ["count", "sum", "mean", "min", "max", "std", "last"]


def fold(name: str, values: list[float], start_ts: int = 0) -> list:
    agg = get_aggregator(name)
    state = agg.init()
    for i, v in enumerate(values):
        agg.update(state, v, start_ts + i)
    return state


def reference(name: str, values: list[float]) -> object:
    present = [v for v in values if v is not None]
    if name == "count":
        return len(values)
    if name == "sum":
        return float(sum(present))
    if name == "mean":
        return sum(present) / len(present) if present else None
    if name == "min":
        return min(present) if present else None
    if name == "max":
        return max(present) if present else None
    if name == "std":
        return float(np.std(present, ddof=1)) if len(present) > 1 else None
    if name == "last":
        return present[-1] if present else None
    raise AssertionError(name)


@pytest.mark.parametrize("name", NUMERIC)
def test_update_matches_reference(name: str) -> None:
    rng = random.Random(1)
    values = [rng.uniform(-50, 50) for _ in range(200)]
    result = get_aggregator(name).finalize(fold(name, values))
    expected = reference(name, values)
    assert result == pytest.approx(expected, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("name", NUMERIC)
def test_merge_of_partitions_equals_sequential(name: str) -> None:
    """Monoid law: any split into contiguous partitions merges to the same result."""
    rng = random.Random(2)
    values = [rng.uniform(0, 1e6) for _ in range(300)]
    agg = get_aggregator(name)
    cuts = sorted(rng.sample(range(1, 299), 7))
    bounds = [0, *cuts, 300]
    merged = agg.init()
    for lo, hi in itertools.pairwise(bounds):
        agg.merge_into(merged, fold(name, values[lo:hi], start_ts=lo))
    assert agg.finalize(merged) == pytest.approx(agg.finalize(fold(name, values)), rel=1e-9)


@pytest.mark.parametrize("name", NUMERIC)
def test_identity_and_missing_values(name: str) -> None:
    agg = get_aggregator(name)
    state = fold(name, [1.0, None, float("nan"), 3.0])  # type: ignore[list-item]
    before = agg.finalize(state)
    agg.merge_into(state, agg.init())
    assert agg.finalize(state) == before
    empty = agg.init()
    agg.merge_into(empty, state)
    assert agg.finalize(empty) == before


def test_count_counts_rows_and_sum_skips_missing() -> None:
    assert get_aggregator("count").finalize(fold("count", [None, 1.0, None])) == 3  # type: ignore[list-item]
    assert get_aggregator("sum").finalize(fold("sum", [None, 1.0, 2.5])) == 3.5  # type: ignore[list-item]


def test_empty_results() -> None:
    assert get_aggregator("count").finalize(get_aggregator("count").init()) == 0
    assert get_aggregator("sum").finalize(get_aggregator("sum").init()) == 0.0
    for name in ("mean", "min", "max", "std", "last"):
        agg = get_aggregator(name)
        assert agg.finalize(agg.init()) is None
    assert get_aggregator("std").finalize(fold("std", [5.0])) is None


def test_std_is_stable_for_large_offsets() -> None:
    values = [1e9 + v for v in (4.0, 7.0, 13.0, 16.0)]
    assert get_aggregator("std").finalize(fold("std", values)) == pytest.approx(
        np.std([4, 7, 13, 16], ddof=1), rel=1e-9
    )


def test_last_prefers_latest_timestamp_regardless_of_order() -> None:
    agg = get_aggregator("last")
    state = agg.init()
    agg.update(state, "new", 200)
    agg.update(state, "old", 100)
    assert agg.finalize(state) == "new"
    other = agg.init()
    agg.update(other, "newest", 300)
    assert agg.finalize(agg.merge(state, other)) == "newest"
    assert agg.finalize(agg.merge(other, state)) == "newest"


@pytest.mark.parametrize("n", [0, 1, 10, 500, 20_000])
def test_hyperloglog_accuracy(n: int) -> None:
    agg = CountDistinct()
    state = agg.init()
    for i in range(n):
        agg.update(state, f"item-{i}", 0)
        agg.update(state, f"item-{i}", 0)  # duplicates do not count
    estimate = agg.finalize(state)
    if n <= 10:
        assert estimate == n
    else:
        assert abs(estimate - n) / n < 0.05


def test_hyperloglog_merge_is_union() -> None:
    agg = CountDistinct()
    a, b = agg.init(), agg.init()
    for i in range(3000):
        agg.update(a, i, 0)
    for i in range(2000, 5000):
        agg.update(b, i, 0)
    union = agg.finalize(agg.merge(a, b))
    assert abs(union - 5000) / 5000 < 0.05
    assert agg.finalize(agg.merge(a, b)) == agg.finalize(agg.merge(b, a))


def test_registry_lookup() -> None:
    assert set(AGGREGATORS) == {
        "count",
        "sum",
        "mean",
        "min",
        "max",
        "std",
        "last",
        "count_distinct",
    }
    with pytest.raises(KeyError, match="unknown aggregation"):
        get_aggregator("median")
    assert math.isclose(get_aggregator("mean").finalize([3.0, 2]), 1.5)
