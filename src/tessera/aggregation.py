"""Mergeable aggregation states.

Every aggregation is a monoid: ``init`` is the identity, ``update`` folds one
value in, and ``merge`` combines two partial states. Tiles store partial
states, and a window is the merge of the tiles it covers. States are plain
JSON-compatible lists so they can live in any key-value store.

The same objects run offline and online, which is the basis of the
training/serving parity guarantee (ADR-0002).
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

State = list[Any]


def _is_missing(value: Any) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


class Aggregator(ABC):
    """One aggregation function. Implementations must be stateless."""

    name: str

    @abstractmethod
    def init(self) -> State: ...

    @abstractmethod
    def update(self, state: State, value: Any, ts: int) -> None:
        """Fold ``value`` (observed at ``ts``) into ``state`` in place."""

    @abstractmethod
    def merge_into(self, acc: State, other: State) -> None:
        """Merge ``other`` into ``acc`` in place."""

    @abstractmethod
    def finalize(self, state: State) -> Any: ...

    def merge(self, a: State, b: State) -> State:
        acc = list(a)
        self.merge_into(acc, b)
        return acc


class Count(Aggregator):
    name = "count"

    def init(self) -> State:
        return [0]

    def update(self, state: State, value: Any, ts: int) -> None:
        state[0] += 1

    def merge_into(self, acc: State, other: State) -> None:
        acc[0] += other[0]

    def finalize(self, state: State) -> int:
        return int(state[0])


class Sum(Aggregator):
    name = "sum"

    def init(self) -> State:
        return [0.0]

    def update(self, state: State, value: Any, ts: int) -> None:
        if not _is_missing(value):
            state[0] += float(value)

    def merge_into(self, acc: State, other: State) -> None:
        acc[0] += other[0]

    def finalize(self, state: State) -> float:
        return float(state[0])


class Mean(Aggregator):
    name = "mean"

    def init(self) -> State:
        return [0.0, 0]

    def update(self, state: State, value: Any, ts: int) -> None:
        if not _is_missing(value):
            state[0] += float(value)
            state[1] += 1

    def merge_into(self, acc: State, other: State) -> None:
        acc[0] += other[0]
        acc[1] += other[1]

    def finalize(self, state: State) -> float | None:
        return float(state[0]) / state[1] if state[1] else None


class Min(Aggregator):
    name = "min"

    def init(self) -> State:
        return [None]

    def update(self, state: State, value: Any, ts: int) -> None:
        if not _is_missing(value) and (state[0] is None or value < state[0]):
            state[0] = float(value)

    def merge_into(self, acc: State, other: State) -> None:
        if other[0] is not None and (acc[0] is None or other[0] < acc[0]):
            acc[0] = other[0]

    def finalize(self, state: State) -> float | None:
        value: float | None = state[0]
        return value


class Max(Aggregator):
    name = "max"

    def init(self) -> State:
        return [None]

    def update(self, state: State, value: Any, ts: int) -> None:
        if not _is_missing(value) and (state[0] is None or value > state[0]):
            state[0] = float(value)

    def merge_into(self, acc: State, other: State) -> None:
        if other[0] is not None and (acc[0] is None or other[0] > acc[0]):
            acc[0] = other[0]

    def finalize(self, state: State) -> float | None:
        value: float | None = state[0]
        return value


class Std(Aggregator):
    """Sample standard deviation with Welford updates and Chan's parallel merge.

    State is ``[n, mean, M2]``. Numerically stable for large magnitudes where
    the naive sum-of-squares formula cancels catastrophically.
    """

    name = "std"

    def init(self) -> State:
        return [0, 0.0, 0.0]

    def update(self, state: State, value: Any, ts: int) -> None:
        if _is_missing(value):
            return
        x = float(value)
        state[0] += 1
        delta = x - state[1]
        state[1] += delta / state[0]
        state[2] += delta * (x - state[1])

    def merge_into(self, acc: State, other: State) -> None:
        n_b = other[0]
        if n_b == 0:
            return
        n_a = acc[0]
        if n_a == 0:
            acc[0], acc[1], acc[2] = other[0], other[1], other[2]
            return
        n = n_a + n_b
        delta = other[1] - acc[1]
        acc[1] = acc[1] + delta * n_b / n
        acc[2] = acc[2] + other[2] + delta * delta * n_a * n_b / n
        acc[0] = n

    def finalize(self, state: State) -> float | None:
        if state[0] < 2:
            return None
        return math.sqrt(max(state[2], 0.0) / (state[0] - 1))


class Last(Aggregator):
    """Most recent value by event time. Ties go to the later-applied value."""

    name = "last"

    def init(self) -> State:
        return [None, None]

    def update(self, state: State, value: Any, ts: int) -> None:
        if _is_missing(value):
            return
        if state[0] is None or ts >= state[0]:
            state[0], state[1] = ts, value

    def merge_into(self, acc: State, other: State) -> None:
        if other[0] is not None and (acc[0] is None or other[0] >= acc[0]):
            acc[0], acc[1] = other[0], other[1]

    def finalize(self, state: State) -> Any:
        return state[1]


class CountDistinct(Aggregator):
    """Approximate distinct count with HyperLogLog (precision 12, ~1.6% error).

    The register set is stored sparsely as ``[[index, rank], ...]`` because
    most entities see few distinct values per tile. Merging takes the
    element-wise maximum, which makes the sketch exactly mergeable.
    """

    name = "count_distinct"
    precision = 12

    @property
    def _m(self) -> int:
        return 1 << self.precision

    def init(self) -> State:
        return []

    def _hash(self, value: Any) -> int:
        digest = hashlib.blake2b(str(value).encode(), digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def _set(self, state: State, index: int, rank: int) -> None:
        for pair in state:
            if pair[0] == index:
                if rank > pair[1]:
                    pair[1] = rank
                return
        state.append([index, rank])

    def update(self, state: State, value: Any, ts: int) -> None:
        if _is_missing(value):
            return
        h = self._hash(value)
        index = h >> (64 - self.precision)
        rest = h & ((1 << (64 - self.precision)) - 1)
        rank = (64 - self.precision) - rest.bit_length() + 1
        self._set(state, index, rank)

    def merge_into(self, acc: State, other: State) -> None:
        if not other:
            return
        if not acc:
            acc.extend([list(p) for p in other])
            return
        lookup = {p[0]: p for p in acc}
        for index, rank in other:
            existing = lookup.get(index)
            if existing is None:
                pair = [index, rank]
                acc.append(pair)
                lookup[index] = pair
            elif rank > existing[1]:
                existing[1] = rank

    def finalize(self, state: State) -> int:
        if not state:
            return 0
        m = self._m
        alpha = 0.7213 / (1 + 1.079 / m)
        harmonic = (m - len(state)) + sum(2.0**-rank for _, rank in state)
        estimate = alpha * m * m / harmonic
        zeros = m - len(state)
        if estimate <= 2.5 * m and zeros:
            estimate = m * math.log(m / zeros)  # linear counting for small cardinalities
        return round(float(estimate))


AGGREGATORS: dict[str, Aggregator] = {
    agg.name: agg for agg in (Count(), Sum(), Mean(), Min(), Max(), Std(), Last(), CountDistinct())
}


def get_aggregator(name: str) -> Aggregator:
    try:
        return AGGREGATORS[name]
    except KeyError:
        raise KeyError(f"unknown aggregation {name!r}") from None
