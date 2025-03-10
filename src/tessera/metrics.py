"""A dependency-free Prometheus registry (counters, gauges, histograms).

Kept in-house so the stream worker image does not need prometheus_client,
and so tests can assert on metric values directly.
"""

from __future__ import annotations

import bisect
import threading
from collections import defaultdict

LabelKey = tuple[tuple[str, str], ...]

DEFAULT_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)


def _key(labels: dict[str, str] | None) -> LabelKey:
    return tuple(sorted((labels or {}).items()))


def _fmt(name: str, key: LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    pairs = key + extra
    if not pairs:
        return name
    body = ",".join(f'{k}="{v}"' for k, v in pairs)
    return f"{name}{{{body}}}"


class Counter:
    def __init__(self, name: str, help_text: str) -> None:
        self.name, self.help = name, help_text
        self._values: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def inc(self, amount: float = 1.0, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counters only go up")
        with self._lock:
            self._values[_key(labels)] += amount

    def value(self, **labels: str) -> float:
        return self._values.get(_key(labels), 0.0)

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} counter"]
        lines += [f"{_fmt(self.name, k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Gauge:
    def __init__(self, name: str, help_text: str) -> None:
        self.name, self.help = name, help_text
        self._values: dict[LabelKey, float] = {}
        self._lock = threading.Lock()

    def set(self, value: float, **labels: str) -> None:
        with self._lock:
            self._values[_key(labels)] = value

    def value(self, **labels: str) -> float | None:
        return self._values.get(_key(labels))

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} gauge"]
        lines += [f"{_fmt(self.name, k)} {v}" for k, v in sorted(self._values.items())]
        return lines


class Histogram:
    def __init__(
        self, name: str, help_text: str, buckets: tuple[float, ...] = DEFAULT_BUCKETS
    ) -> None:
        self.name, self.help = name, help_text
        self.buckets = tuple(sorted(buckets))
        self._counts: dict[LabelKey, list[int]] = {}
        self._sums: dict[LabelKey, float] = defaultdict(float)
        self._lock = threading.Lock()

    def observe(self, value: float, **labels: str) -> None:
        key = _key(labels)
        with self._lock:
            counts = self._counts.setdefault(key, [0] * (len(self.buckets) + 1))
            counts[bisect.bisect_left(self.buckets, value)] += 1
            self._sums[key] += value

    def count(self, **labels: str) -> int:
        return sum(self._counts.get(_key(labels), []))

    def render(self) -> list[str]:
        lines = [f"# HELP {self.name} {self.help}", f"# TYPE {self.name} histogram"]
        for key, counts in sorted(self._counts.items()):
            running = 0
            for bound, n in zip(self.buckets, counts):
                running += n
                lines.append(f"{_fmt(self.name + '_bucket', key, (('le', str(bound)),))} {running}")
            running += counts[-1]
            lines.append(f"{_fmt(self.name + '_bucket', key, (('le', '+Inf'),))} {running}")
            lines.append(f"{_fmt(self.name + '_sum', key)} {self._sums[key]}")
            lines.append(f"{_fmt(self.name + '_count', key)} {running}")
        return lines


class Registry:
    def __init__(self) -> None:
        self._metrics: dict[str, Counter | Gauge | Histogram] = {}

    def counter(self, name: str, help_text: str) -> Counter:
        metric = self._metrics.setdefault(name, Counter(name, help_text))
        assert isinstance(metric, Counter)
        return metric

    def gauge(self, name: str, help_text: str) -> Gauge:
        metric = self._metrics.setdefault(name, Gauge(name, help_text))
        assert isinstance(metric, Gauge)
        return metric

    def histogram(self, name: str, help_text: str) -> Histogram:
        metric = self._metrics.setdefault(name, Histogram(name, help_text))
        assert isinstance(metric, Histogram)
        return metric

    def render(self) -> str:
        lines: list[str] = []
        for name in sorted(self._metrics):
            lines += self._metrics[name].render()
        return "\n".join(lines) + "\n"
