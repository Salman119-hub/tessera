"""Stream ingestion: consume events, update online state, commit offsets.

Processing is at-least-once. Offsets are committed only after the batch's
state writes succeed, and tile updates are deduplicated by event id, so a
crash between write and commit replays harmlessly. Events beyond a view's
allowed lateness, and records that fail schema validation, go to a
dead-letter sink with the reason, for inspection and replay.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from tessera.metrics import Registry
from tessera.online.store import OnlineStore
from tessera.online.writer import IngestResult, OnlineWriter
from tessera.repository import Repository
from tessera.timeutil import now

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Message:
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: bytes


class EventBus(Protocol):
    def poll(self, max_records: int, timeout_s: float) -> list[Message]: ...

    def commit(self, messages: list[Message]) -> None: ...

    def close(self) -> None: ...


class InMemoryBus:
    """A partitioned in-process log with committed offsets per partition."""

    def __init__(self, partitions: int = 4) -> None:
        self.partitions = partitions
        self._log: dict[tuple[str, int], list[Message]] = defaultdict(list)
        self._position: dict[tuple[str, int], int] = defaultdict(int)
        self.committed: dict[tuple[str, int], int] = defaultdict(int)
        self.topics: set[str] = set()
        self._lock = threading.Lock()

    def subscribe(self, topics: Iterable[str]) -> None:
        self.topics.update(topics)

    def publish(self, topic: str, value: dict[str, Any] | bytes, key: str | None = None) -> None:
        payload = value if isinstance(value, bytes) else json.dumps(value).encode()
        key_bytes = key.encode() if key is not None else None
        partition = (sum(key_bytes) if key_bytes else 0) % self.partitions
        with self._lock:
            log_ = self._log[(topic, partition)]
            log_.append(Message(topic, partition, len(log_), key_bytes, payload))

    def poll(self, max_records: int, timeout_s: float = 0.0) -> list[Message]:
        out: list[Message] = []
        with self._lock:
            for (topic, partition), messages in sorted(self._log.items()):
                if topic not in self.topics:
                    continue
                pos = self._position[(topic, partition)]
                take = messages[pos : pos + max_records - len(out)]
                out.extend(take)
                self._position[(topic, partition)] = pos + len(take)
                if len(out) >= max_records:
                    break
        return out

    def commit(self, messages: list[Message]) -> None:
        with self._lock:
            for m in messages:
                tp = (m.topic, m.partition)
                self.committed[tp] = max(self.committed[tp], m.offset + 1)

    def rewind_to_committed(self) -> None:
        """Simulate a consumer restart: uncommitted messages are redelivered."""
        with self._lock:
            for tp in list(self._position):
                self._position[tp] = self.committed[tp]

    def lag(self) -> int:
        with self._lock:
            return sum(len(msgs) - self.committed[tp] for tp, msgs in self._log.items())

    def close(self) -> None:
        return None


class DeadLetterSink(Protocol):
    def write(self, source: str, record: Any, reason: str) -> None: ...


class MemoryDeadLetter:
    def __init__(self) -> None:
        self.items: list[tuple[str, Any, str]] = []

    def write(self, source: str, record: Any, reason: str) -> None:
        self.items.append((source, record, reason))


class JsonlDeadLetter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, source: str, record: Any, reason: str) -> None:
        line = json.dumps({"source": source, "reason": reason, "record": record}, default=str)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


@dataclass
class BatchStats:
    messages: int = 0
    undecodable: int = 0
    ingest: IngestResult = field(default_factory=IngestResult)

    def to_dict(self) -> dict[str, Any]:
        return {"messages": self.messages, "undecodable": self.undecodable, **self.ingest.to_dict()}


class StreamProcessor:
    """Consumes source topics and keeps aggregate views fresh in the online store."""

    def __init__(
        self,
        repo: Repository,
        store: OnlineStore,
        bus: EventBus,
        *,
        dead_letter: DeadLetterSink | None = None,
        metrics: Registry | None = None,
        batch_size: int = 500,
        poll_timeout_s: float = 1.0,
        clock: Callable[[], int] = now,
    ) -> None:
        self.repo = repo
        self.writer = OnlineWriter(repo, store)
        self.bus = bus
        self.dead_letter = dead_letter or MemoryDeadLetter()
        self.batch_size = batch_size
        self.poll_timeout_s = poll_timeout_s
        self.clock = clock
        self.topic_sources: dict[str, str] = {}
        for source in repo.sources.values():
            topic = getattr(source, "topic", "")
            if topic and repo.aggregate_views_for_source(source.name):
                self.topic_sources[topic] = source.name
        self.metrics = metrics or Registry()
        self._events = self.metrics.counter(
            "tessera_stream_events_total", "Events processed by outcome."
        )
        self._lag = self.metrics.gauge(
            "tessera_stream_event_lag_seconds", "Wall clock minus newest event time, per source."
        )
        self._batch_seconds = self.metrics.histogram(
            "tessera_stream_batch_seconds", "Time to process one polled batch."
        )
        self._stop = threading.Event()

    @property
    def topics(self) -> list[str]:
        return sorted(self.topic_sources)

    def process_batch(self, messages: list[Message]) -> BatchStats:
        stats = BatchStats(messages=len(messages))
        by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for message in messages:
            source = self.topic_sources.get(message.topic)
            if source is None:
                continue
            try:
                record = json.loads(message.value)
                if not isinstance(record, dict):
                    raise ValueError("event is not a JSON object")
            except ValueError as exc:
                stats.undecodable += 1
                self.dead_letter.write(source, message.value.decode(errors="replace"), str(exc))
                self._events.inc(source=source, outcome="undecodable")
                continue
            by_source[source].append(record)
        for source, batch in by_source.items():
            result = self.writer.ingest(source, batch)
            for record, reason in result.invalid:
                self.dead_letter.write(source, record, f"schema: {reason}")
            for outcome, count in result.status_counts.items():
                self._events.inc(count, source=source, outcome=outcome)
            if result.invalid:
                self._events.inc(len(result.invalid), source=source, outcome="invalid")
            if result.max_event_ts is not None:
                self._lag.set(max(0, self.clock() - result.max_event_ts), source=source)
            stats.ingest.merge(result)
        return stats

    def run_once(self) -> BatchStats:
        messages = self.bus.poll(self.batch_size, self.poll_timeout_s)
        if not messages:
            return BatchStats()
        started = time.perf_counter()
        stats = self.process_batch(messages)
        self.bus.commit(messages)  # only after state is durable
        self._batch_seconds.observe(time.perf_counter() - started)
        return stats

    def run(self, *, max_batches: int | None = None, idle_sleep_s: float = 0.2) -> None:
        batches = 0
        while not self._stop.is_set():
            stats = self.run_once()
            if stats.messages:
                batches += 1
                log.info("processed batch", extra={"stats": stats.to_dict()})
            elif idle_sleep_s:
                self._stop.wait(idle_sleep_s)
            if max_batches is not None and batches >= max_batches:
                break
        self.bus.close()

    def stop(self) -> None:
        self._stop.set()
