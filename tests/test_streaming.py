from __future__ import annotations

from typing import Any

from tessera.metrics import Registry
from tessera.online.server import FeatureServer
from tessera.online.store import InMemoryStore
from tessera.repository import Repository
from tessera.streaming import InMemoryBus, JsonlDeadLetter, MemoryDeadLetter, StreamProcessor

from .conftest import T0, click, service


def make(repo: Repository) -> tuple[InMemoryStore, InMemoryBus, StreamProcessor, MemoryDeadLetter]:
    store, bus, dlq = InMemoryStore(), InMemoryBus(partitions=3), MemoryDeadLetter()
    processor = StreamProcessor(
        repo, store, bus, dead_letter=dlq, batch_size=50, clock=lambda: T0 + 10_000
    )
    bus.subscribe(processor.topics)
    return store, bus, processor, dlq


def features(repo: Repository, store: InMemoryStore, t: int) -> list[dict[str, Any]]:
    rows = [{"user_id": "u1", "price": 1.0}, {"user_id": "u2", "price": 1.0}]
    return FeatureServer(repo, store).get_features(service, rows, request_time=t)


def test_topics_are_derived_from_repository(repo: Repository) -> None:
    _, _, processor, _ = make(repo)
    assert processor.topics == ["clicks.v1"]


def test_processes_and_commits(repo: Repository, click_events: list[dict[str, Any]]) -> None:
    _store, bus, processor, _ = make(repo)
    for event in click_events:
        bus.publish("clicks.v1", event, key=event["user_id"])
    bus.publish("unrelated.topic", {"x": 1})
    total = 0
    while (stats := processor.run_once()).messages:
        total += stats.ingest.applied
    assert total == len(click_events)
    assert bus.lag() == 1  # only the unsubscribed topic remains
    assert processor.metrics.counter("tessera_stream_events_total", "").value(
        source="clicks", outcome="applied"
    ) == len(click_events)
    lag = processor.metrics.gauge("tessera_stream_event_lag_seconds", "").value(source="clicks")
    assert lag is not None and lag >= 0
    assert "tessera_stream_batch_seconds_count" in processor.metrics.render()


def test_redelivery_after_crash_is_harmless(
    repo: Repository, click_events: list[dict[str, Any]]
) -> None:
    """Crash after writing state but before committing: the batch is redelivered."""
    store, bus, processor, _ = make(repo)
    for event in click_events:
        bus.publish("clicks.v1", event, key=event["user_id"])
    messages = bus.poll(120)
    processor.process_batch(messages)  # state written ...
    bus.rewind_to_committed()  # ... but the commit never happened
    rejected = 0
    while (stats := processor.run_once()).messages:
        rejected += stats.ingest.duplicates + stats.ingest.late
    assert rejected == 120  # every redelivered event is recognised, none re-applied
    t = max(e["ts"] for e in click_events) + 1
    reference_store, reference_bus, reference, _ = make(repo)
    for event in click_events:
        reference_bus.publish("clicks.v1", event, key=event["user_id"])
    while reference.run_once().messages:
        pass
    assert features(repo, store, t) == features(repo, reference_store, t)


def test_dead_letters(repo: Repository, tmp_path: Any) -> None:
    _store, bus, processor, dlq = make(repo)
    bus.publish("clicks.v1", b"{not json")
    bus.publish("clicks.v1", b"[1, 2]")
    bus.publish("clicks.v1", {"event_id": "x", "user_id": "u1", "value": "abc", "ts": T0})
    bus.publish("clicks.v1", click(1, "u1", T0 + 7200, 1.0))
    bus.publish("clicks.v1", click(2, "u1", T0, 1.0), key="u1")  # late
    stats = processor.run_once()
    assert stats.undecodable == 2
    assert stats.ingest.applied == 1
    assert len(dlq.items) == 3
    assert any("schema" in reason for _, _, reason in dlq.items)
    # Late events are counted, not dead-lettered: they are expected under disorder.
    assert stats.ingest.late == 1
    sink = JsonlDeadLetter(tmp_path / "dlq" / "dead.jsonl")
    sink.write("clicks", {"a": 1}, "reason")
    assert "reason" in (tmp_path / "dlq" / "dead.jsonl").read_text()


def test_run_loop_stops(repo: Repository) -> None:
    store, bus, processor, _ = make(repo)
    bus.publish("clicks.v1", click(1, "u1", T0, 1.0))
    processor.run(max_batches=1, idle_sleep_s=0)
    processor.stop()
    processor.run(idle_sleep_s=0)  # returns immediately once stopped
    assert store.count(repo.view("user_clicks").namespace) == 1


def test_metrics_registry_rendering() -> None:
    registry = Registry()
    counter = registry.counter("requests_total", "Requests.")
    counter.inc(route="/a")
    counter.inc(2, route="/a")
    gauge = registry.gauge("depth", "Depth.")
    gauge.set(4.5)
    hist = registry.histogram("latency_seconds", "Latency.")
    hist.observe(0.003)
    hist.observe(7.0)
    text = registry.render()
    assert 'requests_total{route="/a"} 3.0' in text
    assert "depth 4.5" in text
    assert 'latency_seconds_bucket{le="0.005"} 1' in text
    assert 'latency_seconds_bucket{le="+Inf"} 2' in text
    assert hist.count() == 2
    import pytest

    with pytest.raises(ValueError, match="only go up"):
        counter.inc(-1)
