"""Kafka adapter for :class:`~tessera.streaming.StreamProcessor`.

Requires ``pip install tessera[kafka]`` (confluent-kafka). Works with any
Kafka-protocol broker, including Azure Event Hubs' Kafka endpoint, MSK and
Redpanda. Auto-commit is disabled; offsets are committed explicitly after
each batch is persisted.
"""

from __future__ import annotations

from typing import Any

from tessera.streaming import Message


class KafkaBus:  # pragma: no cover - exercised against a real broker in integration tests
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
        **config: Any,
    ) -> None:
        try:
            from confluent_kafka import Consumer
        except ImportError as exc:
            raise ImportError("KafkaBus requires `pip install tessera[kafka]`") from exc
        settings = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "partition.assignment.strategy": "cooperative-sticky",
            **config,
        }
        self._consumer = Consumer(settings)
        self._consumer.subscribe(topics)

    def poll(self, max_records: int, timeout_s: float) -> list[Message]:
        raw = self._consumer.consume(num_messages=max_records, timeout=timeout_s)
        out: list[Message] = []
        for msg in raw:
            if msg.error():
                continue
            out.append(
                Message(msg.topic(), msg.partition(), msg.offset(), msg.key(), msg.value() or b"")
            )
        return out

    def commit(self, messages: list[Message]) -> None:
        if not messages:
            return
        from confluent_kafka import TopicPartition

        latest: dict[tuple[str, int], int] = {}
        for m in messages:
            tp = (m.topic, m.partition)
            latest[tp] = max(latest.get(tp, -1), m.offset)
        offsets = [TopicPartition(t, p, o + 1) for (t, p), o in latest.items()]
        self._consumer.commit(offsets=offsets, asynchronous=False)

    def close(self) -> None:
        self._consumer.close()


def event_hubs_config(connection_string: str) -> dict[str, Any]:
    """librdkafka settings for Azure Event Hubs' Kafka endpoint (SASL/PLAIN over TLS)."""
    return {
        "security.protocol": "SASL_SSL",
        "sasl.mechanisms": "PLAIN",
        "sasl.username": "$ConnectionString",
        "sasl.password": connection_string,
    }
