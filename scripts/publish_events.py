"""Replay example transactions into Kafka for the local docker-compose stack.

Events are keyed by card so that each card's events land on one partition,
which is what the stream processor's per-entity ordering relies on.

Usage: ``python scripts/publish_events.py [--brokers localhost:19092] [--limit 2000]``
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

DEFAULT = Path(__file__).resolve().parents[1] / "examples" / "fraud" / "data" / "transactions.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brokers", default="localhost:19092")
    parser.add_argument("--topic", default="payments.transactions.v1")
    parser.add_argument("--file", default=str(DEFAULT))
    parser.add_argument("--limit", type=int, default=0, help="0 means all events")
    parser.add_argument("--rate", type=float, default=0, help="events per second; 0 is unthrottled")
    args = parser.parse_args()

    from confluent_kafka import Producer

    producer = Producer({"bootstrap.servers": args.brokers, "linger.ms": 20, "acks": "all"})
    sent = 0
    with open(args.file, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row_out = {**row, "amount": float(row["amount"])}
            producer.produce(args.topic, key=row["card_id"], value=json.dumps(row_out))
            sent += 1
            if sent % 1000 == 0:
                producer.poll(0)
            if args.rate:
                time.sleep(1 / args.rate)
            if args.limit and sent >= args.limit:
                break
    producer.flush(30)
    print(f"published {sent} events to {args.topic}")


if __name__ == "__main__":
    main()
