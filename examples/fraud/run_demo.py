"""End-to-end demo: train offline, serve online, then prove the two agree.

1. Build a point-in-time training set for every transaction before the final
   day, train the baseline model, and promote it through the gate.
2. Backfill the online store from the archive up to the start of the final day.
3. Replay the final day as a live stream. Each transaction is scored through
   the feature server *before* its own event is ingested, as a real
   authorisation would be, and every served vector is logged.
4. Audit the log: recompute every served vector offline at the same instant
   and compare.
5. Repeat the replay with a fraction of events delivered late, and show the
   audit attributing the resulting skew.

Usage: ``python examples/fraud/run_demo.py [--workdir .tessera-demo] [--late-fraction 0.02]``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from generate_data import generate  # noqa: E402

from tessera.definitions import SnapshotFeatureView  # noqa: E402
from tessera.jobs import backfill  # noqa: E402
from tessera.monitoring import feature_drift, skew_audit  # noqa: E402
from tessera.offline import get_historical_features  # noqa: E402
from tessera.online.server import FeatureServer, MemoryFeatureLogger  # noqa: E402
from tessera.online.store import InMemoryStore  # noqa: E402
from tessera.online.writer import OnlineWriter  # noqa: E402
from tessera.repository import Repository  # noqa: E402
from tessera.schema import TS_COLUMN, coerce_frame, read_source_frame  # noqa: E402
from tessera.streaming import InMemoryBus, StreamProcessor  # noqa: E402
from tessera.training import ModelRegistry, PromotionGate, TrainConfig, train_model  # noqa: E402

REQUEST_FIELDS = ("amount", "merchant_country", "channel")


def replay_live_day(
    repo: Repository,
    frames: dict[str, pd.DataFrame],
    live_start: int,
    *,
    late_fraction: float = 0.0,
    late_delay: int = 1200,
    seed: int = 0,
) -> tuple[MemoryFeatureLogger, dict[str, Any]]:
    """Backfill to ``live_start``, then stream and score the remaining events."""
    store = InMemoryStore()
    history = {name: frame[frame[TS_COLUMN] < live_start] for name, frame in frames.items()}
    backfill(repo, store, frames=history)
    logger = MemoryFeatureLogger()
    clock = {"now": live_start}
    server = FeatureServer(repo, store, logger=logger, clock=lambda: clock["now"])
    bus = InMemoryBus(partitions=4)
    processor = StreamProcessor(repo, store, bus, batch_size=1000, clock=lambda: clock["now"])
    bus.subscribe(processor.topics)

    live = frames["transactions"][frames["transactions"][TS_COLUMN] >= live_start]
    profiles = frames["card_profiles"]
    pending_profiles = profiles[profiles[TS_COLUMN] >= live_start].sort_values(TS_COLUMN)
    profile_view = repo.view("card_profile")
    assert isinstance(profile_view, SnapshotFeatureView)
    cdc_writer = OnlineWriter(repo, store)

    rng = np.random.default_rng(seed)
    delayed: list[tuple[int, dict[str, Any]]] = []
    raw_columns = [
        "event_id",
        "card_id",
        "merchant_id",
        "amount",
        "merchant_country",
        "channel",
        "ts",
    ]
    service = repo.service("fraud_v1")
    started = time.perf_counter()
    for row in live[[*raw_columns, TS_COLUMN]].to_dict("records"):
        t = int(row[TS_COLUMN])
        clock["now"] = t
        # Deliver delayed events whose delivery time has come, then CDC updates.
        due = [e for when, e in delayed if when <= t]
        delayed = [(when, e) for when, e in delayed if when > t]
        for event in due:
            bus.publish("payments.transactions.v1", event, key=event["card_id"])
        ready = pending_profiles[pending_profiles[TS_COLUMN] < t]
        if not ready.empty:
            cdc_writer.materialize_snapshot(profile_view, ready)
            pending_profiles = pending_profiles[pending_profiles[TS_COLUMN] >= t]
        processor.run_once()
        # Score the authorisation before its own event exists anywhere.
        server.get_features(
            service,
            [
                {
                    "card_id": row["card_id"],
                    "merchant_id": row["merchant_id"],
                    **{k: row[k] for k in REQUEST_FIELDS},
                }
            ],
            request_time=t,
        )
        event = {k: row[k] for k in raw_columns}
        if rng.random() < late_fraction:
            delayed.append((t + late_delay, event))
        else:
            bus.publish("payments.transactions.v1", event, key=event["card_id"])
    elapsed = time.perf_counter() - started
    stats = {
        "live_events": len(live),
        "served": len(logger.rows),
        "delayed_events": round(late_fraction * len(live)) if late_fraction else 0,
        "seconds": round(elapsed, 2),
        "ms_per_request": round(1000 * elapsed / max(1, len(live)), 3),
    }
    return logger, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workdir", default=str(Path(".tessera-demo")))
    parser.add_argument("--late-fraction", type=float, default=0.02)
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 if the clean replay shows any skew (CI)"
    )
    args = parser.parse_args()
    work = Path(args.workdir)
    data = HERE / "data"
    if not (data / "transactions.csv").exists():
        print("generating data:", generate(data))

    repo = Repository.from_path(HERE / "features.py")
    print(repo.apply(work / "registry.json").render(), "\n")
    service = repo.service("fraud_v1")
    frames = {name: read_source_frame(src) for name, src in repo.sources.items()}
    spine = pd.read_csv(data / "spine.csv")
    spine_ts = coerce_frame(
        spine.rename(columns={"timestamp": "ts"}), repo.sources["transactions"]
    )[TS_COLUMN]
    live_start = int(spine_ts.max()) - int(spine_ts.max()) % 86_400

    # 1. Offline training on history.
    started = time.perf_counter()
    history_spine = spine[spine_ts.to_numpy() < live_start]
    dataset = get_historical_features(service, history_spine, sources=frames)
    build_seconds = time.perf_counter() - started
    result = train_model("card_fraud", service, dataset, TrainConfig(label="is_fraud"))
    registry = ModelRegistry(work / "models")
    version = registry.register(result.artifact)
    gate = PromotionGate().evaluate(result.artifact.metrics, None)
    if gate.promote:
        registry.set_alias("card_fraud", "production", version)
    print(
        f"training set: {len(dataset)} rows x {len(service.output_columns)} features "
        f"in {build_seconds:.1f}s"
    )
    print("test metrics:", json.dumps({k: round(v, 4) for k, v in result.artifact.metrics.items()}))
    print("gate:", "promote" if gate.promote else f"blocked {gate.reasons}")

    # Ablation: what the model achieves without any windowed (stateful) feature.
    stateful_derived = ("txn_context__amount_vs_card_mean", "txn_context__new_card_history")
    without_aggregates = tuple(
        c
        for c in service.output_columns
        if c.startswith(("card_txn__", "merchant_txn__")) or c in stateful_derived
    )
    baseline = train_model(
        "card_fraud_request_only",
        service,
        dataset,
        TrainConfig(label="is_fraud", exclude=without_aggregates),
    )
    print("ablation (test set):")
    for label, metrics in (
        ("all 23 features", result.artifact.metrics),
        (f"request + profile only ({23 - len(without_aggregates)})", baseline.artifact.metrics),
    ):
        print(
            f"  {label:32s} ROC AUC {metrics['roc_auc']:.4f}  AP {metrics['average_precision']:.4f}"
            f"  recall@1%FPR {metrics['recall_at_1pct_fpr']:.4f}"
        )
    print()

    # 2-4. Clean live replay and audit.
    logger, stats = replay_live_day(repo, frames, live_start)
    print("clean replay:", stats)
    report = skew_audit(service, [r.to_dict() for r in logger.rows], sources=frames)
    print(f"skew audit: {report.rows} vectors, mismatch rate {report.mismatch_rate:.6f}\n")
    if args.strict and (report.mismatch_rate > 0 or report.rows == 0):
        for f in report.worst(3):
            print(f"  {f.feature}: {f.examples}", file=sys.stderr)
        sys.exit(1)

    drift = feature_drift(result.artifact, [r.features for r in logger.rows])
    print("top drift (PSI):", [(d.feature, d.psi, d.status) for d in drift[:3]], "\n")

    # 5. Late delivery.
    if args.late_fraction > 0:
        logger, stats = replay_live_day(repo, frames, live_start, late_fraction=args.late_fraction)
        print(f"replay with {args.late_fraction:.0%} of events delayed 20 min:", stats)
        report = skew_audit(service, [r.to_dict() for r in logger.rows], sources=frames)
        print(f"skew audit: mismatch rate {report.mismatch_rate:.6f}; worst features:")
        for f in report.worst(4):
            print(f"  {f.feature:45s} {f.mismatches:4d}/{f.compared} ({f.mismatch_rate:.2%})")


if __name__ == "__main__":
    main()
