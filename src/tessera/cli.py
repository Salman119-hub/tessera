"""Command-line interface: ``tessera <command> --help``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from tessera import __version__
from tessera.config import Settings, load_settings
from tessera.errors import TesseraError
from tessera.repository import Repository, load_registry


def _settings(args: argparse.Namespace) -> Settings:
    settings = load_settings(args.config)
    if getattr(args, "repo", None):
        settings = settings.model_copy(update={"repo_path": args.repo})
    return settings


def _print(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _write_frame(frame: pd.DataFrame, path: str) -> None:
    if "://" not in path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    if path.endswith(".csv"):
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)


def _read_frame(path: str) -> pd.DataFrame:
    """CSV by extension; anything else is Parquet (a file or a partitioned directory).

    Remote URLs (``abfss://``, ``s3://``) work when the matching fsspec
    filesystem is installed, e.g. ``tessera[azure]``.
    """
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def cmd_plan(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    plan = repo.plan(load_registry(settings.registry_path))
    print(plan.render())
    return 0 if plan.empty or not args.detailed_exitcode else 2


def cmd_apply(args: argparse.Namespace) -> int:
    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    plan = repo.apply(settings.registry_path)
    print(plan.render())
    print(f"Registry written to {settings.registry_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from tessera.jobs import validate_sources

    repo = Repository.from_path(_settings(args).repo_path)
    reports = validate_sources(repo)
    _print([r.to_dict() for r in reports])
    return 0 if all(r.passed for r in reports) else 1


def cmd_backfill(args: argparse.Namespace) -> int:
    from tessera.jobs import backfill
    from tessera.online.store import build_store

    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    store = build_store(settings.online_store)
    try:
        results = backfill(repo, store, views=args.view or None)
    finally:
        store.close()
    _print([r.to_dict() for r in results])
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    from tessera.offline import get_historical_features

    repo = Repository.from_path(_settings(args).repo_path)
    service = repo.service(args.service)
    spine = _read_frame(args.spine)
    frame = get_historical_features(service, spine, timestamp_column=args.timestamp_column)
    _write_frame(frame, args.out)
    print(f"Wrote {len(frame)} rows x {len(service.output_columns)} features to {args.out}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from tessera.training import ModelRegistry, PromotionGate, TrainConfig, train_model

    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    service = repo.service(args.service)
    dataset = _read_frame(args.dataset)
    config = TrainConfig(
        label=args.label,
        timestamp_column=args.timestamp_column,
        l2=args.l2,
        cost_fp=args.cost_fp,
        cost_fn=args.cost_fn,
        gap=args.gap,
    )
    result = train_model(args.name, service, dataset, config)
    registry = ModelRegistry(settings.model_dir)
    version = registry.register(result.artifact)
    try:
        production = registry.load(args.name, "production").metrics
    except KeyError:
        production = None
    decision = PromotionGate().evaluate(result.artifact.metrics, production)
    promoted = False
    if args.promote and decision.promote:
        registry.set_alias(args.name, "production", version)
        promoted = True
    run_id = None
    if args.mlflow_experiment:
        from tessera.mlops import log_training_run

        run_id = log_training_run(result.artifact, experiment=args.mlflow_experiment)
    _print(
        {
            "model": args.name,
            "version": version,
            "service": f"{service.name}@{service.fingerprint}",
            "test_metrics": result.artifact.metrics,
            "gate": {"promote": decision.promote, "reasons": list(decision.reasons)},
            "promoted": promoted,
            "mlflow_run_id": run_id,
        }
    )
    return 0 if decision.promote else 3


def cmd_promote(args: argparse.Namespace) -> int:
    from tessera.training import ModelRegistry

    registry = ModelRegistry(_settings(args).model_dir)
    registry.set_alias(args.name, args.alias, args.version)
    print(f"{args.name}: {args.alias} -> v{args.version}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:  # pragma: no cover - starts a server
    import uvicorn

    from tessera.api import create_app

    app = create_app(_settings(args))
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_stream(args: argparse.Namespace) -> int:  # pragma: no cover - needs a broker
    import signal

    from tessera.kafka import KafkaBus
    from tessera.online.store import build_store
    from tessera.streaming import JsonlDeadLetter, StreamProcessor

    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    store = build_store(settings.online_store)
    topics = sorted({getattr(s, "topic", "") for s in repo.sources.values()} - {""})
    bus = KafkaBus(
        settings.kafka.bootstrap_servers, settings.kafka.group_id, topics, **settings.kafka.extra
    )
    processor = StreamProcessor(
        repo,
        store,
        bus,
        dead_letter=JsonlDeadLetter(settings.dead_letter),
        batch_size=args.batch_size,
    )
    signal.signal(signal.SIGTERM, lambda *_: processor.stop())
    processor.run(max_batches=args.max_batches)
    return 0


def cmd_skew_audit(args: argparse.Namespace) -> int:
    from tessera.monitoring import skew_audit
    from tessera.online.server import read_feature_log

    settings = _settings(args)
    repo = Repository.from_path(settings.repo_path)
    log_path = args.log or settings.feature_log
    if not log_path:
        raise TesseraError("no feature log configured; pass --log")
    report = skew_audit(repo.service(args.service), read_feature_log(log_path))
    _print(report.to_dict())
    return 0 if report.mismatch_rate <= args.max_mismatch_rate else 4


def cmd_drift(args: argparse.Namespace) -> int:
    from tessera.monitoring import feature_drift
    from tessera.online.server import read_feature_log
    from tessera.training import ModelRegistry

    settings = _settings(args)
    artifact = ModelRegistry(settings.model_dir).load(args.name, args.alias)
    log_path = args.log or settings.feature_log
    if not log_path:
        raise TesseraError("no feature log configured; pass --log")
    rows = [r["features"] for r in read_feature_log(log_path) if r["service"] == artifact.service]
    drift = feature_drift(artifact, rows, min_rows=args.min_rows)
    _print([{**d.__dict__, "status": d.status} for d in drift])
    return 0 if all(d.status != "alert" for d in drift) else 5


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tessera", description=__doc__)
    parser.add_argument("--version", action="version", version=f"tessera {__version__}")
    parser.add_argument("--config", help="YAML settings file (or TESSERA_CONFIG)")
    parser.add_argument("--repo", help="feature definitions file (overrides config)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="show definition changes against the registry")
    p.add_argument("--detailed-exitcode", action="store_true", help="exit 2 when there are changes")
    p.set_defaults(func=cmd_plan)

    sub.add_parser("apply", help="write the registry snapshot").set_defaults(func=cmd_apply)
    sub.add_parser("validate", help="run data quality checks on sources").set_defaults(
        func=cmd_validate
    )

    p = sub.add_parser("backfill", help="rebuild online state from source archives")
    p.add_argument("--view", action="append", help="limit to a view (repeatable)")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("build-dataset", help="point-in-time training set from a spine")
    p.add_argument("--service", required=True)
    p.add_argument("--spine", required=True, help="CSV/Parquet with keys, timestamp, label")
    p.add_argument("--out", required=True)
    p.add_argument("--timestamp-column", default="timestamp")
    p.set_defaults(func=cmd_build_dataset)

    p = sub.add_parser("train", help="train, register and (optionally) promote a model")
    p.add_argument("--name", required=True)
    p.add_argument("--service", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--label", default="label")
    p.add_argument("--timestamp-column", default="timestamp")
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--cost-fp", type=float, default=1.0)
    p.add_argument("--cost-fn", type=float, default=25.0)
    p.add_argument("--gap", default="1d")
    p.add_argument("--promote", action="store_true", help="promote if the gate passes")
    p.add_argument("--mlflow-experiment", help="also log the run to this MLflow experiment")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("promote", help="point an alias at a model version")
    p.add_argument("--name", required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--alias", default="production")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container default
    p.add_argument("--port", type=int, default=8080)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("stream", help="run the Kafka stream processor")
    p.add_argument("--batch-size", type=int, default=500)
    p.add_argument("--max-batches", type=int)
    p.set_defaults(func=cmd_stream)

    p = sub.add_parser("skew-audit", help="compare served features with offline recomputation")
    p.add_argument("--service", required=True)
    p.add_argument("--log", help="feature log (JSONL); defaults to settings.feature_log")
    p.add_argument("--max-mismatch-rate", type=float, default=0.001)
    p.set_defaults(func=cmd_skew_audit)

    p = sub.add_parser("drift", help="PSI of served features against the training reference")
    p.add_argument("--name", required=True)
    p.add_argument("--alias", default="production")
    p.add_argument("--log")
    p.add_argument("--min-rows", type=int, default=100)
    p.set_defaults(func=cmd_drift)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        code: int = args.func(args)
    except TesseraError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
