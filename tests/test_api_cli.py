from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from tessera.api import Container, create_app
from tessera.cli import main
from tessera.config import ServedModel, load_settings
from tessera.errors import SkewGuardError
from tessera.online.store import InMemoryStore
from tessera.repository import Repository
from tessera.training import ModelRegistry, TrainConfig, train_model

from .conftest import EXAMPLE


@pytest.fixture
def workspace(example_data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("TESSERA_EXAMPLE_DATA", str(example_data))
    config = tmp_path / "tessera.yaml"
    config.write_text(
        "\n".join(
            [
                f"repo_path: {(EXAMPLE / 'features.py').as_posix()}",
                "registry_path: state/registry.json",
                "online_store: sqlite:///state/online.db",
                "model_dir: state/models",
                "feature_log: state/feature_log.jsonl",
                "models: [{name: card_fraud}]",
            ]
        )
    )
    return tmp_path


def run(*args: str) -> int:
    return main(list(args))


class TestConfig:
    def test_layers_and_relative_paths(self, workspace: Path) -> None:
        env = {
            "TESSERA_API_KEYS": '["k1"]',
            "TESSERA_KAFKA__GROUP_ID": "g2",
            "TESSERA_LOG_LEVEL": "DEBUG",
        }
        settings = load_settings(workspace / "tessera.yaml", environ=env)
        assert settings.registry_path == str((workspace / "state" / "registry.json").resolve())
        assert settings.online_store == f"sqlite:///{(workspace / 'state' / 'online.db').resolve()}"
        assert settings.api_keys == ["k1"]
        assert settings.kafka.group_id == "g2"
        assert settings.models == [ServedModel(name="card_fraud")]
        assert load_settings(environ={}).online_store.startswith("sqlite:///")

    def test_rejects_non_mapping(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n")
        with pytest.raises(ValueError, match="mapping"):
            load_settings(bad, environ={})


@pytest.mark.slow
def test_cli_workflow_and_api(
    workspace: Path, example_data: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cfg = str(workspace / "tessera.yaml")
    assert run("--config", cfg, "plan", "--detailed-exitcode") == 2
    assert run("--config", cfg, "apply") == 0
    assert run("--config", cfg, "plan", "--detailed-exitcode") == 0
    assert run("--config", cfg, "validate") == 0
    capsys.readouterr()
    assert run("--config", cfg, "backfill") == 0
    backfilled = {r["view"]: r for r in json.loads(capsys.readouterr().out)}
    assert set(backfilled) == {"card_txn", "merchant_txn", "card_profile"}
    assert backfilled["card_profile"]["entities"] == 250

    dataset = workspace / "dataset.csv"
    assert (
        run(
            "--config",
            cfg,
            "build-dataset",
            "--service",
            "fraud_v1",
            "--spine",
            str(example_data / "spine.csv"),
            "--out",
            str(dataset),
        )
        == 0
    )
    assert len(pd.read_csv(dataset)) == len(pd.read_csv(example_data / "spine.csv"))
    capsys.readouterr()
    code = run(
        "--config",
        cfg,
        "train",
        "--name",
        "card_fraud",
        "--service",
        "fraud_v1",
        "--dataset",
        str(dataset),
        "--label",
        "is_fraud",
        "--promote",
    )
    trained = json.loads(capsys.readouterr().out)
    assert code == 0, trained["gate"]
    assert trained["promoted"] and trained["test_metrics"]["roc_auc"] > 0.9
    assert run("--config", cfg, "promote", "--name", "card_fraud", "--version", "1") == 0

    settings = load_settings(cfg, environ={"TESSERA_API_KEYS": '["secret"]'})
    client = TestClient(create_app(settings))
    headers = {"x-api-key": "secret"}
    assert client.get("/healthz").json()["status"] == "ok"
    assert client.get("/readyz").json()["models"] == ["card_fraud"]
    assert client.get("/v1/registry").status_code == 401
    assert "fraud_v1" in client.get("/v1/registry", headers=headers).json()["services"]

    row = {
        "card_id": "c00001",
        "merchant_id": "m0003",
        "amount": 950.0,
        "merchant_country": "BR",
        "channel": "online",
    }
    features = client.post(
        "/v1/features", headers=headers, json={"service": "fraud_v1", "rows": [row]}
    )
    assert features.status_code == 200
    vector = features.json()["features"][0]
    assert len(vector) == 23 and vector["txn_context__is_online"] is True
    scored = client.post(
        "/v1/models/card_fraud/score", headers=headers, json={"rows": [row], "explain": True}
    )
    body = scored.json()
    assert scored.status_code == 200 and body["version"] == 1
    assert 0 <= body["results"][0]["score"] <= 1 and len(body["results"][0]["explanation"]) == 5

    ingest = client.post(
        "/v1/ingest/transactions",
        headers=headers,
        json={
            "records": [
                {
                    "event_id": "live-1",
                    "card_id": "c00001",
                    "merchant_id": "m0003",
                    "amount": 10.0,
                    "merchant_country": "US",
                    "channel": "pos",
                    "ts": "2030-01-01T00:00:00Z",
                },
                {
                    "event_id": "live-2",
                    "card_id": "c00001",
                    "amount": "abc",
                    "ts": "2030-01-01T00:00:00Z",
                },
            ]
        },
    )
    assert ingest.json()["applied"] == 2  # one event, two views
    assert ingest.json()["invalid"] == 1 and ingest.json()["errors"]

    assert (
        client.post(
            "/v1/features", headers=headers, json={"service": "nope", "rows": [row]}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/v1/features",
            headers=headers,
            json={"service": "fraud_v1", "rows": [{"card_id": "c1"}]},
        ).status_code
        == 422
    )
    assert (
        client.post("/v1/models/other/score", headers=headers, json={"rows": [row]}).status_code
        == 404
    )
    assert (
        client.post("/v1/ingest/nope", headers=headers, json={"records": [{}]}).status_code == 404
    )
    metrics = client.get("/metrics").text
    assert "tessera_scores_total" in metrics and "tessera_http_request_seconds_bucket" in metrics
    assert scored.headers["x-request-id"]

    capsys.readouterr()
    assert (
        run("--config", cfg, "skew-audit", "--service", "fraud_v1", "--max-mismatch-rate", "1") == 0
    )
    audit = json.loads(capsys.readouterr().out)
    assert audit["rows"] >= 2
    assert run("--config", cfg, "drift", "--name", "card_fraud", "--min-rows", "1") in (0, 5)


def test_skew_guard_blocks_startup(
    example_repo: Repository, example_data: Path, tmp_path: Path
) -> None:
    spine = pd.read_csv(example_data / "spine.csv")
    from tessera.offline import get_historical_features

    service = example_repo.service("fraud_v1")
    ds = get_historical_features(service, spine.iloc[:3000])
    artifact = train_model(
        "card_fraud", service, ds, TrainConfig(label="is_fraud", gap="1h")
    ).artifact
    artifact.service_fingerprint = "deadbeef0000"
    registry = ModelRegistry(tmp_path / "models")
    version = registry.register(artifact)
    registry.set_alias("card_fraud", "production", version)
    settings = load_settings(environ={}).model_copy(
        update={
            "repo_path": str(EXAMPLE / "features.py"),
            "model_dir": str(tmp_path / "models"),
            "feature_log": None,
            "models": [ServedModel(name="card_fraud")],
        }
    )
    with pytest.raises(SkewGuardError):
        Container.build(settings, store=InMemoryStore())


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--repo", str(tmp_path / "missing.py"), "plan") == 1
    assert "not found" in capsys.readouterr().err
