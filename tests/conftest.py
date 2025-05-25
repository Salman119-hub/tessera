from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from tessera import (
    Aggregate,
    AggregateFeatureView,
    BatchSource,
    Entity,
    FeatureService,
    Field,
    SnapshotFeatureView,
    StreamSource,
    on_demand,
)
from tessera.definitions import Check
from tessera.repository import Repository

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "fraud"
T0 = 1_788_000_000 - 1_788_000_000 % 86_400  # a midnight UTC

user = Entity("user", join_key="user_id")

clicks = StreamSource(
    name="clicks",
    path="unused.csv",
    topic="clicks.v1",
    timestamp_field="ts",
    event_id_field="event_id",
    allowed_lateness="5m",
    schema=(
        Field("event_id", "string"),
        Field("user_id", "string"),
        Field("value", "float"),
        Field("item", "string"),
    ),
    checks=(Check.not_null("user_id"), Check.between("value", 0, 1000)),
)

profiles = BatchSource(
    name="profiles",
    path="unused.csv",
    timestamp_field="updated_at",
    schema=(Field("user_id", "string"), Field("tier", "string"), Field("score", "float")),
)

user_clicks = AggregateFeatureView(
    name="user_clicks",
    entity=user,
    source=clicks,
    granularity="1m",
    aggregates=(
        Aggregate(None, "count", "10m"),
        Aggregate("value", "sum", "1h"),
        Aggregate("value", "max", "1h"),
        Aggregate("value", "mean", "1d"),
        Aggregate("value", "std", "1d"),
        Aggregate("item", "last", "1d"),
        Aggregate("item", "count_distinct", "1h"),
        Aggregate("value", "min", "3d"),
    ),
)

user_profile = SnapshotFeatureView(
    name="user_profile",
    entity=user,
    source=profiles,
    fields=(Field("tier", "string"), Field("score", "float")),
    ttl="2d",
)


@on_demand(
    request=(Field("price", "float"),),
    outputs=(Field("price_vs_mean", "float"), Field("has_history", "bool")),
    depends_on=(user_clicks,),
)
def click_context(row: dict[str, Any]) -> dict[str, Any]:
    mean = row["user_clicks__value_mean_1d"]
    return {
        "price_vs_mean": row["price"] / mean if mean else 0.0,
        "has_history": mean is not None,
    }


service = FeatureService(
    name="recs_v1",
    features=(user_clicks, user_profile.select("score"), click_context),
)


@pytest.fixture
def repo() -> Repository:
    return Repository([service])


def click(i: int, user_id: str, ts: int, value: float, item: str = "a") -> dict[str, Any]:
    return {"event_id": f"e{i}", "user_id": user_id, "value": value, "item": item, "ts": ts}


@pytest.fixture
def click_events() -> list[dict[str, Any]]:
    """Deterministic events for two users over two days."""
    import random

    rng = random.Random(3)
    events = []
    for i in range(400):
        user_id = "u1" if i % 3 else "u2"
        ts = T0 + rng.randint(0, 2 * 86_400)
        events.append(click(i, user_id, ts, round(rng.uniform(1, 100), 2), rng.choice("abcdefg")))
    return sorted(events, key=lambda e: e["ts"])


@pytest.fixture
def profile_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"user_id": "u1", "tier": "gold", "score": 0.9, "updated_at": T0 - 3600},
            {"user_id": "u1", "tier": "gold", "score": 0.7, "updated_at": T0 + 86_400},
            {"user_id": "u2", "tier": "basic", "score": 0.2, "updated_at": T0 - 5 * 86_400},
        ]
    )


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def example_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small generated copy of the fraud example dataset."""
    out = tmp_path_factory.mktemp("fraud_data")
    generator = load_module(EXAMPLE / "generate_data.py", "fraud_generate_data")
    generator.generate(out, n_cards=250, n_merchants=80, days=10, seed=11)
    return out


@pytest.fixture
def example_repo(example_data: Path, monkeypatch: pytest.MonkeyPatch) -> Repository:
    monkeypatch.setenv("TESSERA_EXAMPLE_DATA", str(example_data))
    return Repository.from_path(EXAMPLE / "features.py")
