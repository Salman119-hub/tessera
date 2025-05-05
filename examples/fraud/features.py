"""Feature definitions for card-transaction fraud scoring.

Load with ``tessera --repo examples/fraud/features.py plan``.

Data lives in ``examples/fraud/data`` (generate it with
``python examples/fraud/generate_data.py``).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

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

DATA = Path(os.environ.get("TESSERA_EXAMPLE_DATA", Path(__file__).parent / "data"))

card = Entity("card", join_key="card_id", description="A payment card (PAN token).")
merchant = Entity("merchant", join_key="merchant_id", description="An acquiring merchant.")

transactions = StreamSource(
    name="transactions",
    path=str(DATA / "transactions.csv"),
    topic="payments.transactions.v1",
    timestamp_field="ts",
    event_id_field="event_id",
    allowed_lateness="10m",
    schema=(
        Field("event_id", "string"),
        Field("card_id", "string"),
        Field("merchant_id", "string"),
        Field("amount", "float"),
        Field("merchant_country", "string"),
        Field("channel", "string"),
    ),
    checks=(
        Check.not_null("event_id"),
        Check.unique("event_id"),
        Check.not_null("card_id"),
        Check.between("amount", 0, 50_000),
        Check.in_set("channel", ["pos", "online"]),
        Check.regex("merchant_country", r"[A-Z]{2}", mostly=0.99),
    ),
    description="Authorisation events, archived from Kafka to the lake hourly.",
)

card_profiles = BatchSource(
    name="card_profiles",
    path=str(DATA / "card_profiles.csv"),
    timestamp_field="updated_at",
    schema=(
        Field("card_id", "string"),
        Field("home_country", "string"),
        Field("credit_limit", "float"),
        Field("account_age_days", "int"),
    ),
    checks=(
        Check.not_null("card_id"),
        Check.between("credit_limit", 100, 100_000),
        Check.between("account_age_days", 0, 30_000),
    ),
    description="Daily CDC snapshot of the card master table.",
)


def _flags(row: dict[str, Any]) -> dict[str, Any]:
    hour = (row["_ts"] % 86_400) // 3600
    return {
        **row,
        "is_online": 1.0 if row["channel"] == "online" else 0.0,
        "is_night": 1.0 if hour < 6 else 0.0,
    }


card_txn = AggregateFeatureView(
    name="card_txn",
    entity=card,
    source=transactions,
    granularity="5m",
    transform=_flags,
    aggregates=(
        Aggregate(None, "count", "10m"),
        Aggregate(None, "count", "1h"),
        Aggregate(None, "count", "1d"),
        Aggregate("amount", "sum", "1h"),
        Aggregate("amount", "max", "1d"),
        Aggregate("amount", "mean", "30d"),
        Aggregate("amount", "std", "30d"),
        Aggregate("is_online", "mean", "30d"),
        Aggregate("is_online", "sum", "1h"),
        Aggregate("is_night", "sum", "1d"),
        Aggregate("merchant_id", "count_distinct", "1d"),
    ),
    owner="risk-ml",
    description="Card velocity and spend-profile features.",
)

merchant_txn = AggregateFeatureView(
    name="merchant_txn",
    entity=merchant,
    source=transactions,
    granularity="5m",
    aggregates=(
        Aggregate(None, "count", "1h"),
        Aggregate("card_id", "count_distinct", "1h"),
        Aggregate("amount", "mean", "1h"),
        Aggregate("amount", "mean", "7d"),
    ),
    owner="risk-ml",
    description="Merchant traffic shape; card-testing attacks show many cards, small amounts.",
)

card_profile = SnapshotFeatureView(
    name="card_profile",
    entity=card,
    source=card_profiles,
    fields=(
        Field("home_country", "string"),
        Field("credit_limit", "float"),
        Field("account_age_days", "int"),
    ),
    ttl="45d",
    owner="risk-ml",
)


@on_demand(
    request=(
        Field("amount", "float"),
        Field("merchant_country", "string"),
        Field("channel", "string"),
    ),
    outputs=(
        Field("amount_vs_card_mean", "float"),
        Field("amount_to_limit", "float"),
        Field("is_foreign", "bool"),
        Field("is_online", "bool"),
        Field("new_card_history", "bool"),
        Field("log_amount", "float"),
    ),
    depends_on=(card_txn, card_profile),
)
def txn_context(row: dict[str, Any]) -> dict[str, Any]:
    """Request-time context: how unusual this amount is for the card, and where it happens."""
    amount = row["amount"] or 0.0
    mean_30d = row["card_txn__amount_mean_30d"]
    limit = row["card_profile__credit_limit"]
    home = row["card_profile__home_country"]
    return {
        "amount_vs_card_mean": amount / mean_30d if mean_30d else 1.0,
        "amount_to_limit": amount / limit if limit else None,
        "is_foreign": home is not None and row["merchant_country"] != home,
        "is_online": row["channel"] == "online",
        "new_card_history": mean_30d is None,
        "log_amount": math.log1p(amount),
    }


fraud_v1 = FeatureService(
    name="fraud_v1",
    features=(
        card_txn,
        merchant_txn,
        card_profile.select("credit_limit", "account_age_days"),
        txn_context,
    ),
    owner="risk-ml",
    description="Real-time card fraud model inputs.",
)
