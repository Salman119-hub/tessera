"""Generate a synthetic card-transaction dataset with two fraud patterns.

* **Account takeover**: a compromised card makes a burst of 3-8 transactions
  within about 45 minutes, mostly online, larger than its usual spend, often
  abroad and at night.
* **Card testing**: a few merchants are used to test stolen cards, so many
  distinct cards make tiny online purchases there within an hour.

Legitimate traffic includes travel, occasional large purchases and night
owls, so neither pattern can be found from one transaction alone. That is
the point: velocity and profile features carry the signal.

Usage: ``python examples/fraud/generate_data.py [--cards 1200 --days 28]``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COUNTRIES = np.array(["US", "GB", "DE", "FR", "CA", "MX", "BR", "JP", "IN", "AU"])
HOME_WEIGHTS = np.array([0.70, 0.10, 0.08, 0.06, 0.06, 0, 0, 0, 0, 0])
HOUR_WEIGHTS = np.array(
    [0.4, 0.2, 0.1, 0.1, 0.1, 0.3, 1, 2, 3, 4, 4, 5, 6, 5, 4, 4, 5, 6, 6, 5, 4, 3, 2, 1]
)
HOUR_WEIGHTS = HOUR_WEIGHTS / HOUR_WEIGHTS.sum()


def generate(
    out_dir: str | Path,
    *,
    n_cards: int = 1200,
    n_merchants: int = 300,
    days: int = 28,
    seed: int = 7,
    start: str = "2026-09-01",
) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    t0 = int(pd.Timestamp(start, tz="UTC").timestamp())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Merchants: mostly domestic (US), popularity follows a power law.
    merchant_ids = np.array([f"m{i:04d}" for i in range(n_merchants)])
    merchant_country = rng.choice(
        COUNTRIES, size=n_merchants, p=[0.6, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02]
    )
    popularity = 1.0 / np.arange(1, n_merchants + 1) ** 0.8
    popularity /= popularity.sum()
    testing_merchants = rng.choice(np.arange(n_merchants // 2, n_merchants), size=3, replace=False)

    # Cards.
    card_ids = np.array([f"c{i:05d}" for i in range(n_cards)])
    home = rng.choice(COUNTRIES, size=n_cards, p=HOME_WEIGHTS)
    credit_limit = np.round(np.clip(rng.lognormal(np.log(5000), 0.6, n_cards), 300, 50_000), -2)
    account_age = rng.integers(20, 4000, n_cards)
    daily_rate = rng.gamma(2.0, 0.65, n_cards)
    spend_median = rng.lognormal(np.log(45), 0.5, n_cards)
    online_prop = rng.beta(2, 5, n_cards)

    rows: list[tuple[int, str, str, float, str, str, int]] = []

    def merchant_for(card: int, foreign: bool) -> int:
        if foreign:
            candidates = np.flatnonzero(merchant_country != home[card])
        else:
            candidates = np.flatnonzero(merchant_country == home[card])
        if len(candidates) == 0:
            candidates = np.arange(n_merchants)
        weights = popularity[candidates] / popularity[candidates].sum()
        return int(rng.choice(candidates, p=weights))

    for c in range(n_cards):
        regulars = [merchant_for(c, False) for _ in range(6)]
        travelling = rng.random() < 0.08
        trip_day = rng.integers(0, days)
        for day in range(days):
            for _ in range(rng.poisson(daily_rate[c])):
                ts = t0 + day * 86_400 + int(rng.choice(24, p=HOUR_WEIGHTS)) * 3600
                ts += int(rng.integers(0, 3600))
                abroad = travelling and trip_day <= day < trip_day + 4
                m = (
                    merchant_for(c, True)
                    if abroad
                    else (
                        regulars[rng.integers(0, 6)]
                        if rng.random() < 0.7
                        else merchant_for(c, False)
                    )
                )
                amount = spend_median[c] * rng.lognormal(0, 0.7)
                if rng.random() < 0.01:
                    amount *= rng.uniform(5, 15)  # legitimate big-ticket purchase
                channel = "online" if rng.random() < online_prop[c] else "pos"
                rows.append(
                    (ts, card_ids[c], merchant_ids[m], amount, str(merchant_country[m]), channel, 0)
                )

    # Account takeover bursts.
    compromised = rng.choice(n_cards, size=max(1, int(0.04 * n_cards)), replace=False)
    for c in compromised:
        night = rng.random() < 0.5
        hour = int(rng.integers(0, 5)) if night else int(rng.integers(9, 22))
        start_ts = t0 + int(rng.integers(3, days)) * 86_400 + hour * 3600
        start_ts += int(rng.integers(0, 3600))
        foreign = rng.random() < 0.7
        ts = start_ts
        for _ in range(int(rng.integers(3, 9))):
            ts += int(rng.integers(60, 600))
            m = merchant_for(c, foreign)
            amount = spend_median[c] * rng.uniform(2, 6) * rng.lognormal(0, 0.3)
            channel = "online" if rng.random() < 0.85 else "pos"
            rows.append(
                (ts, card_ids[c], merchant_ids[m], amount, str(merchant_country[m]), channel, 1)
            )

    # Card-testing attacks at a few merchants.
    for _ in range(max(1, days // 4)):
        m = int(rng.choice(testing_merchants))
        attack_start = t0 + int(rng.integers(1, days)) * 86_400 + int(rng.integers(0, 86_400))
        victims = rng.choice(n_cards, size=int(rng.integers(25, 60)), replace=False)
        for c in victims:
            ts = attack_start + int(rng.integers(0, 3600))
            amount = float(rng.uniform(0.5, 3.0))
            rows.append(
                (ts, card_ids[c], merchant_ids[m], amount, str(merchant_country[m]), "online", 1)
            )

    frame = pd.DataFrame(
        rows,
        columns=[
            "ts_epoch",
            "card_id",
            "merchant_id",
            "amount",
            "merchant_country",
            "channel",
            "is_fraud",
        ],
    )
    frame = frame.sort_values(["ts_epoch", "card_id"], kind="stable").reset_index(drop=True)
    frame["amount"] = frame["amount"].round(2)
    frame["event_id"] = [f"e{i:07d}" for i in range(len(frame))]
    frame["ts"] = pd.to_datetime(frame["ts_epoch"], unit="s", utc=True).dt.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    tx_cols = ["event_id", "card_id", "merchant_id", "amount", "merchant_country", "channel", "ts"]
    frame[tx_cols].to_csv(out / "transactions.csv", index=False)

    spine = frame[
        ["event_id", "card_id", "merchant_id", "amount", "merchant_country", "channel", "is_fraud"]
    ].copy()
    spine["timestamp"] = frame["ts"]
    spine.to_csv(out / "spine.csv", index=False)

    # Card master snapshots: one row at issue, plus credit-limit changes mid-period.
    profile_rows = []
    for c in range(n_cards):
        issued = t0 - 86_400
        profile_rows.append((card_ids[c], home[c], credit_limit[c], int(account_age[c]), issued))
        if rng.random() < 0.15:
            change_day = int(rng.integers(1, days))
            changed = t0 + change_day * 86_400 + 3600
            profile_rows.append(
                (
                    card_ids[c],
                    home[c],
                    round(credit_limit[c] * 1.5, -2),
                    int(account_age[c]) + change_day,
                    changed,
                )
            )
    profiles = pd.DataFrame(
        profile_rows,
        columns=["card_id", "home_country", "credit_limit", "account_age_days", "updated_epoch"],
    )
    profiles["updated_at"] = pd.to_datetime(
        profiles["updated_epoch"], unit="s", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    profiles.drop(columns=["updated_epoch"]).to_csv(out / "card_profiles.csv", index=False)
    return {
        "transactions": len(frame),
        "fraud": int(frame["is_fraud"].sum()),
        "cards": n_cards,
        "profiles": len(profiles),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--cards", type=int, default=1200)
    parser.add_argument("--merchants", type=int, default=300)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    summary = generate(
        args.out, n_cards=args.cards, n_merchants=args.merchants, days=args.days, seed=args.seed
    )
    print(summary)


if __name__ == "__main__":
    main()
