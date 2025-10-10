# Architecture

## Problem

A real-time model is only as good as the agreement between the features it
was trained on and the features it is served. The usual setup computes them
twice: SQL or Spark for training, and a hand-written stream job and cache for
serving. The two drift apart in small ways, such as an off-by-one window
boundary, `NaN` versus `null`, a late event counted in one path and not the
other. The model then underperforms in production for reasons that never show
up in offline evaluation.

Tessera removes the second implementation. Features are declared once, and
one aggregation engine produces both the training values and the serving
values.

## Components

```
                 features.py (definitions)
                         │  plan / apply  ──► registry.json (fingerprints)
          ┌──────────────┴───────────────┐
          ▼                              ▼
  OFFLINE (batch)                   ONLINE (real time)
  archive + spine                   Kafka / Event Hubs
     │                                   │
     ▼                                   ▼
  replay through TileEngine         StreamProcessor ──► dead letters
     │                                   │  (dedupe, lateness, CAS writes)
     ▼                                   ▼
  training set ──► train ──► gate   online store (SQLite / Redis)
                     │                   │
                     ▼                   ▼
               model registry ───► FeatureServer ──► /v1/score
                (skew guard)             │
                                         ▼
                                    feature log ──► skew audit / drift
```

### Definitions (`tessera.definitions`, `tessera.repository`)

Entities, sources (batch tables or streams with an archive), and three kinds
of feature view:

| View | Computes | Offline | Online |
|---|---|---|---|
| `AggregateFeatureView` | windowed `count/sum/mean/min/max/std/last/count_distinct` | replay through `TileEngine` | `TileEngine` state per entity |
| `SnapshotFeatureView` | latest row of a slowly changing table, with a TTL | as-of join, strictly before *t* | latest row per entity |
| `OnDemandFeatureView` | a Python function of request data and other features | same function per row | same function per request |

A `FeatureService` is the versioned feature vector a model consumes. Every
definition has a canonical spec and a 12-hex fingerprint. Row transforms and
on-demand functions are fingerprinted by source code, so editing a function
body counts as a definition change.

### The tile engine (`tessera.tiles`, `tessera.aggregation`)

Each aggregation is a mergeable monoid with JSON-serialisable state. HLL
sketches are stored sparsely, and the standard deviation uses Welford updates
with Chan's merge. Events update one tile per *resolution*: the view's
granularity plus each aggregate's hop. A read for window `W` and hop `H` at
time `t` covers `[align(t, H) - W, t)`. It merges coarse tiles for the bulk
of the window, finer tiles for the remainder, and replays raw events for the
current granularity tile (see [ADR-0001](adr/0001-tiled-sawtooth-windows.md)).

The raw buffer does double duty. It makes the head of the window exact, and
it deduplicates by event id over the lateness horizon. Tiles and raw events
are pruned relative to the entity's newest event, so state per entity is
bounded by the longest window, not by history.

### Offline (`tessera.offline`)

`get_historical_features(service, spine)` sorts each entity's events and spine
rows by time, applies events strictly before each spine timestamp, and reads
the same `TileEngine`. Parity therefore holds by construction rather than by
testing two implementations against each other
([ADR-0002](adr/0002-offline-by-replay.md)). Snapshot views use
`merge_asof` with `allow_exact_matches=False` and a TTL. On-demand views
receive `None` (never `NaN`) for missing values, exactly as they do online.

The Spark job (`pipelines/spark/build_training_set.py`) distributes the same
functions with `cogroup().applyInPandas`, partitioned by entity key.

### Online (`tessera.online`, `tessera.streaming`)

* **Store.** `read` (batched) and `write` with compare-and-set on a per-key
  version. Implementations: in-memory, SQLite in WAL mode, and Redis (an atomic
  Lua CAS script).
* **Writer.** Groups a batch by entity, reads each state once, applies the
  events, and writes conditionally. Keys that lose a race are re-read and
  re-applied. Dedup by event id makes re-application safe
  ([ADR-0003](adr/0003-cas-writes-and-at-least-once.md)).
* **Stream processor.** Poll, validate against the source schema, apply, and
  commit offsets only after the writes succeed. Schema failures and
  undecodable messages go to a dead-letter sink. Events later than
  `allowed_lateness` relative to the entity's newest event are counted and
  dropped.
* **Feature server.** One batched store read per stateful view regardless of
  batch size, then the TTL check, then on-demand views. Every served vector
  is logged with its request time and the service fingerprint.

The online namespace for a view is `name:fingerprint`. A changed definition
reads and writes a fresh, empty namespace and never sees state built under
the old definition. `tessera plan` flags such views as needing a backfill
([ADR-0004](adr/0004-fingerprinted-namespaces-and-skew-guard.md)).

### Training (`tessera.training`)

* Out-of-time split with a gap before the cutoff. Labels such as chargebacks
  mature late, and windowed features near the cutoff overlap the test period.
* A deterministic baseline: median imputation with missing indicators,
  signed-log compression of heavy-tailed columns, standardisation, and
  L2-regularised logistic regression fitted by Newton's method with Armijo
  backtracking.
* A cost-based decision threshold, chosen on training data only.
* Metrics that matter under imbalance: average precision, recall at 1% FPR
  and calibration (ECE), alongside ROC AUC.
* A promotion gate with absolute floors plus tolerances against the current
  production model.
* Artifacts are a single JSON file: weights, preprocessing, drift reference
  bins and the service fingerprint. Nothing is pickled
  ([ADR-0005](adr/0005-json-artifacts-and-a-linear-baseline.md)).

### Monitoring (`tessera.monitoring`)

* **Skew audit.** Rebuilds a spine from the feature log (entities, request
  fields, request time) and recomputes it offline. Each feature is compared
  with a relative tolerance of 1e-6. Parity is the design goal, so any
  mismatch has an operational cause: lateness drops, a stream outage, missing
  CDC, or a deploy without a backfill.
* **Drift.** PSI of served values against the training quantile bins stored
  in the artifact.

## Measured behaviour

Measured on the bundled synthetic dataset (1,200 cards, 300 merchants, 28 days,
43,970 transactions, 1.2% fraud) on a single laptop core, Python 3.13:

| What | Result |
|---|---|
| Point-in-time training set, 42,446 rows x 23 features | 6.8 s |
| Online feature retrieval, single row, in process (memory / SQLite) | p50 0.85 / 0.92 ms, p99 2.0 / 2.1 ms |
| Online retrieval, batch of 100 rows (memory / SQLite) | 22 / 26 ms |
| Clean live-day replay: online vs offline, 1,524 vectors x 23 features | 0 mismatches |
| Same replay with 2% of events delivered 20 min late | 0.50% of values differ, concentrated in merchant features |

The latency figures exclude HTTP and network and come from one machine. Treat
them as an order of magnitude, not a capacity plan. A Redis store adds a
network round trip per view.

## Scaling notes

* Stream throughput scales with partitions. Key the topic by the entity of
  the busiest view so that one partition owns each entity. CAS then almost
  never conflicts and ordering per entity is preserved.
* A view keyed by a different entity than the topic key (here `merchant_txn`
  on a card-keyed topic) sees cross-partition writes. CAS keeps it correct,
  and retries absorb the contention. For very hot keys, re-key into a
  dedicated topic.
* State per entity is bounded by the longest window. Hot entities (a busy
  merchant) carry larger raw buffers, because the buffer covers lateness plus
  one granularity tile.
* Offline replay is roughly linear in events plus spine rows. The Spark job
  parallelises it by entity.

## Deployment (Azure)

Terraform (`infra/terraform`) provisions:

* AKS with workload identity and the Key Vault CSI driver
* Event Hubs Standard (Kafka endpoint, auto-inflate)
* Azure Cache for Redis (`noeviction`)
* ADLS Gen2 for archives, training sets, models and feature logs
* Key Vault, ACR and Log Analytics
* One user-assigned identity per workload, federated to its Kubernetes
  service account

The Helm chart (`deploy/helm/tessera`) runs:

* **API.** HPA, PDB, zone spread, an init container that syncs model
  artifacts, and a read-only root filesystem.
* **Stream processor.** Replicas at most the partition count, with a
  60-second drain on shutdown.

Airflow orchestrates the daily validate, build (Databricks), train/gate,
skew audit and drift tasks.
