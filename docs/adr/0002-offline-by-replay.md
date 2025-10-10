# ADR-0002: Compute offline features by replaying the online engine

**Status:** Accepted

## Context

The conventional design computes training features in SQL or Spark window
functions and serving features in a stream job. Keeping two implementations
equivalent is a permanent tax. Each new aggregation, null rule or boundary
convention has to be built twice and kept in agreement, and the differences
surface as unexplained production underperformance rather than as errors.

## Decision

Offline computation feeds each entity's events, in event-time order, into
the same `TileEngine` the stream processor uses, and reads it just before
each spine timestamp. There is no second implementation of window semantics.
The Spark job distributes this per entity key. It does not reimplement it in
Spark SQL.

## Consequences

* Parity holds by construction. The remaining differences are operational:
  events dropped as late online, outages, missing CDC. The skew audit
  measures exactly those.
* Offline throughput is per-event Python: about 6.8 s for 42k rows with 15
  aggregate features on one core. Large backfills are parallelised by entity
  on Spark instead of being vectorised. We accepted this cost for
  correctness. A vectorised fast path for `sum/count/mean` (cumulative sums)
  is a possible optimisation, but it would need its own parity tests.
* Floating-point sums can differ in the last bits when online events arrive
  out of order. The audit uses a relative tolerance of 1e-6.
