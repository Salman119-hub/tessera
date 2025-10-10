# ADR-0001: Tiled aggregation with sawtooth windows

**Status:** Accepted

## Context

Fraud features need fresh short windows (transactions in the last 10 minutes)
and long ones (mean spend over 30 days) for the same entity, served in
milliseconds. Exact sliding windows need every raw event in the window. For
a 30-day window on a busy merchant that is too much state to read per
request. Pure tumbling windows are cheap, but their values jump at
boundaries and leave the most recent events out.

## Decision

* Store per-entity partial aggregates (tiles) at a small set of resolutions:
  the view's granularity plus one hop per aggregate.
* The default hop is the granularity for windows up to 2 hours, 1 hour for
  windows up to 2 days, and 1 day beyond that.
* Define the value at time `t` for window `W` and hop `H` as covering
  `[align(t, H) - W, t)`. The window length varies between `W` and `W + H`, a
  sawtooth.
* Cover the tail between `align(t, H)` and `t` with progressively finer tiles,
  and cover the current granularity tile with raw events filtered to `< t`.
  The most recent events are therefore always counted, and an event at
  exactly `t` (the one being scored) never is.

## Consequences

* A read merges at most about `W/H` tiles for its main span plus a bounded
  number of filler tiles. That is about 70 merges for a 30-day window, instead
  of thousands of raw events.
* The semantics are a definition, not an approximation. The offline engine
  uses the same definition, and tests check the engine against a brute-force
  implementation of the interval at random times.
* Users see a sawtooth for long windows. The hop is visible in the
  definition, can be overridden per aggregate, and is part of the
  fingerprint.
* Every aggregation must be mergeable. Exact medians and percentiles are
  therefore not offered; a sketch-based quantile would be the extension.
