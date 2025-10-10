# ADR-0003: Compare-and-set state writes with at-least-once delivery

**Status:** Accepted

## Context

Each event updates a per-entity blob with a read-modify-write. Two writers
updating the same entity concurrently silently lose one update under
last-write-wins. Concurrent writers happen during consumer-group rebalances,
or when a view is keyed differently from the topic. Exactly-once delivery
across Kafka and the store would need transactional machinery that neither
Redis nor Event Hubs offers cheaply.

## Decision

* Every stored value carries a version, and writes are conditional on the
  version that was read. This is a Lua script on Redis and a guarded `UPDATE`
  on SQLite.
* The writer re-reads and re-applies the batch's events for keys that lose
  the race, up to `max_retries`.
* Offsets are committed only after state writes succeed. A crash means
  redelivery, never loss.
* Redelivered events are recognised by event id, using the raw buffer that
  covers the lateness horizon, or rejected as late beyond it. In both cases
  they are not applied twice.

## Consequences

* Correct under concurrency and crashes without distributed transactions.
* Producers must supply stable event ids. The source schema requires the
  field.
* Under heavy contention on one key, retries add latency. The scaling notes
  recommend keying topics by the hottest view's entity.
