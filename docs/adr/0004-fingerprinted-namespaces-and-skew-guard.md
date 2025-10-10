# ADR-0004: Fingerprinted namespaces and a deploy-time skew guard

**Status:** Accepted

## Context

Feature definitions change: a window widens, a transform is fixed. Two
failure modes follow. If the change is applied to existing online state,
the stored tiles were built under the old definition, so values are wrong
until the window rolls over. If a model trained on the old definition is
served features computed under the new one, it receives inputs with a
different meaning. Neither failure raises an error.

## Decision

* Each view's online namespace is `name:fingerprint`. A changed definition
  starts from an empty namespace, and `tessera plan` marks it
  `[backfill required]`.
* Model artifacts record the fingerprint of the feature service they were
  trained on. The API checks it against the live definitions at startup and
  refuses to start on a mismatch (`SkewGuardError`).
* The skew audit skips logged vectors whose service fingerprint differs from
  the current one and reports how many it skipped.

## Consequences

* A definition change is safe by default. The deploy sequence is
  apply → backfill → retrain → deploy, and the platform enforces the order
  rather than relying on people to remember it.
* Old namespaces are not deleted automatically. They expire with a Redis TTL
  if one is configured, or need a cleanup job.
* Cosmetic edits to a transform's source (a comment, a rename) also change
  the fingerprint. We accept the false positive in exchange for never
  missing a real change.
