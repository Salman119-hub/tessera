# Runbook

## Skew audit above budget (`skew_audit` task red)

The daily audit replays yesterday's served vectors offline. Use `--max-mismatch-rate`, default 0.1%.

1. Open the report and find the worst features (`features[].mismatch_rate`).
   Look at `examples` for the entity and request time of each one.
2. Work out the pattern from which features are affected:
   * **Only one view's features**, usually short windows: events were dropped
     as late, or the stream lagged. Check
     `tessera_stream_events_total{outcome="late"}` and
     `tessera_stream_event_lag_seconds` around those request times. If late
     drops are structural (a producer that buffers), raise the source's
     `allowed_lateness`. That is a definition change and needs a backfill.
   * **Snapshot features only**: the CDC load was missing or delayed for
     those entities. Re-run the materialisation.
   * **Everything, starting at a deploy**: a definition changed without a
     backfill. See "Plan shows backfill required" below.
3. The online value is the one the model actually used. If the mismatch
   affected decisions, record the window of affected requests for the risk
   team.

## Stream lag rising

`tessera_stream_event_lag_seconds` measures wall-clock time minus the newest
event time per source.

* Check that the replica count is no greater than the partition count.
  Extra consumers sit idle.
* Check `tessera_stream_batch_seconds`. If batches are slow, the store is the
  usual bottleneck: check Redis CPU and latency.
* A spike in `outcome="invalid"` means a producer deployed a breaking
  change. The dead-letter log has the schema error for each record. Records
  can be replayed once the producer is fixed, because writes are idempotent
  by event id.
* Scaling the deployment rebalances partitions cooperatively, so there is no
  stop-the-world pause.

## Plan shows `[backfill required]`

The view's fingerprint changed, so it reads an empty online namespace until
backfilled.

1. `tessera apply` (records the new registry).
2. `tessera backfill --view <name>`. The stream processor writes to the new
   namespace only after the deploy, so the two do not race.
3. Retrain any model on a service that includes the view. The skew guard
   blocks the old model from loading against the new definitions.
4. Deploy. API pods refuse to start (`SkewGuardError`) if step 3 was skipped.

## API pods crash-loop with `SkewGuardError`

The promoted model was trained on different definitions than the ones
deployed. Either roll the definitions back to the model's fingerprint
(recorded in `model.json` under `service_fingerprint`), or train and promote
a model on the current ones. Do not bypass the check: the model would score
inputs whose meaning has changed.

## Promotion gate rejected a candidate (`train_and_gate` exit 3)

Production keeps serving the previous version. The CLI output lists the
reasons.

* **Floor breaches** (AUC, AP, ECE) usually point to label or data
  problems. Check `validate` output and the label pipeline's freshness.
* **Regression against production** with healthy floors: compare feature
  importance between versions. Check drift (`tessera drift`) for a shifted
  input. A real change in traffic may justify a manual promotion
  (`tessera promote`) after review. Record the decision.

## Redis memory high

The cache runs `noeviction`, so writes fail loudly instead of evicting
feature state silently.

* Old namespaces from earlier fingerprints are the usual cause. Delete
  `tessera:<view>:<old fingerprint>:*` once no deployed model uses them.
* Hot entities with large raw buffers come next. These are views with high
  `allowed_lateness` on busy keys.
* Then scale the cache tier.
