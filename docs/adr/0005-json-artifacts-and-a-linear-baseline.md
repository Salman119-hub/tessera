# ADR-0005: JSON model artifacts and a linear baseline

**Status:** Accepted

## Context

The platform's value is the features, not the model class. Teams will bring
gradient-boosted trees or neural models. The built-in model exists to make
the end-to-end loop real (train, gate, serve, monitor) and to provide an
honest baseline. Pickled artifacts tie loading to exact library versions and
execute code on load.

## Decision

* The built-in model is L2-regularised logistic regression with explicit
  preprocessing. It is fitted by Newton's method with Armijo backtracking,
  because pure Newton steps diverge on near-separable fraud data.
* The artifact is one JSON file: weights, imputation values, scaling, drift
  reference bins, the decision threshold, metrics and the service
  fingerprint.
* The `ModelArtifact` interface (`score`, `check_compatible`) is the
  extension point for other model types.

## Consequences

* Artifacts can be read in code review, diffed, and loaded safely by any
  version of the code that knows the schema.
* Per-feature contributions (`weight x standardised value`) give a cheap
  explanation for each score, which the API returns on request.
* A linear model cannot represent interactions. On the bundled synthetic data
  (`run_demo.py`), it reaches average precision 0.957 with the full feature
  set, against 0.703 with request and profile features only. The windowed
  features carry the signal, which is the point the example makes. Real
  traffic is noisier than the generator, so expect lower absolute numbers.
