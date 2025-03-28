"""Optional MLflow tracking for training runs (``pip install tessera[mlflow]``).

The file-based :class:`~tessera.training.ModelRegistry` stays the source of
truth for serving, because the skew guard needs the feature-service
fingerprint next to the weights. MLflow receives a copy of each run for
experiment comparison, tagged with that fingerprint so the two can be joined.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from tessera.training.artifact import ModelArtifact


def log_training_run(
    artifact: ModelArtifact,
    *,
    experiment: str,
    tracking_uri: str | None = None,
    extra_params: dict[str, Any] | None = None,
) -> str:
    """Log params, test metrics and the artifact JSON. Returns the MLflow run id."""
    try:
        import mlflow
    except ImportError as exc:
        raise ImportError("MLflow logging requires `pip install tessera[mlflow]`") from exc
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=f"{artifact.name}-v{artifact.version}") as run:
        mlflow.set_tags(
            {
                "tessera.service": artifact.service,
                "tessera.service_fingerprint": artifact.service_fingerprint,
                "tessera.model_version": str(artifact.version),
            }
        )
        params = {k: v for k, v in artifact.training.items() if isinstance(v, (int, float, str))}
        mlflow.log_params({**params, **(extra_params or {})})
        mlflow.log_metrics({k: float(v) for k, v in artifact.metrics.items()})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(artifact.to_dict(), indent=2), encoding="utf-8")
            mlflow.log_artifact(str(path), artifact_path="tessera")
        return str(run.info.run_id)
