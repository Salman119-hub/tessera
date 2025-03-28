"""Model artifacts and a file-based registry with stage aliases.

An artifact records the feature service *fingerprint* it was trained on.
Serving refuses to load it against different live definitions (the skew
guard), which turns a silent training/serving mismatch into a deploy-time
error.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tessera import __version__
from tessera.definitions import FeatureService
from tessera.errors import SkewGuardError
from tessera.training.model import FloatArray, LogisticModel, Preprocessor

ARTIFACT_FILE = "model.json"
_NAME_RE = re.compile(r"^[a-z][a-z0-9_\-]{0,63}$")


@dataclass
class ModelArtifact:
    name: str
    service: str
    service_fingerprint: str
    features: list[str]
    preprocessor: Preprocessor
    model: LogisticModel
    threshold: float
    metrics: dict[str, float] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    version: int | None = None
    tessera_version: str = __version__

    def predict_proba_rows(self, rows: Sequence[dict[str, Any]]) -> FloatArray:
        return self.model.predict_proba(self.preprocessor.transform_rows(rows))

    def score(self, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        probs = self.predict_proba_rows(rows)
        return [
            {"score": float(p), "decision": "review" if p >= self.threshold else "approve"}
            for p in probs
        ]

    def check_compatible(self, service: FeatureService) -> None:
        if service.name != self.service:
            raise SkewGuardError(
                f"model {self.name!r} was trained on service {self.service!r}, not {service.name!r}"
            )
        if service.fingerprint != self.service_fingerprint:
            raise SkewGuardError(
                f"model {self.name!r} v{self.version} was trained on {self.service}@"
                f"{self.service_fingerprint}, but the live definition is "
                f"{service.fingerprint}. Retrain, or roll back the definition change."
            )

    def explain(self, row: dict[str, Any], top: int = 5) -> list[dict[str, Any]]:
        """Per-feature contributions (weight x standardised value), largest first."""
        x = self.preprocessor.transform_rows([row])[0]
        contributions = np.asarray(self.model.weights) * x
        names = self.preprocessor.output_names
        order = np.argsort(-np.abs(contributions))[:top]
        return [
            {"feature": names[i], "contribution": round(float(contributions[i]), 4)} for i in order
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "service": self.service,
            "service_fingerprint": self.service_fingerprint,
            "features": self.features,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "training": self.training,
            "tessera_version": self.tessera_version,
            "preprocessor": self.preprocessor.to_dict(),
            "model": {"type": "logistic", **self.model.to_dict()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelArtifact:
        model_data = dict(data["model"])
        if model_data.pop("type", "logistic") != "logistic":
            raise ValueError("unsupported model type")
        return cls(
            name=data["name"],
            version=data.get("version"),
            service=data["service"],
            service_fingerprint=data["service_fingerprint"],
            features=list(data["features"]),
            threshold=float(data["threshold"]),
            metrics=dict(data.get("metrics", {})),
            training=dict(data.get("training", {})),
            tessera_version=data.get("tessera_version", "unknown"),
            preprocessor=Preprocessor.from_dict(data["preprocessor"]),
            model=LogisticModel.from_dict(model_data),
        )

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        file = path / ARTIFACT_FILE
        file.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return file

    @classmethod
    def load(cls, directory: str | Path) -> ModelArtifact:
        path = Path(directory)
        file = path / ARTIFACT_FILE if path.is_dir() else path
        return cls.from_dict(json.loads(file.read_text(encoding="utf-8")))


class ModelRegistry:
    """``<root>/<name>/v<N>/model.json`` plus ``<root>/<name>/aliases.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _model_dir(self, name: str) -> Path:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid model name {name!r}")
        return self.root / name

    def versions(self, name: str) -> list[int]:
        directory = self._model_dir(name)
        if not directory.exists():
            return []
        return sorted(
            int(p.name[1:]) for p in directory.iterdir() if p.is_dir() and p.name[1:].isdigit()
        )

    def register(self, artifact: ModelArtifact) -> int:
        directory = self._model_dir(artifact.name)
        directory.mkdir(parents=True, exist_ok=True)
        version = (max(self.versions(artifact.name), default=0)) + 1
        artifact.version = version
        target = directory / f"v{version}"
        tmp = directory / f".v{version}.tmp"
        artifact.save(tmp)
        os.replace(tmp, target)
        return version

    def aliases(self, name: str) -> dict[str, int]:
        file = self._model_dir(name) / "aliases.json"
        if not file.exists():
            return {}
        data: dict[str, int] = json.loads(file.read_text(encoding="utf-8"))
        return data

    def set_alias(self, name: str, alias: str, version: int) -> None:
        if version not in self.versions(name):
            raise KeyError(f"model {name!r} has no version {version}")
        aliases = self.aliases(name)
        aliases[alias] = version
        file = self._model_dir(name) / "aliases.json"
        tmp = file.with_suffix(".tmp")
        tmp.write_text(json.dumps(aliases, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, file)

    def resolve(self, name: str, ref: str | int = "production") -> int:
        if isinstance(ref, int) or str(ref).isdigit():
            version = int(ref)
            if version not in self.versions(name):
                raise KeyError(f"model {name!r} has no version {version}")
            return version
        aliases = self.aliases(name)
        if ref == "latest":
            versions = self.versions(name)
            if not versions:
                raise KeyError(f"model {name!r} has no versions")
            return versions[-1]
        if ref not in aliases:
            raise KeyError(f"model {name!r} has no alias {ref!r}")
        return aliases[ref]

    def load(self, name: str, ref: str | int = "production") -> ModelArtifact:
        version = self.resolve(name, ref)
        return ModelArtifact.load(self._model_dir(name) / f"v{version}")
