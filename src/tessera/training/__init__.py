"""Training-set evaluation, a baseline model, artifacts and the model registry."""

from tessera.training.artifact import ModelArtifact, ModelRegistry
from tessera.training.model import LogisticModel, Preprocessor
from tessera.training.train import PromotionGate, TrainConfig, TrainResult, train_model

__all__ = [
    "LogisticModel",
    "ModelArtifact",
    "ModelRegistry",
    "Preprocessor",
    "PromotionGate",
    "TrainConfig",
    "TrainResult",
    "train_model",
]
