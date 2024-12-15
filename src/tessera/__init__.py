"""Tessera: a feature platform for real-time machine learning.

Feature definitions are declared once in Python. The same tiled aggregation
code computes them offline (point-in-time correct training sets) and online
(low-latency serving), so training and serving values agree exactly.
"""

from tessera.definitions import (
    Aggregate,
    AggregateFeatureView,
    BatchSource,
    Entity,
    FeatureService,
    Field,
    OnDemandFeatureView,
    SnapshotFeatureView,
    StreamSource,
    on_demand,
)

__version__ = "0.9.0"

__all__ = [
    "Aggregate",
    "AggregateFeatureView",
    "BatchSource",
    "Entity",
    "FeatureService",
    "Field",
    "OnDemandFeatureView",
    "SnapshotFeatureView",
    "StreamSource",
    "__version__",
    "on_demand",
]
