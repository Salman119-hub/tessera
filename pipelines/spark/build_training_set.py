"""Point-in-time training sets on Spark (Databricks, EMR, Synapse or HDInsight).

The single-node engine in :mod:`tessera.offline` is the reference
implementation. This job distributes *the same code*: rows are partitioned by
entity key and each partition runs ``compute_aggregate_features`` or
``compute_snapshot_features`` inside ``cogroup().applyInPandas``, so a Spark
training set matches the online path exactly, just as the local one does.

Usage::

    spark-submit pipelines/spark/build_training_set.py \\
        --repo abfss://features@<account>.dfs.core.windows.net/repo/features.py \\
        --service fraud_v1 \\
        --spine abfss://lake@<account>.dfs.core.windows.net/labels/2026-11/ \\
        --out abfss://lake@<account>.dfs.core.windows.net/training/fraud_v1/2026-11/ \\
        --source transactions=abfss://lake@.../archive/transactions/ \\
        --source card_profiles=abfss://lake@.../cdc/card_profiles/

Each ``--source`` overrides a definition's ``path`` with a Parquet location.
"""

from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

from tessera.definitions import (
    AggregateFeatureView,
    BatchSource,
    FeatureService,
    OnDemandFeatureView,
    SnapshotFeatureView,
)
from tessera.offline import (
    compute_aggregate_features,
    compute_on_demand_features,
    compute_snapshot_features,
)
from tessera.repository import Repository
from tessera.schema import TS_COLUMN, coerce_frame, coerce_value
from tessera.timeutil import series_to_epoch_seconds

ROW_ID = "_row_id"
SPARK_TYPES: dict[str, T.DataType] = {
    "int": T.LongType(),
    "float": T.DoubleType(),
    "string": T.StringType(),
    "bool": T.BooleanType(),
}


def feature_schema(view: Any) -> T.StructType:
    fields = [T.StructField(ROW_ID, T.LongType(), False)]
    for name in view.feature_names:
        fields.append(
            T.StructField(view.qualified(name), SPARK_TYPES[view.feature_dtype(name)], True)
        )
    return T.StructType(fields)


def _typed(frame: pd.DataFrame, view: Any) -> pd.DataFrame:
    """Normalise object columns so Arrow can serialise them against the schema."""
    out = frame.copy()
    for name in view.feature_names:
        column = view.qualified(name)
        dtype = view.feature_dtype(name)
        out[column] = [coerce_value(v, dtype) for v in out[column]]
    return out


def view_features(
    spine: DataFrame,
    events: DataFrame,
    view: AggregateFeatureView | SnapshotFeatureView,
    timestamp_column: str,
) -> DataFrame:
    key = view.entity.join_key
    source: BatchSource = view.source

    def compute(spine_pdf: pd.DataFrame, events_pdf: pd.DataFrame) -> pd.DataFrame:
        if spine_pdf.empty:
            return pd.DataFrame(columns=[ROW_ID, *map(view.qualified, view.feature_names)])
        spine_pdf = spine_pdf.copy()
        spine_pdf[TS_COLUMN] = series_to_epoch_seconds(spine_pdf[timestamp_column])
        if events_pdf.empty:
            events_pdf = pd.DataFrame(columns=[*source.columns, source.timestamp_field])
        typed_events = coerce_frame(events_pdf, source)
        if isinstance(view, AggregateFeatureView):
            features = compute_aggregate_features(view, typed_events, spine_pdf)
        else:
            features = compute_snapshot_features(view, typed_events, spine_pdf)
        features.insert(0, ROW_ID, spine_pdf[ROW_ID].to_numpy())
        return _typed(features, view)

    columns = [*source.columns, source.timestamp_field]
    return (
        spine.select(ROW_ID, key, timestamp_column)
        .groupBy(key)
        .cogroup(events.select(*dict.fromkeys(columns)).groupBy(key))
        .applyInPandas(compute, schema=feature_schema(view))
    )


def build(
    spark: SparkSession,
    service: FeatureService,
    spine: DataFrame,
    source_paths: dict[str, str],
    timestamp_column: str = "timestamp",
) -> DataFrame:
    spine = spine.withColumn(ROW_ID, F.monotonically_increasing_id())
    result = spine
    sources: dict[str, DataFrame] = {}
    for view in service.views:
        if isinstance(view, (AggregateFeatureView, SnapshotFeatureView)):
            name = view.source.name
            if name not in sources:
                sources[name] = spark.read.parquet(source_paths.get(name, view.source.path))
            features = view_features(spine, sources[name], view, timestamp_column)
            result = result.join(features, on=ROW_ID, how="left")

    on_demand = [v for v in service.views if isinstance(v, OnDemandFeatureView)]
    if on_demand:
        schema = result.schema
        for view in on_demand:
            for f in view.outputs:
                schema = schema.add(view.qualified(f.name), SPARK_TYPES[f.dtype], True)

        def apply_on_demand(batches: Any) -> Any:
            for pdf in batches:
                for view in on_demand:
                    pdf = pd.concat(
                        [pdf, _typed(compute_on_demand_features(view, pdf), view)], axis=1
                    )
                yield pdf

        result = result.mapInPandas(apply_on_demand, schema=schema)

    selected = [c for c in spine.columns if c != ROW_ID] + service.output_columns
    return result.select(*selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--service", required=True)
    parser.add_argument("--spine", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--timestamp-column", default="timestamp")
    parser.add_argument("--source", action="append", default=[], help="name=path override")
    parser.add_argument("--partitions", type=int, default=400)
    args = parser.parse_args()

    spark = (
        SparkSession.builder.appName(f"tessera-training-set-{args.service}")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.shuffle.partitions", str(args.partitions))
        .getOrCreate()
    )
    repo = Repository.from_path(args.repo)
    service = repo.service(args.service)
    paths = dict(item.split("=", 1) for item in args.source)
    spine = spark.read.parquet(args.spine)
    frame = build(spark, service, spine, paths, args.timestamp_column)
    (frame.write.mode("overwrite").option("compression", "zstd").parquet(args.out))
    spark.stop()


if __name__ == "__main__":
    main()
