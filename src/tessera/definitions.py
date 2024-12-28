"""Declarative feature definitions.

Users describe *what* a feature is: an entity, a source, an aggregation and a
window. Tessera decides *how* to compute it offline and online. Every
definition has a canonical ``spec()`` and a ``fingerprint`` derived from it,
so the registry can tell exactly which definitions changed between releases.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import re
import textwrap
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from tessera.errors import DefinitionError
from tessera.timeutil import format_duration, parse_duration

DType = Literal["int", "float", "string", "bool"]
DTYPES: frozenset[str] = frozenset({"int", "float", "string", "bool"})
AggFunction = Literal["count", "sum", "mean", "min", "max", "std", "last", "count_distinct"]
AGG_FUNCTIONS: frozenset[str] = frozenset(
    {"count", "sum", "mean", "min", "max", "std", "last", "count_distinct"}
)

Row = dict[str, Any]
RowTransform = Callable[[Row], Row]


def fingerprint(spec: Any) -> str:
    """Stable 12-hex-digit hash of a JSON-serialisable spec."""
    canonical = json.dumps(spec, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def callable_fingerprint(func: Callable[..., Any] | None) -> str | None:
    """Hash a function by its source so behaviour changes change the fingerprint."""
    if func is None:
        return None
    try:
        source = textwrap.dedent(inspect.getsource(func))
    except (OSError, TypeError):
        source = f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"
    return hashlib.sha256(source.encode()).hexdigest()[:12]


_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _check_name(kind: str, name: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise DefinitionError(f"{kind} name {name!r} must be a snake_case identifier")
    if "__" in name:
        raise DefinitionError(f"{kind} name {name!r} must not contain '__' (reserved separator)")


@dataclass(frozen=True)
class Field:
    """A typed column of a source, a snapshot feature, or a request input."""

    name: str
    dtype: DType = "float"
    description: str = ""

    def __post_init__(self) -> None:
        _check_name("field", self.name)
        if self.dtype not in DTYPES:
            raise DefinitionError(f"field {self.name!r}: unknown dtype {self.dtype!r}")

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype}


@dataclass(frozen=True)
class Entity:
    """The thing features describe, identified by one join key (card, user, merchant)."""

    name: str
    join_key: str
    description: str = ""

    def __post_init__(self) -> None:
        _check_name("entity", self.name)
        _check_name("join key", self.join_key)

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "join_key": self.join_key}


@dataclass(frozen=True)
class Check:
    """A data quality expectation on a source column (see :mod:`tessera.quality`)."""

    kind: Literal["not_null", "between", "in_set", "unique", "regex"]
    column: str
    args: tuple[Any, ...] = ()
    mostly: float = 1.0
    severity: Literal["error", "warn"] = "error"

    @staticmethod
    def not_null(column: str, *, mostly: float = 1.0, severity: str = "error") -> Check:
        return Check("not_null", column, (), mostly, severity)  # type: ignore[arg-type]

    @staticmethod
    def between(
        column: str,
        low: float | None,
        high: float | None,
        *,
        mostly: float = 1.0,
        severity: str = "error",
    ) -> Check:
        return Check("between", column, (low, high), mostly, severity)  # type: ignore[arg-type]

    @staticmethod
    def in_set(
        column: str, values: Sequence[Any], *, mostly: float = 1.0, severity: str = "error"
    ) -> Check:
        return Check("in_set", column, tuple(values), mostly, severity)  # type: ignore[arg-type]

    @staticmethod
    def unique(column: str, *, severity: str = "error") -> Check:
        return Check("unique", column, (), 1.0, severity)  # type: ignore[arg-type]

    @staticmethod
    def regex(column: str, pattern: str, *, mostly: float = 1.0, severity: str = "error") -> Check:
        return Check("regex", column, (pattern,), mostly, severity)  # type: ignore[arg-type]

    def spec(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "column": self.column,
            "args": list(self.args),
            "mostly": self.mostly,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class BatchSource:
    """A table of timestamped records in the lake (CSV or Parquet)."""

    name: str
    path: str
    timestamp_field: str
    schema: tuple[Field, ...]
    checks: tuple[Check, ...] = ()
    description: str = ""

    def __post_init__(self) -> None:
        _check_name("source", self.name)
        names = [f.name for f in self.schema]
        if len(set(names)) != len(names):
            raise DefinitionError(f"source {self.name!r} has duplicate schema fields")
        for check in self.checks:
            if check.column not in names and check.column != self.timestamp_field:
                raise DefinitionError(
                    f"source {self.name!r}: check on unknown column {check.column!r}"
                )

    @property
    def columns(self) -> list[str]:
        return [f.name for f in self.schema]

    def field(self, name: str) -> Field:
        for f in self.schema:
            if f.name == name:
                return f
        raise DefinitionError(f"source {self.name!r} has no field {name!r}")

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "batch",
            "name": self.name,
            "timestamp_field": self.timestamp_field,
            "schema": [f.spec() for f in self.schema],
        }


@dataclass(frozen=True)
class StreamSource(BatchSource):
    """An event stream (Kafka topic). ``path`` points at its archive in the lake.

    The archive is what offline computation and backfills read, which is why a
    stream source also behaves as a batch source.
    """

    topic: str = ""
    event_id_field: str = "event_id"
    allowed_lateness: str = "10m"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.topic:
            raise DefinitionError(f"stream source {self.name!r} needs a topic")
        if self.event_id_field not in self.columns:
            raise DefinitionError(
                f"stream source {self.name!r}: event_id_field {self.event_id_field!r} "
                "is not in the schema"
            )
        parse_duration(self.allowed_lateness)

    @property
    def lateness_seconds(self) -> int:
        return parse_duration(self.allowed_lateness)

    def spec(self) -> dict[str, Any]:
        return {
            **super().spec(),
            "kind": "stream",
            "topic": self.topic,
            "event_id_field": self.event_id_field,
            "allowed_lateness": self.allowed_lateness,
        }


Source = BatchSource | StreamSource


def default_hop(window: int, granularity: int) -> int:
    """Sawtooth hop policy: short windows refresh at ``granularity``, long ones hourly or daily.

    A coarser hop for long windows bounds the number of tiles merged per read
    (see ADR-0001). The hop always divides the window and is a multiple of the
    granularity; otherwise the granularity itself is used.
    """
    for candidate, max_window in ((granularity, 2 * 3600), (3600, 2 * 86_400), (86_400, None)):
        if max_window is not None and window > max_window:
            continue
        if candidate % granularity == 0 and window % candidate == 0:
            return candidate
    return granularity


@dataclass(frozen=True)
class Aggregate:
    """``function(column)`` over a trailing window, e.g. ``Aggregate("amount", "sum", "1h")``.

    ``column`` is ``None`` for ``count`` (count events). ``hop`` overrides the
    sawtooth refresh interval; see :func:`default_hop`.
    """

    column: str | None
    function: AggFunction
    window: str
    name: str | None = None
    hop: str | None = None

    def __post_init__(self) -> None:
        if self.function not in AGG_FUNCTIONS:
            raise DefinitionError(f"unknown aggregation function {self.function!r}")
        if self.column is None and self.function != "count":
            raise DefinitionError(f"aggregation {self.function!r} needs a column")
        parse_duration(self.window)
        if self.hop is not None:
            parse_duration(self.hop)
        if self.name is not None:
            _check_name("feature", self.name)

    @property
    def feature_name(self) -> str:
        if self.name:
            return self.name
        base = self.column or "events"
        return f"{base}_{self.function}_{format_duration(self.window_seconds)}"

    @property
    def window_seconds(self) -> int:
        return parse_duration(self.window)

    def spec(self, hop: int) -> dict[str, Any]:
        return {
            "name": self.feature_name,
            "column": self.column,
            "function": self.function,
            "window": self.window_seconds,
            "hop": hop,
        }


class FeatureView:
    """Base class for everything a :class:`FeatureService` can reference."""

    name: str
    description: str
    owner: str
    tags: dict[str, str]

    @property
    def feature_names(self) -> list[str]:
        raise NotImplementedError

    def feature_dtype(self, feature: str) -> str:
        raise NotImplementedError

    def spec(self) -> dict[str, Any]:
        raise NotImplementedError

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.spec())

    @property
    def namespace(self) -> str:
        """Online-store namespace. Includes the fingerprint so a changed
        definition never reads state written under the old one."""
        return f"{self.name}:{self.fingerprint}"

    def select(self, *features: str) -> FeatureSelection:
        return FeatureSelection(self, tuple(features))

    def qualified(self, feature: str) -> str:
        return f"{self.name}__{feature}"


@dataclass(frozen=True, eq=False)
class AggregateFeatureView(FeatureView):
    """Windowed aggregations over a source, keyed by an entity."""

    name: str
    entity: Entity
    source: Source
    aggregates: tuple[Aggregate, ...]
    granularity: str = "5m"
    transform: RowTransform | None = None
    description: str = ""
    owner: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_name("feature view", self.name)
        if not self.aggregates:
            raise DefinitionError(f"view {self.name!r} has no aggregates")
        gran = self.granularity_seconds
        names = [a.feature_name for a in self.aggregates]
        if len(set(names)) != len(names):
            raise DefinitionError(f"view {self.name!r} has duplicate feature names")
        for agg in self.aggregates:
            hop = self.hop_for(agg)
            if hop % gran != 0:
                raise DefinitionError(
                    f"{self.name}.{agg.feature_name}: hop {hop}s is not a multiple of "
                    f"granularity {gran}s"
                )
            if agg.window_seconds % hop != 0:
                raise DefinitionError(
                    f"{self.name}.{agg.feature_name}: window {agg.window} is not a multiple "
                    f"of its hop {format_duration(hop)}"
                )
        hops = sorted(self.resolutions, reverse=True)
        for coarse, fine in itertools.pairwise(hops):
            if coarse % fine != 0:
                raise DefinitionError(
                    f"view {self.name!r}: tile resolutions {format_duration(coarse)} and "
                    f"{format_duration(fine)} do not nest"
                )
        if self.entity.join_key not in self.source.columns:
            raise DefinitionError(
                f"view {self.name!r}: source {self.source.name!r} lacks join key "
                f"{self.entity.join_key!r}"
            )
        if self.transform is None:
            for agg in self.aggregates:
                if agg.column is not None and agg.column not in self.source.columns:
                    raise DefinitionError(
                        f"{self.name}.{agg.feature_name}: unknown column {agg.column!r}"
                    )

    @property
    def granularity_seconds(self) -> int:
        return parse_duration(self.granularity)

    def hop_for(self, agg: Aggregate) -> int:
        if agg.hop is not None:
            return parse_duration(agg.hop)
        return default_hop(agg.window_seconds, self.granularity_seconds)

    @property
    def resolutions(self) -> list[int]:
        """Distinct tile widths, finest first. The granularity is always included."""
        hops = {self.hop_for(a) for a in self.aggregates} | {self.granularity_seconds}
        return sorted(hops)

    @property
    def input_columns(self) -> list[str]:
        cols: list[str] = []
        for agg in self.aggregates:
            if agg.column is not None and agg.column not in cols:
                cols.append(agg.column)
        return cols

    @property
    def feature_names(self) -> list[str]:
        return [a.feature_name for a in self.aggregates]

    def feature_dtype(self, feature: str) -> str:
        for agg in self.aggregates:
            if agg.feature_name == feature:
                if agg.function in ("count", "count_distinct"):
                    return "int"
                if agg.function == "last" and agg.column is not None:
                    try:
                        return self.source.field(agg.column).dtype
                    except DefinitionError:
                        return "float"
                return "float"
        raise DefinitionError(f"view {self.name!r} has no feature {feature!r}")

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "aggregate",
            "name": self.name,
            "entity": self.entity.spec(),
            "source": self.source.spec(),
            "granularity": self.granularity_seconds,
            "aggregates": [a.spec(self.hop_for(a)) for a in self.aggregates],
            "transform": callable_fingerprint(self.transform),
        }


@dataclass(frozen=True, eq=False)
class SnapshotFeatureView(FeatureView):
    """Latest-value features from a slowly changing table, valid for ``ttl``."""

    name: str
    entity: Entity
    source: BatchSource
    fields: tuple[Field, ...]
    ttl: str = "30d"
    description: str = ""
    owner: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_name("feature view", self.name)
        parse_duration(self.ttl)
        if self.entity.join_key not in self.source.columns:
            raise DefinitionError(
                f"view {self.name!r}: source {self.source.name!r} lacks join key "
                f"{self.entity.join_key!r}"
            )
        for f in self.fields:
            if f.name not in self.source.columns:
                raise DefinitionError(f"view {self.name!r}: unknown source column {f.name!r}")

    @property
    def ttl_seconds(self) -> int:
        return parse_duration(self.ttl)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def feature_dtype(self, feature: str) -> str:
        for f in self.fields:
            if f.name == feature:
                return f.dtype
        raise DefinitionError(f"view {self.name!r} has no feature {feature!r}")

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "snapshot",
            "name": self.name,
            "entity": self.entity.spec(),
            "source": self.source.spec(),
            "fields": [f.spec() for f in self.fields],
            "ttl": self.ttl_seconds,
        }


@dataclass(frozen=True, eq=False)
class OnDemandFeatureView(FeatureView):
    """Row-level features computed at request time from request data and other features.

    ``func`` receives a dict holding the request fields and the qualified names
    (``view__feature``) of the features in ``depends_on``, and returns a dict
    with one entry per output. The same function runs offline and online.
    """

    name: str
    request: tuple[Field, ...]
    outputs: tuple[Field, ...]
    func: Callable[[Row], Row]
    depends_on: tuple[FeatureView, ...] = ()
    description: str = ""
    owner: str = ""
    tags: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _check_name("feature view", self.name)
        if not self.outputs:
            raise DefinitionError(f"on-demand view {self.name!r} has no outputs")

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.outputs]

    def feature_dtype(self, feature: str) -> str:
        for f in self.outputs:
            if f.name == feature:
                return f.dtype
        raise DefinitionError(f"view {self.name!r} has no feature {feature!r}")

    def compute(self, row: Row) -> Row:
        result = self.func(row)
        missing = [f.name for f in self.outputs if f.name not in result]
        if missing:
            raise DefinitionError(f"on-demand view {self.name!r} did not return {missing}")
        return {f.name: result[f.name] for f in self.outputs}

    def spec(self) -> dict[str, Any]:
        return {
            "kind": "on_demand",
            "name": self.name,
            "request": [f.spec() for f in self.request],
            "outputs": [f.spec() for f in self.outputs],
            "depends_on": sorted(v.name for v in self.depends_on),
            "func": callable_fingerprint(self.func),
        }


def on_demand(
    *,
    request: Sequence[Field],
    outputs: Sequence[Field],
    depends_on: Sequence[FeatureView] = (),
    name: str | None = None,
    description: str = "",
) -> Callable[[Callable[[Row], Row]], OnDemandFeatureView]:
    """Decorator form of :class:`OnDemandFeatureView`."""

    def wrap(func: Callable[[Row], Row]) -> OnDemandFeatureView:
        return OnDemandFeatureView(
            name=name or func.__name__,
            request=tuple(request),
            outputs=tuple(outputs),
            func=func,
            depends_on=tuple(depends_on),
            description=description or (inspect.getdoc(func) or ""),
        )

    return wrap


@dataclass(frozen=True)
class FeatureSelection:
    """A subset of a view's features, as referenced by a service."""

    view: FeatureView
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        unknown = set(self.features) - set(self.view.feature_names)
        if unknown:
            raise DefinitionError(f"view {self.view.name!r} has no features {sorted(unknown)}")


FeatureRef = FeatureView | FeatureSelection


@dataclass(frozen=True, eq=False)
class FeatureService:
    """The versioned feature vector a model consumes."""

    name: str
    features: tuple[FeatureRef, ...]
    description: str = ""
    owner: str = ""

    def __post_init__(self) -> None:
        _check_name("feature service", self.name)
        seen: set[str] = set()
        for ref in self.features:
            view = ref.view if isinstance(ref, FeatureSelection) else ref
            if view.name in seen:
                raise DefinitionError(f"service {self.name!r} references {view.name!r} twice")
            seen.add(view.name)
        for view in self.views:
            if isinstance(view, OnDemandFeatureView):
                missing = [d.name for d in view.depends_on if d.name not in seen]
                if missing:
                    raise DefinitionError(
                        f"service {self.name!r}: on-demand view {view.name!r} depends on "
                        f"{missing}, which the service does not include"
                    )

    def selections(self) -> list[FeatureSelection]:
        out: list[FeatureSelection] = []
        for ref in self.features:
            if isinstance(ref, FeatureSelection):
                out.append(ref)
            else:
                out.append(FeatureSelection(ref, tuple(ref.feature_names)))
        return out

    @property
    def views(self) -> list[FeatureView]:
        return [s.view for s in self.selections()]

    @property
    def output_columns(self) -> list[str]:
        return [s.view.qualified(f) for s in self.selections() for f in s.features]

    @property
    def request_fields(self) -> list[Field]:
        fields: dict[str, Field] = {}
        for view in self.views:
            if isinstance(view, OnDemandFeatureView):
                for f in view.request:
                    fields.setdefault(f.name, f)
        return list(fields.values())

    @property
    def entities(self) -> list[Entity]:
        out: dict[str, Entity] = {}
        for view in self.views:
            entity = getattr(view, "entity", None)
            if isinstance(entity, Entity):
                out.setdefault(entity.name, entity)
        return list(out.values())

    def column_dtypes(self) -> dict[str, str]:
        return {
            s.view.qualified(f): s.view.feature_dtype(f)
            for s in self.selections()
            for f in s.features
        }

    def spec(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "features": [
                {
                    "view": s.view.name,
                    "fingerprint": s.view.fingerprint,
                    "features": list(s.features),
                }
                for s in self.selections()
            ],
        }

    @property
    def fingerprint(self) -> str:
        return fingerprint(self.spec())
