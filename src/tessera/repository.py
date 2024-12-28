"""The feature repository: definitions loaded from Python, plus plan/apply.

``tessera plan`` compares the definitions in code with the last applied
registry snapshot and prints what would change, in the spirit of
``terraform plan``. Any change to an aggregate view changes its fingerprint
and therefore its online namespace, so the plan flags it as needing a
backfill before traffic depends on it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from tessera.definitions import (
    AggregateFeatureView,
    BatchSource,
    Entity,
    FeatureService,
    FeatureView,
    OnDemandFeatureView,
    SnapshotFeatureView,
)
from tessera.errors import DefinitionError

REGISTRY_VERSION = 1


@dataclass(frozen=True)
class Change:
    action: Literal["add", "modify", "remove"]
    kind: Literal["entity", "source", "view", "service"]
    name: str
    details: tuple[str, ...] = ()
    requires_backfill: bool = False

    def render(self) -> str:
        symbol = {"add": "+", "modify": "~", "remove": "-"}[self.action]
        suffix = "  [backfill required]" if self.requires_backfill else ""
        head = f"{symbol} {self.kind} {self.name}{suffix}"
        return "\n".join([head, *(f"    {d}" for d in self.details)])


@dataclass(frozen=True)
class Plan:
    changes: tuple[Change, ...] = field(default_factory=tuple)

    @property
    def empty(self) -> bool:
        return not self.changes

    @property
    def backfills(self) -> list[str]:
        return [c.name for c in self.changes if c.requires_backfill]

    def render(self) -> str:
        if self.empty:
            return "No changes. Registry is up to date."
        counts = {
            a: sum(1 for c in self.changes if c.action == a) for a in ("add", "modify", "remove")
        }
        lines = [c.render() for c in self.changes]
        lines.append(
            f"\nPlan: {counts['add']} to add, {counts['modify']} to change, "
            f"{counts['remove']} to remove."
        )
        if self.backfills:
            lines.append(f"Backfill before serving: {', '.join(self.backfills)}")
        return "\n".join(lines)


def _diff_named(old: list[dict[str, Any]], new: list[dict[str, Any]], label: str) -> list[str]:
    old_by = {item["name"]: item for item in old}
    new_by = {item["name"]: item for item in new}
    details = [f"{label} +{n}" for n in new_by if n not in old_by]
    details += [f"{label} -{n}" for n in old_by if n not in new_by]
    details += [f"{label} ~{n}" for n in new_by if n in old_by and new_by[n] != old_by[n]]
    return details


def _view_details(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    details: list[str] = []
    if old.get("kind") != new.get("kind"):
        return [f"kind {old.get('kind')} -> {new.get('kind')}"]
    if new["kind"] == "aggregate":
        details += _diff_named(old["aggregates"], new["aggregates"], "feature")
        for key in ("granularity", "transform"):
            if old.get(key) != new.get(key):
                details.append(f"{key} changed")
    elif new["kind"] == "snapshot":
        details += _diff_named(old["fields"], new["fields"], "field")
        if old.get("ttl") != new.get("ttl"):
            details.append(f"ttl {old.get('ttl')}s -> {new.get('ttl')}s")
    else:
        details += _diff_named(old["outputs"], new["outputs"], "output")
        if old.get("func") != new.get("func"):
            details.append("function body changed")
        if old.get("request") != new.get("request"):
            details.append("request schema changed")
    if old.get("source") != new.get("source"):
        details.append("source schema changed")
    if old.get("entity") != new.get("entity"):
        details.append("entity changed")
    return details or ["definition changed"]


class Repository:
    """An immutable, validated set of definitions."""

    def __init__(self, objects: Iterable[Any]) -> None:
        self.entities: dict[str, Entity] = {}
        self.sources: dict[str, BatchSource] = {}
        self.views: dict[str, FeatureView] = {}
        self.services: dict[str, FeatureService] = {}
        for obj in objects:
            self._register(obj)
        for service in list(self.services.values()):
            for view in service.views:
                self._register(view)
        for view in list(self.views.values()):
            if isinstance(view, OnDemandFeatureView):
                for dep in view.depends_on:
                    self._register(dep)
        self._validate()

    def _add(self, registry: dict[str, Any], kind: str, obj: Any) -> None:
        existing = registry.get(obj.name)
        if existing is None:
            registry[obj.name] = obj
        elif existing is not obj and existing.spec() != obj.spec():
            raise DefinitionError(f"two different {kind}s are named {obj.name!r}")

    def _register(self, obj: Any) -> None:
        if isinstance(obj, Entity):
            self._add(self.entities, "entity", obj)
        elif isinstance(obj, BatchSource):
            self._add(self.sources, "source", obj)
        elif isinstance(obj, (AggregateFeatureView, SnapshotFeatureView)):
            self._add(self.views, "feature view", obj)
            self._add(self.entities, "entity", obj.entity)
            self._add(self.sources, "source", obj.source)
        elif isinstance(obj, OnDemandFeatureView):
            self._add(self.views, "feature view", obj)
        elif isinstance(obj, FeatureService):
            self._add(self.services, "feature service", obj)

    def _validate(self) -> None:
        names = list(self.views) + list(self.services)
        if len(set(names)) != len(names):
            raise DefinitionError("view and service names must be unique across both")

    @classmethod
    def from_path(cls, path: str | Path) -> Repository:
        """Import a Python definitions file and collect its module-level definitions."""
        file = Path(path).resolve()
        if not file.exists():
            raise DefinitionError(f"definitions file not found: {file}")
        module_name = f"tessera_repo_{abs(hash(str(file)))}"
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            raise DefinitionError(f"cannot import {file}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except DefinitionError:
            raise
        except Exception as exc:
            raise DefinitionError(f"error importing {file}: {exc}") from exc
        objects: list[Any] = []
        for value in vars(module).values():
            if isinstance(value, (list, tuple)):
                objects.extend(value)
            else:
                objects.append(value)
        return cls(objects)

    def view(self, name: str) -> FeatureView:
        try:
            return self.views[name]
        except KeyError:
            raise DefinitionError(f"unknown feature view {name!r}") from None

    def service(self, name: str) -> FeatureService:
        try:
            return self.services[name]
        except KeyError:
            raise DefinitionError(f"unknown feature service {name!r}") from None

    def aggregate_views_for_source(self, source_name: str) -> list[AggregateFeatureView]:
        return [
            v
            for v in self.views.values()
            if isinstance(v, AggregateFeatureView) and v.source.name == source_name
        ]

    def snapshot(self) -> dict[str, Any]:
        """JSON-serialisable registry state, as written by ``apply``."""
        return {
            "version": REGISTRY_VERSION,
            "entities": {n: e.spec() for n, e in sorted(self.entities.items())},
            "sources": {n: s.spec() for n, s in sorted(self.sources.items())},
            "views": {
                n: {"fingerprint": v.fingerprint, "spec": v.spec()}
                for n, v in sorted(self.views.items())
            },
            "services": {
                n: {"fingerprint": s.fingerprint, "spec": s.spec()}
                for n, s in sorted(self.services.items())
            },
        }

    def plan(self, previous: dict[str, Any] | None) -> Plan:
        previous = previous or {"entities": {}, "sources": {}, "views": {}, "services": {}}
        current = self.snapshot()
        changes: list[Change] = []
        for kind, key in (("entity", "entities"), ("source", "sources")):
            old, new = previous.get(key, {}), current[key]
            for name in new:
                if name not in old:
                    changes.append(Change("add", kind, name))  # type: ignore[arg-type]
                elif old[name] != new[name]:
                    changes.append(Change("modify", kind, name))  # type: ignore[arg-type]
            for name in old:
                if name not in new:
                    changes.append(Change("remove", kind, name))  # type: ignore[arg-type]
        old_views, new_views = previous.get("views", {}), current["views"]
        for name, entry in new_views.items():
            view = self.views[name]
            stateful = isinstance(view, (AggregateFeatureView, SnapshotFeatureView))
            if name not in old_views:
                changes.append(Change("add", "view", name, (), stateful))
            elif old_views[name]["fingerprint"] != entry["fingerprint"]:
                details = tuple(_view_details(old_views[name]["spec"], entry["spec"]))
                changes.append(Change("modify", "view", name, details, stateful))
        changes += [Change("remove", "view", n) for n in old_views if n not in new_views]
        old_services, new_services = previous.get("services", {}), current["services"]
        for name, entry in new_services.items():
            if name not in old_services:
                changes.append(Change("add", "service", name))
            elif old_services[name]["fingerprint"] != entry["fingerprint"]:
                changes.append(
                    Change(
                        "modify",
                        "service",
                        name,
                        ("models trained on the previous version will fail the skew guard",),
                    )
                )
        changes += [Change("remove", "service", n) for n in old_services if n not in new_services]
        return Plan(tuple(changes))

    def apply(self, registry_path: str | Path) -> Plan:
        path = Path(registry_path)
        previous = load_registry(path)
        plan = self.plan(previous)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        return plan


def load_registry(path: str | Path) -> dict[str, Any] | None:
    file = Path(path)
    if not file.exists():
        return None
    data: dict[str, Any] = json.loads(file.read_text(encoding="utf-8"))
    if data.get("version") != REGISTRY_VERSION:
        raise DefinitionError(f"unsupported registry version {data.get('version')!r}")
    return data
