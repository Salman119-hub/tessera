"""Tiled per-entity aggregation state.

Window semantics (identical offline and online)
------------------------------------------------
For an aggregate with window ``W`` and hop ``H``, the value as of time ``t``
covers events with timestamps in::

    [align(t, H) - W, t)

so the window length is a "sawtooth" between ``W`` and ``W + H``. The interval
is half-open: an event at exactly ``t`` is excluded, which keeps the event
being scored out of its own features.

Storage
-------
Each event updates one tile per resolution (the view's distinct hops). A read
merges coarse tiles for the bulk of the window, finer tiles for the remainder,
and replays raw events for the current granularity tile, filtered to ``< t``.
The raw buffer also deduplicates events by id, which makes stream ingestion
idempotent.

Offline computation replays events through this same class, so a training
row and the online read at the same instant see the same value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from tessera.aggregation import Aggregator, State, get_aggregator
from tessera.definitions import AggregateFeatureView, StreamSource
from tessera.timeutil import align

STATE_VERSION = 1


class ApplyStatus(str, Enum):
    APPLIED = "applied"
    DUPLICATE = "duplicate"
    LATE = "late"


@dataclass
class EntityState:
    """All aggregation state for one (feature view, entity) pair."""

    tiles: dict[int, dict[int, list[State]]] = field(default_factory=dict)
    raw: list[list[Any]] = field(default_factory=list)
    max_ts: int | None = None

    def to_json(self) -> bytes:
        payload = {
            "v": STATE_VERSION,
            "m": self.max_ts,
            "t": {
                str(r): {str(s): st for s, st in tiles.items()} for r, tiles in self.tiles.items()
            },
            "r": self.raw,
        }
        return json.dumps(payload, separators=(",", ":")).encode()

    @classmethod
    def from_json(cls, blob: bytes | str) -> EntityState:
        payload = json.loads(blob)
        if payload.get("v") != STATE_VERSION:
            raise ValueError(f"unsupported entity state version {payload.get('v')!r}")
        tiles = {
            int(r): {int(s): st for s, st in by_start.items()}
            for r, by_start in payload["t"].items()
        }
        return cls(tiles=tiles, raw=payload["r"], max_ts=payload["m"])


@dataclass(frozen=True)
class _Slot:
    name: str
    aggregator: Aggregator
    column_index: int | None
    window: int
    hop: int


class TileEngine:
    """Applies events to and reads windows from :class:`EntityState` for one view."""

    def __init__(self, view: AggregateFeatureView) -> None:
        self.view = view
        self.input_columns = view.input_columns
        col_index = {c: i for i, c in enumerate(self.input_columns)}
        self.slots = [
            _Slot(
                name=agg.feature_name,
                aggregator=get_aggregator(agg.function),
                column_index=col_index.get(agg.column) if agg.column else None,
                window=agg.window_seconds,
                hop=view.hop_for(agg),
            )
            for agg in view.aggregates
        ]
        self.resolutions = view.resolutions  # ascending; resolutions[0] is the granularity
        self.base = self.resolutions[0]
        source = view.source
        self.lateness = source.lateness_seconds if isinstance(source, StreamSource) else 0
        # Raw events must cover the head granularity tile and the lateness horizon,
        # which is also the deduplication horizon.
        self.raw_retention = max(self.base, self.lateness) + self.base
        # Resolution r stores the slots whose hop is >= r: their main window when
        # hop == r, or as filler between align(t, hop) and align(t, r).
        self.res_slots: dict[int, list[int]] = {
            r: [i for i, s in enumerate(self.slots) if s.hop >= r] for r in self.resolutions
        }
        self.res_pos: dict[int, dict[int, int]] = {
            r: {slot: pos for pos, slot in enumerate(idx)} for r, idx in self.res_slots.items()
        }
        self.tile_retention: dict[int, int] = {}
        for r in self.resolutions:
            needed = [s.window for s in self.slots if s.hop == r]
            needed += [s.hop for s in self.slots if s.hop > r]
            self.tile_retention[r] = max(needed, default=r)

    @property
    def feature_names(self) -> list[str]:
        return [s.name for s in self.slots]

    def new_state(self) -> EntityState:
        return EntityState()

    def apply(
        self, state: EntityState, ts: int, values: list[Any], event_id: str | None = None
    ) -> ApplyStatus:
        """Fold one event into ``state``. ``values`` is ordered like ``input_columns``."""
        if event_id is not None and any(entry[1] == event_id for entry in state.raw):
            return ApplyStatus.DUPLICATE
        if state.max_ts is not None and ts < state.max_ts - self.lateness:
            return ApplyStatus.LATE
        for r in self.resolutions:
            by_start = state.tiles.setdefault(r, {})
            start = align(ts, r)
            states = by_start.get(start)
            if states is None:
                states = [self.slots[i].aggregator.init() for i in self.res_slots[r]]
                by_start[start] = states
            for pos, i in enumerate(self.res_slots[r]):
                slot = self.slots[i]
                value = values[slot.column_index] if slot.column_index is not None else None
                slot.aggregator.update(states[pos], value, ts)
        state.raw.append([ts, event_id, list(values)])
        previous = state.max_ts
        if previous is None or ts > previous:
            state.max_ts = ts
            if previous is None or align(ts, self.base) != align(previous, self.base):
                self._prune(state)
        return ApplyStatus.APPLIED

    def _prune(self, state: EntityState) -> None:
        assert state.max_ts is not None
        horizon = state.max_ts - self.raw_retention
        if state.raw and state.raw[0][0] < horizon:
            state.raw = [entry for entry in state.raw if entry[0] >= horizon]
        for r, by_start in state.tiles.items():
            cutoff = align(state.max_ts, r) - self.tile_retention.get(r, r)
            stale = [s for s in by_start if s < cutoff]
            for s in stale:
                del by_start[s]

    def _merge_range(
        self, acc: State, slot_index: int, state: EntityState, r: int, lo: int, hi: int
    ) -> None:
        """Merge resolution-``r`` tiles with start in ``[lo, hi)`` into ``acc``, oldest first."""
        if hi <= lo:
            return
        by_start = state.tiles.get(r)
        if not by_start:
            return
        pos = self.res_pos[r][slot_index]
        aggregator = self.slots[slot_index].aggregator
        n_slots = (hi - lo) // r
        if n_slots <= len(by_start):
            starts: Any = range(lo, hi, r)
        else:
            starts = sorted(s for s in by_start if lo <= s < hi)
        for start in starts:
            states = by_start.get(start)
            if states is not None:
                aggregator.merge_into(acc, states[pos])

    def read(self, state: EntityState, t: int) -> dict[str, Any]:
        """Feature values as of ``t`` (events strictly before ``t``)."""
        head = align(t, self.base)
        head_events = [entry for entry in state.raw if head <= entry[0] < t]
        out: dict[str, Any] = {}
        for i, slot in enumerate(self.slots):
            acc = slot.aggregator.init()
            end = align(t, slot.hop)
            self._merge_range(acc, i, state, slot.hop, end - slot.window, end)
            cursor = end
            for r in reversed(self.resolutions):
                if r >= slot.hop:
                    continue
                stop = align(t, r)
                self._merge_range(acc, i, state, r, cursor, stop)
                cursor = stop
            for ts, _event_id, values in head_events:
                value = values[slot.column_index] if slot.column_index is not None else None
                slot.aggregator.update(acc, value, ts)
            out[slot.name] = slot.aggregator.finalize(acc)
        return out

    def empty_values(self) -> dict[str, Any]:
        return self.read(EntityState(), 0)
