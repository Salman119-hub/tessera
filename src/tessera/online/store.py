"""Key-value stores for online state.

The contract is deliberately small: batched reads, and writes guarded by
optimistic concurrency (compare-and-set on a per-key version). The stream
processor partitions by entity, so conflicts are rare. When two writers do
race, the loser re-reads and re-applies instead of silently dropping an
update (ADR-0003).
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from tessera.errors import ConflictError


@dataclass(frozen=True)
class VersionedValue:
    version: int
    value: bytes


@runtime_checkable
class OnlineStore(Protocol):
    def read(self, namespace: str, keys: Iterable[str]) -> dict[str, VersionedValue]:
        """Return the stored values for the keys that exist."""
        ...

    def write(self, namespace: str, key: str, value: bytes, expected_version: int | None) -> int:
        """Store ``value`` if the current version equals ``expected_version``
        (``None`` means the key must not exist). Returns the new version or
        raises :class:`ConflictError`."""
        ...

    def write_many(
        self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
    ) -> dict[str, int]:
        """Conditional writes for several keys. Keys that lose the race are
        omitted from the result; the rest are committed."""
        ...

    def count(self, namespace: str) -> int: ...

    def close(self) -> None: ...


class InMemoryStore:
    """Thread-safe dict-backed store, for tests and single-process demos."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], VersionedValue] = {}
        self._lock = threading.Lock()

    def read(self, namespace: str, keys: Iterable[str]) -> dict[str, VersionedValue]:
        with self._lock:
            out = {}
            for key in keys:
                item = self._data.get((namespace, key))
                if item is not None:
                    out[key] = item
            return out

    def _write_locked(
        self, namespace: str, key: str, value: bytes, expected_version: int | None
    ) -> int:
        current = self._data.get((namespace, key))
        current_version = current.version if current else None
        if current_version != expected_version:
            raise ConflictError(
                f"{namespace}/{key}: expected version {expected_version}, found {current_version}"
            )
        new_version = (current_version or 0) + 1
        self._data[(namespace, key)] = VersionedValue(new_version, value)
        return new_version

    def write(self, namespace: str, key: str, value: bytes, expected_version: int | None) -> int:
        with self._lock:
            return self._write_locked(namespace, key, value, expected_version)

    def write_many(
        self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
    ) -> dict[str, int]:
        with self._lock:
            out = {}
            for key, (value, expected) in items.items():
                try:
                    out[key] = self._write_locked(namespace, key, value, expected)
                except ConflictError:
                    continue
            return out

    def count(self, namespace: str) -> int:
        with self._lock:
            return sum(1 for ns, _ in self._data if ns == namespace)

    def close(self) -> None:
        return None


class SQLiteStore:
    """Durable single-node store on SQLite in WAL mode.

    Suitable for local development, the demo stack and small deployments. A
    connection per thread avoids SQLite's cross-thread restrictions.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._memory_conn: sqlite3.Connection | None = None
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS online_state ("
            " namespace TEXT NOT NULL, key TEXT NOT NULL, version INTEGER NOT NULL,"
            " value BLOB NOT NULL, PRIMARY KEY (namespace, key))"
        )
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        if self.path == ":memory:":
            if self._memory_conn is None:
                self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._memory_conn
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def read(self, namespace: str, keys: Iterable[str]) -> dict[str, VersionedValue]:
        key_list = list(dict.fromkeys(keys))
        out: dict[str, VersionedValue] = {}
        conn = self._conn()
        for i in range(0, len(key_list), 500):
            chunk = key_list[i : i + 500]
            marks = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT key, version, value FROM online_state "  # noqa: S608 - placeholders only
                f"WHERE namespace = ? AND key IN ({marks})",
                [namespace, *chunk],
            ).fetchall()
            for key, version, value in rows:
                out[key] = VersionedValue(int(version), bytes(value))
        return out

    def _write(
        self,
        conn: sqlite3.Connection,
        namespace: str,
        key: str,
        value: bytes,
        expected_version: int | None,
    ) -> int:
        if expected_version is None:
            cur = conn.execute(
                "INSERT OR IGNORE INTO online_state (namespace, key, version, value) "
                "VALUES (?, ?, 1, ?)",
                (namespace, key, value),
            )
            if cur.rowcount != 1:
                raise ConflictError(f"{namespace}/{key}: key already exists")
            return 1
        cur = conn.execute(
            "UPDATE online_state SET version = version + 1, value = ? "
            "WHERE namespace = ? AND key = ? AND version = ?",
            (value, namespace, key, expected_version),
        )
        if cur.rowcount != 1:
            raise ConflictError(f"{namespace}/{key}: version {expected_version} is stale")
        return expected_version + 1

    def write(self, namespace: str, key: str, value: bytes, expected_version: int | None) -> int:
        conn = self._conn()
        try:
            version = self._write(conn, namespace, key, value, expected_version)
        except ConflictError:
            conn.rollback()
            raise
        conn.commit()
        return version

    def write_many(
        self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
    ) -> dict[str, int]:
        conn = self._conn()
        out: dict[str, int] = {}
        for key, (value, expected) in items.items():
            try:
                out[key] = self._write(conn, namespace, key, value, expected)
            except ConflictError:
                continue
        conn.commit()
        return out

    def count(self, namespace: str) -> int:
        row = (
            self._conn()
            .execute("SELECT COUNT(*) FROM online_state WHERE namespace = ?", (namespace,))
            .fetchone()
        )
        return int(row[0])

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None
        if self._memory_conn is not None:
            self._memory_conn.close()
            self._memory_conn = None


def build_store(url: str, **kwargs: Any) -> OnlineStore:
    """Create a store from a URL: ``memory://``, ``sqlite:///path.db`` or ``redis://host:6379/0``."""
    if url.startswith("memory://"):
        return InMemoryStore()
    if url.startswith("sqlite:///"):
        return SQLiteStore(url[len("sqlite:///") :])
    if url.startswith(("redis://", "rediss://")):
        from tessera.online.redis_store import RedisStore

        return RedisStore(url, **kwargs)
    raise ValueError(f"unsupported online store URL: {url!r}")
