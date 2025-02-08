"""Redis-backed online store (``pip install tessera[redis]``).

Each key is a Redis hash ``{prefix}:{namespace}:{key}`` with fields ``v``
(version) and ``d`` (data). Conditional writes run as a Lua script, so the
version check and the write are atomic on the server without WATCH/MULTI
round trips.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tessera.errors import ConflictError
from tessera.online.store import VersionedValue

_CAS_SCRIPT = """
local current = redis.call('HGET', KEYS[1], 'v')
local expected = ARGV[1]
if expected == '' then
  if current then return -1 end
elseif current ~= expected then
  return -1
end
local new = (tonumber(current) or 0) + 1
redis.call('HSET', KEYS[1], 'v', new, 'd', ARGV[2])
if tonumber(ARGV[3]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[3]) end
return new
"""


class RedisStore:
    def __init__(self, url: str, *, prefix: str = "tessera", ttl_seconds: int = 0) -> None:
        try:
            import redis
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError("RedisStore requires `pip install tessera[redis]`") from exc
        self._client: Any = redis.Redis.from_url(url)
        self._prefix = prefix
        self._ttl = ttl_seconds
        self._cas = self._client.register_script(_CAS_SCRIPT)

    def _key(self, namespace: str, key: str) -> str:
        return f"{self._prefix}:{namespace}:{key}"

    def read(self, namespace: str, keys: Iterable[str]) -> dict[str, VersionedValue]:
        key_list = list(dict.fromkeys(keys))
        pipe = self._client.pipeline(transaction=False)
        for key in key_list:
            pipe.hmget(self._key(namespace, key), "v", "d")
        out: dict[str, VersionedValue] = {}
        for key, (version, data) in zip(key_list, pipe.execute()):
            if version is not None and data is not None:
                out[key] = VersionedValue(int(version), bytes(data))
        return out

    def write(self, namespace: str, key: str, value: bytes, expected_version: int | None) -> int:
        expected = "" if expected_version is None else str(expected_version)
        result = int(self._cas(keys=[self._key(namespace, key)], args=[expected, value, self._ttl]))
        if result < 0:
            raise ConflictError(f"{namespace}/{key}: version {expected_version} is stale")
        return result

    def write_many(
        self, namespace: str, items: Mapping[str, tuple[bytes, int | None]]
    ) -> dict[str, int]:
        pipe = self._client.pipeline(transaction=False)
        keys = list(items)
        for key in keys:
            value, expected_version = items[key]
            expected = "" if expected_version is None else str(expected_version)
            self._cas(
                keys=[self._key(namespace, key)], args=[expected, value, self._ttl], client=pipe
            )
        out: dict[str, int] = {}
        for key, result in zip(keys, pipe.execute()):
            if int(result) >= 0:
                out[key] = int(result)
        return out

    def count(self, namespace: str) -> int:
        return sum(1 for _ in self._client.scan_iter(match=f"{self._prefix}:{namespace}:*"))

    def close(self) -> None:
        self._client.close()
