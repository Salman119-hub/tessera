"""Layered settings: defaults, then a YAML file, then ``TESSERA_*`` environment variables.

Nested keys use ``__`` in environment variables, for example
``TESSERA_KAFKA__BOOTSTRAP_SERVERS=broker:9092``. Relative paths in a YAML file
resolve against the file's directory, so a project config works from any
working directory.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

ENV_PREFIX = "TESSERA_"
PATH_KEYS = ("repo_path", "registry_path", "model_dir", "feature_log", "dead_letter")


class ServedModel(BaseModel):
    name: str
    alias: str = "production"


class KafkaSettings(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "tessera-stream"
    extra: dict[str, Any] = Field(default_factory=dict)


class Settings(BaseModel):
    repo_path: str = "features.py"
    registry_path: str = ".tessera/registry.json"
    online_store: str = "sqlite:///.tessera/online.db"
    model_dir: str = ".tessera/models"
    feature_log: str | None = ".tessera/feature_log.jsonl"
    dead_letter: str = ".tessera/dead_letter.jsonl"
    api_keys: list[str] = Field(default_factory=list)
    models: list[ServedModel] = Field(default_factory=list)
    kafka: KafkaSettings = Field(default_factory=KafkaSettings)
    log_level: str = "INFO"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _parse_env_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


def _env_overrides(environ: Mapping[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, raw in environ.items():
        if not name.startswith(ENV_PREFIX) or name == f"{ENV_PREFIX}CONFIG":
            continue
        parts = name[len(ENV_PREFIX) :].lower().split("__")
        cursor = out
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _parse_env_value(raw)
    return out


def _resolve_paths(data: dict[str, Any], base: Path) -> dict[str, Any]:
    out = dict(data)
    for key in PATH_KEYS:
        value = out.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            out[key] = str((base / value).resolve())
    store = out.get("online_store")
    if isinstance(store, str) and store.startswith("sqlite:///"):
        db = store[len("sqlite:///") :]
        if db != ":memory:" and not Path(db).is_absolute():
            out["online_store"] = f"sqlite:///{(base / db).resolve()}"
    return out


def load_settings(
    path: str | Path | None = None, environ: Mapping[str, str] | None = None
) -> Settings:
    env = os.environ if environ is None else environ
    data: dict[str, Any] = {}
    config_path = path or env.get(f"{ENV_PREFIX}CONFIG")
    if config_path:
        file = Path(config_path)
        loaded = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{file} must contain a mapping")
        data = _resolve_paths(loaded, file.resolve().parent)
    data = _deep_merge(data, _env_overrides(env))
    return Settings.model_validate(data)
