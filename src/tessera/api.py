"""HTTP serving API (FastAPI).

Endpoints
---------
``GET  /healthz``                       liveness
``GET  /readyz``                        store reachable and models loaded
``GET  /v1/registry``                   views and services with fingerprints
``POST /v1/features``                   online feature vectors for a service
``POST /v1/ingest/{source}``            push events (low-volume sources, tests)
``POST /v1/models/{name}/score``        features + score + decision
``GET  /metrics``                       Prometheus exposition
"""

from __future__ import annotations

import hmac
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from tessera import __version__
from tessera.config import Settings
from tessera.errors import DefinitionError, SchemaError
from tessera.metrics import Registry
from tessera.online.server import FeatureServer, JsonlFeatureLogger
from tessera.online.store import OnlineStore, build_store
from tessera.online.writer import OnlineWriter
from tessera.repository import Repository
from tessera.training.artifact import ModelArtifact, ModelRegistry

log = logging.getLogger(__name__)


class FeaturesRequest(BaseModel):
    service: str
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
    request_time: int | None = None


class FeaturesResponse(BaseModel):
    service: str
    fingerprint: str
    request_time: int
    features: list[dict[str, Any]]


class IngestRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1, max_length=10_000)


class ScoreRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=1000)
    explain: bool = False


class ScoredRow(BaseModel):
    score: float
    decision: str
    features: dict[str, Any]
    explanation: list[dict[str, Any]] | None = None


class ScoreResponse(BaseModel):
    model: str
    version: int | None
    threshold: float
    results: list[ScoredRow]


@dataclass
class Container:
    """Long-lived dependencies, built once at startup."""

    settings: Settings
    repo: Repository
    store: OnlineStore
    server: FeatureServer
    writer: OnlineWriter
    models: dict[str, ModelArtifact] = field(default_factory=dict)
    metrics: Registry = field(default_factory=Registry)

    @classmethod
    def build(cls, settings: Settings, *, store: OnlineStore | None = None) -> Container:
        repo = Repository.from_path(settings.repo_path)
        online = store or build_store(settings.online_store)
        logger = JsonlFeatureLogger(settings.feature_log) if settings.feature_log else None
        container = cls(
            settings=settings,
            repo=repo,
            store=online,
            server=FeatureServer(repo, online, logger=logger),
            writer=OnlineWriter(repo, online),
        )
        registry = ModelRegistry(settings.model_dir)
        for served in settings.models:
            artifact = registry.load(served.name, served.alias)
            # Skew guard: fail at startup, not at the first bad prediction.
            artifact.check_compatible(repo.service(artifact.service))
            container.models[served.name] = artifact
            log.info("loaded model %s v%s (%s)", served.name, artifact.version, served.alias)
        return container


def create_app(settings: Settings, *, container: Container | None = None) -> FastAPI:
    deps = container or Container.build(settings)
    app = FastAPI(title="Tessera feature serving", version=__version__)
    app.state.container = deps
    metrics = deps.metrics
    requests_total = metrics.counter(
        "tessera_http_requests_total", "HTTP requests by route and status."
    )
    latency = metrics.histogram("tessera_http_request_seconds", "HTTP request latency.")
    rows_served = metrics.counter(
        "tessera_feature_rows_total", "Feature vectors served by service."
    )
    scores_total = metrics.counter("tessera_scores_total", "Scored rows by model and decision.")

    @app.middleware("http")
    async def instrument(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        latency.observe(time.perf_counter() - started, route=path)
        requests_total.inc(route=path, status=str(response.status_code))
        response.headers["x-request-id"] = request_id
        return response

    def authorize(request: Request) -> None:
        keys = deps.settings.api_keys
        if not keys:
            return
        supplied = request.headers.get("x-api-key", "")
        if not any(hmac.compare_digest(supplied, k) for k in keys):
            raise HTTPException(status_code=401, detail="invalid or missing API key")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz")
    def readyz() -> dict[str, Any]:
        try:
            deps.store.read("__readiness__", ["probe"])
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"online store unavailable: {exc}") from exc
        missing = [m.name for m in deps.settings.models if m.name not in deps.models]
        if missing:
            raise HTTPException(status_code=503, detail=f"models not loaded: {missing}")
        return {"status": "ready", "models": sorted(deps.models)}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prometheus() -> str:
        return metrics.render()

    @app.get("/v1/registry", dependencies=[Depends(authorize)])
    def registry() -> dict[str, Any]:
        return {
            "views": {
                name: {"fingerprint": v.fingerprint, "features": v.feature_names}
                for name, v in deps.repo.views.items()
            },
            "services": {
                name: {"fingerprint": s.fingerprint, "columns": s.output_columns}
                for name, s in deps.repo.services.items()
            },
        }

    @app.post("/v1/features", response_model=FeaturesResponse, dependencies=[Depends(authorize)])
    def features(body: FeaturesRequest) -> FeaturesResponse:
        try:
            service = deps.repo.service(body.service)
        except DefinitionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        t = body.request_time if body.request_time is not None else deps.server.clock()
        try:
            vectors = deps.server.get_features(service, body.rows, request_time=t)
        except SchemaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        rows_served.inc(len(vectors), service=service.name)
        return FeaturesResponse(
            service=service.name, fingerprint=service.fingerprint, request_time=t, features=vectors
        )

    @app.post("/v1/ingest/{source}", dependencies=[Depends(authorize)])
    def ingest(source: str, body: IngestRequest) -> dict[str, Any]:
        if source not in deps.repo.sources:
            raise HTTPException(status_code=404, detail=f"unknown source {source!r}")
        result = deps.writer.ingest(source, body.records)
        payload = result.to_dict()
        payload["errors"] = [reason for _, reason in result.invalid[:20]]
        return payload

    @app.post(
        "/v1/models/{name}/score", response_model=ScoreResponse, dependencies=[Depends(authorize)]
    )
    def score(name: str, body: ScoreRequest) -> ScoreResponse:
        artifact = deps.models.get(name)
        if artifact is None:
            raise HTTPException(status_code=404, detail=f"model {name!r} is not loaded")
        try:
            vectors = deps.server.get_features(artifact.service, body.rows)
        except SchemaError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        scored = artifact.score(vectors)
        results = []
        for vector, outcome in zip(vectors, scored):
            scores_total.inc(model=name, decision=outcome["decision"])
            results.append(
                ScoredRow(
                    score=outcome["score"],
                    decision=outcome["decision"],
                    features=vector,
                    explanation=artifact.explain(vector) if body.explain else None,
                )
            )
        return ScoreResponse(
            model=name, version=artifact.version, threshold=artifact.threshold, results=results
        )

    return app
