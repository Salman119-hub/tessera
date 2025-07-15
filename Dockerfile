# syntax=docker/dockerfile:1.7
# One image for the API, the stream processor and batch jobs; the command selects the role.

FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[api,kafka,redis,azure]"

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="tessera" \
      org.opencontainers.image.description="Feature platform for real-time ML" \
      org.opencontainers.image.licenses="MIT"
RUN apt-get update \
 && apt-get install -y --no-install-recommends librdkafka1 tini \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --uid 10001 --create-home --shell /usr/sbin/nologin tessera
COPY --from=build /opt/venv /opt/venv
COPY examples /app/examples
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TESSERA_REPO_PATH=/app/examples/fraud/features.py
WORKDIR /app
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8080/healthz').status != 200)"
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["tessera", "serve", "--host", "0.0.0.0", "--port", "8080"]
