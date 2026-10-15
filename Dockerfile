FROM python:3.11-slim

ARG PIP_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN groupadd --system routetune \
    && useradd --system --gid routetune --home-dir /app routetune

COPY requirements.lock ./
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --require-hashes -r requirements.lock

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

COPY configs ./configs
COPY scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/data /app/artifacts \
    && chown -R routetune:routetune /app

USER routetune

EXPOSE 8787

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8787/health', timeout=3)"]

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["routetune", "serve", "--label-bundle", "/app/artifacts/bootstrap/label-bundle", "--models", "/app/configs/models.docker.toml", "--trace-db", "/app/data/routetune.db", "--host", "0.0.0.0", "--port", "8787"]
