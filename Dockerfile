# HTTP API for the ml-1m serving snapshots (ADR-0014).
#
# The image holds code, configs, and the committed results JSON. It holds no
# MovieLens data and no snapshot: mount them at run time.
#
#   docker build -t movielens-recommender .
#   docker run --rm -p 8000:8000 -v "$PWD/artifacts:/app/artifacts:ro" movielens-recommender
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# LightGBM needs the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU wheel first so the [deep] extra does not pull the CUDA build.
RUN pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install ".[deep,rank,api]"

# Configs and tuned hyperparameters let the same image run build-artifacts.
COPY configs ./configs
COPY results ./results

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/artifacts /app/data \
    && chown -R app /app/artifacts /app/data /app/results
USER app

ENV MOVIELENS_ARTIFACTS=/app/artifacts/ml-1m
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"

CMD ["uvicorn", "--factory", "movielens_recommender.serving.api:create_app_from_env", \
     "--host", "0.0.0.0", "--port", "8000"]
