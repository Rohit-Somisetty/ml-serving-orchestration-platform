FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY src /app/src
COPY data /app/data
COPY artifacts /app/artifacts
COPY seed_artifacts /app/seed_artifacts
COPY outputs /app/outputs
COPY scripts /app/scripts
COPY tests /app/tests
COPY Makefile /app/Makefile

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip \
    && pip install . \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["uvicorn", "ml_platform.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
