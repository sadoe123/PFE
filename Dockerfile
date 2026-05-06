# ── Stage 1 : builder ─────────────────────────────────────────
FROM python:3.11-slim AS builder

ENV POETRY_VERSION=1.8.2 \
    POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# gcc + libsndfile requis pour webrtcvad (compilation C) et soundfile
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libpq-dev gcc python3-dev libsndfile1-dev ffmpeg \
    && curl -sSL https://install.python-poetry.org | python3 - \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml poetry.lock* ./
RUN poetry install --only main --no-root
COPY . .
RUN poetry install --only main

# ── Voice packages — installés via pip (pas dans pyproject.toml) ──
# webrtcvad  : nécessite gcc pour compilation
# openai-whisper : ne supporte pas PEP 517 via Poetry
# piper-tts / vosk / soundfile : ajoutés ici pour cohérence
RUN /app/.venv/bin/pip install --no-cache-dir \
    setuptools \
    webrtcvad==2.0.10 \
    piper-tts==1.4.2 \
    vosk==0.3.45 \
    soundfile \
    openai-whisper==20231117


# ── Stage 2 : runtime ─────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# ── Dépendances système + ODBC SQL Server ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gnupg2 apt-transport-https ca-certificates libpq5 \
    ffmpeg libsndfile1 \
    && for i in 1 2 3; do \
         curl -fsSL --retry 3 --retry-delay 5 \
           https://packages.microsoft.com/keys/microsoft.asc \
           | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
         && break || sleep 10; \
       done \
    && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] https://packages.microsoft.com/debian/12/prod bookworm main" \
       > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
       msodbcsql18 unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /app/.venv      ./.venv
COPY --from=builder /app/core       ./core
COPY --from=builder /app/connectors ./connectors
COPY --from=builder /app/api        ./api
COPY --from=builder /app/db         ./db

# Dossiers modèles IA persistants (montés via volumes docker-compose)
RUN mkdir -p /app/models/piper /app/models/vosk /app/models/whisper /app/models/hf

RUN addgroup --system onepilot \
    && adduser --system --ingroup onepilot onepilot \
    && chown -R onepilot:onepilot /app/models

USER onepilot

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
