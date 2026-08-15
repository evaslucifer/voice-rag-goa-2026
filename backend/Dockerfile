# ==============================================================================
# Voice-Enabled Multilingual RAG Backend - Production Dockerfile
# ==============================================================================

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create non-root user and required directories
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/qdrant_storage && \
    chown -R appuser:appuser /app

# Copy application source code
COPY app ./app
COPY ingestion ./ingestion
COPY data ./data
COPY benchmarks ./benchmarks
COPY .env.example ./.env.example

RUN chown -R appuser:appuser /app

USER appuser

# Expose backend port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start FastAPI application via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
