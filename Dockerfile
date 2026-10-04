FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best.pt

WORKDIR /app

# CPU-only PyTorch keeps the image around 1 GB instead of 5+ GB.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[api]"

# The trained checkpoint is not baked into the image: mount it at /app/models (see README).
RUN useradd --create-home appuser && mkdir -p /app/models && chown appuser /app/models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "polypseg.api:app", "--host", "0.0.0.0", "--port", "8000"]
