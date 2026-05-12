# Single image used by both `docker compose up` locally and HuggingFace
# Spaces in production. Models are exported + quantized inside the image
# at build time (HF Spaces' build sandbox has no access to our local
# models/ directory). docker-compose overlays ./models on top for fast
# local iteration.
#
# Listen port honors $PORT — Spaces sets it to 7860; we default to 7860
# for parity. Override via `docker run -e PORT=8000` or compose.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferbench/ ./inferbench/
COPY configs/ ./configs/

# Bake the FP32 + INT8 ONNX models into the image (~320 MB). Cached as
# a Docker layer so rebuilds without code changes are fast.
RUN python -m inferbench.models.export_model \
    && python -m inferbench.models.quantize_model

EXPOSE 7860

CMD ["sh", "-c", "uvicorn inferbench.server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
