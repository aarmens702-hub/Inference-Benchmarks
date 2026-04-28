# InferBench — minimal CPU image. GPU image is a follow-up in W8.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY inferbench/ ./inferbench/
COPY configs/ ./configs/
COPY models/ ./models/

EXPOSE 8000

CMD ["uvicorn", "inferbench.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
