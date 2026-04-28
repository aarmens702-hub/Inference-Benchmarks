PY := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PIP := .venv/bin/pip

HOST ?= 127.0.0.1
PORT ?= 8000
SCENARIO ?= A

.PHONY: help install venv export-model serve test bench bench-k6 bench-clean clean

help:
	@echo "InferBench Makefile targets:"
	@echo "  make venv          create .venv with python3.11"
	@echo "  make install       install runtime + dev deps into .venv"
	@echo "  make export-model  export DistilBERT-SST2 to ONNX FP32"
	@echo "  make serve         run FastAPI on $(HOST):$(PORT)"
	@echo "  make test          run pytest"
	@echo "  make bench SCENARIO=A   run a benchmark scenario"

venv:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

export-model:
	$(PY) -m inferbench.models.export_model

serve:
	$(UVICORN) inferbench.server.app:app --host $(HOST) --port $(PORT)

test:
	$(PY) -m pytest -q

bench:
	$(PY) -m inferbench.benchmarks.run_benchmark --scenario $(SCENARIO)

bench-k6:
	k6 run inferbench/benchmarks/k6/infer_load_test.js

bench-clean:
	rm -rf results/scratch

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
