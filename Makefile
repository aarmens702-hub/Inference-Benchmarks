PY := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PIP := .venv/bin/pip

HOST ?= 127.0.0.1
PORT ?= 8000
SCENARIO ?= A

.PHONY: help install venv export-model quantize-model evaluate-accuracy \
        serve test lint bench bench-k6 bench-clean clean \
        compose-up compose-down compose-logs ci

help:
	@echo "InferBench Makefile targets:"
	@echo ""
	@echo "  setup:"
	@echo "    venv               create .venv with python3.11"
	@echo "    install            install runtime + dev deps into .venv"
	@echo ""
	@echo "  models:"
	@echo "    export-model       export DistilBERT-SST2 to ONNX FP32"
	@echo "    quantize-model     quantize FP32 ONNX to INT8 dynamic"
	@echo "    evaluate-accuracy  evaluate model on SST-2 validation"
	@echo ""
	@echo "  run / dev:"
	@echo "    serve              run FastAPI on $(HOST):$(PORT)"
	@echo "    test               run pytest"
	@echo "    lint               run ruff"
	@echo "    ci                 lint + test (pre-push gate)"
	@echo ""
	@echo "  benchmark:"
	@echo "    bench SCENARIO=A   run benchmark scenario A..F"
	@echo "    bench-k6           run k6 HTTP load test"
	@echo "    bench-clean        wipe results/scratch"
	@echo ""
	@echo "  docker:"
	@echo "    compose-up         docker compose up --build -d"
	@echo "    compose-down       docker compose down"
	@echo "    compose-logs       tail container logs"

venv:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

export-model:
	$(PY) -m inferbench.models.export_model

quantize-model:
	$(PY) -m inferbench.models.quantize_model

evaluate-accuracy:
	$(PY) -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-fp32 --output results/accuracy/fp32.json
	$(PY) -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-int8 --output results/accuracy/int8.json

serve:
	$(UVICORN) inferbench.server.app:app --host $(HOST) --port $(PORT)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check inferbench tests

ci: lint test

bench:
	$(PY) -m inferbench.benchmarks.run_benchmark --scenario $(SCENARIO)

bench-k6:
	k6 run inferbench/benchmarks/k6/infer_load_test.js

bench-clean:
	rm -rf results/scratch

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f --tail=100

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
