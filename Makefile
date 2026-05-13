PY := .venv/bin/python
UVICORN := .venv/bin/uvicorn
PIP := .venv/bin/pip

HOST ?= 127.0.0.1
PORT ?= 8000
SCENARIO ?= A

HF_SPACE ?= huggingface.co/spaces/Aarmen/inferbench
HF_REMOTE_NAME ?= space

.PHONY: help install venv export-model export-model-fp16 quantize-model evaluate-accuracy \
        serve serve-gpu test lint bench bench-gpu bench-k6 bench-clean clean \
        gpu-smoke gpu-install \
        compose-up compose-down compose-logs ci \
        space-remote deploy-space

help:
	@echo "InferBench Makefile targets:"
	@echo ""
	@echo "  setup:"
	@echo "    venv               create .venv with python3.11"
	@echo "    install            install runtime + dev deps into .venv"
	@echo ""
	@echo "  models:"
	@echo "    export-model       export DistilBERT-SST2 to ONNX FP32"
	@echo "    export-model-fp16  convert FP32 ONNX to FP16 (for CUDA EP)"
	@echo "    quantize-model     quantize FP32 ONNX to INT8 dynamic"
	@echo "    evaluate-accuracy  evaluate model on SST-2 validation"
	@echo ""
	@echo "  gpu (see docs/HARDWARE.md):"
	@echo "    gpu-install        swap onnxruntime -> onnxruntime-gpu in .venv"
	@echo "    gpu-smoke          verify CUDAExecutionProvider is reachable"
	@echo "    serve-gpu          serve with configs/server-gpu.yaml"
	@echo "    bench-gpu SCENARIO=B HARDWARE=a2000-ada  bench against GPU server"
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
	@echo ""
	@echo "  hf spaces:"
	@echo "    space-remote       add huggingface as a git remote ($(HF_REMOTE_NAME) -> $(HF_SPACE))"
	@echo "    deploy-space       push main to the huggingface remote"

venv:
	python3.11 -m venv .venv
	$(PIP) install --upgrade pip

install:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

export-model:
	$(PY) -m inferbench.models.export_model

export-model-fp16:
	$(PY) -m inferbench.models.export_model --precision fp16

quantize-model:
	$(PY) -m inferbench.models.quantize_model

evaluate-accuracy:
	$(PY) -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-fp32 --output results/accuracy/fp32.json
	$(PY) -m inferbench.models.evaluate_accuracy --model-dir models/distilbert-sst2-int8 --output results/accuracy/int8.json

serve:
	$(UVICORN) inferbench.server.app:app --host $(HOST) --port $(PORT)

serve-gpu:
	INFERBENCH_CONFIG=configs/server-gpu.yaml $(UVICORN) inferbench.server.app:app --host $(HOST) --port $(PORT)

# GPU bring-up: install onnxruntime-gpu in the existing venv (mutually
# exclusive with the CPU onnxruntime wheel — uninstall first). The >=1.22 pin
# is for ORT's preload_dlls() API which inferbench.engine.model_runner uses on
# Windows to find CUDA dependency DLLs from torch's bundled libs.
gpu-install:
	$(PIP) uninstall -y onnxruntime
	$(PIP) install "onnxruntime-gpu>=1.22" onnxconverter-common

gpu-smoke:
	$(PY) -m inferbench.tools.gpu_smoke

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check inferbench tests

ci: lint test

bench:
	$(PY) -m inferbench.benchmarks.run_benchmark --scenario $(SCENARIO)

# GPU bench: tags the run with HARDWARE (default a2000-ada). Assumes the
# GPU server is already running (in another shell): `make serve-gpu`.
HARDWARE ?= a2000-ada
bench-gpu:
	$(PY) -m inferbench.benchmarks.run_benchmark --scenario $(SCENARIO) --hardware $(HARDWARE)

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

space-remote:
	@if git remote get-url $(HF_REMOTE_NAME) >/dev/null 2>&1; then \
		echo "remote $(HF_REMOTE_NAME) already set to: $$(git remote get-url $(HF_REMOTE_NAME))"; \
	else \
		git remote add $(HF_REMOTE_NAME) https://$(HF_SPACE); \
		echo "added remote $(HF_REMOTE_NAME) -> https://$(HF_SPACE)"; \
	fi

deploy-space: space-remote
	git push $(HF_REMOTE_NAME) main

clean:
	rm -rf .pytest_cache __pycache__ */__pycache__ */*/__pycache__
