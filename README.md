# InferBench

> A single-node ML inference-serving harness for ONNX Runtime — measures
> how dynamic batching, caching, quantization, request scheduling, and
> execution-provider choice trade off latency, throughput, and accuracy
> on real traffic.

[![CI](https://github.com/aarmens702-hub/Inference-Benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/aarmens702-hub/Inference-Benchmarks/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.25-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-48%20passing-brightgreen)

![Headline result: bench(B) sweep](results/headline/concurrency_sweep.png)

> Same model, same load. INT8 + dynamic batching takes peak throughput
> from **240 → 670 req/s** while p99 at c=64 collapses from **2.5 s →
> 110 ms**. Detailed breakdown in [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

---

## TL;DR — what's in the box

| Component | What it does | Where |
|---|---|---|
| **FastAPI server** | Async `/infer`, `/health`, `/metrics`, `/admin/cache/reset` | [`inferbench/server`](inferbench/server) |
| **Dynamic batcher** | Async queue + size/wait flush + per-request futures | [`engine/batcher.py`](inferbench/engine/batcher.py) |
| **Backpressure** | Bounded queue → `429`, request timeout → `504` | [`engine/request_queue.py`](inferbench/engine/request_queue.py) |
| **Prediction cache** | Thread-safe LRU keyed by `sha1(model‖precision‖input)` | [`engine/cache.py`](inferbench/engine/cache.py) |
| **Adaptive controller** | SLO-driven, saturation-aware batch tuner | [`engine/controller.py`](inferbench/engine/controller.py) |
| **Quantization** | ORT dynamic INT8 + SST-2 accuracy harness | [`models/`](inferbench/models) |
| **Backends** | CPU / CUDA / CoreML / TensorRT EP chains with auto-fallback | [`docs/BACKENDS.md`](docs/BACKENDS.md) |
| **Benchmark suite** | 6 scenarios + k6 + matplotlib charts + markdown reports | [`benchmarks/`](inferbench/benchmarks) |

48 tests, ruff-clean, GitHub Actions gating both. Full architecture
diagram in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Quickstart

```bash
make venv install      # python3.11 venv + deps
make export-model      # download DistilBERT-SST2 → ONNX FP32 (~256 MB)
make quantize-model    # → ONNX INT8 dynamic (~64 MB)

make serve             # FastAPI on http://127.0.0.1:8000
make bench SCENARIO=A  # measured run → results/run_NNN/

make ci                # ruff + pytest, no model required
```

Or run in a container:

```bash
make compose-up        # docker compose up --build -d
```

`configs/server.yaml` controls model, backend, batching, cache, and
controller. `configs/benchmark.yaml` defines the six scenarios.

---

## Headline results

DistilBERT-SST2 on Apple M-series CPU, 8 cores, ONNX Runtime 1.25.

### 1. Quantization tradeoff

![Quantization tradeoff](results/headline/quant_tradeoff.png)

Dynamic INT8 over `quantize_dynamic` — no calibration data needed.
[`results/quant_tradeoff/summary.md`](results/quant_tradeoff/summary.md)
has the per-class breakdown and the joined latency table.

### 2. Cache value (Scenario E, 50 rps, FP32)

![Cache hit latency](results/headline/cache_hit_latency.png)

p50 collapses **24×** from 19.6 ms → 0.81 ms at 80 % repeat ratio —
sub-millisecond median because hits skip both queue and inference.
Hits are isolated per ratio via [`/admin/cache/reset`](inferbench/server/routes.py).
[`results/run_006/`](results/run_006)

### 3. Cold start (Scenario F, INT8 + batching)

| metric | value |
|---|---:|
| cold (1st request) | 22.46 ms |
| warm p50 / p99 | 14.11 / 15.01 ms |
| ratio | **1.59 ×** |

Spawned-server measurement with `INFERBENCH_SKIP_WARMUP=1` so we
observe the unwarmed first forward pass. [`results/run_009/`](results/run_009)

---

## Benchmark scenarios

| ID | Name | What it tests |
|----|------|---------------|
| **A** | Single-request baseline | Raw model latency, no queueing |
| **B** | Fixed-concurrency sweep | Closed-loop at {1, 4, 8, 16, 32, 64} clients |
| **C** | Poisson arrivals | Open-loop traffic at λ rps |
| **D** | Spike test | 10 → 100 → 10 rps with unique inputs |
| **E** | Cache-heavy | 0 / 20 / 50 / 80 % repeat ratio |
| **F** | Cold start | First-request latency on a fresh server |

Each `bench` invocation writes a self-contained directory:

```
results/run_NNN/
├── results.json         # full metrics
├── summary.md           # human-readable
├── raw_latencies.json   # per-request samples (replottable)
├── config.snapshot.yaml # exact server config used
└── charts/              # latency_cdf.png and/or sweep curves
```

Every committed run reproduces from its own snapshot.

---

## Design tradeoffs (the interview-grade highlights)

Full narrative in [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

### Static batching has three regimes, not one

`bench(B)` showed that static batching at `max=16, wait=10ms` is:

- **Pure overhead at low concurrency** (c=1: p99 +60 % vs no batching)
  because solo requests pay the full 10 ms wait
- **The biggest win at mid concurrency** (c=8–32: p99 -65 %) where
  inference cost amortizes across a full batch
- **Catastrophic at saturation** (c=64: p99 +410 % unbounded queue)

The W7 adaptive controller was meant to tune the knobs by regime.

### Adaptive batching's real domain isn't saturation

`bench(C-adaptive)` regressed against static at λ=peak rate — both the
naive (`shrink on SLO violation`) and saturation-aware (`if qsize > k *
batch_size, grow batch instead`) versions did worse than holding the
defaults. Documented honestly in
[run_011](results/run_011) and [run_012](results/run_012).

> The right tool at saturation is load shedding, not batch tuning.
> Adaptive batching's domain is moderate load with bursts —
> harvest throughput in calm periods, protect the tail during spikes.

### INT8 alone resolves the saturation cliff

The c=64 row of bench(B): FP32 stalls at 154 req/s with 2.5 s p99 (queue
unboundedly grows). INT8 runs the same workload at 613 req/s with 111 ms
p99 — drain rate now exceeds arrival rate everywhere we tested. Sometimes
the right answer to "what should the controller do?" is "swap precision."

### CoreML EP isn't a free win on Apple Silicon

`bench(B-coreml)` ([run_013](results/run_013)): only 287 of 417 graph
nodes have CoreML kernels, so the graph is partitioned and the cross-EP
copy/sync overhead dominates the small per-token compute on a model this
small. ~6× slower at c=1, completely unstable at c≥4 (97–100 % errors).
For DistilBERT-class models on M-series, **CPU EP wins**.

### Two real bugs the benchmarks surfaced

- `_collect_batch` deadline used `first.submitted_at + max_wait_ms`,
  which is *already past* under saturation → batches degenerated to
  size 1 even with deep queues. Fixed: drain `get_nowait` after
  deadline. **p99 at λ=200 dropped 1.71 s → 0.29 s** from one commit.
- `error_rate = errors / successes` returned 352 % the first time
  CoreML melted. Now `errors / (errors + successes)`, capped at 1.

---

## Repo tour

```
inferbench/
├── server/         FastAPI app, routes, schemas
├── engine/         model_runner, batcher, queue, cache,
│                   controller, metrics
├── models/         export_model, quantize_model, evaluate_accuracy
├── benchmarks/     run_benchmark, workloads, k6/
└── reports/        generate_report, plot_results, quant_compare

configs/            server.yaml + benchmark.yaml
docs/               ARCHITECTURE.md, BACKENDS.md, TRADEOFFS.md
results/run_*/      every committed benchmark, with snapshot + charts
results/headline/   README cover charts
results/accuracy/   FP32 + INT8 SST-2 evaluation sidecars
results/quant_tradeoff/  joined comparison report
tests/              48 tests, asyncio + integration via TestClient
```

---

## Limitations

- Single-node only.
- No streaming inference (one `InferResponse` per `/infer`).
- Cold-start measurement underestimates real-world cold start
  (Python process is already up; misses import/session-construction).
- Adaptive controller is empirical, not formally tuned.
- TensorRT and CUDA paths exist in code but were never executed —
  no NVIDIA host on the dev machine. See [`docs/BACKENDS.md`](docs/BACKENDS.md).

---

## Future work

- gRPC transport
- Real TensorRT execution-provider run
- Comparison against NVIDIA Triton Inference Server (highest-leverage
  next step for credibility)
- A second model — MobileNetV2 image classifier — to prove the harness
  is model-agnostic
- Static (calibration-based) INT8

---

## License

MIT. See [LICENSE](LICENSE).
