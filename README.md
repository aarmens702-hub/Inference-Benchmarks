# InferBench

Local inference-serving benchmark and optimizer for ONNX models on FastAPI + ONNX Runtime.

## What it is

InferBench exposes an ONNX model behind a FastAPI server and measures how
**dynamic batching, caching, quantization, request scheduling, and execution
provider choice** affect p50 / p95 / p99 latency and throughput under
realistic load.

## Why it matters

Inference serving is a latency / throughput tradeoff problem. Larger batches
improve throughput but hurt tail latency. Quantization is faster but trades
accuracy. A queue protects against bursts but can hide saturation. This
project measures and tunes those tradeoffs end to end on a single node.

## Features

- FastAPI server with async request handling, lifespan-driven setup
- Dynamic batching with configurable `max_batch_size` / `max_wait_ms`
- Bounded queue with HTTP 429 backpressure + `request_timeout_ms` → 504
- LRU prediction cache (sha1-keyed, sub-ms hits) + `/admin/cache/reset`
- ONNX Runtime execution providers: CPU / CUDA / CoreML / TensorRT
  (auto-fallback chain, [BACKENDS.md](docs/BACKENDS.md))
- ONNX FP32 export + dynamic INT8 quantization scripts
- SST-2 accuracy evaluation harness (overall + per-class)
- p50 / p95 / p99 / throughput / queue-time / inference-time / cache-hit
  metrics reported on every response and via `/metrics`
- Adaptive batching controller (saturation-aware SLO tuning)
- Six benchmark scenarios + k6 HTTP load test
- Reproducible per-run JSON, Markdown, raw samples, and matplotlib charts

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the request flow
diagram and module map.

```
client → FastAPI /infer → cache → DynamicBatcher → ModelRunner → ORT (CPU / CUDA / CoreML / TRT)
                                       ↑
                       AdaptiveBatchController (every 5s: read p95, tune knobs)
```

## Quickstart

```bash
make venv                       # python3.11 venv
make install                    # runtime + dev deps
make export-model               # download DistilBERT-SST2 + export to ONNX FP32
make quantize-model             # ORT dynamic INT8

make serve                      # FastAPI on :8000
make bench SCENARIO=A           # measured run, results/run_NNN/

make ci                         # ruff lint + pytest (no model required)

docker compose up --build -d    # containerised serve, mounts ./models
```

`configs/server.yaml` is the source of truth for which model, backend,
batching policy, cache, and controller settings the server uses.
`configs/benchmark.yaml` defines scenarios A..F.

## Benchmark Scenarios

| ID | Name                      | Purpose                                          |
|----|---------------------------|--------------------------------------------------|
| A  | Single-request baseline   | Raw model latency, no queueing                   |
| B  | Fixed concurrency sweep   | Closed-loop at 1, 4, 8, 16, 32, 64 clients       |
| C  | Poisson arrivals          | Realistic open-loop traffic at λ rps             |
| D  | Spike test                | 10 → 100 → 10 req/s burst with unique inputs     |
| E  | Cache-heavy workload      | 0% / 20% / 50% / 80% repeat ratio                |
| F  | Cold start                | Spawned-server first-request latency vs warm     |

Each `bench` invocation writes a self-contained directory:

```
results/run_NNN/
  results.json            full metrics
  summary.md              human-readable
  raw_latencies.json      per-request samples (for replots)
  config.snapshot.yaml    exact server config used
  charts/                 latency_cdf.png and/or sweep curves
```

## Example Results (DistilBERT-SST2, M-series CPU)

### Quantization tradeoff

| metric              | FP32      | INT8 dyn. | delta    |
|---------------------|----------:|----------:|---------:|
| model size          | 255.5 MB  |  64.3 MB  | -74.9 %  |
| SST-2 val accuracy  | 91.06 %   | 90.71 %   | -0.34 pp |
| examples / sec (b=16) | 87.2    | 249.5     | +186 %   |

Full report: [results/quant_tradeoff/summary.md](results/quant_tradeoff/summary.md)

### Concurrency sweep (bench(B), p99 latency)

| c  | FP32, no batch | FP32, static batch | INT8, static batch |
|---:|---------------:|-------------------:|-------------------:|
|  1 |      20.98 ms  |          33.53 ms  |          17.01 ms  |
|  4 |      42.25 ms  |          66.25 ms  |          20.35 ms  |
|  8 |     130.60 ms  |          45.94 ms  |          24.76 ms  |
| 16 |     307.53 ms  |         105.20 ms  |          35.61 ms  |
| 64 |     490.10 ms  |        2498.09 ms  |         110.74 ms  |

INT8 alone takes p99 at c=64 from 2.5 s → 110 ms because the drain
rate now exceeds the arrival rate.

### Cache value (bench(E), 50 rps, FP32)

| repeat ratio | p50 ms |
|--------------|-------:|
| 0 %          | 19.57  |
| 50 %         | 15.22  |
| 80 %         |  0.81  |

24× p50 reduction at 80 % repeat — sub-millisecond cache-hit median.

### Cold start (bench(F), INT8 + batching)

| metric        | value     |
|---------------|----------:|
| cold (1 req)  |  22.46 ms |
| warm p99      |  15.01 ms |
| ratio         |  1.59 ×   |

## Design Decisions

- **Dynamic batching saves the tail**: the c=64 row above is the
  motivating case. See `bench(B)` commits for the static batch
  comparison and the W3 commit message for the algorithm details.
- **Backpressure over unbounded queues**: at saturation, queueing
  silently destroys tail latency (run_002 c=64 hit 2.5 s p99). The
  bounded queue + 429 keeps the failure mode loud.
- **INT8 dynamic over static**: dynamic quantization needs no
  calibration data and gets within ~0.5 pp of FP32 on SST-2. Static
  quantization left as an exercise (see `quantize_model.py`).
- **Adaptive batching is for moderate load, not saturation**:
  `bench(C-adaptive)` showed both naive and saturation-aware
  controllers regress at λ ≈ peak rate. The right tool at saturation
  is load shedding — adaptive batching lives in the regime above
  baseline but with slack.

## Limitations

- Single-node only.
- No streaming inference (one InferResponse per /infer call).
- Cold-start measurement amortizes most setup via lifespan warmup
  by default; `INFERBENCH_SKIP_WARMUP=1` bypasses it for Scenario F.
- Adaptive controller is empirical, not formally tuned — see W7
  commits for limitations.

## Future Work

- gRPC transport
- TensorRT execution provider real run (we only have notes —
  needs an NVIDIA host)
- Comparison against NVIDIA Triton Inference Server
- Multi-model multiplexing
- Static INT8 with calibration

## Repo Tour

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — diagrams and module map
- [docs/BACKENDS.md](docs/BACKENDS.md) — execution-provider tradeoffs
- [docs/TRADEOFFS.md](docs/TRADEOFFS.md) — design decisions deep-dive
- `configs/` — server and benchmark YAML
- `results/run_*/` — every committed benchmark run
- `inferbench/` — implementation, see ARCHITECTURE.md

## License

MIT
