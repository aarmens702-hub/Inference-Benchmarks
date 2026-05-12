---
title: InferBench
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: FP32 vs INT8 DistilBERT side by side
---

# InferBench

> A single-node ML inference-serving harness for ONNX Runtime — measures
> how dynamic batching, caching, quantization, request scheduling, and
> execution-provider choice trade off latency, throughput, and accuracy
> on real traffic.

[![CI](https://github.com/aarmens702-hub/Inference-Benchmarks/actions/workflows/ci.yml/badge.svg)](https://github.com/aarmens702-hub/Inference-Benchmarks/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-huggingface%20spaces-yellow)](https://huggingface.co/spaces/Aarmen/inferbench)
![Python](https://img.shields.io/badge/python-3.11-blue)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.25-orange)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-55%20passing-brightgreen)

**Live demo:** [huggingface.co/spaces/Aarmen/inferbench](https://huggingface.co/spaces/Aarmen/inferbench) — paste a sentence, watch FP32 vs INT8 give the same answer with very different latency.

![Headline result: bench(B) sweep](results/headline/concurrency_sweep.png)

> Same model, same load. INT8 + dynamic batching takes peak throughput
> from **240 → 670 req/s** while p99 at c=64 collapses from **2.5 s →
> 110 ms**. Detailed breakdown in [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md).

---

## Table of contents

1. [What this project actually is](#1-what-this-project-actually-is)
2. [The problem it solves](#2-the-problem-it-solves)
3. [How the system fits together](#3-how-the-system-fits-together)
4. [The optimization techniques, explained](#4-the-optimization-techniques-explained)
5. [The benchmark suite](#5-the-benchmark-suite)
6. [Headline findings (with charts)](#6-headline-findings-with-charts)
7. [Quickstart](#7-quickstart)
8. [Repo tour](#8-repo-tour)
9. [Limitations and future work](#9-limitations-and-future-work)

---

## 1. What this project actually is

InferBench is two things stacked together:

1. **An inference server.** A FastAPI service that loads a pretrained
   ONNX model (DistilBERT-SST2 sentiment classifier by default) and
   exposes a `POST /infer` endpoint. Send it text, get back POSITIVE or
   NEGATIVE plus a confidence score and a full timing breakdown.

2. **A benchmark harness.** Six experiments that hammer the server
   under different traffic patterns and measure exactly how it behaves.
   Each experiment writes a self-contained, reproducible result folder.

The point isn't the model. The point is **measuring the engineering
decisions around the model** — batching, queueing, caching, model
precision, hardware backend choice — under realistic load. Every claim
in this README points to a `results/run_NNN/` directory you can
re-run from its committed config snapshot.

---

## 2. The problem it solves

Putting an AI model into production is a tradeoff problem disguised as
a deployment problem. You want all of:

- **Low latency** for individual users
- **High throughput** to handle many users at once
- **Low cost** (small model, cheap hardware)
- **Stable behavior** under traffic spikes

These goals fight each other:

```
faster for one user      ──────►  fewer users handled
handle more users        ──────►  individual responses get slower
shrink the model         ──────►  some accuracy lost
ignore traffic spikes    ──────►  server falls over silently
```

InferBench is a lab where you can pick a configuration, run an
experiment, and measure where on that tradeoff curve you actually
land. It implements the standard production techniques (batching,
caching, quantization, backpressure, adaptive control) and the
benchmark harness tells you which ones help, when, and by how much.

---

## 3. How the system fits together

A request takes this path through the server:

```
                ┌──────────────────┐
                │  HTTP client     │  curl / k6 / closed-loop / Poisson
                │  (load gen)      │  / spike / cache_repeat
                └────────┬─────────┘
                         │ POST /infer  {"inputs": ["..."]}
                         ▼
            ┌─────────────────────────────┐
            │   FastAPI route /infer      │
            │                             │
            │   1. PredictionCache.get()  │──── HIT ────► return immediately
            │      (sub-millisecond)      │              (skip queue, skip model)
            │                             │
            │   2. batcher.submit(text)   │
            │      └─► await future       │
            └────────┬────────────────────┘
                     ▼
        ┌───────────────────────────┐
        │   RequestQueue            │  bounded; HTTP 429 if full
        │   (asyncio.Queue)         │  request_timeout_ms → HTTP 504
        └────────┬──────────────────┘
                 ▼
   ┌──────────────────────────────┐
   │   DynamicBatcher (asyncio)   │  collects until size or wait limit
   │   - max_batch_size           │  e.g. {16 reqs} or {10 ms elapsed}
   │   - max_wait_ms              │
   └────────┬─────────────────────┘
            ▼
 ┌───────────────────────────────────────┐
 │  asyncio.to_thread(runner.run, batch) │  CPU-bound, off the event loop
 └────────┬──────────────────────────────┘
          ▼
┌────────────────────────────┐
│  ONNX Runtime              │  CPU / CUDA / CoreML / TensorRT
│  InferenceSession.run()    │  (chosen at boot from server.yaml)
└────────────────────────────┘

Optional sidecar:  AdaptiveBatchController, runs every 5s,
reads observed p95 latency and queue depth, mutates
batcher.max_batch_size and max_wait_ms.
```

Every response carries a full timing breakdown — see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the file-level map.

### What "latency" means

Total response time has three pieces:

```
┌──────────── latency_ms (e2e wall clock) ────────────────┐
│                                                          │
│  ┌── queue_wait_ms ──┐  ┌── inference_ms ──┐  ┌── ... ──┐│
│  │ time from submit  │  │ ORT session.run  │  │ server  ││
│  │ to batch flush    │  │ for whole batch  │  │ overhead││
│  └───────────────────┘  └─ (shared by    ─┘  └─────────┘│
│                            all in batch)                 │
└──────────────────────────────────────────────────────────┘
```

This decomposition is the single most useful diagnostic in the project.
If the model is slow, fix the model. If the queue is the problem, fix
the queue. They look identical from the outside but require completely
different fixes.

![Latency decomposition](results/headline/latency_decomposition.png)

At low percentiles, end-to-end ≈ inference time. At the tail (p99),
queue wait dominates — meaning the model isn't actually slow, the
*system* is just overloaded. That's the entire reason batching and
backpressure matter.

---

## 4. The optimization techniques, explained

This is the meat of the project. Each technique below has a one-line
intuition, a diagram or chart, and a pointer to the experiment that
measured it.

### 4.1 Dynamic batching

**Intuition:** GPUs and modern CPUs are happier doing one big matrix
multiply than four small ones. So instead of running the model
immediately when a request arrives, wait a few milliseconds, see if
more requests show up, and run them all together in a single batch.

![Batching concept](results/headline/batching_concept.png)

Two knobs:

- `max_batch_size` — flush when this many requests are queued
- `max_wait_ms` — flush after this much time, even if batch isn't full

The tension: bigger batches → more throughput, but the *first* request
in each batch waits longer. There's a sweet spot.

**Result (bench(B), [run_002](results/run_002) vs [run_003](results/run_003)):**

| concurrency | no batching p99 | static batching p99 | delta            |
|------------:|----------------:|--------------------:|-----------------:|
|  1          |    20.98 ms     |       33.53 ms      |  +60 % (worse)   |
|  4          |    42.25 ms     |       66.25 ms      |  +57 % (worse)   |
|  8          |   130.60 ms     |       45.94 ms      |  **-65 %**       |
| 16          |   307.53 ms     |      105.20 ms      |  **-66 %**       |
| 32          |   372.12 ms     |      290.89 ms      |  **-22 %**       |
| 64          |   490.10 ms     |    2 498.09 ms      | +410 % (broken)  |

Three regimes show up clearly:

- **Low concurrency**: batching is overhead — solo requests pay the full
  10 ms wait for friends who never come.
- **Mid concurrency** (the sweet spot): the queue stays full enough that
  batches actually fill, and tail latency drops by ~65 %.
- **Saturation**: with no upper bound on the queue, items pile up faster
  than batches drain. Latency explodes. This is what backpressure (§4.3)
  is for.

### 4.2 Prediction cache

**Intuition:** if someone asks the same thing twice, don't re-run the
model — return the saved answer.

```
client request: "this movie sucks"
                       │
                       ▼
       ┌─────────────────────────┐
       │ key = sha1(model_id ||  │
       │           precision ||  │
       │           input_text)   │
       └────────┬────────────────┘
                ▼
       ┌──────────────┐    HIT (≈ 0 ms) ──► return cached prediction
       │  LRU cache   │────►
       │  (4096 keys) │    MISS  ──► run model, cache the result
       └──────────────┘
```

Why include model_id and precision in the key? Because if you swap from
FP32 to INT8, the prediction *might* differ slightly — using stale
cached values would silently leak FP32 answers into your INT8 deployment.

**Result (bench(E), [run_006](results/run_006)):**

![Cache hit ratio vs latency](results/headline/cache_hit_latency.png)

p50 collapses **24×** — from 19.6 ms to 0.81 ms — when 80 % of requests
are repeats. That's because cache hits skip both the queue *and* the
model entirely.

### 4.3 Backpressure: bounded queue + request timeout

**Intuition:** when a queue has no upper bound, two failure modes hide
behind each other. The server *looks* responsive (it's accepting
requests, no errors!) but new arrivals wait further and further behind
older ones until the latency for any user is unacceptable. By the time
you notice, you're already five minutes deep into a queue.

The fix is to make the failure mode loud:

- `queue.max_size: 1024` — once the queue hits this, the next
  `put_nowait` raises `QueueOverflowError`, and the route returns
  **HTTP 429 Too Many Requests**. Clients learn immediately that the
  server is overloaded.
- `queue.request_timeout_ms: 5000` — wraps the per-request wait in
  `asyncio.wait_for`. If a request has been queued longer than this,
  it returns **HTTP 504 Gateway Timeout** instead of holding a slot
  forever.

```
   incoming requests           bounded queue (max=1024)
      ► ► ►                    ┌──────────────────┐
      ► ► ►   ──────────►      │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │ ──► batcher
      ► ► ►                    └──────────────────┘
      ► ► ►   ◄── HTTP 429      (full → reject new)
      ► ► ►   ◄── HTTP 504      (waited too long → drop)
```

The 410 % p99 regression at c=64 in §4.1 is the failure mode this
prevents.

### 4.4 Quantization (INT8)

**Intuition:** the model's weights are stored as 32-bit floats by
default. If you convert them to 8-bit integers, the model is 4×
smaller, runs ~3× faster on CPU, and loses a tiny bit of accuracy.
ONNX Runtime can do this with one function call (`quantize_dynamic`)
and no calibration data.

![Quantization tradeoff](results/headline/quant_tradeoff.png)

**Per-class breakdown** ([results/quant_tradeoff/](results/quant_tradeoff)):

| class    | FP32 acc | INT8 acc | delta         |
|----------|---------:|---------:|--------------:|
| negative |  89.02 % |  89.49 % | **+0.47 pp**  |
| positive |  93.02 % |  91.89 % | **-1.13 pp**  |

**Three examples flipped out of 872.** For 4× size reduction and ~3×
throughput, that's the right side of the trade for almost any
deployment that isn't accuracy-critical.

Even better: INT8 alone resolves the c=64 saturation cliff from §4.1.
The same workload that crushed FP32 (2.5 s p99) runs at 110 ms p99
with INT8 because drain rate (670 req/s) now exceeds arrival rate
everywhere we tested.

### 4.5 Execution providers (CPU / GPU / accelerators)

**Intuition:** ONNX Runtime can target different backends — vanilla CPU,
NVIDIA CUDA, NVIDIA TensorRT (a more aggressive CUDA), or Apple's
CoreML. The `model.backend` config picks one.

```yaml
model:
  backend: onnxruntime-cpu      # default
  # backend: onnxruntime-cuda      # NVIDIA GPU
  # backend: onnxruntime-coreml    # Apple ANE / GPU
  # backend: onnxruntime-tensorrt  # NVIDIA + TRT compile
```

Each backend is a *priority chain*: CoreML falls back to CPU for any
node CoreML can't handle. `resolve_providers()` fails loudly if none
are available, so a misconfigured CUDA box doesn't silently demote to
CPU and run 50× slower without you noticing. Full details in
[`docs/BACKENDS.md`](docs/BACKENDS.md).

We tested CoreML on this Mac. It was a surprise:

| backend            | c=1 mean | c≥4 stability | notes                              |
|--------------------|---------:|---------------|------------------------------------|
| onnxruntime-cpu    |   3.3 ms | stable        | Best on M-series for transformers. |
| onnxruntime-coreml |  17.0 ms | **broken**    | 287/417 nodes covered → partitioned graph; 97-100 % errors at concurrent load. |

CoreML EP is great for big CNNs where ANE shines, but for a small
transformer it can't avoid splitting the graph between CPU and CoreML
and the cross-device copy overhead destroys the win. Documented honestly
in [run_013](results/run_013).

### 4.6 Adaptive batching controller

**Intuition:** static `max_batch_size=16, max_wait_ms=10` is a guess.
Could the server tune those itself based on observed performance?

The controller runs as a separate asyncio task that wakes every 5
seconds, reads recent p95 latency from the batcher's rolling window,
and decides:

```
every 5s:
    if p95 > SLO:
        if queue is deep (saturated):
            grow batch (drain queue faster)
        else:
            shrink wait (let solo requests through)
    elif p95 < 0.7 * SLO and queue has stuff in it:
        grow both (harvest throughput from slack)
    else:
        hold
```

Bounded steps (`±2` each tick), min/max clamps, and a "no-data" tick
when fewer than 32 samples have been observed since boot — together
they prevent oscillation. There are 10 unit tests verifying these
properties.

**Result at λ=200 (saturation), [run_010](results/run_010) vs [run_011](results/run_011) vs [run_012](results/run_012):**

![Adaptive vs static](results/headline/adaptive_vs_static.png)

Both the naive (v1) and saturation-aware (v2) controllers **regressed**
against just leaving the defaults alone. This is the most interesting
finding in the project:

> At λ ≈ peak rate, no batching policy can fix the underlying capacity
> shortage. The right tool at saturation is **load shedding** (§4.3),
> not knob-tuning. Adaptive batching's real domain is **moderate load
> with bursts** — calm periods to harvest throughput, spikes to
> protect the tail against. That's a future experiment.

Most tutorials ship adaptive controllers without this caveat. Building
one and *measuring its limits* is the systems-engineering value of the
exercise. Full discussion in
[`docs/TRADEOFFS.md` §6](docs/TRADEOFFS.md).

---

## 5. The benchmark suite

Six scenarios stress the server in different ways. All write to
`results/run_NNN/` with a results JSON, summary markdown, raw samples,
config snapshot, and PNG charts.

| ID | Name | What it measures | Workload |
|----|------|------------------|----------|
| **A** | Single-request baseline | Raw latency, no queueing | 1 client, sequential |
| **B** | Concurrency sweep | Latency vs throughput across load | Closed-loop at {1, 4, 8, 16, 32, 64} clients |
| **C** | Poisson arrivals | Open-loop traffic at fixed mean rate | Exp(λ) interarrivals at λ rps |
| **D** | Spike test | Server response to sudden burst | 10 → 100 → 10 rps, 30 s each, unique inputs |
| **E** | Cache-heavy | Cache value at varying repeat rates | 0 / 20 / 50 / 80 % hot-key probability |
| **F** | Cold start | First-request latency on fresh server | Spawn fresh process, time first request |

**Closed-loop vs open-loop** is an important distinction. Closed-loop
clients send the next request only after they get a response — backpressure
is automatic. Open-loop clients (Poisson, spike) send on a schedule
regardless of how the server is doing — the server *can't* slow them
down, so it must reject them or fall over. Real internet traffic looks
like the open-loop case.

Per-run output structure:

```
results/run_NNN/
├── results.json         # full metrics, machine-readable
├── summary.md           # human-readable, what the commit message quotes
├── raw_latencies.json   # per-request samples (for replots, CDFs)
├── config.snapshot.yaml # exact server config used (for reproducibility)
└── charts/              # latency_cdf.png and/or sweep curves
```

The config snapshot is the key. Six months from now, anyone can take
`config.snapshot.yaml`, drop it in `configs/server.yaml`, and rerun
the same scenario to get statistically equivalent numbers.

---

## 6. Headline findings (with charts)

The committed runs tell a coherent story. In order of impact:

### 6.1 INT8 quantization is a near-free 4× win

256 MB → 64 MB, accuracy 91.06 % → 90.71 %, throughput 87 → 250
examples/sec at batch=16. See §4.4.

### 6.2 Static batching halves p99 in the mid-load regime

c=8: 130 ms → 46 ms p99. c=16: 308 ms → 105 ms. See §4.1 and
[run_002](results/run_002) vs [run_003](results/run_003).

### 6.3 Caching cuts p50 by 24× when traffic repeats

At 80 % repeat rate, p50 drops from 19.57 ms to 0.81 ms because hits
skip everything. See §4.2 and [run_006](results/run_006).

### 6.4 Adaptive batching is not the right tool at saturation

Two controller versions both regressed at λ=200. Documented honestly,
because it's a useful negative result. See §4.6.

### 6.5 Apple CoreML EP can be slower than CPU and unstable under concurrency

For DistilBERT-class models on M-series, CPU wins. See §4.5 and
[run_013](results/run_013).

### 6.6 Two real bugs the benchmarks surfaced

- **Batcher saturation bug**: `_collect_batch` deadline used the first
  request's submitted-at time, which is in the past under saturation.
  Result: every batch was size 1 even with deep queues. Fix: drain
  available items via `get_nowait` once the deadline passes. **Single
  commit dropped p99 at λ=200 from 1.71 s to 0.29 s.**
- **Error-rate math**: `errors / successes` returned 352 % when CoreML
  melted. Now `errors / (errors + successes)`, capped at 1 by
  construction.

---

## 7. Quickstart

```bash
make venv install      # python3.11 venv + deps
make export-model      # download DistilBERT-SST2 → ONNX FP32 (~256 MB)
make quantize-model    # → ONNX INT8 dynamic (~64 MB)

make serve             # FastAPI on http://127.0.0.1:8000
make bench SCENARIO=A  # measured run → results/run_NNN/

make ci                # ruff + pytest, no model required
```

### Or run in a container

```bash
make compose-up        # docker compose up --build -d
curl http://localhost:8000/health
make compose-down
```

### Open the Gradio demo locally

```bash
make serve             # FastAPI + /demo mounted
open http://127.0.0.1:8000/demo
```

`server.yaml` defines `models:` with both `fp32` and `int8`. The demo
fans the input out to both in parallel and renders a latency badge per
side. To ship API-only, set `demo.enabled: false`.

### Deploy to HuggingFace Spaces

The `Dockerfile.spaces` variant bakes the FP32 + INT8 models into the
image at build time so the Space boots ready to serve. After creating
the Space (see [docs/SPACE_README.md](docs/SPACE_README.md)):

```bash
make space-remote      # one-time: add huggingface as a git remote
make deploy-space      # push main → Space builds and goes live
```

### Hit the API directly

```bash
curl -X POST http://localhost:8000/infer \
  -H 'Content-Type: application/json' \
  -d '{"inputs":["this movie was surprisingly good"]}'
```

Sample response:

```json
{
  "request_id": "abc123",
  "predictions": [{"label": "POSITIVE", "score": 0.9999}],
  "latency_ms": 18.2,
  "queue_wait_ms": 10.5,
  "inference_ms": 7.0,
  "batch_size": 1,
  "cache_hit": false,
  "model_backend": "onnxruntime-cpu",
  "model_precision": "fp32"
}
```

### Watch the server adapt

```bash
curl http://localhost:8000/metrics | jq
# {
#   "cache":   {"hits": 1, "misses": 1, "hit_ratio": 0.5, ...},
#   "batcher": {"queue_size": 0, "avg_batch_size": 1.0,
#               "max_batch_size": 16, "max_wait_ms": 10.0,
#               "batch_size_histogram": {"1": 1}},
#   "controller": { "tick": 4, "last_action": "hold", ... }
# }
```

`configs/server.yaml` is the single source of truth — model, backend,
batching, cache, controller. `configs/benchmark.yaml` defines the
scenarios.

---

## 8. Repo tour

```
inferbench/
├── server/
│   ├── app.py        FastAPI factory + lifespan
│   ├── routes.py     /infer, /health, /metrics, /admin/cache/reset
│   └── schemas.py    Pydantic request / response types
│
├── engine/
│   ├── model_runner.py   ORT InferenceSession + tokenizer wrapper
│   ├── batcher.py        DynamicBatcher (asyncio task, size/wait flush)
│   ├── request_queue.py  bounded queue + QueueOverflowError → 429
│   ├── cache.py          thread-safe LRU, sha1 keys
│   ├── controller.py     AdaptiveBatchController, saturation-aware
│   └── metrics.py        single source of truth for percentiles
│
├── models/
│   ├── export_model.py        HuggingFace → ONNX FP32 (optimum)
│   ├── quantize_model.py      ORT dynamic INT8
│   └── evaluate_accuracy.py   SST-2 validation harness
│
├── benchmarks/
│   ├── run_benchmark.py   CLI dispatcher for scenarios A..F
│   ├── workloads.py       closed_loop / poisson / spike / cache_repeat
│   └── k6/                infer_load_test.js (HTTP-level)
│
└── reports/
    ├── generate_report.py  results.json + summary.md per run
    ├── plot_results.py     latency CDFs + sweep curves
    └── quant_compare.py    FP32 vs INT8 joined report

configs/                server.yaml + benchmark.yaml
docs/                   ARCHITECTURE.md, BACKENDS.md, TRADEOFFS.md
results/run_NNN/        every committed benchmark, snapshot + charts
results/headline/       README cover charts
results/accuracy/       SST-2 evaluation sidecars
results/quant_tradeoff/ joined comparison report
tests/                  48 tests, asyncio + integration via TestClient
.github/workflows/      ruff lint + pytest CI
```

Read the docs in this order if you're new:

1. [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — module map and
   request flow with diagrams
2. [`docs/BACKENDS.md`](docs/BACKENDS.md) — execution-provider chains
   and the "pick the right backend" table
3. [`docs/TRADEOFFS.md`](docs/TRADEOFFS.md) — the full design
   narrative, every claim keyed to a committed run

---

## 9. Limitations and future work

**Honest limitations:**

- Single-node only. No replicas, no load balancer, no distributed serving.
- No streaming inference (one `InferResponse` per `/infer` call).
- Cold-start measurement underestimates real-world cold start — the
  Python process is already running, so we miss the import /
  session-construction cost a Kubernetes pod boot would pay.
- Adaptive controller is empirical, not formally tuned. The
  no-oscillation property is asserted with tests, not proven.
- TensorRT and CUDA paths exist in code but were never executed —
  no NVIDIA host on the dev machine.

**Future work, ranked by leverage:**

1. **Comparison against NVIDIA Triton Inference Server.** Same
   scenarios, same model, side-by-side. This is the credibility
   multiplier — proves the harness numbers aren't off in the absolute
   compared to a battle-tested reference.
2. **Real TensorRT run** on an NVIDIA host. The code path exists; only
   the actual compile + bench commit is missing.
3. **A second model** — MobileNetV2 image classifier — to prove the
   harness isn't DistilBERT-coupled. Image preprocessing is different
   enough that this would catch real coupling bugs.
4. **gRPC transport.** REST is fine for the harness, but gRPC is what
   production inference servers actually speak.
5. **Static (calibration-based) INT8.** Dynamic INT8 covered the
   tradeoff narrative; static would gain ~0.1 pp accuracy at the
   cost of needing a calibration set.
6. **Multi-model multiplexing.** Different problem (routing, isolation,
   per-model batching), bigger scope.

---

## License

MIT. See [LICENSE](LICENSE).
