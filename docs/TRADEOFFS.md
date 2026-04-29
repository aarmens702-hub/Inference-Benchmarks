# Design Tradeoffs and Findings

This document is the technical narrative of the InferBench project,
keyed to the committed benchmark runs. Every claim here points to a
`results/run_*/` directory you can reproduce from the snapshotted
config.

Hardware throughout: Apple M-series, 8 CPUs, macOS 15.5, Python 3.11,
ONNX Runtime 1.25, DistilBERT-SST2.

---

## 1. Why the project exists

The interesting work in modern model deployment isn't training; it's
**how you serve a fixed model under variable load**. Serving introduces
problems the training stack hides: queueing dynamics, batch formation,
tail latency, backpressure, model precision tradeoffs, execution
provider stability. InferBench is a single-node lab where every one of
those knobs is observable and tunable.

---

## 2. Dynamic batching: the headline tradeoff

`bench(B)` runs a closed-loop concurrency sweep at {1, 4, 8, 16, 32, 64}
clients for 30 s each.

### 2.1 No batching baseline (run_002)

| c  | p50    | p95    | p99    | rps   |
|----|--------|--------|--------|-------|
| 1  |   6.66 |  11.60 |  20.98 | 130.4 |
| 4  |  15.55 |  25.11 |  42.25 | 240.0 |
| 8  |  34.70 |  81.66 | 130.60 | 195.3 |
| 16 |  81.73 | 198.78 | 307.53 | 171.5 |
| 32 | 129.22 | 252.86 | 372.12 | 233.7 |
| 64 | 266.43 | 411.65 | 490.10 | 237.0 |

Throughput peaks at c=4 (~240 req/s). Higher concurrency adds queue
depth, not utilisation: every concurrent request serializes through the
same ORT session, so additional clients only inflate tail latency.

### 2.2 Static batching, max=16/wait=10ms (run_003)

| c  | p50    | p95    | p99     | rps   | vs no-batch p99       |
|----|--------|--------|---------|-------|-----------------------|
| 1  |  17.10 |  24.67 |   33.53 |  54.2 | +60 % (worse)         |
| 4  |  25.88 |  38.68 |   66.25 | 135.1 | +57 % (worse)         |
| 8  |  38.65 |  41.15 |   45.94 | 205.8 | -65 % (better)        |
| 16 |  91.80 |  99.17 |  105.20 | 188.5 | -66 % (better)        |
| 32 | 190.82 | 206.64 |  290.89 | 164.8 | -22 % (better)        |
| 64 | 337.00 | 804.72 | 2498.09 | 154.4 | +410 % (catastrophic) |

Three regimes:

- **Low concurrency (c=1, 4)**: batching is pure overhead. A solo
  request waits the full `max_wait_ms=10ms` before flushing alone.
- **Mid concurrency (c=8 to 32)**: batching halves p99. The p99/p50
  ratio collapses from ~3.8× (no batch) to ~1.1× — much tighter tail.
- **Saturation (c=64)**: unbounded queue. Items arrive faster than
  batches of 16 can drain; p99 explodes to 2.5 s. Motivates W4.

The takeaway: static batching is strictly an SLA-relevant tool only
when concurrency lives in the mid range. Outside that window it
either adds overhead (low load) or fails loudly (saturation).

---

## 3. Backpressure (W4)

The `c=64` regression above is the unbounded-queue failure mode. W4
added:

- `queue_max_size` → `put_nowait` raises `QueueOverflowError` → HTTP 429
- `request_timeout_ms` → `asyncio.wait_for` → HTTP 504

`bench(D)` (run_005) ran a 10 → 100 → 10 rps spike for 90 s with
`queue_max_size=1024`. Queue absorbed the spike (1 timeout out of 3599)
but tail latency still climbed to 2.4 s during the 100 rps phase.
The honest takeaway: with a generous queue limit, backpressure trades
errors for tail latency — both are losses, but at least the failure is
detectable. A tighter queue would shed load earlier.

---

## 4. Caching (W4)

`bench(E)` (run_006) sweeps repeat ratio {0, 20, 50, 80} % at 50 rps,
30 s each.

| repeat ratio | p50 ms | mean ms |
|-------------:|-------:|--------:|
| 0 %          |  19.57 |   20.66 |
| 20 %         |  20.24 |  131.53 |
| 50 %         |  15.22 |   11.62 |
| 80 %         |   0.81 |    5.48 |

p50 collapses from 19.6 ms → 0.81 ms at 80 % repeat — sub-millisecond
median because cache hits skip both queue and inference. The p95 spike
at 20 % repeat is workload variance (single 30 s window); deserves a
longer-window re-run before being treated as a real signal.

W6 added `/admin/cache/reset` and Scenario E now resets between sub-runs,
so `observed_cache_hit_ratio` in the report finally matches the
configured ratio within sample noise.

---

## 5. Quantization (W5)

INT8 dynamic quantization via ORT (`quantize_dynamic` + `QuantType.QInt8`).
No calibration set required.

| metric              | FP32      | INT8      | delta    |
|---------------------|----------:|----------:|---------:|
| model size          | 255.5 MB  |  64.3 MB  | -74.9 %  |
| SST-2 val accuracy  | 91.06 %   | 90.71 %   | -0.34 pp |
| examples / sec      |  87.2     | 249.5     | +186 %   |
| p99 @ c=64 (bench B)| 2498 ms   | 110.7 ms  | -95.6 %  |

Per-class accuracy: negative -0.5 pp, positive -0.0 pp (3 examples
total flipped out of 872).

**INT8 alone resolves the saturation cliff.** With FP32 the c=64 row
of bench(B) hit 2.5 s p99 because the drain rate was below the arrival
rate. With INT8 the same workload runs at p99=111 ms because drain rate
(670 req/s peak) now exceeds arrival rate at every tested concurrency.
Sometimes the right answer to "what should the controller do?" is
"swap precision."

Static (calibration-based) INT8 remains as future work — the dynamic
path already gets the size + speed wins for the project's purposes.

---

## 6. Adaptive batching (W7)

The most architecturally interesting feature. Spec'd as: every 5 s,
read p95, step `max_batch_size` and `max_wait_ms` up or down.

### 6.1 Naive controller (bench(C-adaptive-v1), run_011)

Decision rule:

```
if p95 > slo:    shrink batch + shrink wait
elif p95 < 0.7*slo and qsize is high:    grow
else:    hold
```

At λ=200 (≈ peak FP32 throughput): **the controller regressed.** Static
gave p99=291 ms; adaptive-v1 gave p99=921 ms.

Why: when p95 climbed because the queue was deep (saturation), the
controller shrank `max_batch_size`, cutting drain throughput. The queue
grew faster, p95 spiked further, the controller shrank more — a
positive feedback loop in the wrong direction.

### 6.2 Saturation-aware controller (bench(C-adaptive-v2), run_012)

```
if p95 > slo:
    if qsize > 2 * max_batch_size:
        grow batch (drain faster), keep wait    # saturation branch
    else:
        keep batch, shrink wait                  # batch-wait jitter
elif p95 < 0.7*slo and qsize > max_batch_size/2:
    grow both
else:
    hold
```

Result: still worse than static (p99 = 1.59 s vs 0.29 s). The
controller correctly detected saturation and grew batches from 16 to
22, but FP32 inference time scales sublinearly with batch size — going
16 → 22 made each forward pass ~30 % longer while drain rate barely
budged. Per-batch latency went up; head-of-line waits got worse.

### 6.3 The honest finding

**At λ ≈ peak rate, adaptive batching is the wrong tool.** No batching
policy can convert "more arrivals than the model can process" into
acceptable latency. The right answer at saturation is load shedding,
which W4 already provides. Adaptive batching's real domain is **moderate
load with bursty arrivals** — calm periods to harvest throughput, spikes
to protect against. That regime is a future experiment (e.g. λ=150
with a step burst to 250 on FP32).

What survives:

- The controller infrastructure itself (bounded steps, `min_samples`,
  saturation detection, decision history exposed via `/metrics`)
- The negative result, which is itself a useful interview talking point:
  "I built it, ran it, and learned its real domain isn't what the spec
  hand-waved."

---

## 7. Execution providers (W8)

Backend chains with auto-fallback, then bench(B-coreml) ran on the
M-series Mac to test CoreML.

| backend            | c=1 mean | c=8 stability | per-token notes                         |
|--------------------|---------:|---------------|-----------------------------------------|
| onnxruntime-cpu    |  3.31 ms | stable        | Best on M-series for transformer-class. |
| onnxruntime-coreml | 17.00 ms | broken        | 287/417 nodes supported -> partitioned graph; 97-100 % errors at c≥4. |
| onnxruntime-cuda   |    n/a   | n/a           | No NVIDIA host on this dev machine.     |
| onnxruntime-tensorrt |  n/a   | n/a           | Documented in BACKENDS.md, no run.      |

CoreML EP isn't a free win on Apple Silicon for this workload. Two
distinct findings:

1. **Latency**: copy / sync overhead between CoreML and CPU partitions
   dominates for small per-request compute.
2. **Concurrency**: shared `InferenceSession.run` from multiple
   `asyncio.to_thread` workers fails. Likely needs a session-per-worker
   pool to be usable.

For DistilBERT-class models on Apple Silicon, **CPU EP wins**. The
benchmark suite stays on CPU for everything except this one
diagnostic run.

---

## 8. What changed about the system that wasn't in the spec

Two real bugs the benchmarks surfaced:

### 8.1 Batcher under saturation (fixed in run_010 commit chain)

Original `_collect_batch` used `deadline = first.submitted_at + max_wait_ms`.
Under saturation, `first` had already been queued longer than `max_wait_ms`
when the batcher pulled it, so the inner loop exited immediately even with
items waiting. Result: every batch was size 1 (`avg_batch_size=1.0` even
with a deep queue).

Fix: when the deadline has elapsed, drain available items via `get_nowait`
without further waiting. Low-load behavior unchanged.

This single fix dropped p99 at λ=200 from 1.71 s to 0.29 s.

### 8.2 Error-rate math (fixed in W8 fix commit)

`ThroughputStats.error_rate` was `errors / successes`, which produced
percentages > 100 % the first time CoreML melted down (1 991 errors / 565
successes = 352 %). Now `errors / (errors + successes)`, bounded in [0, 1]
by construction.

---

## 9. What's still loose

- **No real GPU run.** TRT documented in `docs/BACKENDS.md`, never executed.
- **Single model.** A second, structurally different model (e.g.
  MobileNetV2 image classifier) would test the genericity claims.
- **Adaptive controller has no formal proof of stability.** The
  no-oscillation property is empirical, not bounded.
- **Cold-start measurement underestimates real-world cold start.**
  The `INFERBENCH_SKIP_WARMUP` env var bypasses the in-process warmup
  but the Python process is already long-running, so we miss the
  imports / session-construction cost that dominates real cold start
  (e.g. a Kubernetes pod boot).

---

## 10. Sentence-level summary

> InferBench is a serving-side ML systems harness for ONNX Runtime
> that measures the latency / throughput / accuracy / cost of every
> standard knob (batch size, wait time, queue depth, request timeout,
> cache, precision, execution provider, adaptive controller) on real
> traffic patterns, and the answer to most of them is "depends on the
> regime, here are the numbers." The headline result: INT8 dynamic
> quantization plus dynamic batching gets DistilBERT-SST2 from 240
> req/s to 670 req/s on an M-series CPU with -0.34 pp accuracy and
> p99 unchanged.
