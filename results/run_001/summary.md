# Benchmark Run: scenario A

**Scenario**: single-request-baseline
**Started**: 2026-04-28T20:41:39.654456+00:00
**Duration**: 30.0s
**Backend**: onnxruntime-cpu | **Precision**: fp32

## End-to-end latency (ms)

| count | mean | p50 | p90 | p95 | p99 | min | max |
|------:|-----:|----:|----:|----:|----:|----:|----:|
| 4564 | 6.56 | 6.20 | 8.58 | 9.68 | 11.70 | 4.15 | 58.33 |

## Inference-only latency (ms)

| p50 | p95 | p99 |
|----:|----:|----:|
| 4.98 | 7.85 | 9.57 |

## Throughput

- **Requests**: 4564
- **Throughput**: 152.1 req/s
- **Errors**: 0 (0.00%)

## Environment

- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
