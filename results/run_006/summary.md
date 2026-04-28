# Benchmark Sweep: scenario E

**Scenario**: cache-heavy
**Started**: 2026-04-28T22:32:39.984911+00:00
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: cache_repeat

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| repeat=0.00 | 1541 | 19.57 | 27.15 | 30.32 | 35.65 | 20.66 | 51.3 | 0.00 |
| repeat=0.20 | 1431 | 20.24 | 385.87 | 1049.42 | 1581.25 | 131.53 | 47.7 | 0.00 |
| repeat=0.50 | 1448 | 15.22 | 23.82 | 28.00 | 33.96 | 11.62 | 48.3 | 0.00 |
| repeat=0.80 | 1476 | 0.81 | 19.84 | 21.96 | 31.53 | 5.48 | 49.1 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
