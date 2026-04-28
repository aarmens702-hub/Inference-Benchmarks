# Benchmark Sweep: scenario C

**Scenario**: poisson-arrivals
**Started**: 2026-04-28T21:08:41.795876+00:00
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: open_loop_poisson

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| poisson@50rps | 1485 | 20.34 | 28.40 | 31.30 | 36.53 | 21.51 | 49.5 | 0.00 |
| poisson@100rps | 2947 | 1753.69 | 3046.13 | 3804.45 | 4229.86 | 1577.46 | 97.4 | 0.00 |
| poisson@200rps | 5987 | 1900.69 | 1940.32 | 1952.83 | 1983.22 | 1741.76 | 135.2 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
