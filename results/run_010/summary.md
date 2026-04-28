# Benchmark Sweep: scenario C

**Scenario**: poisson-arrivals
**Started**: 2026-04-28T23:13:39.227417+00:00
**Backend**: onnxruntime-cpu | **Precision**: fp32
**Workload**: open_loop_poisson

## Latency vs concurrency

| concurrency | n | p50 ms | p90 ms | p95 ms | p99 ms | mean ms | rps | err % |
|------------:|--:|-------:|-------:|-------:|-------:|--------:|----:|------:|
| poisson@200rps | 12022 | 45.83 | 147.65 | 210.73 | 291.26 | 67.19 | 200.3 | 0.00 |

## Environment
- Host: `Mac-2.lan`
- Platform: `macOS-15.5-arm64-arm-64bit`
- Python: `3.11.14`
- CPU count: 8
