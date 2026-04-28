# Quantization Tradeoff: FP32 vs INT8 (dynamic)

Model: `distilbert-base-uncased-finetuned-sst-2-english` on ONNX Runtime CPU EP
FP32 bench: `B` from `2026-04-28T21:01:38.562572+00:00`
INT8 bench: `B` from `2026-04-28T22:47:43.905885+00:00`

## Headline

| metric              | FP32              | INT8              | delta            |
|---------------------|------------------:|------------------:|------------------|
| model size          |         255.5 MB |          64.3 MB | -74.9%            |
| accuracy (SST-2 val)|          91.06 % |          90.71 % | -0.34 pp          |
| examples / sec      |             87.2 |            249.5 | +186.2%            |

## Per-class accuracy

| class    | FP32        | INT8        |
|----------|------------:|------------:|
| negative |      89.02% |      89.49% |
| positive |      93.02% |      91.89% |

## Latency / throughput at concurrency (bench(B))

|  c | p50 FP32 | p50 INT8 | p95 FP32 | p95 INT8 | p99 FP32 | p99 INT8 | rps FP32 | rps INT8 |
|---:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|---------:|
|  1 |    17.10 |    14.76 |    24.67 |    16.17 |    33.53 |    17.01 |     54.2 |     67.5 |
|  4 |    25.88 |    18.24 |    38.68 |    19.50 |    66.25 |    20.35 |    135.1 |    219.4 |
|  8 |    38.65 |    21.80 |    41.15 |    23.41 |    45.94 |    24.76 |    205.8 |    365.6 |
| 16 |    91.80 |    24.64 |    99.17 |    27.31 |   105.20 |    35.61 |    188.5 |    670.9 |
| 32 |   190.82 |    51.11 |   206.64 |    53.92 |   290.89 |    56.09 |    164.8 |    623.3 |
| 64 |   337.00 |   103.90 |   804.72 |   107.57 |  2498.09 |   110.74 |    154.4 |    613.1 |

## Takeaways

- **Size**: INT8 is 4.0x smaller than FP32 — pure consequence of 1-byte vs 4-byte weights.
- **Accuracy**: -0.34 percentage points on SST-2 val. 3 examples flipped out of 872.
- **Throughput** (batched eval, batch=16): INT8 is 2.86x faster.
- **Tail latency** under concurrent load: see table above. Ratio holds across the concurrency range, with the biggest win at the higher-c rows where batched inference dominates.
