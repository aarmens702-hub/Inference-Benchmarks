// k6 load test for InferBench POST /infer.
//
// Usage:
//   k6 run inferbench/benchmarks/k6/infer_load_test.js
//   k6 run -e BASE_URL=http://127.0.0.1:8000 -e VUS=16 -e DURATION=60s \
//          -e CACHE_HIT_RATIO=0.5 inferbench/benchmarks/k6/infer_load_test.js
//
// Flags:
//   BASE_URL          target server (default http://127.0.0.1:8000)
//   VUS               virtual users (closed-loop concurrency)
//   DURATION          test duration, e.g. 60s, 5m
//   RAMPUP            ramp-up duration (default 5s)
//   CACHE_HIT_RATIO   probability of repeating a fixed input
//                     (used to drive Scenario E without changing the
//                     server). Default 0.0.
//
// Thresholds fail the run if p95 > 200ms or error rate > 1%.

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8000';
const VUS = parseInt(__ENV.VUS || '16');
const DURATION = __ENV.DURATION || '60s';
const RAMPUP = __ENV.RAMPUP || '5s';
const CACHE_HIT_RATIO = parseFloat(__ENV.CACHE_HIT_RATIO || '0.0');

const TEXTS = [
    'this movie was surprisingly good',
    'absolute waste of time',
    'the plot was clever and the acting was solid',
    'i would not recommend this to anyone',
    'an unexpected gem with great cinematography',
    'boring, predictable, and far too long',
    'a delightful watch from start to finish',
    'the dialogue felt forced and the pacing was off',
];
const HOT_KEY = 'this is a fixed cache-warming input';

export const options = {
    stages: [
        { duration: RAMPUP, target: VUS },
        { duration: DURATION, target: VUS },
    ],
    thresholds: {
        http_req_failed: ['rate<0.01'],
        http_req_duration: ['p(95)<200'],
    },
    tags: {
        scenario: __ENV.SCENARIO || 'k6-load',
        cache_hit_ratio: String(CACHE_HIT_RATIO),
    },
};

const inferenceMs = new Trend('infer_inference_ms');
const queueWaitMs = new Trend('infer_queue_wait_ms');
const cacheHits = new Rate('infer_cache_hit_ratio');

function pickText() {
    if (CACHE_HIT_RATIO > 0 && Math.random() < CACHE_HIT_RATIO) {
        return HOT_KEY;
    }
    return TEXTS[Math.floor(Math.random() * TEXTS.length)];
}

export default function () {
    const payload = JSON.stringify({ inputs: [pickText()] });
    const params = {
        headers: { 'Content-Type': 'application/json' },
        timeout: '10s',
    };
    const res = http.post(`${BASE_URL}/infer`, payload, params);

    const ok = check(res, {
        'status 200': (r) => r.status === 200,
    });

    if (ok && res.body) {
        try {
            const body = JSON.parse(res.body);
            if (typeof body.inference_ms === 'number') inferenceMs.add(body.inference_ms);
            if (typeof body.queue_wait_ms === 'number') queueWaitMs.add(body.queue_wait_ms);
            cacheHits.add(body.cache_hit === true ? 1 : 0);
        } catch (e) {
            // body not JSON — ignored
        }
    }
}
