"""Smoke tests for ModelRunner against the exported DistilBERT-SST2 ONNX.

Skipped automatically if the model isn't exported yet (CI without GPU /
without prior `make export-model` shouldn't fail to collect).
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODEL_DIR = Path("models/distilbert-sst2-fp32")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").exists(),
    reason="Run `make export-model` first.",
)


@pytest.fixture(scope="module")
def runner():
    from inferbench.engine.model_runner import ModelRunner

    r = ModelRunner(MODEL_DIR)
    r.warmup(n=2)
    return r


def test_runner_returns_one_prediction_per_input(runner):
    out = runner.run(["this movie was surprisingly good"])
    assert len(out.predictions) == 1
    assert out.batch_size == 1


def test_runner_predicts_positive_sentiment(runner):
    out = runner.run(["this movie was surprisingly good"])
    pred = out.predictions[0]
    assert pred.label == "POSITIVE"
    assert pred.score > 0.9


def test_runner_predicts_negative_sentiment(runner):
    out = runner.run(["absolute waste of time"])
    pred = out.predictions[0]
    assert pred.label == "NEGATIVE"
    assert pred.score > 0.9


def test_runner_handles_batch(runner):
    out = runner.run(["a delightful film", "boring and predictable"])
    assert out.batch_size == 2
    assert [p.label for p in out.predictions] == ["POSITIVE", "NEGATIVE"]


def test_runner_records_timing(runner):
    out = runner.run(["sample text"])
    assert out.inference_ms > 0
    assert out.preprocess_ms >= 0
    assert out.postprocess_ms >= 0


def test_runner_rejects_empty_batch(runner):
    with pytest.raises(ValueError):
        runner.run([])
