"""Gradio side-by-side comparator at /demo.

Sync-handler implementation. Mounted Gradio under FastAPI has been
unreliable with async handlers that await futures created elsewhere on
the asyncio loop (batcher.submit) — the futures never resolved through
Gradio's queue worker. This version skips the batcher entirely and
calls `entry.runner.run([text])` synchronously inside Gradio's worker
thread. For a demo with ~1 user at a time that's strictly better:
no queueing latency, no event-loop coupling.

Mounted into the FastAPI app via gr.mount_gradio_app(app, demo, path="/demo").
"""

from __future__ import annotations

import html
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import gradio as gr
from fastapi import FastAPI

from inferbench.server.registry import ModelEntry, ModelRegistry

DEFAULT_EXAMPLES = [
    "this movie was surprisingly good",
    "boring, predictable, and far too long",
    "an unexpected gem with great cinematography",
    "the dialogue felt forced and the pacing was off",
    "a delightful watch from start to finish",
]

_LABEL_COLOURS = {"POSITIVE": "#1f9d55", "NEGATIVE": "#c53030"}


@dataclass
class _Result:
    label: str
    score: float
    latency_ms: float
    inference_ms: float
    precision: str
    backend: str


def _latency_tag(ms: float) -> tuple[str, str]:
    if ms < 20:
        return "#1f9d55", "fast"
    if ms < 100:
        return "#d69e2e", "moderate"
    return "#c53030", "slow"


def _result_html(model_name: str, result: _Result) -> str:
    label_colour = _LABEL_COLOURS.get(result.label, "#4a5568")
    lat_colour, lat_tag = _latency_tag(result.latency_ms)
    return (
        f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:16px;"
        f"background:#fafafa;'>"
        f"<div style='font-size:12px;color:#718096;text-transform:uppercase;"
        f"letter-spacing:0.05em;margin-bottom:8px;'>"
        f"{html.escape(model_name)} · {html.escape(result.precision)}"
        f"</div>"
        f"<div style='font-size:28px;font-weight:600;color:{label_colour};"
        f"margin-bottom:4px;'>{html.escape(result.label)}</div>"
        f"<div style='font-size:14px;color:#4a5568;margin-bottom:12px;'>"
        f"score {result.score:.4f}</div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;font-size:12px;'>"
        f"<span style='background:{lat_colour};color:#fff;padding:3px 8px;"
        f"border-radius:4px;'>{result.latency_ms:.1f} ms ({lat_tag})</span>"
        f"<span style='background:#edf2f7;color:#2d3748;padding:3px 8px;"
        f"border-radius:4px;'>inference {result.inference_ms:.1f} ms</span>"
        f"<span style='background:#edf2f7;color:#2d3748;padding:3px 8px;"
        f"border-radius:4px;'>{html.escape(result.backend)}</span>"
        f"</div></div>"
    )


def _error_html(model_name: str, message: str) -> str:
    return (
        f"<div style='border:1px solid #fed7d7;background:#fff5f5;border-radius:8px;"
        f"padding:16px;'>"
        f"<div style='font-size:12px;color:#718096;text-transform:uppercase;"
        f"margin-bottom:8px;'>{html.escape(model_name)}</div>"
        f"<div style='color:#c53030;font-weight:500;'>{html.escape(message)}</div>"
        f"</div>"
    )


def _run_one(entry: ModelEntry, text: str) -> _Result:
    """Sync call into the model runner. Records wall-clock latency around the
    forward pass and reports the runner's own inference_ms separately.
    """
    t0 = time.perf_counter()
    run_result = entry.runner.run([text])
    latency_ms = (time.perf_counter() - t0) * 1000.0
    prediction = run_result.predictions[0]
    return _Result(
        label=prediction.label,
        score=prediction.score,
        latency_ms=latency_ms,
        inference_ms=run_result.inference_ms,
        precision=entry.runner.metadata.precision,
        backend=entry.runner.metadata.backend,
    )


def build_demo(app: FastAPI) -> gr.Blocks:
    # Two-worker pool so both models run in parallel inside the sync handler.
    # Sized per-process; ample for the demo's traffic profile.
    pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="demo-infer")

    def compare(text: str) -> tuple[str, str]:
        text = (text or "").strip()
        if not text:
            return (
                _error_html("input required", "type some text and press compare"),
                "",
            )

        registry: ModelRegistry = app.state.registry
        entries = registry.all()
        if len(entries) < 2:
            return (
                _error_html(
                    "not enough models",
                    "the demo expects at least two models registered",
                ),
                "",
            )

        left, right = entries[0], entries[1]
        try:
            future_left = pool.submit(_run_one, left, text)
            future_right = pool.submit(_run_one, right, text)
            result_left = future_left.result(timeout=15)
            result_right = future_right.result(timeout=15)
        except Exception as exc:
            return (
                _error_html(left.name, f"error: {exc}"),
                _error_html(right.name, f"error: {exc}"),
            )

        return (
            _result_html(left.name, result_left),
            _result_html(right.name, result_right),
        )

    with gr.Blocks(title="InferBench — FP32 vs INT8") as demo:
        gr.Markdown(
            "# InferBench live demo\n"
            "Same input, two models, two latency numbers. The labels should match; "
            "the latency badges shouldn't. INT8 quantization is ~4× smaller and ~3× "
            "faster than FP32 with -0.34 pp accuracy on SST-2."
        )
        with gr.Row():
            text_input = gr.Textbox(
                label="text",
                placeholder="paste a movie review or a one-line sentence",
                lines=2,
                scale=4,
            )
            submit_btn = gr.Button("compare", variant="primary", scale=1)
        gr.Examples(examples=[[ex] for ex in DEFAULT_EXAMPLES], inputs=text_input)

        with gr.Row():
            left_panel = gr.HTML()
            right_panel = gr.HTML()

        submit_btn.click(fn=compare, inputs=text_input, outputs=[left_panel, right_panel])
        text_input.submit(fn=compare, inputs=text_input, outputs=[left_panel, right_panel])

    return demo
