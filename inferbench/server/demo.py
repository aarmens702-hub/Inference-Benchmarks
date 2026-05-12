"""Gradio side-by-side comparator at /demo.

Posts the same input to two models from the registry in parallel and
shows label, score, and a latency badge per side. The latency colour
encoding makes the FP32-vs-INT8 tradeoff visible at a glance.

Mounted into the FastAPI app via gr.mount_gradio_app(app, demo, path="/demo").
"""

from __future__ import annotations

import asyncio
import html
from datetime import datetime

import gradio as gr
from fastapi import FastAPI

from inferbench.server.inference import (
    InferConfig,
    InferOverloaded,
    InferTimedOut,
    run_inference,
)
from inferbench.server.registry import ModelEntry, ModelRegistry
from inferbench.server.schemas import InferResponse

DEFAULT_EXAMPLES = [
    "this movie was surprisingly good",
    "boring, predictable, and far too long",
    "an unexpected gem with great cinematography",
    "the dialogue felt forced and the pacing was off",
    "a delightful watch from start to finish",
]

_LABEL_COLOURS = {"POSITIVE": "#1f9d55", "NEGATIVE": "#c53030"}


def _latency_tag(ms: float) -> tuple[str, str]:
    if ms < 20:
        return "#1f9d55", "fast"
    if ms < 100:
        return "#d69e2e", "moderate"
    return "#c53030", "slow"


def _result_html(model_name: str, response: InferResponse) -> str:
    pred = response.predictions[0]
    label_colour = _LABEL_COLOURS.get(pred.label, "#4a5568")
    lat_colour, lat_tag = _latency_tag(response.latency_ms)
    cache = "cache hit" if response.cache_hit else f"batch={response.batch_size}"
    return (
        f"<div style='border:1px solid #e2e8f0;border-radius:8px;padding:16px;"
        f"background:#fafafa;'>"
        f"<div style='font-size:12px;color:#718096;text-transform:uppercase;"
        f"letter-spacing:0.05em;margin-bottom:8px;'>"
        f"{html.escape(model_name)} · {html.escape(response.model_precision)}"
        f"</div>"
        f"<div style='font-size:28px;font-weight:600;color:{label_colour};"
        f"margin-bottom:4px;'>{html.escape(pred.label)}</div>"
        f"<div style='font-size:14px;color:#4a5568;margin-bottom:12px;'>"
        f"score {pred.score:.4f}</div>"
        f"<div style='display:flex;gap:8px;flex-wrap:wrap;font-size:12px;'>"
        f"<span style='background:{lat_colour};color:#fff;padding:3px 8px;"
        f"border-radius:4px;'>{response.latency_ms:.1f} ms ({lat_tag})</span>"
        f"<span style='background:#edf2f7;color:#2d3748;padding:3px 8px;"
        f"border-radius:4px;'>inference {response.inference_ms:.1f} ms</span>"
        f"<span style='background:#edf2f7;color:#2d3748;padding:3px 8px;"
        f"border-radius:4px;'>queue {response.queue_wait_ms:.1f} ms</span>"
        f"<span style='background:#edf2f7;color:#2d3748;padding:3px 8px;"
        f"border-radius:4px;'>{cache}</span>"
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


def _history_cell(name: str, response: InferResponse | None) -> str:
    if response is None:
        return f"{name}: —"
    pred = response.predictions[0]
    return f"{name}: {pred.label} ({pred.score:.2f}) · {response.latency_ms:.1f}ms"


async def _safe_infer(entry: ModelEntry, text: str, cfg: InferConfig) -> tuple[InferResponse | None, str]:
    try:
        response = await run_inference(entry, [text], cfg)
        return response, _result_html(entry.name, response)
    except InferOverloaded:
        return None, _error_html(entry.name, "queue full — try again in a moment")
    except InferTimedOut:
        return None, _error_html(entry.name, "request timed out")
    except Exception as exc:  # surface unexpected errors to the UI
        return None, _error_html(entry.name, f"error: {exc}")


async def _compare(
    text: str,
    history: list[list[str]],
    app: FastAPI,
) -> tuple[str, str, list[list[str]]]:
    text = (text or "").strip()
    if not text:
        return (
            _error_html("input required", "type some text and press submit"),
            "",
            history,
        )

    registry: ModelRegistry = app.state.registry
    entries = registry.all()
    if len(entries) < 2:
        return (
            _error_html(
                "not enough models",
                "the demo expects at least two models registered (e.g. fp32 and int8)",
            ),
            "",
            history,
        )

    cfg = InferConfig(
        cache=app.state.cache,
        request_timeout_ms=app.state.request_timeout_ms,
    )
    left, right = entries[0], entries[1]

    (resp_left, html_left), (resp_right, html_right) = await asyncio.gather(
        _safe_infer(left, text, cfg),
        _safe_infer(right, text, cfg),
    )

    snippet = text if len(text) <= 60 else text[:57] + "..."
    row = [
        datetime.now().strftime("%H:%M:%S"),
        snippet,
        _history_cell(left.name, resp_left),
        _history_cell(right.name, resp_right),
    ]
    new_history = (history + [row])[-20:]
    return html_left, html_right, new_history


def build_demo(app: FastAPI) -> gr.Blocks:
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
            submit = gr.Button("compare", variant="primary", scale=1)
        gr.Examples(examples=[[ex] for ex in DEFAULT_EXAMPLES], inputs=text_input)

        with gr.Row():
            left_panel = gr.HTML()
            right_panel = gr.HTML()

        gr.Markdown("### recent comparisons")
        history_state = gr.State([])
        history_table = gr.Dataframe(
            headers=["time", "input", "model 1", "model 2"],
            value=[],
            interactive=False,
            wrap=True,
        )

        async def _submit_handler(text: str, history: list[list[str]]):
            left_html, right_html, new_history = await _compare(text, history, app)
            return left_html, right_html, new_history, new_history

        submit.click(
            fn=_submit_handler,
            inputs=[text_input, history_state],
            outputs=[left_panel, right_panel, history_state, history_table],
        )
        text_input.submit(
            fn=_submit_handler,
            inputs=[text_input, history_state],
            outputs=[left_panel, right_panel, history_state, history_table],
        )
    # Required for async handlers when Gradio is mounted into FastAPI —
    # without it, async .click() chains fire but never propagate updates
    # back to the UI components. default_concurrency_limit=4 caps how
    # many compare requests can run in parallel against the registry.
    demo.queue(default_concurrency_limit=4)
    return demo
