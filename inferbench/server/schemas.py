"""Pydantic schemas for the /infer endpoint."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class InferRequest(BaseModel):
    inputs: list[str] = Field(..., min_length=1, description="One or more text inputs to classify.")
    request_id: Optional[str] = None


class PredictionItem(BaseModel):
    label: str
    score: float


class InferResponse(BaseModel):
    request_id: str
    predictions: list[PredictionItem]
    latency_ms: float
    queue_wait_ms: float
    inference_ms: float
    batch_size: int
    cache_hit: bool
    model_backend: str
    model_precision: str


class HealthResponse(BaseModel):
    status: str
    model_id: str
    backend: str
    precision: str
