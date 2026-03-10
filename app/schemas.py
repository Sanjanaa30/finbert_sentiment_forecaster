from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    headline: str | list[str]


class ScoreItem(BaseModel):
    headline: str
    label: Literal["negative", "neutral", "positive"]
    score: float = Field(description="P(positive) - P(negative)")
    probabilities: dict[str, float] = Field(
        description="Class probabilities: negative/neutral/positive"
    )


class ScoreResponse(BaseModel):
    results: list[ScoreItem]
    timestamp: datetime
    model_version: str


class ScoreSingleResponse(BaseModel):
    result: ScoreItem
    timestamp: datetime
    model_version: str


class ForecastRequest(BaseModel):
    # Option A: send precomputed feature values directly
    features: dict[str, float]
    # Option B later (optional): add date field and compute features inside API
    # date: str | None = None


class ForecastResponse(BaseModel):
    forecast_class: Literal["down", "up"]
    probabilities: dict[str, float] = Field(description="down/up probabilities")
    timestamp: datetime
    model_version: str
