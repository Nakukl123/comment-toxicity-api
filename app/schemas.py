"""Request/response contracts. Pydantic validates every request automatically,
so bad input is rejected with a clear 422 error before it ever reaches the model."""
from pydantic import BaseModel, Field, field_validator

from app.config import MAX_BATCH_SIZE, MAX_TEXT_CHARS


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS,
                      examples=["You are an idiot and nobody likes you"])

    @field_validator("text")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text must not be blank")
        return v


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)

    @field_validator("texts")
    @classmethod
    def check_items(cls, v: list[str]) -> list[str]:
        for t in v:
            if not t.strip():
                raise ValueError("texts must not contain blank strings")
            if len(t) > MAX_TEXT_CHARS:
                raise ValueError(f"each text must be at most {MAX_TEXT_CHARS} characters")
        return v


class Prediction(BaseModel):
    scores: dict[str, float]
    flagged_labels: list[str]
    is_toxic: bool


class PredictResponse(Prediction):
    model_version: str
    latency_ms: float


class BatchPredictResponse(BaseModel):
    results: list[Prediction]
    model_version: str
    latency_ms: float
