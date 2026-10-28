"""FastAPI service that serves the toxicity model.

Run locally:   uvicorn app.main:app --reload
Docs (auto):   http://localhost:8000/docs
"""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request

from app import config
from app.model import KerasToxicityPredictor, Predictor
from app.monitoring import Monitor
from app.schemas import (BatchPredictRequest, BatchPredictResponse, PredictRequest,
                         PredictResponse)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("toxicity.api")


def attach(app: FastAPI, predictor: Predictor) -> None:
    """Wire a predictor + monitor onto the app. Tests call this with a fake predictor."""
    app.state.predictor = predictor
    app.state.monitor = Monitor(
        log_path=config.PREDICTION_LOG,
        window=config.DRIFT_WINDOW,
        alert_ratio=config.DRIFT_ALERT_RATIO,
        labels=predictor.labels,
        baseline=predictor.baseline,
        to_stdout=config.PREDICTION_LOG_STDOUT,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model once, when the server starts (not on every request).
    if not getattr(app.state, "predictor", None):
        attach(app, KerasToxicityPredictor())
    yield


app = FastAPI(
    title="Comment Toxicity API",
    description="Multi-label toxicity classifier (BiLSTM) served as a REST API.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_predictor(request: Request) -> Predictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return predictor


def get_monitor(request: Request) -> Monitor:
    return request.app.state.monitor


@app.get("/health")
def health(predictor: Predictor = Depends(get_predictor)):
    """Liveness check. Hosting platforms and load balancers poll this."""
    return {"status": "ok", "model_version": predictor.version}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest,
            predictor: Predictor = Depends(get_predictor),
            monitor: Monitor = Depends(get_monitor)):
    start = time.perf_counter()
    try:
        result = predictor.predict([req.text])[0]
    except Exception:
        monitor.record_error()
        log.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    latency = round((time.perf_counter() - start) * 1000, 2)
    monitor.record([req.text], [result], latency, predictor.version)
    return {**result, "model_version": predictor.version, "latency_ms": latency}


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest,
                  predictor: Predictor = Depends(get_predictor),
                  monitor: Monitor = Depends(get_monitor)):
    start = time.perf_counter()
    try:
        results = predictor.predict(req.texts)
    except Exception:
        monitor.record_error()
        log.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
    latency = round((time.perf_counter() - start) * 1000, 2)
    monitor.record(req.texts, results, latency, predictor.version)
    return {"results": results, "model_version": predictor.version, "latency_ms": latency}


@app.get("/metrics")
def metrics(predictor: Predictor = Depends(get_predictor),
            monitor: Monitor = Depends(get_monitor)):
    """Operational + model-health numbers: traffic, errors, latency, drift alerts."""
    return {"model_version": predictor.version, **monitor.snapshot()}
