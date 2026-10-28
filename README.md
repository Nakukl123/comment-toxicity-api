---
title: Comment Toxicity API
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# Comment Toxicity API

A multi-label toxicity classifier (Bidirectional LSTM, TensorFlow/Keras) trained on the
Jigsaw Toxic Comment dataset and **served as a containerised REST API** with input
validation, automated tests, CI, prediction logging and drift monitoring.

It scores a comment on six labels: `toxic`, `severe_toxic`, `obscene`, `threat`,
`insult`, `identity_hate`.

> Started as a notebook + Streamlit capstone; this repo is the production version of it.

## Architecture

```
client ──HTTP/JSON──▶ FastAPI (app/main.py)
                         │  Pydantic validation (app/schemas.py) → 422 on bad input
                         ▼
                      Predictor (app/model.py)
                         │  model loaded once at startup, inference compiled with tf.function
                         │  text → Keras Tokenizer → pad to 200 → Embedding → LSTM(64) → 6 sigmoids
                         ▼
                      Monitor (app/monitoring.py)
                            JSONL prediction log (no raw text stored)
                            rolling label rates vs training baseline → drift alerts
```

The model and its `tokenizer.json` are versioned together in S3 (`v1/`), so the exact
tokenizer a model was trained with is always deployed with it.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check + model version |
| POST | `/predict` | Score one comment |
| POST | `/predict/batch` | Score up to 64 comments |
| GET | `/metrics` | Requests, errors, p50/p95 latency, live label rates, drift alerts |
| GET | `/docs` | Interactive Swagger UI (auto-generated) |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "you are an idiot"}'
```
```json
{
  "scores": {"toxic": 0.96, "severe_toxic": 0.01, "obscene": 0.53, "threat": 0.20,
             "insult": 0.99, "identity_hate": 0.00},
  "flagged_labels": ["toxic", "obscene", "insult"],
  "is_toxic": true,
  "model_version": "v20261027-0736",
  "latency_ms": 12.4
}
```

## Model results

Trained on the Jigsaw toxic-comment data (multi-label, 6 classes). Fill in from your notebook:

| Metric | Score |
|---|---|
| Weighted F1 | _from notebook_ |
| Precision / Recall | _from notebook_ |

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put toxicity_model.keras, tokenizer.json, model_meta.json in model_store/
uvicorn app.main:app --reload        # open http://localhost:8000/docs
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

Tests use a fake predictor, so they run in under a second without TensorFlow. They
cover the happy path, input validation, batch limits, error handling, privacy of the
prediction log and the drift alert.

## Docker

```bash
docker build -t toxicity-api .
docker run -p 7860:7860 toxicity-api      # http://localhost:7860/docs
```

The image runs as a non-root user and has a built-in `HEALTHCHECK` against `/health`.

## CI

`.github/workflows/ci.yml` runs on every push and pull request: install → `pytest` →
`docker build`. A broken change shows a red ✗ on the commit before it can be deployed.

## Configuration

All settings are environment variables (see `app/config.py`), so the same image runs
anywhere:

| Variable | Default | Meaning |
|---|---|---|
| `TOXICITY_THRESHOLD` | `0.5` | Score at which a label is flagged |
| `MAX_TEXT_CHARS` | `5000` | Longest accepted comment |
| `MAX_BATCH_SIZE` | `64` | Most comments per batch call |
| `DRIFT_WINDOW` | `500` | Recent predictions used for drift checks |
| `DRIFT_ALERT_RATIO` | `3` | Alert when a live label rate exceeds training rate × this |

## Engineering notes

- **Inference latency:** calling the LSTM eagerly took ~700 ms per request. Wrapping
  inference in a `tf.function` with a fixed input signature cut it to ~10–15 ms.
- **Load once:** the model is loaded and warmed up at startup, not per request.
- **Privacy:** the prediction log stores text length and scores, never the comment itself.
- **Drift:** live flag rates per label are compared with the training baseline saved in
  `model_meta.json`; a sustained jump suggests the incoming data has shifted or the model
  needs retraining.

## Next steps

- Replace the LSTM with a fine-tuned DistilBERT and compare AUC vs latency
- Persist metrics to Prometheus/Grafana instead of in-memory counters
- Add a feedback endpoint so moderators' corrections become retraining data
