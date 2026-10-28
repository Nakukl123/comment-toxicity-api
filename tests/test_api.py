"""API tests. They use a FAKE predictor, so they run in seconds and don't need
TensorFlow or the trained model file: CI tests the service logic, not the weights."""
import pytest
from fastapi.testclient import TestClient

from app import config
from app import main
from app.model import to_result

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


class FakePredictor:
    labels = LABELS
    version = "test-v0"
    baseline = {label: 0.05 for label in LABELS}

    def predict(self, texts):
        # "idiot" anywhere -> toxic + insult; everything else clean.
        out = []
        for t in texts:
            hot = "idiot" in t.lower()
            scores = [0.9 if hot and l in ("toxic", "insult") else 0.01 for l in LABELS]
            out.append(to_result(scores, LABELS, 0.5))
        return out


@pytest.fixture
def client(tmp_path):
    config.PREDICTION_LOG = str(tmp_path / "predictions.jsonl")
    main.attach(main.app, FakePredictor())
    with TestClient(main.app) as c:
        yield c
    main.app.state.predictor = None


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": "test-v0"}


def test_predict_toxic(client):
    r = client.post("/predict", json={"text": "you are an idiot"})
    assert r.status_code == 200
    body = r.json()
    assert body["is_toxic"] is True
    assert set(body["flagged_labels"]) == {"toxic", "insult"}
    assert set(body["scores"]) == set(LABELS)
    assert body["model_version"] == "test-v0"


def test_predict_clean(client):
    r = client.post("/predict", json={"text": "thanks for the helpful edit"})
    assert r.status_code == 200
    assert r.json()["is_toxic"] is False
    assert r.json()["flagged_labels"] == []


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "x" * 6000}])
def test_predict_rejects_bad_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json={"texts": ["idiot", "nice work"]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert [x["is_toxic"] for x in results] == [True, False]


def test_batch_rejects_too_many(client):
    texts = ["hello"] * (config.MAX_BATCH_SIZE + 1)
    assert client.post("/predict/batch", json={"texts": texts}).status_code == 422


def test_predictions_are_logged_without_raw_text(client):
    client.post("/predict", json={"text": "secret comment idiot"})
    log_text = open(config.PREDICTION_LOG).read()
    assert "secret comment" not in log_text       # privacy: raw text never stored
    assert '"text_chars": 20' in log_text


def test_stdout_logging_for_cloud_log_services(tmp_path, capsys):
    config.PREDICTION_LOG = str(tmp_path / "p.jsonl")
    config.PREDICTION_LOG_STDOUT = True
    try:
        main.attach(main.app, FakePredictor())
        with TestClient(main.app) as c:
            c.post("/predict", json={"text": "private words idiot"})
        out = capsys.readouterr().out
        assert '"event": "prediction"' in out
        assert "private words" not in out
    finally:
        config.PREDICTION_LOG_STDOUT = False
        main.app.state.predictor = None


def test_metrics_counts_and_drift_alert(client):
    # 60 toxic predictions >> 5% training baseline -> drift alert expected
    for _ in range(60):
        client.post("/predict", json={"text": "idiot"})
    m = client.get("/metrics").json()
    assert m["total_requests"] == 60
    assert m["live_positive_rate"]["toxic"] == 1.0
    assert any(a.startswith("toxic") for a in m["drift_alerts"])
    assert m["latency_ms"]["p50"] is not None


def test_model_error_returns_500(client):
    class Broken(FakePredictor):
        def predict(self, texts):
            raise RuntimeError("boom")

    main.attach(main.app, Broken())
    r = client.post("/predict", json={"text": "hello"})
    assert r.status_code == 500
    assert client.get("/metrics").json()["total_errors"] == 1
