"""Loads the trained model ONCE at startup and serves predictions.

Loading per request would add seconds of latency to every call; loading once
and keeping the model in memory is the standard serving pattern."""
import json
import logging
from typing import Protocol

import numpy as np

from app.config import META_PATH, MODEL_PATH, THRESHOLD, TOKENIZER_PATH

log = logging.getLogger("toxicity.model")


class Predictor(Protocol):
    labels: list[str]
    version: str
    baseline: dict[str, float]

    def predict(self, texts: list[str]) -> list[dict]: ...


def to_result(scores: np.ndarray, labels: list[str], threshold: float) -> dict:
    score_map = {label: round(float(s), 4) for label, s in zip(labels, scores)}
    flagged = [label for label, s in score_map.items() if s >= threshold]
    return {"scores": score_map, "flagged_labels": flagged, "is_toxic": bool(flagged)}


class KerasToxicityPredictor:
    """LSTM model + Keras Tokenizer (the setup used in the original capstone).

    text -> tokenizer -> integer sequence -> pad to the model's input length -> LSTM -> 6 scores
    """

    def __init__(self, model_path: str = MODEL_PATH, meta_path: str = META_PATH,
                 tokenizer_path: str = TOKENIZER_PATH, threshold: float = THRESHOLD):
        import tensorflow as tf  # imported here so tests can run without TensorFlow
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.preprocessing.text import tokenizer_from_json

        log.info("Loading model from %s", model_path)
        self.model = tf.keras.models.load_model(model_path)
        self.maxlen = int(self.model.input_shape[1])   # 200 for this model
        with open(tokenizer_path) as f:
            self.tokenizer = tokenizer_from_json(f.read())
        with open(meta_path) as f:
            meta = json.load(f)
        self.labels = meta["labels"]
        self.version = meta["model_version"]
        self.baseline = meta.get("baseline_positive_rate", {})
        self.threshold = threshold
        self._pad = pad_sequences
        # Compile inference into a TF graph: an LSTM run eagerly step-by-step is ~50x slower.
        self._infer = tf.function(
            lambda x: self.model(x, training=False),
            input_signature=[tf.TensorSpec(shape=[None, self.maxlen], dtype=tf.int32)],
        )
        self.predict(["warm-up"])   # first call builds the graph; do it before real traffic
        log.info("Model %s loaded and warmed up (maxlen=%d, labels=%s)",
                 self.version, self.maxlen, self.labels)

    def predict(self, texts: list[str]) -> list[dict]:
        seqs = self.tokenizer.texts_to_sequences(texts)
        padded = self._pad(seqs, maxlen=self.maxlen).astype("int32")
        scores = self._infer(padded).numpy()
        return [to_result(row, self.labels, self.threshold) for row in scores]
