"""Lightweight production monitoring.

1. Every prediction is logged to a JSONL file (one JSON object per line), so it can
   later be loaded into pandas / a dashboard / a warehouse for analysis.
   Privacy: we log text LENGTH and scores, never the raw comment.
2. A rolling window of recent predictions is kept in memory and compared with the
   label rates seen in training. If, say, "threat" is suddenly flagged 3x more often
   than in training, the incoming data has probably changed (data drift) or the
   model is misbehaving -> /metrics reports a drift alert.
"""
import json
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone


class Monitor:
    def __init__(self, log_path: str, window: int, alert_ratio: float,
                 labels: list[str], baseline: dict[str, float], to_stdout: bool = False):
        self.log_path = log_path
        self.to_stdout = to_stdout
        self.alert_ratio = alert_ratio
        self.labels = labels
        self.baseline = baseline
        self.recent = deque(maxlen=window)        # flagged-label lists, newest last
        self.latencies = deque(maxlen=window)
        self.total_requests = 0
        self.total_errors = 0
        self.started_at = time.time()
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def record(self, texts: list[str], results: list[dict], latency_ms: float,
               model_version: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.total_requests += 1
            self.latencies.append(latency_ms)
            with open(self.log_path, "a") as f:
                for text, res in zip(texts, results):
                    self.recent.append(res["flagged_labels"])
                    line = json.dumps({
                        "event": "prediction",
                        "ts": now,
                        "model_version": model_version,
                        "text_chars": len(text),
                        "scores": res["scores"],
                        "is_toxic": res["is_toxic"],
                        "latency_ms": latency_ms,
                    })
                    f.write(line + "\n")
                    if self.to_stdout:
                        print(line, flush=True)

    def record_error(self) -> None:
        with self._lock:
            self.total_errors += 1
        if self.to_stdout:
            print(json.dumps({"event": "prediction_error",
                              "ts": datetime.now(timezone.utc).isoformat()}), flush=True)

    def snapshot(self) -> dict:
        with self._lock:
            n = len(self.recent)
            live_rate = {
                label: (sum(label in flags for flags in self.recent) / n if n else 0.0)
                for label in self.labels
            }
            lat = sorted(self.latencies)

        drift_alerts = []
        if n >= 50:   # don't raise alarms on a handful of requests
            for label in self.labels:
                base = self.baseline.get(label)
                if base and live_rate[label] > base * self.alert_ratio:
                    drift_alerts.append(
                        f"{label}: live rate {live_rate[label]:.3f} vs training {base:.3f}"
                    )

        def pct(p: float) -> float | None:
            return round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None

        return {
            "uptime_s": round(time.time() - self.started_at),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "window_predictions": n,
            "latency_ms": {"p50": pct(0.50), "p95": pct(0.95)},
            "live_positive_rate": {k: round(v, 4) for k, v in live_rate.items()},
            "training_positive_rate": self.baseline,
            "drift_alerts": drift_alerts,
        }
