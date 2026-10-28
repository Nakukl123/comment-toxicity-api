"""All settings come from environment variables, so the same code runs on a laptop,
in Docker, or on a cloud host without edits (one of the 12-factor app principles)."""
import os

MODEL_DIR = os.getenv("MODEL_DIR", "model_store")
MODEL_PATH = os.path.join(MODEL_DIR, "toxicity_model.keras")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")

THRESHOLD = float(os.getenv("TOXICITY_THRESHOLD", "0.5"))
MAX_TEXT_CHARS = int(os.getenv("MAX_TEXT_CHARS", "5000"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "64"))

# Monitoring
PREDICTION_LOG = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
# Also print each prediction record to stdout, so Docker can ship it to CloudWatch / any log service
PREDICTION_LOG_STDOUT = os.getenv("PREDICTION_LOG_STDOUT", "false").lower() == "true"
DRIFT_WINDOW = int(os.getenv("DRIFT_WINDOW", "500"))          # last N predictions
DRIFT_ALERT_RATIO = float(os.getenv("DRIFT_ALERT_RATIO", "3"))  # live rate vs training rate
