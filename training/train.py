"""
Train the comment-toxicity model and save it as ONE end-to-end file.

The text-vectorisation layer is saved INSIDE the model, so the deployed API
can send raw strings straight in. No separate tokenizer/pickle to keep in sync.

Usage:
    python training/train.py --data data/train.csv --epochs 3
    python training/train.py --data data/train.csv --sample 20000 --epochs 1   # quick run

Expects a text column (comment_text / comment / text) plus any of the 6 Jigsaw label
columns. 5-label versions of the dataset work too.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMNS = ["comment_text", "comment", "text"]
MAX_TOKENS = 50_000      # vocabulary size (kept modest so the model file stays small)
SEQ_LEN = 200            # words per comment after padding/truncation
EMBED_DIM = 64


def build_model(train_texts: np.ndarray, n_labels: int) -> tf.keras.Model:
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN, name="vectorizer"
    )
    vectorizer.adapt(train_texts)

    inputs = tf.keras.Input(shape=(1,), dtype=tf.string, name="text")
    x = vectorizer(inputs)
    x = tf.keras.layers.Embedding(MAX_TOKENS, EMBED_DIM, name="embedding")(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32), name="bilstm")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_labels, activation="sigmoid", name="labels")(x)

    model = tf.keras.Model(inputs, outputs, name="toxicity_bilstm")
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True, name="auc")],
    )
    return model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to Jigsaw train.csv")
    p.add_argument("--out", default="model_store", help="Folder to save the model in")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--sample", type=int, default=0, help="Use only N rows (0 = all)")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Accept common names for the text column
    text_col = next((c for c in TEXT_COLUMNS if c in df.columns), None)
    if text_col is None:
        raise SystemExit(f"No text column found. Expected one of {TEXT_COLUMNS}; got {list(df.columns)}")
    # Use whichever of the known labels the CSV actually has (5 or 6 both work)
    labels = [l for l in LABELS if l in df.columns]
    if not labels:
        raise SystemExit(f"No label columns found. Expected some of {LABELS}; got {list(df.columns)}")
    print(f"Text column: {text_col} | Labels: {labels}")

    df = df.dropna(subset=[text_col])
    if args.sample:
        df = df.sample(n=min(args.sample, len(df)), random_state=42)

    X = df[text_col].astype(str).to_numpy()
    y = df[labels].clip(0, 1).to_numpy(dtype="float32")
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(X_tr, len(labels))
    # Keras can't take numpy "object" arrays of Python strings; convert to tf.string tensors.
    X_tr_t = tf.constant(X_tr.reshape(-1, 1))
    X_val_t = tf.constant(X_val.reshape(-1, 1))
    model.fit(
        X_tr_t, y_tr,
        validation_data=(X_val_t, y_val),
        epochs=args.epochs, batch_size=args.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True)],
    )

    # ---- Evaluate per label (ROC-AUC is the standard metric for this dataset) ----
    val_pred = model.predict(X_val_t, verbose=0)
    per_label_auc = {}
    for i, label in enumerate(labels):
        if len(np.unique(y_val[:, i])) == 2:          # AUC needs both classes present
            per_label_auc[label] = round(float(roc_auc_score(y_val[:, i], val_pred[:, i])), 4)
        else:
            per_label_auc[label] = None

    os.makedirs(args.out, exist_ok=True)
    model_path = os.path.join(args.out, "toxicity_model.keras")
    model.save(model_path)

    # Metadata travels with the model: the API reports the version, and the
    # monitoring layer compares live prediction rates against this training baseline.
    version = datetime.now(timezone.utc).strftime("v%Y%m%d-%H%M")
    meta = {
        "model_version": version,
        "labels": labels,
        "trained_on_rows": int(len(X_tr)),
        "validation_rows": int(len(X_val)),
        "val_roc_auc": per_label_auc,
        "baseline_positive_rate": {
            label: round(float(y_tr[:, i].mean()), 5) for i, label in enumerate(labels)
        },
    }
    with open(os.path.join(args.out, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved {model_path}  ({version})")
    print(json.dumps(per_label_auc, indent=2))


if __name__ == "__main__":
    main()
