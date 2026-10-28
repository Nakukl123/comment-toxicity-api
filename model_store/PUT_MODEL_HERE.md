The trained model is not stored in git.

- Train it with `training/train.py` (see the guide), which writes
  `toxicity_model.keras` and `model_meta.json` here.
- In production the server downloads both files from S3 into this folder
  before `docker build`.
