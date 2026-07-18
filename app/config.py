"""Application configuration."""

from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"


class Config:
    SECRET_KEY = "sugar-spike-predictor-dev"
    DIABETIC_MODEL_PATH = ARTIFACTS_DIR / "glucose_prediction_model_diabetic.pkl"
    NON_DIABETIC_MODEL_PATH = ARTIFACTS_DIR / "glucose_prediction_model_non_diabetic.pkl"
    # Reported RMSE from specialized training runs (mg/dL).
    DIABETIC_RMSE = 28.5
    NON_DIABETIC_RMSE = 18.7
    A1C_DIABETIC_THRESHOLD = 6.5
