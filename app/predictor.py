"""Model loading and prediction orchestration."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from flask import current_app

from app.features import adjust_prediction, build_feature_row, estimate_accuracy
from app.recommendations import get_recommendation

logger = logging.getLogger(__name__)

_diabetic_model = None
_non_diabetic_model = None
_models_loaded = False
_model_error: str | None = None


def models_status() -> tuple[bool, str | None]:
    return _models_loaded, _model_error


def load_models() -> None:
    global _diabetic_model, _non_diabetic_model, _models_loaded, _model_error

    diabetic_path = Path(current_app.config["DIABETIC_MODEL_PATH"])
    non_diabetic_path = Path(current_app.config["NON_DIABETIC_MODEL_PATH"])

    if not diabetic_path.exists() or not non_diabetic_path.exists():
        missing = [str(p) for p in (diabetic_path, non_diabetic_path) if not p.exists()]
        _models_loaded = False
        _model_error = f"Model files not found: {', '.join(missing)}. Train models or place pickles in artifacts/."
        logger.warning(_model_error)
        return

    try:
        _diabetic_model = joblib.load(diabetic_path)
        _non_diabetic_model = joblib.load(non_diabetic_path)
        _models_loaded = True
        _model_error = None
        logger.info("Specialized models loaded with joblib")
    except Exception as joblib_error:  # noqa: BLE001
        try:
            with diabetic_path.open("rb") as left, non_diabetic_path.open("rb") as right:
                _diabetic_model = pickle.load(left)
                _non_diabetic_model = pickle.load(right)
            _models_loaded = True
            _model_error = None
            logger.info("Specialized models loaded with pickle")
        except Exception as pickle_error:  # noqa: BLE001
            _models_loaded = False
            _model_error = f"Could not load models: {joblib_error}; {pickle_error}"
            logger.error(_model_error)


def predict_glucose(
    person: dict[str, Any],
    meal: dict[str, Any],
    current_glucose: float,
    glucose_trend: float,
) -> dict[str, Any]:
    if not _models_loaded:
        raise RuntimeError(_model_error or "Models are not loaded.")

    threshold = float(current_app.config["A1C_DIABETIC_THRESHOLD"])
    is_diabetic = float(person.get("a1c", 0)) >= threshold
    features = build_feature_row(person, meal, current_glucose, glucose_trend)
    frame = pd.DataFrame([features])

    if is_diabetic:
        model = _diabetic_model
        base_rmse = float(current_app.config["DIABETIC_RMSE"])
    else:
        model = _non_diabetic_model
        base_rmse = float(current_app.config["NON_DIABETIC_RMSE"])

    raw = float(model.predict(frame)[0])
    prediction = adjust_prediction(raw, features)
    accuracy = estimate_accuracy(base_rmse, features, is_diabetic)

    return {
        "prediction": round(prediction, 1),
        "current_glucose": round(current_glucose, 1),
        "is_diabetic": "Yes" if is_diabetic else "No",
        "meal_type": meal.get("meal_type", "Unknown"),
        "carbs": meal.get("carbs", 0),
        "accuracy": round(accuracy),
        "message": get_recommendation(prediction, is_diabetic, meal),
        "expected_range_low": round(prediction - base_rmse, 1),
        "expected_range_high": round(prediction + base_rmse, 1),
    }
