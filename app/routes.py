"""HTTP routes."""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request

from app.predictor import load_models, models_status, predict_glucose

bp = Blueprint("main", __name__)


@bp.before_app_request
def _ensure_models_loaded() -> None:
    loaded, _ = models_status()
    if not loaded:
        load_models()


@bp.get("/")
def home():
    loaded, error = models_status()
    versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
    }
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        versions["sklearn"] = "unknown"

    return render_template(
        "index.html",
        models_loaded=loaded,
        model_error=error,
        versions=versions,
    )


@bp.post("/predict")
def predict():
    loaded, error = models_status()
    if not loaded:
        message = error or "Models not loaded."
        if request.is_json:
            return jsonify({"error": message}), 500
        return render_template(
            "error.html",
            error_title="Model Loading Error",
            error_message=message,
        )

    try:
        person = {
            "age": float(request.form.get("age", 0)),
            "gender": request.form.get("gender", "Unknown"),
            "bmi": float(request.form.get("bmi", 0)),
            "a1c": float(request.form.get("a1c", 0)),
            "fasting_glucose": float(request.form.get("fasting_glucose", 0)),
            "insulin_level": float(request.form.get("insulin_level", 0)),
            "heart_rate": float(request.form.get("heart_rate", 70)),
        }
        meal = {
            "meal_type": request.form.get("meal_type", "Unknown"),
            "calories": float(request.form.get("calories", 0)),
            "carbs": float(request.form.get("carbs", 0)),
            "protein": float(request.form.get("protein", 0)),
            "fat": float(request.form.get("fat", 0)),
            "fiber": float(request.form.get("fiber", 0)),
        }
        try:
            current_glucose = float(request.form.get("current_glucose", 100))
            glucose_trend = float(request.form.get("glucose_trend", 0))
        except (TypeError, ValueError):
            current_glucose = 100.0
            glucose_trend = 0.0

        result = predict_glucose(person, meal, current_glucose, glucose_trend)
        return render_template("result.html", result=result)
    except Exception as exc:  # noqa: BLE001
        current_app.logger.exception("Prediction failed")
        details = traceback.format_exc()
        if request.is_json:
            return jsonify({"error": str(exc)}), 500
        return render_template(
            "error.html",
            error_title="Prediction Error",
            error_message=f"Error during prediction: {exc}",
            technical_details=details,
        )


@bp.get("/health")
def health():
    loaded, error = models_status()
    return jsonify({"status": "ok" if loaded else "degraded", "models_loaded": loaded, "error": error})
