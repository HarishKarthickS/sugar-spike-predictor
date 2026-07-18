"""HTTP routes."""

from __future__ import annotations

import sys
import traceback

import numpy as np
import pandas as pd
from flask import Blueprint, current_app, jsonify, render_template, request

from app.predictor import load_models, models_status, predict_glucose
from app.validation import parse_prediction_form

bp = Blueprint("main", __name__)


def _home_context(**extra):
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

    context = {
        "models_loaded": loaded,
        "model_error": error,
        "versions": versions,
        "errors": {},
        "values": {},
        "form_error": None,
    }
    context.update(extra)
    return context


@bp.before_app_request
def _ensure_models_loaded() -> None:
    loaded, _ = models_status()
    if not loaded:
        load_models()


@bp.get("/")
def home():
    return render_template("index.html", **_home_context())


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

    payload, errors, values = parse_prediction_form(request.form)
    if errors:
        if request.is_json:
            return jsonify({"error": "Validation failed", "fields": errors}), 400
        return (
            render_template(
                "index.html",
                **_home_context(
                    errors=errors,
                    values=values,
                    form_error="Please fix the highlighted fields and try again.",
                ),
            ),
            400,
        )

    try:
        assert payload is not None
        result = predict_glucose(
            payload["person"],
            payload["meal"],
            payload["current_glucose"],
            payload["glucose_trend"],
        )
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
