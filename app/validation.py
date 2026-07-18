"""Server-side form validation for prediction inputs."""

from __future__ import annotations

from typing import Any


FIELD_RULES: dict[str, dict[str, Any]] = {
    "age": {"label": "Age", "min": 1, "max": 120, "step": 1},
    "bmi": {"label": "BMI", "min": 10, "max": 60, "step": 0.1},
    "a1c": {"label": "A1c (%)", "min": 3.5, "max": 15, "step": 0.1},
    "fasting_glucose": {"label": "Fasting glucose (mg/dL)", "min": 50, "max": 400, "step": 1},
    "insulin_level": {"label": "Insulin level", "min": 0, "max": 300, "step": 0.1},
    "heart_rate": {"label": "Heart rate (bpm)", "min": 40, "max": 200, "step": 1},
    "current_glucose": {"label": "Current glucose (mg/dL)", "min": 40, "max": 600, "step": 1},
    "glucose_trend": {"label": "Glucose trend (mg/dL per hour)", "min": -30, "max": 30, "step": 0.1},
    "calories": {"label": "Calories", "min": 0, "max": 3000, "step": 1},
    "carbs": {"label": "Carbohydrates (g)", "min": 0, "max": 500, "step": 0.1},
    "protein": {"label": "Protein (g)", "min": 0, "max": 300, "step": 0.1},
    "fat": {"label": "Fat (g)", "min": 0, "max": 300, "step": 0.1},
    "fiber": {"label": "Fiber (g)", "min": 0, "max": 100, "step": 0.1},
}

ALLOWED_GENDERS = {"Male", "Female", "Other"}
ALLOWED_MEAL_TYPES = {"Breakfast", "Lunch", "Dinner", "Snack"}


def _parse_number(raw: str | None, field: str) -> tuple[float | None, str | None]:
    rule = FIELD_RULES[field]
    label = rule["label"]
    if raw is None or str(raw).strip() == "":
        return None, f"{label} is required."
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return None, f"{label} must be a number."
    if value != value:  # NaN
        return None, f"{label} must be a number."
    if value < rule["min"] or value > rule["max"]:
        return None, f"{label} must be between {rule['min']} and {rule['max']}."
    return value, None


def parse_prediction_form(form) -> tuple[dict[str, Any] | None, dict[str, str], dict[str, Any]]:
    """
    Validate form data.

    Returns (payload, errors, values).
    payload is None when validation fails.
    values always contains echoed fields for re-rendering the form.
    """
    errors: dict[str, str] = {}
    values: dict[str, Any] = {}

    gender = (form.get("gender") or "").strip()
    values["gender"] = gender
    if gender not in ALLOWED_GENDERS:
        errors["gender"] = "Select a valid gender."

    meal_type = (form.get("meal_type") or "").strip()
    values["meal_type"] = meal_type
    if meal_type not in ALLOWED_MEAL_TYPES:
        errors["meal_type"] = "Select a valid meal type."

    numbers: dict[str, float] = {}
    for field in FIELD_RULES:
        raw = form.get(field)
        values[field] = "" if raw is None else str(raw)
        value, error = _parse_number(raw, field)
        if error:
            errors[field] = error
        elif value is not None:
            numbers[field] = value
            values[field] = value

    if "fiber" in numbers and "carbs" in numbers and numbers["fiber"] > numbers["carbs"]:
        errors["fiber"] = "Fiber cannot be greater than carbohydrates."

    if all(k in numbers for k in ("calories", "carbs", "protein", "fat")):
        estimated = 4 * numbers["carbs"] + 4 * numbers["protein"] + 9 * numbers["fat"]
        # Allow slack for rounding / alcohol / labeling variance.
        if numbers["calories"] > 0 and estimated > numbers["calories"] * 1.6 + 50:
            errors["calories"] = (
                "Calories look too low for the entered carbs/protein/fat. "
                "Check the nutrition values."
            )
        if numbers["calories"] >= 50 and estimated < numbers["calories"] * 0.35:
            errors["calories"] = (
                "Calories look high for the entered macros. "
                "Check carbs, protein, fat, or calories."
            )

    if "current_glucose" in numbers and "fasting_glucose" in numbers:
        if abs(numbers["current_glucose"] - numbers["fasting_glucose"]) > 350:
            errors["current_glucose"] = (
                "Current glucose and fasting glucose differ by an unusually large amount. "
                "Please verify both readings."
            )

    if errors:
        return None, errors, values

    payload = {
        "person": {
            "age": numbers["age"],
            "gender": gender,
            "bmi": numbers["bmi"],
            "a1c": numbers["a1c"],
            "fasting_glucose": numbers["fasting_glucose"],
            "insulin_level": numbers["insulin_level"],
            "heart_rate": numbers["heart_rate"],
        },
        "meal": {
            "meal_type": meal_type,
            "calories": numbers["calories"],
            "carbs": numbers["carbs"],
            "protein": numbers["protein"],
            "fat": numbers["fat"],
            "fiber": numbers["fiber"],
        },
        "current_glucose": numbers["current_glucose"],
        "glucose_trend": numbers["glucose_trend"],
    }
    return payload, {}, values
