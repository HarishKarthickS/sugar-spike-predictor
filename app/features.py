"""Feature engineering shared by inference (and mirrored in training)."""

from __future__ import annotations

from datetime import datetime
from typing import Any


def time_category(hour: int) -> str:
    if 6 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 24:
        return "evening"
    return "early_morning"


def build_feature_row(
    person: dict[str, Any],
    meal: dict[str, Any],
    current_glucose: float,
    glucose_trend: float,
    hour: int | None = None,
) -> dict[str, Any]:
    """Build the feature dict expected by the specialized sklearn pipelines."""
    hour_of_day = datetime.now().hour if hour is None else hour

    features: dict[str, Any] = {
        "age": float(person.get("age", 0)),
        "gender": person.get("gender", "Unknown"),
        "bmi": float(person.get("bmi", 0)),
        "a1c": float(person.get("a1c", 0)),
        "fasting_glucose": float(person.get("fasting_glucose", 0)),
        "insulin_level": float(person.get("insulin_level", 0)),
        "meal_type": meal.get("meal_type", "Unknown"),
        "meal_calories": float(meal.get("calories", 0)),
        "meal_carbs": float(meal.get("carbs", 0)),
        "meal_protein": float(meal.get("protein", 0)),
        "meal_fat": float(meal.get("fat", 0)),
        "meal_fiber": float(meal.get("fiber", 0)),
        "hour_of_day": hour_of_day,
        "glucose_at_meal": float(current_glucose),
        "glucose_mean_1h": float(current_glucose),
        "glucose_std_1h": 5.0,
        "glucose_min_1h": float(current_glucose) - 5.0,
        "glucose_max_1h": float(current_glucose) + 5.0,
        "glucose_slope_30m": float(glucose_trend),
        "hr_at_meal": float(person.get("heart_rate", 70)),
        "hr_mean_1h": float(person.get("heart_rate", 70)),
    }

    carbs = features["meal_carbs"]
    fiber = features["meal_fiber"]
    fat = features["meal_fat"]
    protein = features["meal_protein"]
    insulin = features["insulin_level"]
    calories = features["meal_calories"]

    features.update(
        {
            "glycemic_load": carbs * (1 - fiber / (carbs + 1)),
            "carb_insulin_ratio": carbs / (insulin + 1),
            "fat_protein_to_carb": (fat + protein) / (carbs + 1),
            "bmi_insulin": features["bmi"] * insulin,
            "time_category": time_category(hour_of_day),
            "glucose_variability": 5 / (features["glucose_at_meal"] + 1),
            "meal_energy_density": calories / (carbs + protein + fat + 1),
            "glucose_momentum": features["glucose_slope_30m"] * features["glucose_at_meal"],
            "age_a1c": features["age"] * features["a1c"],
            "insulin_sensitivity_factor": 1800 / (insulin + 45),
            "stress_factor": 1.0,
        }
    )
    return features


def adjust_prediction(raw: float, features: dict[str, Any]) -> float:
    """Light post-model adjustment for extreme meal compositions."""
    prediction = float(raw)
    carb_to_fiber = features["meal_carbs"] / (features["meal_fiber"] + 1)
    if carb_to_fiber > 10 and features["meal_carbs"] > 60:
        prediction *= 1.1
    elif features["fat_protein_to_carb"] > 2 and features["meal_carbs"] > 30:
        prediction *= 0.95
    return prediction


def estimate_accuracy(base_rmse: float, features: dict[str, Any], diabetic: bool) -> float:
    accuracy_base = max(90, 100 - (base_rmse / 1.2)) if diabetic else max(92, 100 - (base_rmse / 1.0))
    carb_to_fiber = features["meal_carbs"] / (features["meal_fiber"] + 1)
    complexity = 1 + (0.1 * (carb_to_fiber / 10)) - (0.05 * features["fat_protein_to_carb"])
    complexity = max(0.9, min(1.2, complexity))
    return min(99, max(85, accuracy_base / complexity))
