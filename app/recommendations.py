"""Meal recommendations based on predicted glucose."""

from __future__ import annotations

from typing import Any


def get_recommendation(predicted_glucose: float, is_diabetic: bool, meal: dict[str, Any]) -> str:
    carbs = float(meal.get("carbs", 0))
    fiber = float(meal.get("fiber", 0))
    protein = float(meal.get("protein", 0))
    fat = float(meal.get("fat", 0))

    carb_quality = (
        "low"
        if fiber == 0 or (carbs / (fiber + 0.001)) > 10
        else "moderate"
        if (carbs / (fiber + 0.001)) > 5
        else "high"
    )
    fat_protein_balance = (fat + protein) / (carbs + 1)

    if is_diabetic:
        if predicted_glucose > 180:
            base = "Your predicted glucose is higher than recommended."
            if carb_quality == "low" and carbs > 50:
                return (
                    f"{base} Consider reducing carbohydrates (currently {carbs}g) "
                    "or replacing them with higher-fiber options. Adding protein and fat can slow absorption."
                )
            if fat_protein_balance < 0.5:
                return (
                    f"{base} Try increasing protein and healthy fats relative to carbs. "
                    "Discuss insulin adjustment with your healthcare provider."
                )
            return (
                f"{base} This meal may require insulin adjustment. "
                "Moderate activity after eating can help lower glucose."
            )
        if predicted_glucose > 140:
            base = "Your predicted glucose is somewhat elevated."
            if carb_quality == "low":
                return (
                    f"{base} Add more fiber (currently {fiber}g). "
                    "Whole grains, vegetables, and legumes help."
                )
            return (
                f"{base} A 15–20 minute walk after eating can help. "
                f"Your fiber intake ({fiber}g) is already moderating the response."
            )
        return (
            "Your predicted glucose is in a good post-meal range for a diabetic individual. "
            f"The protein+fat to carb ratio ({fat_protein_balance:.1f}) looks helpful."
        )

    if predicted_glucose > 140:
        base = "Your predicted glucose is higher than typical for a non-diabetic person."
        if carb_quality == "low":
            return (
                f"{base} Reduce simple carbohydrates (currently {carbs}g) "
                f"or add more fiber (currently {fiber}g)."
            )
        return f"{base} Balance carbs with more protein and healthy fats, or reduce portion size."
    if predicted_glucose > 120:
        return (
            "Your predicted glucose is slightly elevated. "
            f"Current protein+fat to carbs ratio is {fat_protein_balance:.1f}; "
            "aim for at least 0.8 for better control."
        )
    return (
        "Your predicted glucose is within a normal post-meal range. "
        f"This meal ({carbs}g carbs, {protein}g protein, {fat}g fat) looks well balanced."
    )
