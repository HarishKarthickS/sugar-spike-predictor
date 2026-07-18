from app.features import adjust_prediction, build_feature_row, estimate_accuracy, time_category


def test_time_category_buckets():
    assert time_category(7) == "morning"
    assert time_category(12) == "midday"
    assert time_category(15) == "afternoon"
    assert time_category(20) == "evening"
    assert time_category(2) == "early_morning"


def test_time_category_boundaries():
    assert time_category(6) == "morning"
    assert time_category(11) == "midday"
    assert time_category(14) == "afternoon"
    assert time_category(18) == "evening"
    assert time_category(0) == "early_morning"
    assert time_category(5) == "early_morning"


def test_build_feature_row_includes_engineered_fields():
    person = {
        "age": 45,
        "gender": "F",
        "bmi": 28,
        "a1c": 6.7,
        "fasting_glucose": 130,
        "insulin_level": 15,
        "heart_rate": 75,
    }
    meal = {
        "meal_type": "Lunch",
        "calories": 600,
        "carbs": 60,
        "protein": 25,
        "fat": 20,
        "fiber": 8,
    }
    row = build_feature_row(person, meal, current_glucose=120, glucose_trend=0.2, hour=12)
    assert row["time_category"] == "midday"
    assert "glycemic_load" in row
    assert "carb_insulin_ratio" in row
    assert row["meal_carbs"] == 60
    assert row["glucose_momentum"] == 0.2 * 120
    assert row["age_a1c"] == 45 * 6.7
    assert row["bmi_insulin"] == 28 * 15


def test_glycemic_load_formula():
    row = build_feature_row(
        {"age": 30, "bmi": 22, "a1c": 5.2, "insulin_level": 10, "heart_rate": 70},
        {"calories": 400, "carbs": 50, "protein": 20, "fat": 10, "fiber": 10, "meal_type": "Snack"},
        100,
        0,
        hour=10,
    )
    expected = 50 * (1 - 10 / (50 + 1))
    assert abs(row["glycemic_load"] - expected) < 1e-9


def test_adjust_and_accuracy():
    features = build_feature_row(
        {"age": 40, "bmi": 24, "a1c": 5.2, "insulin_level": 10, "heart_rate": 70},
        {"calories": 500, "carbs": 80, "protein": 10, "fat": 10, "fiber": 2, "meal_type": "Dinner"},
        100,
        0.1,
        hour=19,
    )
    bumped = adjust_prediction(140, features)
    assert bumped >= 140
    score = estimate_accuracy(18.7, features, diabetic=False)
    assert 85 <= score <= 99


def test_adjust_prediction_high_fat_protein_lowers():
    features = build_feature_row(
        {"age": 40, "bmi": 24, "a1c": 5.2, "insulin_level": 10, "heart_rate": 70},
        {"calories": 700, "carbs": 40, "protein": 50, "fat": 50, "fiber": 12, "meal_type": "Dinner"},
        100,
        0,
        hour=19,
    )
    adjusted = adjust_prediction(160, features)
    assert adjusted < 160


def test_adjust_prediction_unchanged_for_balanced_meal():
    features = build_feature_row(
        {"age": 40, "bmi": 24, "a1c": 5.2, "insulin_level": 10, "heart_rate": 70},
        {"calories": 400, "carbs": 40, "protein": 20, "fat": 15, "fiber": 8, "meal_type": "Lunch"},
        100,
        0,
        hour=12,
    )
    assert adjust_prediction(130, features) == 130


def test_estimate_accuracy_clamped():
    simple = build_feature_row(
        {"age": 30, "bmi": 22, "a1c": 5.0, "insulin_level": 8, "heart_rate": 70},
        {"calories": 300, "carbs": 20, "protein": 25, "fat": 15, "fiber": 10, "meal_type": "Snack"},
        90,
        0,
        hour=10,
    )
    assert estimate_accuracy(1.0, simple, diabetic=True) <= 99
    complex_meal = build_feature_row(
        {"age": 30, "bmi": 22, "a1c": 5.0, "insulin_level": 8, "heart_rate": 70},
        {"calories": 900, "carbs": 200, "protein": 5, "fat": 5, "fiber": 1, "meal_type": "Dinner"},
        90,
        0,
        hour=19,
    )
    assert estimate_accuracy(50.0, complex_meal, diabetic=False) >= 85
