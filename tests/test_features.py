from app.features import adjust_prediction, build_feature_row, estimate_accuracy, time_category


def test_time_category_buckets():
    assert time_category(7) == "morning"
    assert time_category(12) == "midday"
    assert time_category(15) == "afternoon"
    assert time_category(20) == "evening"
    assert time_category(2) == "early_morning"


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
