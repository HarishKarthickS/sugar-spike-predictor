from app.recommendations import get_recommendation


def _meal(**overrides):
    meal = {"carbs": 60, "fiber": 8, "protein": 25, "fat": 20}
    meal.update(overrides)
    return meal


def test_diabetic_high_glucose_low_fiber_carbs():
    msg = get_recommendation(200, True, _meal(carbs=70, fiber=2))
    assert "higher than recommended" in msg
    assert "reducing carbohydrates" in msg


def test_diabetic_high_glucose_low_fat_protein_balance():
    msg = get_recommendation(190, True, _meal(carbs=40, fiber=8, protein=5, fat=5))
    assert "protein and healthy fats" in msg
    assert "healthcare provider" in msg


def test_diabetic_elevated_with_good_fiber():
    msg = get_recommendation(150, True, _meal(carbs=40, fiber=10))
    assert "somewhat elevated" in msg
    assert "walk" in msg


def test_diabetic_good_range():
    msg = get_recommendation(120, True, _meal())
    assert "good post-meal range" in msg


def test_non_diabetic_high_low_carb_quality():
    msg = get_recommendation(150, False, _meal(carbs=80, fiber=2))
    assert "higher than typical" in msg
    assert "simple carbohydrates" in msg or "fiber" in msg


def test_non_diabetic_slightly_elevated():
    msg = get_recommendation(130, False, _meal())
    assert "slightly elevated" in msg


def test_non_diabetic_normal_range():
    msg = get_recommendation(100, False, _meal(carbs=45, protein=20, fat=15))
    assert "normal post-meal range" in msg
    assert "45g carbs" in msg
