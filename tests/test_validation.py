from app.validation import parse_prediction_form


class _Form(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def _valid_form(**overrides):
    data = {
        "age": "35",
        "gender": "Female",
        "bmi": "24.5",
        "a1c": "5.6",
        "fasting_glucose": "95",
        "insulin_level": "12",
        "heart_rate": "72",
        "current_glucose": "110",
        "glucose_trend": "0",
        "meal_type": "Lunch",
        "calories": "550",
        "carbs": "55",
        "protein": "25",
        "fat": "18",
        "fiber": "8",
    }
    data.update(overrides)
    return _Form(data)


def test_valid_form_parses():
    payload, errors, _values = parse_prediction_form(_valid_form())
    assert errors == {}
    assert payload is not None
    assert payload["person"]["age"] == 35
    assert payload["meal"]["carbs"] == 55


def test_rejects_out_of_range_age():
    _payload, errors, _values = parse_prediction_form(_valid_form(age="200"))
    assert "age" in errors


def test_rejects_fiber_gt_carbs():
    _payload, errors, _values = parse_prediction_form(_valid_form(carbs="10", fiber="20"))
    assert "fiber" in errors


def test_rejects_invalid_gender():
    _payload, errors, _values = parse_prediction_form(_valid_form(gender="Nope"))
    assert "gender" in errors


def test_rejects_inconsistent_calories():
    _payload, errors, _values = parse_prediction_form(
        _valid_form(calories="50", carbs="200", protein="50", fat="50")
    )
    assert "calories" in errors
