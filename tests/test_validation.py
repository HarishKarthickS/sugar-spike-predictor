from app.validation import FIELD_RULES, parse_prediction_form


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


def test_rejects_missing_required_field():
    _payload, errors, _values = parse_prediction_form(_valid_form(bmi=""))
    assert "bmi" in errors
    assert "required" in errors["bmi"].lower()


def test_rejects_non_numeric():
    _payload, errors, _values = parse_prediction_form(_valid_form(carbs="abc"))
    assert "carbs" in errors
    assert "number" in errors["carbs"].lower()


def test_rejects_fiber_gt_carbs():
    _payload, errors, _values = parse_prediction_form(_valid_form(carbs="10", fiber="20"))
    assert "fiber" in errors


def test_rejects_invalid_gender():
    _payload, errors, _values = parse_prediction_form(_valid_form(gender="Nope"))
    assert "gender" in errors


def test_rejects_invalid_meal_type():
    _payload, errors, _values = parse_prediction_form(_valid_form(meal_type="Brunch"))
    assert "meal_type" in errors


def test_rejects_inconsistent_calories_too_low():
    _payload, errors, _values = parse_prediction_form(
        _valid_form(calories="50", carbs="200", protein="50", fat="50")
    )
    assert "calories" in errors


def test_rejects_inconsistent_calories_too_high():
    _payload, errors, _values = parse_prediction_form(
        _valid_form(calories="2000", carbs="10", protein="5", fat="5")
    )
    assert "calories" in errors


def test_rejects_extreme_glucose_gap():
    _payload, errors, _values = parse_prediction_form(
        _valid_form(fasting_glucose="70", current_glucose="500")
    )
    assert "current_glucose" in errors


def test_preserves_values_on_error():
    _payload, errors, values = parse_prediction_form(_valid_form(age="999", carbs="40"))
    assert "age" in errors
    assert values["carbs"] == 40 or values["carbs"] == "40"


def test_boundary_values_accepted():
    payload, errors, _values = parse_prediction_form(
        _valid_form(age="1", bmi="10", a1c="3.5", heart_rate="40", glucose_trend="-30")
    )
    assert errors == {}
    assert payload is not None
    assert payload["person"]["age"] == 1


def test_field_rules_cover_core_inputs():
    expected = {
        "age",
        "bmi",
        "a1c",
        "fasting_glucose",
        "insulin_level",
        "heart_rate",
        "current_glucose",
        "glucose_trend",
        "calories",
        "carbs",
        "protein",
        "fat",
        "fiber",
    }
    assert set(FIELD_RULES) == expected
