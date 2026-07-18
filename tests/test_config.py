from pathlib import Path

from app.config import ARTIFACTS_DIR, Config, ROOT_DIR


def test_paths_resolve_under_repo():
    assert ROOT_DIR.name == "sugar-spike-predictor" or (ROOT_DIR / "run.py").exists()
    assert ARTIFACTS_DIR == ROOT_DIR / "artifacts"
    assert Config.DIABETIC_MODEL_PATH.parent == ARTIFACTS_DIR
    assert Config.NON_DIABETIC_MODEL_PATH.parent == ARTIFACTS_DIR


def test_model_artifacts_exist():
    assert Path(Config.DIABETIC_MODEL_PATH).exists()
    assert Path(Config.NON_DIABETIC_MODEL_PATH).exists()


def test_rmse_constants_are_positive():
    assert Config.DIABETIC_RMSE > 0
    assert Config.NON_DIABETIC_RMSE > 0
    assert Config.DIABETIC_RMSE > Config.NON_DIABETIC_RMSE
    assert Config.A1C_DIABETIC_THRESHOLD == 6.5
