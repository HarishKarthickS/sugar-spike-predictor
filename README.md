# Sugar Spike Predictor

Open-source post-meal blood glucose prediction. Given personal health metrics and meal
nutrition, the app estimates glucose about two hours later using **population-specific**
models (diabetic vs non-diabetic).

**Live demo:** [https://sugar-spike-predictor-92g4.onrender.com/](https://sugar-spike-predictor-92g4.onrender.com/)

> Not medical advice. For research and education only. Always follow your clinician’s guidance.

## Features

- Feature engineering: glycemic load, carb–insulin ratio, fat/protein–carb balance, glucose momentum, age×A1c, time-of-day bins
- Specialized models: **XGBoost** for diabetic cohort, **HistGradientBoosting** for non-diabetic
- sklearn `Pipeline` + `ColumnTransformer` (impute → scale / one-hot → feature selection → regressor)
- Flask web UI with client + server form validation, recommendations, and expected RMSE ranges
- Trainable from CGM + bio CSV data (optional — pickles in `artifacts/` ship with the repo)

## Project layout

```text
app/                 Flask app (routes, validation, features, predictor, templates, static)
artifacts/           Trained model pickles (joblib)
training/            Data prep + baseline / specialized training scripts
data/                Place raw CGM/bio CSVs here when retraining (gitignored contents)
docs/assets/         Plots and docs assets
tests/               Unit tests
run.py               Local entrypoint
```

## Quick start

Use **Python 3.10–3.12** (Render uses 3.11.9 via `runtime.txt`).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

Open http://localhost:5000 — health probe: http://localhost:5000/health

Production-style serve:

```bash
gunicorn -b 0.0.0.0:5000 "run:app"
```

## Form validation

Inputs are checked in the browser and again on the server:

- Required fields and numeric ranges (age, BMI, A1c, glucose, macros, etc.)
- Gender / meal type must be from the allowed lists
- Fiber cannot exceed carbohydrates
- Calories are cross-checked against carbs/protein/fat energy estimate
- Invalid submissions re-render the form with field errors (values preserved)

## Training (optional)

`data/` is empty in git on purpose — raw CGM/bio CSVs are local-only. The deployed app loads models from `artifacts/`.

1. Put `bio.csv` and `CGMacros-*.csv` under `data/`.
2. `python training/data_preparation.py`
3. `python training/train_baseline.py`
4. `python training/train_specialized.py` (writes pickles into `artifacts/`)

## Stack

Python, pandas, numpy, scikit-learn, XGBoost, Flask, gunicorn, matplotlib/seaborn

## Deploy

Hosted on Render: [https://sugar-spike-predictor-92g4.onrender.com/](https://sugar-spike-predictor-92g4.onrender.com/)

Config in-repo: `render.yaml`, `Procfile`, `runtime.txt`. Pushing to `main` redeploys when the Render service is linked to this GitHub repo.

## License

MIT — see [LICENSE](LICENSE).
