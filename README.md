# Sugar Spike Predictor

Open-source post-meal blood glucose prediction. Given personal health metrics and meal
nutrition, the app estimates glucose about two hours later using **population-specific**
models (diabetic vs non-diabetic).

> Not medical advice. For research and education only. Always follow your clinician’s guidance.

## Features

- Feature engineering: glycemic load, carb–insulin ratio, fat/protein–carb balance, glucose momentum, age×A1c, time-of-day bins
- Specialized models: **XGBoost** for diabetic cohort, **HistGradientBoosting** for non-diabetic
- sklearn `Pipeline` + `ColumnTransformer` (impute → scale / one-hot → feature selection → regressor)
- Flask web UI with recommendations and expected RMSE ranges
- Trainable from CGM + bio CSV data

## Project layout

```text
app/                 Flask app (routes, features, predictor, templates, static)
artifacts/           Trained model pickles (joblib)
training/            Data prep + baseline / specialized training scripts
data/                Place raw CGM/bio CSVs here (gitignored contents)
docs/assets/         Plots and docs assets
tests/               Unit tests
run.py               Local entrypoint
```

## Quick start

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
python run.py
```

Open http://localhost:5000

Health check: http://localhost:5000/health

Production-style serve:

```bash
gunicorn -b 0.0.0.0:5000 "run:app"
```

## Training

1. Put `bio.csv` and `CGMacros-*.csv` under `data/`.
2. Merge:

```bash
python training/data_preparation.py
```

3. Baseline model:

```bash
python training/train_baseline.py
```

4. Specialized diabetic / non-diabetic models (writes into `artifacts/`):

```bash
python training/train_specialized.py
```

## Stack

Python, pandas, numpy, scikit-learn, XGBoost, Flask, gunicorn, matplotlib/seaborn

## License

MIT — see [LICENSE](LICENSE).
