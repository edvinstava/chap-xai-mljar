# chap-xai-mljar

A [CHAP](https://dhis2-chap.github.io/chap-core/)-compatible external forecasting model built with **MLJAR AutoML** and native **SHAP explainability** outputs.

This repository trains a regression model for `disease_cases`, generates holdout metrics, and writes SHAP artifacts that CHAP can consume.

## What this project includes

- A CHAP-compatible `train.py` / `predict.py` pair
- MLJAR AutoML training with model selection across multiple algorithms and eval metrics
- Sequential forecasting logic that supports lag and rolling features
- Native SHAP outputs:
  - `shap_values.csv` from `predict.py` (per-row contributions)
  - `<model>.shap_values.csv` and `<model>.shap_summary.png` from `train.py`
- `MLproject` configured with `provides_native_shap: true`

## Quickstart

```bash
git clone https://github.com/edvinstava/chap-xai-miljar.git
cd chap-xai-mljar
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Smoke-test with the example data:

```bash
python train.py input/trainData.csv output/model.bin
python predict.py output/model.bin input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

Expected training artifacts include:

- `output/model.bin`
- `output/model.bin.metrics.json`
- `output/model.bin.shap_values.csv`
- `output/model.bin.shap_summary.png`

Prediction writes:

- `output/predictions.csv`
- `shap_values.csv`

## Project behavior and modeling approach

### 1 — Model training (`train.py`)

`train.py`:

- engineers lag/rolling/climate interaction features
- evaluates AutoML candidates with `rmse`, `mae`, and `mape`
- picks the eval metric using a combined holdout score
- retrains a final model on all training rows
- stores payload (`model`, `features`, `lags`, `metrics`, `target_transform`)

### 2 — Feature engineering (`train.py` + `predict.py`)

Features are built in `engineer_features()` in `train.py` and reconstructed
row-by-row in `_build_features_for_row()` in `predict.py`. Both functions must
stay in sync — if you add a feature in one, add the same logic in the other.

The default features are:
- Climate: `rainfall`, `mean_temperature`
- Calendar: `month_sin`, `month_cos` (cyclic encoding)
- Autoregressive: `disease_cases_lag_1/2/3`
- Rolling and derived: `cases_roll_mean_3/6`, `cases_diff_1`, `cases_growth`, `cases_per_100k`
- Interactions/history: `rain_temp_interaction`, `rainfall_roll_mean_3`, `temp_roll_mean_3`, `location_case_mean_prior`, `location_case_std_prior`

### 3 — Prediction (`predict.py`)

`predict.py`:

- loads the saved training payload
- performs iterative, per-location forecasting so lag features are updated with each predicted step
- writes `sample_0` predictions to output CSV
- writes `shap_values.csv` for native CHAP explainability integration

### 4 — SHAP explainability

`write_native_shap()` in `predict.py` tries explainers in this order:

| Explainer | Best for |
|-----------|----------|
| `TreeExplainer` | XGBoost, LightGBM, RandomForest, ExtraTrees |
| `LinearExplainer` | LinearRegression, Ridge, Lasso, ElasticNet |
| `KernelExplainer` | Any model (slower, uses sampled background) |

The output `shap_values.csv` has columns:
```
location, time_period, expected_value, shap__<feature>..., value__<feature>...
```
CHAP reads this file automatically when `provides_native_shap: true` is set in `MLproject`.

## Data format

CHAP passes harmonised CSVs with these columns:

| Column | Description |
|--------|-------------|
| `time_period` | Month string (`YYYY-MM`) |
| `location` | Org unit / region label |
| `rainfall` | Monthly rainfall (mm) |
| `mean_temperature` | Monthly mean temperature (°C) |
| `population` | Population (optional) |
| `disease_cases` | Target — reported cases (train only) |

`historic_data` (passed to predict) has the same schema and provides context
for autoregressive features. `future_data` has the covariate columns but
`disease_cases` is empty for the rows to forecast.

## MLproject metadata

Current metadata and model name are defined in `MLproject`. If publishing this as your own model package, update:

- `name`
- `meta_data.display_name`
- `meta_data.description`
- `meta_data.author`
- `meta_data.contact_email`

## Running with CHAP

```bash
CHAP_FORCE_NATIVE_SHAP=true chap evaluate \
  --model-name /path/to/chap-xai-mljar \
  --dataset-name ISIMIP_dengue_harmonized \
  --dataset-country brazil \
  --report-filename report.pdf \
  --ignore-environment \
  --debug
```

## Publishing to GitHub

```bash
git init
git add MLproject train.py predict.py requirements.txt pyproject.toml README.md
git add input/   # optional: small fixture CSVs for smoke-testing
git commit -m "Initial CHAP XAI MLJAR model"
git branch -M main
git remote add origin https://github.com/edvinstava/chap-xai-miljar.git
git push -u origin main
```

---

Based on the [chap-xai-lag-forecast](https://github.com/dhis2-chap) reference implementation.
See also: [chap_auto_ewars](https://github.com/chap-models/chap_auto_ewars) for an R-style example.
