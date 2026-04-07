# chap-xai-template

A minimal, forkable template for building [CHAP](https://dhis2-chap.github.io/chap-core/)-compatible external models with **native SHAP support**.

Clone or fork this repository, swap out the model, and you have a fully explainable disease-forecasting model that integrates with CHAP's XAI pipeline out of the box.

## What this template gives you

- A working `train.py` / `predict.py` pair that CHAP can call directly
- Autoregressive lag features and rolling statistics as a sensible starting point
- A `write_native_shap()` helper that auto-detects your model type and writes `shap_values.csv` with per-row SHAP contributions — no extra work required
- `MLproject` wired up with `provides_native_shap: true`

## Quickstart

```bash
git clone https://github.com/YOUR_USER/chap-xai-template
cd chap-xai-template
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Smoke-test with the example data:

```bash
python train.py input/trainData.csv output/model.bin
python predict.py output/model.bin input/trainData.csv input/futureClimateData.csv output/predictions.csv
```

## How to adapt this to your own model

### 1 — Swap the model (`train.py`)

Open `train.py` and find the `build_model()` function:

```python
def build_model(n_rows):
    """Return an untrained scikit-learn compatible estimator."""
    if n_rows < 30:
        return LinearRegression()
    return XGBRegressor(...)
```

Replace the body with any [shap-supported](https://shap.readthedocs.io/en/latest/) estimator:

```python
# Example: LightGBM
from lightgbm import LGBMRegressor
def build_model(n_rows):
    return LGBMRegressor(n_estimators=300, learning_rate=0.05)

# Example: Random Forest
from sklearn.ensemble import RandomForestRegressor
def build_model(n_rows):
    return RandomForestRegressor(n_estimators=200, random_state=42)

# Example: Ridge regression
from sklearn.linear_model import Ridge
def build_model(n_rows):
    return Ridge(alpha=1.0)
```

The rest of `train.py` and all of `predict.py` work unchanged.

### 2 — Adjust features (`train.py` + `predict.py`)

Features are built in `engineer_features()` in `train.py` and reconstructed
row-by-row in `_build_features_for_row()` in `predict.py`. Both functions must
stay in sync — if you add a feature in one, add the same logic in the other.

The default features are:
- Climate: `rainfall`, `mean_temperature`
- Calendar: `month_sin`, `month_cos` (cyclic encoding)
- Autoregressive: `disease_cases_lag_1/2/3`
- Rolling: `cases_roll_mean_3/6`, `cases_diff_1`, `cases_growth`, `cases_per_100k`

Remove or add features as needed for your use case.

### 3 — SHAP (nothing to change)

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

### 4 — Update metadata (`MLproject`)

Edit the `meta_data` block:

```yaml
meta_data:
  display_name: My XAI model
  description: >
    One-line description of what makes your model different.
  author: Your Name
  contact_email: your@email.com
```

Also update the `name` field at the top (used as the model ID inside CHAP).

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

## Running with CHAP

```bash
CHAP_FORCE_NATIVE_SHAP=true chap evaluate \
  --model-name /path/to/chap-xai-template \
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
git commit -m "Initial CHAP XAI model template"
git branch -M main
git remote add origin https://github.com/YOUR_USER/chap-xai-template.git
git push -u origin main
```

---

Based on the [chap-xai-lag-forecast](https://github.com/dhis2-chap) reference implementation.
See also: [chap_auto_ewars](https://github.com/chap-models/chap_auto_ewars) for an R-style example.
