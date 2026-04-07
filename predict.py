"""
predict.py for chap-xai-mljar.

Generates CHAP-compatible forecasts and writes native SHAP explanations to
`shap_values.csv` with columns:
  location, time_period, expected_value, shap__<feature>..., value__<feature>...
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
import shap
from supervised.automl import AutoML


# ---------------------------------------------------------------------------
# NATIVE SHAP OUTPUT  ← keep as-is; works for any shap-supported model
# ---------------------------------------------------------------------------

def _extract_shap_arrays(shap_result):
    base = None
    values = shap_result
    if hasattr(shap_result, "values"):
        values = shap_result.values
    if hasattr(shap_result, "base_values"):
        base = shap_result.base_values
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 0]
    if base is not None:
        base = np.asarray(base)
    return values, base


def write_native_shap(model, x_df, out_df, out_paths):
    """
    Compute per-row SHAP values and write shap_values.csv.

    Tries explainers in order of preference:
      1. TreeExplainer  — fast, exact; for XGBoost / LightGBM / RandomForest
      2. LinearExplainer — fast, exact; for linear models
      3. KernelExplainer — slow, model-agnostic fallback (sampled background)

    The output CSV has columns:
      location, time_period, expected_value, shap__<feature>..., value__<feature>...
    """
    if x_df.empty:
        return

    feature_cols = list(x_df.columns)

    def _candidate_models(m):
        candidates = [m]
        for attr in ["model", "_model", "best_model", "_best_model", "_final_model"]:
            val = getattr(m, attr, None)
            if val is not None:
                candidates.append(val)
        return candidates

    sv = None
    expected = None
    for candidate in _candidate_models(model):
        try:
            explainer = shap.TreeExplainer(candidate)
            shap_result = explainer(x_df)
            sv, base = _extract_shap_arrays(shap_result)
            if base is None:
                ev = np.asarray(explainer.expected_value).reshape(-1)
                expected = np.full(len(x_df), float(ev[0]))
            else:
                expected = np.asarray(base).reshape(-1)
            break
        except Exception:
            continue

    if sv is None:
        for candidate in _candidate_models(model):
            try:
                explainer = shap.LinearExplainer(candidate, x_df)
                shap_result = explainer(x_df)
                sv, base = _extract_shap_arrays(shap_result)
                if base is None:
                    ev = np.asarray(explainer.expected_value).reshape(-1)
                    expected = np.full(len(x_df), float(ev[0]))
                else:
                    expected = np.asarray(base).reshape(-1)
                break
            except Exception:
                continue

    if sv is None:
        try:
            background = shap.sample(x_df, min(50, len(x_df)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background)
            shap_result = explainer.shap_values(x_df)
            sv, _ = _extract_shap_arrays(shap_result)
            ev = np.asarray(explainer.expected_value).reshape(-1)
            expected = np.full(len(x_df), float(ev[0]))
        except Exception:
            print("Warning: unable to compute SHAP values with Tree, Linear, or Kernel explainers.")
            return

    if isinstance(sv, list):
        sv = sv[0]
    sv = np.asarray(sv)
    if sv.shape == (len(feature_cols), len(x_df)):
        sv = sv.T
    if sv.shape != (len(x_df), len(feature_cols)):
        print(f"Warning: SHAP shape {sv.shape} does not match {(len(x_df), len(feature_cols))}.")
        return

    expected = np.asarray(expected)
    if expected.ndim == 0:
        expected = np.full(len(x_df), float(expected))
    elif expected.ndim > 1:
        expected = expected.reshape(-1)
    if len(expected) != len(x_df):
        expected = np.full(len(x_df), float(expected[0]))

    shap_df = pd.DataFrame(sv, columns=[f"shap__{c}" for c in feature_cols])
    val_df = x_df.reset_index(drop=True).rename(columns={c: f"value__{c}" for c in feature_cols})

    result = pd.concat([
        out_df[['location', 'time_period']].reset_index(drop=True),
        pd.Series(expected, name='expected_value'),
        shap_df,
        val_df,
    ], axis=1)
    if isinstance(out_paths, str):
        out_paths = [out_paths]
    for out_path in out_paths:
        result.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING  ← keep in sync with train.py
# ---------------------------------------------------------------------------

def _build_features_for_row(row, state, lags, defaults, feature_cols):
    """
    Reconstruct the same feature vector used at training time, but from
    the rolling state maintained per-location during sequential prediction.
    """
    feat = {
        'rainfall': row['rainfall'],
        'mean_temperature': row['mean_temperature'],
        'month_sin': np.sin(2 * np.pi * row['time_period'].month / 12.0),
        'month_cos': np.cos(2 * np.pi * row['time_period'].month / 12.0),
    }

    cases_1 = state['disease_cases'].tail(1)
    cases_2 = state['disease_cases'].tail(2)
    cases_3 = state['disease_cases'].tail(3)
    cases_6 = state['disease_cases'].tail(6)

    feat['cases_roll_mean_3'] = float(cases_3.mean()) if len(cases_3) else float(defaults['disease_cases'])
    feat['cases_roll_mean_6'] = float(cases_6.mean()) if len(cases_6) else float(defaults['disease_cases'])

    if len(cases_2) >= 2:
        last_case = float(cases_2.iloc[-1])
        prev_case = float(cases_2.iloc[-2])
        feat['cases_diff_1'] = last_case - prev_case
        feat['cases_growth'] = last_case / (prev_case + 1.0)
    elif len(cases_1) == 1:
        last_case = float(cases_1.iloc[-1])
        feat['cases_diff_1'] = 0.0
        feat['cases_growth'] = last_case / (float(defaults['disease_cases']) + 1.0)
    else:
        feat['cases_diff_1'] = 0.0
        feat['cases_growth'] = 0.0

    pop_value = row['population'] if pd.notna(row.get('population')) else defaults['population']
    base_cases = float(cases_1.iloc[-1]) if len(cases_1) else float(defaults['disease_cases'])
    feat['cases_per_100k'] = base_cases / (float(pop_value) + 1.0) * 1e5
    feat['population'] = float(pop_value)
    feat['rain_temp_interaction'] = float(row['rainfall']) * float(row['mean_temperature'])
    rain_3 = state['rainfall'].tail(3)
    temp_3 = state['mean_temperature'].tail(3)
    feat['rainfall_roll_mean_3'] = float(rain_3.mean()) if len(rain_3) else float(defaults['rainfall'])
    feat['temp_roll_mean_3'] = float(temp_3.mean()) if len(temp_3) else float(defaults['mean_temperature'])
    feat['location_case_mean_prior'] = float(state['disease_cases'].mean()) if len(state) else float(defaults['disease_cases'])
    feat['location_case_std_prior'] = float(state['disease_cases'].std()) if len(state) > 1 else 0.0

    for lag in lags:
        key = f'disease_cases_lag_{lag}'
        if len(state) >= lag:
            feat[key] = float(state['disease_cases'].iloc[-lag])
        else:
            feat[key] = float(defaults['disease_cases'])

    return pd.DataFrame([{c: feat.get(c, 0.0) for c in feature_cols}])


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    payload = joblib.load(model_fn)

    if not isinstance(payload, dict) or 'model' not in payload:
        # Minimal payload: bare estimator (no lags / feature engineering)
        model = payload
        future_df = pd.read_csv(future_climatedata_fn)
        if 'population' not in future_df.columns:
            future_df['population'] = 0.0
        X = future_df[['rainfall', 'mean_temperature', 'population']]
        future_df['sample_0'] = model.predict(X)
        shap_out = future_df[['location', 'time_period']].copy()
        shap_out['time_period'] = pd.to_datetime(shap_out['time_period']).dt.strftime('%Y-%m')
        shap_out_path = os.path.join(
            os.path.dirname(os.path.abspath(predictions_fn)),
            "shap_values.csv",
        )
        write_native_shap(model, X, shap_out, [shap_out_path, "shap_values.csv"])
        future_df.to_csv(predictions_fn, index=False)
        print("Predictions:", future_df['sample_0'].tolist())
        return

    model = payload['model']
    automl_results_path = payload.get('automl_results_path')
    if automl_results_path and not hasattr(model, "predict"):
        model = AutoML(results_path=automl_results_path)
    feature_cols = payload['features']
    lags = payload.get('lags', [])
    target_transform = payload.get('target_transform')

    future_df = pd.read_csv(future_climatedata_fn)
    historic_df = pd.read_csv(historic_data_fn)

    future_df['time_period'] = pd.to_datetime(future_df['time_period'])
    historic_df['time_period'] = pd.to_datetime(historic_df['time_period'])
    historic_df = historic_df.sort_values(['location', 'time_period']).reset_index(drop=True)
    future_df = future_df.sort_values(['location', 'time_period']).reset_index(drop=True)

    for col in ['population', 'rainfall', 'mean_temperature']:
        if col not in historic_df.columns:
            historic_df[col] = np.nan
        historic_df[col] = pd.to_numeric(historic_df[col], errors='coerce')
    if 'disease_cases' not in historic_df.columns:
        historic_df['disease_cases'] = 0.0
    historic_df['disease_cases'] = pd.to_numeric(historic_df['disease_cases'], errors='coerce').fillna(0.0)

    if 'population' not in future_df.columns:
        future_df['population'] = historic_df['population'].mean() if len(historic_df) else 0.0
    future_df['population'] = pd.to_numeric(future_df['population'], errors='coerce')

    global_defaults = {
        col: float(historic_df[col].mean()) if len(historic_df) else 0.0
        for col in ['rainfall', 'mean_temperature', 'population', 'disease_cases']
    }
    by_loc_defaults = (
        historic_df.groupby('location')[['rainfall', 'mean_temperature', 'population', 'disease_cases']]
        .mean()
        .to_dict(orient='index')
    )

    state_by_loc = {
        loc: grp[['time_period', 'rainfall', 'mean_temperature', 'population', 'disease_cases']]
        .copy().reset_index(drop=True)
        for loc, grp in historic_df.groupby('location')
    }

    x_rows = []
    preds = []
    for idx, row in future_df.iterrows():
        loc = row['location']
        defaults = by_loc_defaults.get(loc, global_defaults)
        if loc not in state_by_loc:
            state_by_loc[loc] = pd.DataFrame(
                columns=['time_period', 'rainfall', 'mean_temperature', 'population', 'disease_cases']
            )

        x = _build_features_for_row(row, state_by_loc[loc], lags, defaults, feature_cols)
        x_rows.append(x.iloc[0].to_dict())

        y_raw = float(model.predict(x)[0])
        y_hat = float(np.clip(np.expm1(y_raw), 0, None)) if target_transform == 'log1p' else y_raw
        preds.append(y_hat)
        future_df.at[idx, 'sample_0'] = y_hat

        new_row = pd.DataFrame([{
            'time_period': row['time_period'],
            'rainfall': row['rainfall'],
            'mean_temperature': row['mean_temperature'],
            'population': row.get('population', defaults['population']),
            'disease_cases': y_hat,
        }])
        state_by_loc[loc] = pd.concat([state_by_loc[loc], new_row], ignore_index=True)

    future_df['time_period'] = future_df['time_period'].dt.strftime('%Y-%m')
    x_pred_df = pd.DataFrame(x_rows)[feature_cols]
    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = x_pred_df[col].values
    shap_out_df = future_df[['location', 'time_period']].copy()
    shap_out_path = os.path.join(
        os.path.dirname(os.path.abspath(predictions_fn)),
        "shap_values.csv",
    )
    write_native_shap(model, x_pred_df, shap_out_df, [shap_out_path, "shap_values.csv"])

    future_df.to_csv(predictions_fn, index=False)
    print("Predictions:", preds)
    return np.array(preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict with a CHAP-compatible model.')
    parser.add_argument('model_fn', type=str, help='Path to the trained model artifact.')
    parser.add_argument('historic_data_fn', type=str, help='Path to historic data CSV.')
    parser.add_argument('future_climatedata_fn', type=str, help='Path to future climate data CSV.')
    parser.add_argument('predictions_fn', type=str, help='Path to write predictions CSV.')
    args = parser.parse_args()
    predict(args.model_fn, args.historic_data_fn, args.future_climatedata_fn, args.predictions_fn)
