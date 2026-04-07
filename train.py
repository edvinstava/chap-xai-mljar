"""
train.py — CHAP external model template with native SHAP support.

How to adapt this to your own model
------------------------------------
1. Edit the FEATURE ENGINEERING section to add, remove, or transform features.
2. Replace the MODEL section with any scikit-learn compatible estimator that
   shap supports (XGBoost, LightGBM, RandomForest, LinearRegression, etc.).
3. Keep the SHAP section as-is — it auto-detects the model type.

Column names expected by CHAP
-------------------------------
  time_period     — month string (YYYY-MM)
  location        — org unit / region label
  rainfall        — numeric covariate
  mean_temperature— numeric covariate
  population      — numeric covariate (optional)
  disease_cases   — target (may contain gaps; zeros used for missing)
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from supervised.automl import AutoML
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

def _load_training_dataframe(csv_fn):
    raw_df = pd.read_csv(csv_fn)
    if "Unnamed: 0" in raw_df.columns:
        raw_df = raw_df.drop(columns=["Unnamed: 0"])
    if "population" not in raw_df.columns:
        raw_df["population"] = np.nan
    return raw_df


# ---------------------------------------------------------------------------
# FEATURE ENGINEERING  ← customize this section
# ---------------------------------------------------------------------------

def engineer_features(df):
    """
    Build model features from the harmonised CHAP dataframe.

    Returns (df_with_features, feature_cols, lags) where `feature_cols` is
    the ordered list of column names the model will be trained on.

    Add or remove features here freely — just keep feature_cols consistent
    with what predict.py reconstructs at inference time.
    """
    df = df.copy()
    df['time_period'] = pd.to_datetime(df['time_period'])
    df = df.sort_values(['location', 'time_period']).reset_index(drop=True)

    # Cyclic calendar encoding
    df['month_sin'] = np.sin(2 * np.pi * df['time_period'].dt.month / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['time_period'].dt.month / 12.0)

    df['disease_cases'] = df['disease_cases'].fillna(0)
    df['population'] = pd.to_numeric(df['population'], errors='coerce')
    df['rainfall'] = pd.to_numeric(df['rainfall'], errors='coerce')
    df['mean_temperature'] = pd.to_numeric(df['mean_temperature'], errors='coerce')

    # Autoregressive lags (skip if group too small)
    candidate_lags = [1, 2, 3]
    min_group_size = df.groupby('location').size().min()
    lags = [lag for lag in candidate_lags if lag < min_group_size]

    lag_features = []
    for lag in lags:
        feat = f'disease_cases_lag_{lag}'
        df[feat] = df.groupby('location')['disease_cases'].shift(lag)
        lag_features.append(feat)

    # Rolling statistics (shifted to avoid leakage)
    df['cases_diff_1'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.diff(1).shift(1))
    )
    df['cases_roll_mean_3'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df['cases_roll_mean_6'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).rolling(6, min_periods=1).mean())
    )
    df['cases_growth'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1) / (s.shift(2) + 1.0))
    )
    df['cases_per_100k'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1))
        / (df['population'] + 1.0)
    ) * 1e5
    df['rain_temp_interaction'] = df['rainfall'] * df['mean_temperature']
    df['rainfall_roll_mean_3'] = (
        df.groupby('location')['rainfall']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df['temp_roll_mean_3'] = (
        df.groupby('location')['mean_temperature']
        .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    )
    df['location_case_mean_prior'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    df['location_case_std_prior'] = (
        df.groupby('location')['disease_cases']
        .transform(lambda s: s.shift(1).expanding(min_periods=2).std())
    )
    df['location_case_std_prior'] = df['location_case_std_prior'].fillna(0.0)

    feature_cols = [
        'rainfall',
        'mean_temperature',
        'population',
        'month_sin',
        'month_cos',
        'cases_diff_1',
        'cases_roll_mean_3',
        'cases_roll_mean_6',
        'cases_growth',
        'cases_per_100k',
        'rain_temp_interaction',
        'rainfall_roll_mean_3',
        'temp_roll_mean_3',
        'location_case_mean_prior',
        'location_case_std_prior',
    ] + lag_features

    return df, feature_cols, lags


# ---------------------------------------------------------------------------
# MODEL  ← swap your estimator here
# ---------------------------------------------------------------------------

def build_automl(results_path, total_time_limit=180, eval_metric="rmse"):
    return AutoML(
        mode="Perform",
        ml_task="regression",
        eval_metric=eval_metric,
        total_time_limit=total_time_limit,
        algorithms=["Linear", "Random Forest", "Extra Trees", "LightGBM", "Xgboost"],
        explain_level=0,
        train_ensemble=True,
        stack_models=False,
        validation_strategy={
            "validation_type": "split",
            "train_ratio": 0.8,
            "shuffle": False,
            "stratify": False,
        },
        random_state=42,
        results_path=results_path,
    )


def _compute_holdout_metrics(y_true, y_hat, y_true_log, y_hat_log):
    nonzero = np.abs(y_true) > 1e-6
    if np.any(nonzero):
        epsilon = 1.0
        ape = np.abs((y_true[nonzero] - y_hat[nonzero]) / (y_true[nonzero] + epsilon))
        mape_pct = float(np.mean(ape) * 100.0)
        within_20pct = float(np.mean(ape <= 0.20))
    else:
        mape_pct = float('nan')
        within_20pct = float('nan')
    return {
        'mae': float(mean_absolute_error(y_true, y_hat)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_hat))),
        'r2': float(r2_score(y_true, y_hat)),
        'rmse_log': float(np.sqrt(mean_squared_error(y_true_log, y_hat_log))),
        'mape_pct_nonzero_targets': mape_pct,
        'within_20pct_accuracy_nonzero_targets': within_20pct,
    }


def _multi_metric_score(candidate_metrics, baseline_metrics):
    rmse_ratio = candidate_metrics['rmse'] / max(baseline_metrics['rmse'], 1e-8)
    mae_ratio = candidate_metrics['mae'] / max(baseline_metrics['mae'], 1e-8)
    mape_ratio = candidate_metrics['mape_pct_nonzero_targets'] / max(baseline_metrics['mape_pct_nonzero_targets'], 1e-8)
    within_gap = 1.0 - candidate_metrics['within_20pct_accuracy_nonzero_targets']
    return (0.35 * rmse_ratio) + (0.25 * mae_ratio) + (0.25 * mape_ratio) + (0.15 * within_gap)


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train(csv_fn, model_fn):
    df = _load_training_dataframe(csv_fn)
    df, feature_cols, lags = engineer_features(df)

    train_df = df.dropna(subset=feature_cols + ['disease_cases'])

    # Target: log1p transform stabilises variance for count data.
    # Remove / change if your target is not a count.
    y_full = train_df['disease_cases']
    y_full_log = np.log1p(y_full)

    # Hold-out split for metrics (last 6 periods per location)
    test_horizon = 6
    split_idx = train_df.groupby('location').cumcount(ascending=False)
    test_mask = split_idx < test_horizon
    train_mask = ~test_mask

    X_train = train_df.loc[train_mask, feature_cols]
    y_train = train_df.loc[train_mask, 'disease_cases']
    X_test = train_df.loc[test_mask, feature_cols]
    y_test = train_df.loc[test_mask, 'disease_cases']
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # --- Evaluation model (on train split) ---
    metrics = {'test_rows': 0, 'mae': float('nan'), 'rmse': float('nan'),
               'r2': float('nan'), 'rmse_log': float('nan'),
               'mape_pct_nonzero_targets': float('nan'),
               'within_20pct_accuracy_nonzero_targets': float('nan')}

    selected_eval_metric = "rmse"
    if len(X_train) > 0 and len(X_test) > 0:
        baseline_log = np.full(len(y_test_log), float(np.mean(y_train_log)))
        baseline = np.clip(np.expm1(baseline_log), 0, None)
        baseline_metrics = _compute_holdout_metrics(
            y_test.to_numpy(), baseline, y_test_log.to_numpy(), baseline_log
        )

        candidates = []
        for eval_metric in ["rmse", "mae", "mape"]:
            eval_results_path = f"{model_fn}.eval_automl_{eval_metric}"
            eval_model = build_automl(
                eval_results_path, total_time_limit=75, eval_metric=eval_metric
            )
            eval_model.fit(X_train, y_train_log)

            y_hat_log = eval_model.predict(X_test)
            y_hat = np.clip(np.expm1(y_hat_log), 0, None)
            candidate_metrics = _compute_holdout_metrics(
                y_test.to_numpy(), y_hat, y_test_log.to_numpy(), y_hat_log
            )
            candidate_metrics["test_rows"] = int(len(y_test))
            score = _multi_metric_score(candidate_metrics, baseline_metrics)
            candidates.append((score, eval_metric, candidate_metrics))

        candidates.sort(key=lambda x: x[0])
        _, selected_eval_metric, metrics = candidates[0]
        metrics["selected_eval_metric"] = selected_eval_metric
    else:
        print("Warning: not enough data for a train/test split — skipping hold-out metrics.")

    # --- Final model (on all data) ---
    final_results_path = f"{model_fn}.automl"
    model = build_automl(
        final_results_path, total_time_limit=300, eval_metric=selected_eval_metric
    )
    model.fit(train_df[feature_cols], y_full_log)

    # --- SHAP (training-time summary) ---
    if os.getenv("TRAIN_SHAP_SUMMARY", "0") == "1":
        shap_plot_fn = f"{model_fn}.shap_summary.png"
        shap_values_fn = f"{model_fn}.shap_values.csv"
        shap_sample = train_df[feature_cols].sample(n=min(30, len(train_df)), random_state=42)
        try:
            explainer = shap.Explainer(model.predict, shap_sample)
            sv_obj = explainer(shap_sample, max_evals=min(31, 2 * shap_sample.shape[1] + 1))
            sv = sv_obj.values
            pd.DataFrame({'feature': feature_cols, 'mean_abs_shap': np.abs(sv).mean(axis=0)}) \
                .sort_values('mean_abs_shap', ascending=False) \
                .to_csv(shap_values_fn, index=False)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(sv, shap_sample, feature_names=feature_cols, show=False)
            plt.tight_layout()
            plt.savefig(shap_plot_fn, dpi=200)
            plt.close()
        except Exception:
            pass

    # --- Save artifacts ---
    payload = {
        'model': model,
        'features': feature_cols,
        'lags': lags,
        'model_type': "MLJAR_AutoML",
        'automl_results_path': final_results_path,
        'metrics': metrics,
        'selected_eval_metric': selected_eval_metric,
        'target_transform': 'log1p',
    }
    joblib.dump(payload, model_fn)
    metrics_fn = f"{model_fn}.metrics.json"
    with open(metrics_fn, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"Model  : {payload['model_type']} | rows: {len(train_df)} | lags: {lags}")
    print(f"Metrics: MAE={metrics['mae']:.3f}  RMSE={metrics['rmse']:.3f}  "
          f"R2={metrics['r2']:.3f}  MAPE%={metrics['mape_pct_nonzero_targets']:.2f}")
    print(f"Saved  : {os.path.basename(model_fn)}, {os.path.basename(metrics_fn)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a CHAP-compatible model with native SHAP.')
    parser.add_argument('csv_fn', type=str, help='Path to CHAP training CSV.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model artifact.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)
