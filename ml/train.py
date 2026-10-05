"""RenewAI - Phase 2: Machine Learning Model Training & Evaluation
==================================================================
Trains an XGBoost Regressor (with automatic RandomForest fallback)
using a strict chronological 80/20 time-series split.
Evaluates MAE, RMSE, R2, creates evaluation plots, and persists the
model along with validation error distributions for confidence estimation.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Attempt to import XGBoost, fallback if unavailable
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from ml.preprocess import (
    load_raw_data,
    preprocess_data,
    split_time_series,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    OUTPUT_DIR
)

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "renewai_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "evaluation_metrics.json")


def train_solar_model(df: pd.DataFrame = None):
    """Train XGBoost / RandomForest model on solar dataset."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if df is None:
        raw_df = load_raw_data()
        df = preprocess_data(raw_df, verbose=False)

    print("[RenewAI ML] Performing chronological 80/20 train-test split...")
    X_train, X_test, y_train, y_test, train_times, test_times = split_time_series(df, test_size=0.2)

    model = None
    model_type = ""

    if XGBOOST_AVAILABLE:
        try:
            print("[RenewAI ML] Training XGBoost Regressor (preferred)...")
            model = xgb.XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                n_jobs=-1
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_test, y_test)],
                verbose=False
            )
            model_type = "XGBoost Regressor"
        except Exception as e:
            print(f"[RenewAI ML] XGBoost training encountered issue: {e}. Falling back to RandomForest...")
            model = None

    if model is None:
        print("[RenewAI ML] Training RandomForestRegressor (robust fallback)...")
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        model_type = "RandomForest Regressor"

    # Evaluate on Train and Test sets
    y_train_pred = np.clip(model.predict(X_train), 0.0, None)
    y_test_pred = np.clip(model.predict(X_test), 0.0, None)

    # Calculate metrics
    train_mae = float(mean_absolute_error(y_train, y_train_pred))
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    train_r2 = float(r2_score(y_train, y_train_pred))

    test_mae = float(mean_absolute_error(y_test, y_test_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_test_pred)))
    test_r2 = float(r2_score(y_test, y_test_pred))

    # Calculate test residuals for reliability scoring
    test_residuals = np.abs(y_test.values - y_test_pred)
    residual_p50 = float(np.percentile(test_residuals, 50))
    residual_p75 = float(np.percentile(test_residuals, 75))
    residual_p90 = float(np.percentile(test_residuals, 90))
    max_power_seen = float(df[TARGET_COLUMN].max())

    print("\n" + "=" * 50)
    print(f"       RENEWAI MODEL EVALUATION ({model_type})")
    print("=" * 50)
    print(f"Training Set Metrics  -> MAE: {train_mae:.2f} W | RMSE: {train_rmse:.2f} W | R²: {train_r2:.4f}")
    print(f"Testing Set Metrics   -> MAE: {test_mae:.2f} W | RMSE: {test_rmse:.2f} W | R²: {test_r2:.4f}")
    print(f"Validation Residuals  -> Median: {residual_p50:.2f} W | 75th: {residual_p75:.2f} W | 90th: {residual_p90:.2f} W")
    print("=" * 50)

    # Visualizations
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Actual vs Predicted Time-Series Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_times.values, y_test.values, label="Actual Power (W)", color="#2563EB", linewidth=1.5, alpha=0.85)
    ax.plot(test_times.values, y_test_pred, label="Predicted Power (W)", color="#DC2626", linestyle="--", linewidth=1.5, alpha=0.85)
    ax.set_title(f"RenewAI: Actual vs Predicted Solar Power (Test Set, R²={test_r2:.3f})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (Holdout Horizon)", fontsize=11)
    ax.set_ylabel("Power Output (W)", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    act_pred_path = os.path.join(OUTPUT_DIR, "actual_vs_predicted.png")
    fig.savefig(act_pred_path, dpi=200)
    plt.close(fig)
    print(f"[RenewAI Viz] Saved: {act_pred_path}")

    # 2. Feature Importance Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        sorted_feats = [FEATURE_COLUMNS[i] for i in indices]
        sorted_imps = [importances[i] for i in indices]
        ax.barh(sorted_feats[::-1], sorted_imps[::-1], color="#10B981")
        ax.set_title("RenewAI Feature Importance in Power Forecasting", fontsize=12, fontweight="bold")
        ax.set_xlabel("Relative Importance Score", fontsize=10)
    plt.tight_layout()
    feat_imp_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    fig.savefig(feat_imp_path, dpi=200)
    plt.close(fig)
    print(f"[RenewAI Viz] Saved: {feat_imp_path}")

    # Save Bundle
    model_bundle = {
        "model": model,
        "model_type": model_type,
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "metrics": {
            "train_mae": train_mae,
            "train_rmse": train_rmse,
            "train_r2": train_r2,
            "test_mae": test_mae,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "residual_p50": residual_p50,
            "residual_p75": residual_p75,
            "residual_p90": residual_p90,
            "max_power": max_power_seen
        },
        "feature_stats": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max())
            }
            for col in FEATURE_COLUMNS
        }
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[RenewAI ML] Model bundle saved to: {MODEL_PATH}")

    # Save Metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(model_bundle["metrics"], f, indent=4)
    print(f"[RenewAI ML] Metrics exported to: {METRICS_PATH}")

    return model_bundle


if __name__ == "__main__":
    train_solar_model()
