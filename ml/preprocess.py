"""RenewAI - Phase 1: Data Preprocessing & Exploratory Analysis
============================================================
Handles loading, cleaning, feature engineering, validation,
and visual exploration of solar and meteorological data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Default data path
DEFAULT_DATA_PATH = os.path.join("data", "Lahore_Single_Panel_Dataset_with_Meteorological_Datas.csv")
CLEANED_DATA_PATH = os.path.join("data", "cleaned_solar_data.csv")
OUTPUT_DIR = "outputs"

TARGET_COLUMN = "Power"
FEATURE_COLUMNS = [
    "Temperature",
    "Prectotland",
    "Rhoa",
    "Irradiance (G)",
    "Irradiance (A)",
    "Cloud",
    "Hour",
    "Day",
    "Month",
    "DayOfWeek"
]


def load_raw_data(file_path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load raw solar dataset from CSV with robust error handling."""
    if not os.path.exists(file_path):
        # Check alternative parent paths if called from subdirectories
        alt_path = os.path.join("..", file_path)
        if os.path.exists(alt_path):
            file_path = alt_path
        else:
            raise FileNotFoundError(
                f"RenewAI Dataset not found at: {os.path.abspath(file_path)}\n"
                "Please verify that the CSV exists in the data/ directory."
            )

    print(f"[RenewAI Preprocess] Loading raw dataset from: {file_path}")
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Clean, sort, handle missing values, and extract time features."""
    df_clean = df.copy()

    # Verify TimeStamp column
    if "TimeStamp" not in df_clean.columns:
        raise KeyError("Required column 'TimeStamp' not found in dataset columns: " + str(df_clean.columns.tolist()))

    # Convert TimeStamp to datetime
    df_clean["TimeStamp"] = pd.to_datetime(df_clean["TimeStamp"], errors="coerce")

    # Drop rows where TimeStamp could not be parsed
    invalid_dates = df_clean["TimeStamp"].isnull().sum()
    if invalid_dates > 0:
        if verbose:
            print(f"[RenewAI Preprocess] Warning: Dropping {invalid_dates} rows with invalid TimeStamp values.")
        df_clean = df_clean.dropna(subset=["TimeStamp"])

    # Sort strictly by timestamp
    df_clean = df_clean.sort_values("TimeStamp").reset_index(drop=True)

    # Remove duplicate timestamp rows if any
    duplicates = df_clean.duplicated(subset=["TimeStamp"]).sum()
    if duplicates > 0:
        if verbose:
            print(f"[RenewAI Preprocess] Dropping {duplicates} duplicate timestamp records.")
        df_clean = df_clean.drop_duplicates(subset=["TimeStamp"], keep="first").reset_index(drop=True)

    # Missing value handling
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    if df_clean[numeric_cols].isnull().sum().sum() > 0:
        if verbose:
            print("[RenewAI Preprocess] Handling missing values using forward/backward fill interpolation.")
        df_clean[numeric_cols] = df_clean[numeric_cols].ffill().bfill()

    # Time Feature Engineering
    df_clean["Hour"] = df_clean["TimeStamp"].dt.hour
    df_clean["Day"] = df_clean["TimeStamp"].dt.day
    df_clean["Month"] = df_clean["TimeStamp"].dt.month
    df_clean["DayOfWeek"] = df_clean["TimeStamp"].dt.dayofweek

    # Verify target column exists
    if TARGET_COLUMN not in df_clean.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in dataset columns: {df_clean.columns.tolist()}")

    # Ensure power values are non-negative
    df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].clip(lower=0.0)

    if verbose:
        print(f"[RenewAI Preprocess] Preprocessing complete. Shape: {df_clean.shape}")
        print(f"[RenewAI Preprocess] Features available: {FEATURE_COLUMNS}")
        print(f"[RenewAI Preprocess] Target variable: '{TARGET_COLUMN}'")

    return df_clean


def get_feature_target_matrices(df: pd.DataFrame):
    """Extract feature matrix X and target series y."""
    missing_feats = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_feats:
        raise KeyError(f"Missing required feature columns: {missing_feats}")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def split_time_series(df: pd.DataFrame, test_size: float = 0.2):
    """Strict chronological time-series train/test split (80% train, 20% test).
    No shuffling to prevent data leakage in temporal forecasting.
    """
    X, y = get_feature_target_matrices(df)
    n_samples = len(df)
    split_idx = int(n_samples * (1.0 - test_size))

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()
    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    train_timestamps = df["TimeStamp"].iloc[:split_idx]
    test_timestamps = df["TimeStamp"].iloc[split_idx:]

    print(f"[RenewAI Split] Train samples: {len(X_train)} ({train_timestamps.min()} to {train_timestamps.max()})")
    print(f"[RenewAI Split] Test samples:  {len(X_test)} ({test_timestamps.min()} to {test_timestamps.max()})")

    return X_train, X_test, y_train, y_test, train_timestamps, test_timestamps


def generate_exploratory_visualizations(df: pd.DataFrame, output_dir: str = OUTPUT_DIR):
    """Generate and save standard exploratory plots for hackathon reporting."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # 1. Power Generation Over Time
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["TimeStamp"], df[TARGET_COLUMN], color="#2563EB", linewidth=1, alpha=0.85)
    ax.set_title("RenewAI: Solar Power Generation Over Time (Lahore Single Panel)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Timestamp", fontsize=11)
    ax.set_ylabel("Power Output (W)", fontsize=11)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "power_over_time.png")
    fig.savefig(plot1_path, dpi=200)
    plt.close(fig)
    print(f"[RenewAI Viz] Saved: {plot1_path}")

    # 2. Irradiance vs Power
    fig, ax = plt.subplots(figsize=(8, 6))
    if "Irradiance (G)" in df.columns:
        scatter = ax.scatter(df["Irradiance (G)"], df[TARGET_COLUMN], c=df["Hour"], cmap="viridis", alpha=0.6, s=18)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Hour of Day", fontsize=10)
        ax.set_xlabel("Global Irradiance (G) [W/m²]", fontsize=11)
    else:
        ax.scatter(range(len(df)), df[TARGET_COLUMN], alpha=0.5)
        ax.set_xlabel("Index")
    ax.set_title("Solar Power Output vs. Global Irradiance", fontsize=13, fontweight="bold")
    ax.set_ylabel("Power Output (W)", fontsize=11)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "irradiance_vs_power.png")
    fig.savefig(plot2_path, dpi=200)
    plt.close(fig)
    print(f"[RenewAI Viz] Saved: {plot2_path}")

    # 3. Cloud vs Power
    if "Cloud" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["Cloud"], df[TARGET_COLUMN], color="#059669", alpha=0.5, s=20)
        ax.set_title("Solar Power Output vs. Cloud Index / Fraction", fontsize=13, fontweight="bold")
        ax.set_xlabel("Cloud Index (0.0 = Clear Sky, 1.0 = Overcast)", fontsize=11)
        ax.set_ylabel("Power Output (W)", fontsize=11)
        plt.tight_layout()
        plot3_path = os.path.join(output_dir, "cloud_vs_power.png")
        fig.savefig(plot3_path, dpi=200)
        plt.close(fig)
        print(f"[RenewAI Viz] Saved: {plot3_path}")

    # 4. Temperature vs Power
    if "Temperature" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df["Temperature"], df[TARGET_COLUMN], color="#DC2626", alpha=0.5, s=20)
        ax.set_title("Solar Power Output vs. Ambient Temperature (°C)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Ambient Temperature (°C)", fontsize=11)
        ax.set_ylabel("Power Output (W)", fontsize=11)
        plt.tight_layout()
        plot4_path = os.path.join(output_dir, "temperature_vs_power.png")
        fig.savefig(plot4_path, dpi=200)
        plt.close(fig)
        print(f"[RenewAI Viz] Saved: {plot4_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("      RENEWAI - PHASE 1: PREPROCESSING & EXPLORATION")
    print("=" * 60)
    raw_df = load_raw_data()
    clean_df = preprocess_data(raw_df)
    clean_df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"[RenewAI Preprocess] Cleaned dataset saved to: {CLEANED_DATA_PATH}")
    generate_exploratory_visualizations(clean_df)
    print("=" * 60)
    print("PHASE 1 COMPLETE: DATA PIPELINE OPERATIONAL")
    print("=" * 60)
