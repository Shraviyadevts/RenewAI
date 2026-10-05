"""RenewAI – AI-Powered Renewable Energy Forecasting & Decision Support System
=============================================================================
Main CLI Runner & Demonstration Pipeline.
Demonstrates live Open-Meteo rolling 24-hour solar forecasting for Lahore,
dynamic surplus classification, opportunity windows, prescriptive recommendations,
and live What-If scenario simulations.
"""

import os
import sys
import argparse
from datetime import datetime

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.train import train_solar_model
from ml.predict import load_trained_model, predict_single, predict_batch
from ml.opportunity import detect_opportunity_window
from ml.recommendation import get_recommendation, run_what_if_simulation
from backend.weather_service import get_live_24h_forecast_dataframe, fetch_open_meteo_raw


def run_live_pipeline(demand: float = 60.0, battery_soc: float = 50.0):
    """Execute the complete live rolling 24-hour RenewAI pipeline."""
    print("\n" + "=" * 65)
    print("      RENEWAI: LIVE 24-HOUR RENEWABLE FORECASTING SYSTEM")
    print("=" * 65)

    # 1. Check Model
    print("\n[STEP 1] Loading Trained Solar Forecasting Model...")
    bundle = load_trained_model()
    metrics = bundle["metrics"]
    print(f"[OK] Model Type: {bundle['model_type']}")
    print(f"[OK] Historical Holdout Benchmark: R2 = {metrics['test_r2']:.4f} | MAE = {metrics['test_mae']:.2f} W | RMSE = {metrics['test_rmse']:.2f} W")

    # 2. Live Weather API Fetch
    print("\n[STEP 2] Fetching Live Meteorological Forecast for Lahore, Pakistan...")
    df_24, meta = get_live_24h_forecast_dataframe(force_refresh=True)
    print(f"[OK] Coordinates: {meta['coordinates']} | Timezone: {meta['timezone']}")
    print(f"[OK] Last Updated: {meta['last_updated']}")
    print(f"[OK] Rolling Horizon: {len(df_24)} hours ({df_24['timestamp'].min()} to {df_24['timestamp'].max()})")

    # 3. Rolling 24-Hour Predictions
    print("\n[STEP 3] Generating Live 24-Hour Solar Power Predictions...")
    pred_df = predict_batch(df_24, bundle=bundle)
    curr_row = pred_df.iloc[0]
    curr_power = float(curr_row["Predicted_Power"])
    curr_rel = float(curr_row["Reliability_Score"])
    peak_power = float(pred_df["Predicted_Power"].max())
    peak_idx = pred_df["Predicted_Power"].idxmax()
    peak_time = pred_df.loc[peak_idx, "timestamp"].strftime("%I:%M %p (%a)")

    print(f"[OK] Current Solar Estimate ({curr_row['timestamp'].strftime('%I:%M %p')}): {curr_power:.1f} W (Reliability: {curr_rel:.1f}%)")
    print(f"[OK] Peak Solar Forecast in Horizon: {peak_power:.1f} W at {peak_time}")

    # 4. Opportunity Window Detection
    print(f"\n[STEP 4] Detecting Renewable Opportunity Window (Expected Demand: {demand:.1f} W)...")
    opp = detect_opportunity_window(pred_df, demand=demand)
    if opp["has_opportunity"]:
        print(f"[OK] Best Opportunity Window: {opp['window_text']}")
        print(f"[OK] Peak Surplus:           +{opp['peak_surplus']:.1f} W at {opp['peak_time_str']}")
        print(f"[OK] Total Net Clean Energy:  {opp['total_surplus_wh']:.1f} Wh over {opp['surplus_hours']} hours")
        print(f"[OK] Deficit Duration:        {opp['deficit_hours']} hours")
    else:
        print(f"[OK] Opportunity Status: {opp['window_text']}")

    # 5. Prescriptive Recommendation & Why Explanation
    print("\n[STEP 5] Generating Prescriptive Decision & 'Why?' Deduction...")
    rec = get_recommendation(
        predicted_power=curr_power,
        demand=demand,
        battery_soc=battery_soc,
        reliability_score=curr_rel
    )
    print(f"[OK] Recommended Action: {rec['action']}")
    print(f"[OK] Operational Reason: {rec['reason']}")
    print(f"[OK] Deduction Formula:  {rec['why']['logic_formula']}")

    # 6. Live What-If Simulation
    print("\n[STEP 6] Running Live What-If Scenario Simulation (Surge Cloud to 85%, Low Battery 20%)...")
    base_feat = dict(curr_row)
    base_feat["Demand"] = demand
    base_feat["Battery_SOC"] = battery_soc
    sim = run_what_if_simulation(
        base_features=base_feat,
        new_cloud_pct=85.0,
        new_demand_w=demand + 40.0,
        new_battery_soc_pct=20.0,
        bundle=bundle
    )
    print(f"[OK] Baseline Condition  -> Power: {sim['base_scenario']['predicted_power']:.1f} W | Action: {sim['base_scenario']['recommendation']['action']}")
    print(f"[OK] Simulated Scenario  -> Power: {sim['simulated_scenario']['predicted_power']:.1f} W | Action: {sim['simulated_scenario']['recommendation']['action']}")
    print(f"[OK] Transition Reason:  {sim['comparison']['why_change']}")

    print("\n" + "=" * 65)
    print("      LIVE RENEWAI PIPELINE VERIFIED SUCCESSFULLY")
    print("=" * 65)
    print("\nTo launch live user interfaces:")
    print("  * Streamlit Dashboard: streamlit run frontend/app.py")
    print("  * FastAPI REST API:   uvicorn backend.main:app --reload")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RenewAI Live Pipeline Runner")
    parser.add_argument("--demand", type=float, default=60.0, help="Expected load demand in Watts (default: 60.0)")
    parser.add_argument("--battery", type=float, default=50.0, help="Current battery SOC percentage (default: 50.0)")
    args = parser.parse_args()

    run_live_pipeline(demand=args.demand, battery_soc=args.battery)
