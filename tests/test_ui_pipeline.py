import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.weather_service import get_live_24h_forecast_dataframe
from ml.predict import load_trained_model, predict_batch
from ml.demand import generate_24h_demand_profile
from ml.battery import simulate_battery_trajectory
from ml.opportunity import detect_opportunity_window, classify_energy_balance

bundle = load_trained_model()
df_24, meta = get_live_24h_forecast_dataframe()
pred_df = predict_batch(df_24, bundle=bundle)

demand_curve = generate_24h_demand_profile(pred_df['timestamp'], baseline_demand_w=60.0)
pred_df['Demand'] = demand_curve
pred_df['Surplus'] = pred_df['Predicted_Power'] - pred_df['Demand']
pred_df['Surplus_Category'] = pred_df['Surplus'].apply(classify_energy_balance)

batt_sim = simulate_battery_trajectory(initial_soc_pct=50.0, capacity_wh=500.0, surplus_series=pred_df['Surplus'].values)
pred_df['Projected_Battery_SOC'] = batt_sim['soc_trajectory_pct']

table_rows = []
for _, row in pred_df.iterrows():
    table_rows.append({
        'Time': row['timestamp'].strftime('%a %I:%M %p'),
        'Solar Forecast': f"{row['Predicted_Power']:.1f} W",
        'Demand': f"{row['Demand']:.1f} W",
        'Surplus / Deficit': f"{row['Surplus']:+.1f} W",
        'Tier': row['Surplus_Category'],
        'Battery SOC': f"{row['Projected_Battery_SOC']:.0f}%"
    })

df_table = pd.DataFrame(table_rows)
print("SUCCESS: Timeline table verified with", len(df_table), "rows.")
print(df_table.head(2))
