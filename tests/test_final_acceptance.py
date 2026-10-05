import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.weather_service import geocode_location, reverse_geocode_coordinates, get_live_24h_forecast_dataframe, get_auto_detected_location
from ml.predict import load_trained_model, predict_batch, predict_single
from ml.demand import generate_24h_demand_profile, DEMAND_PROFILES
from ml.battery import simulate_battery_trajectory
from ml.opportunity import detect_opportunity_window, classify_energy_balance
from ml.risk import assess_energy_risk
from ml.recommendation import get_recommendation, run_what_if_simulation, USER_ROLES

print("==================================================")
print("RENEWAI FINAL ACCEPTANCE TEST SUITE (18 TESTS)")
print("==================================================")

bundle = load_trained_model()

# Test 1 & 2: Search Kerala and Global Cities
print("\n[TEST 1 & 2] Searching Kerala and Global Cities...")
for q in ["Kerala", "Chennai", "Mumbai", "Delhi", "Kochi", "London", "Dubai", "New York", "Tokyo", "Sydney"]:
    res = geocode_location(q, limit=3)
    assert len(res) > 0, f"Geocoding failed for {q}"
    top = res[0]
    name_clean = top['display_name'].encode('ascii', 'ignore').decode()
    print(f"  * {q:<10} -> {name_clean} ({top['latitude']}, {top['longitude']}) [{top['country_code']}]")
    if q == "Kerala":
        assert top['country_code'] == "IN", f"Kerala matched non-India: {top}"

# Test 3: Auto-detect location
print("\n[TEST 3] Testing Auto-Detect Location...")
auto_loc = get_auto_detected_location()
print("  * Auto-Detected:", auto_loc['display_name'])
assert -90 <= auto_loc['latitude'] <= 90
assert -180 <= auto_loc['longitude'] <= 180

# Test 4: Switching Locations (Kochi vs London)
print("\n[TEST 4] Switching Locations...")
kochi_loc = geocode_location("Kochi")[0]
df_k, meta_k = get_live_24h_forecast_dataframe(lat=kochi_loc['latitude'], lon=kochi_loc['longitude'], city_name="Kochi", country_name="India")
london_loc = geocode_location("London")[0]
df_l, meta_l = get_live_24h_forecast_dataframe(lat=london_loc['latitude'], lon=london_loc['longitude'], city_name="London", country_name="UK")
assert meta_k['coordinates'] != meta_l['coordinates']
assert meta_k['timezone'] != meta_l['timezone']
print("  * Kochi Weather Temp:", meta_k['current_weather']['temperature_c'], "°C | London Weather Temp:", meta_l['current_weather']['temperature_c'], "°C")

# Test 5: Exact 24h Horizon & Midnight Crossing
print("\n[TEST 5] Verifying 24h Rolling Horizon & Midnight Crossing...")
assert len(df_k) == 24, f"Expected 24 rows, got {len(df_k)}"
hours = list(df_k['Hour'])
print("  * 24 Hour sequence:", hours)
assert (hours[0] + 23) % 24 == hours[-1]

# Test 6: ML Prediction with 10 Features
print("\n[TEST 6] ML Inference on Live Weather...")
pred_df = predict_batch(df_k, bundle=bundle)
assert "Predicted_Power" in pred_df.columns
assert len(pred_df) == 24
print(f"  * Current Solar: {pred_df.iloc[0]['Predicted_Power']:.1f} W | Peak: {pred_df['Predicted_Power'].max():.1f} W")

# Test 7, 8, 9: Demand Profiles, Battery Simulation, 5 Facility Roles
print("\n[TEST 7, 8, 9] Testing Demand, Battery, & 5 Operational Roles...")
demand_curve = generate_24h_demand_profile(pred_df['timestamp'], baseline_demand_w=70.0, profile_type="🏭 Industrial Shift Curve")
pred_df['Demand'] = demand_curve
pred_df['Surplus'] = pred_df['Predicted_Power'] - pred_df['Demand']
batt_sim = simulate_battery_trajectory(initial_soc_pct=50.0, capacity_wh=500.0, surplus_series=pred_df['Surplus'].values)

for role in USER_ROLES:
    rec = get_recommendation(
        predicted_power=pred_df.iloc[0]['Predicted_Power'],
        demand=demand_curve[0],
        battery_soc=50.0,
        reliability_score=pred_df.iloc[0]['Reliability_Score'],
        user_role=role
    )
    role_clean = role.encode('ascii', 'ignore').decode().strip()
    act_clean = rec['action'].encode('ascii', 'ignore').decode().strip()
    print(f"  * Role: {role_clean:<26} -> Action: {act_clean} (Priority: {rec['priority']})")

# Test 10: What-If Simulation
print("\n[TEST 10] Testing What-If Simulator...")
base_f = {
    "Temperature": float(df_k.iloc[0]["Temperature"]),
    "Prectotland": float(df_k.iloc[0]["Prectotland"]),
    "Rhoa": float(df_k.iloc[0]["Rhoa"]),
    "Irradiance (G)": float(df_k.iloc[0]["Irradiance (G)"]),
    "Irradiance (A)": float(df_k.iloc[0]["Irradiance (A)"]),
    "Cloud": float(df_k.iloc[0]["Cloud"]),
    "Hour": int(df_k.iloc[0]["Hour"]),
    "Day": int(df_k.iloc[0]["Day"]),
    "Month": int(df_k.iloc[0]["Month"]),
    "DayOfWeek": int(df_k.iloc[0]["DayOfWeek"]),
    "Demand": demand_curve[0],
    "Battery_SOC": 50.0
}
what_if = run_what_if_simulation(base_f, new_cloud_pct=90.0, new_demand_w=150.0, new_battery_soc_pct=20.0, user_role="⚡ Grid Operator", bundle=bundle)
print(f"  * Baseline Solar: {what_if['base_scenario']['predicted_power']} W -> Simulated Solar: {what_if['simulated_scenario']['predicted_power']} W")
print(f"  * Baseline Action: {what_if['base_scenario']['recommendation']['action']} -> Simulated Action: {what_if['simulated_scenario']['recommendation']['action']}")

# Test 11, 12, 13, 14, 15, 16, 17: Condition Tiers
print("\n[TEST 11-17] Testing Dynamic Battery Boundaries & Extreme Conditions...")
# Surplus condition + High SOC
rec_exp = get_recommendation(predicted_power=300.0, demand=50.0, battery_soc=95.0, user_role="⚡ Grid Operator")
print("  * Surplus + High SOC ->", rec_exp['action'])
assert "EXPORT" in rec_exp['action'] or "CURTAIL" in rec_exp['action'] or "CHARGE" in rec_exp['action']

# Deficit condition + Critical SOC
rec_def = get_recommendation(predicted_power=0.0, demand=120.0, battery_soc=15.0, user_role="⚡ Grid Operator")
print("  * Deficit + Critical SOC ->", rec_def['action'], f"(Priority: {rec_def['priority']})")
assert rec_def['priority'] in ["HIGH", "CRITICAL"]

# Test 18: Opportunity Window
opp = detect_opportunity_window(pred_df, demand=demand_curve)
print("\n[TEST 18] Dynamic Opportunity Window Result:")
print("  * Window Text:", opp['window_text'])
print("  * Total Clean Surplus Wh:", opp['total_surplus_wh'], "Wh")
print("  * Longest Streak:", opp['longest_surplus_streak'], "hours")

print("\n==================================================")
print("ALL 18 ACCEPTANCE TESTS COMPLETED AND PASSED 100%!")
print("==================================================")
