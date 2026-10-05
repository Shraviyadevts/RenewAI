import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.weather_service import geocode_location, reverse_geocode_coordinates, get_live_24h_forecast_dataframe
from ml.predict import load_trained_model, predict_batch
from ml.demand import generate_24h_demand_profile
from ml.battery import simulate_battery_trajectory
from ml.opportunity import detect_opportunity_window, classify_energy_balance
from ml.risk import assess_energy_risk
from ml.recommendation import get_recommendation, USER_ROLES

print("=== 1. TESTING LOCATION VERIFICATION SUITE ===")
test_places = [
    'Kerala', 'Tamil Nadu', 'Karnataka', 'Maharashtra', 'Delhi',
    'Chennai', 'Bengaluru', 'Mumbai', 'Kochi', 'Coimbatore', 'Hyderabad',
    'London', 'Springfield', 'Tokyo', 'Sydney', 'Cairo'
]

kerala_ok = False
for p in test_places:
    res = geocode_location(p, limit=3)
    assert len(res) > 0, f"Failed to geocode {p}"
    top = res[0]
    disp_clean = top['display_name'].encode('ascii', 'ignore').decode()
    print(f"Query: {p:<15} -> {disp_clean} (Lat: {top['latitude']}, Lon: {top['longitude']}) [{top['country_code']}]")
    if p == 'Kerala':
        assert top['country_code'] == 'IN' and 'India' in top['country'], f"Kerala geocoded incorrectly: {top}"
        kerala_ok = True

assert kerala_ok, "Kerala validation failed!"
print("\n[PASSED] Kerala accurately resolved to Kerala, India!")

print("\n=== 2. TESTING REVERSE GEOCODING ===")
rev = reverse_geocode_coordinates(9.9312, 76.2673)
print("Reverse Geocode (9.9312, 76.2673):", rev['display_name'])
assert 'Kerala' in rev['state'] or 'Kochi' in rev['city']

print("\n=== 3. TESTING END-TO-END DATA FLOW FOR KERALA ===")
bundle = load_trained_model()
k_loc = geocode_location('Kerala')[0]
df_24, meta = get_live_24h_forecast_dataframe(
    lat=k_loc['latitude'],
    lon=k_loc['longitude'],
    city_name=k_loc['city'],
    state_name=k_loc['state'],
    country_name=k_loc['country']
)
assert len(df_24) == 24, f"Expected 24 rows, got {len(df_24)}"
pred_df = predict_batch(df_24, bundle=bundle)
demand_curve = generate_24h_demand_profile(pred_df['timestamp'], baseline_demand_w=75.0, profile_type='⚡ Grid Feeder Curve')
pred_df['Demand'] = demand_curve
pred_df['Surplus'] = pred_df['Predicted_Power'] - pred_df['Demand']
pred_df['Surplus_Category'] = pred_df['Surplus'].apply(classify_energy_balance)

batt_sim = simulate_battery_trajectory(initial_soc_pct=60.0, capacity_wh=600.0, surplus_series=pred_df['Surplus'].values)
opp = detect_opportunity_window(pred_df, demand=demand_curve)
risk = assess_energy_risk(predicted_power=pred_df.iloc[0]['Predicted_Power'], demand=demand_curve[0], battery_soc=60.0, reliability_score=pred_df.iloc[0]['Reliability_Score'])
rec = get_recommendation(predicted_power=pred_df.iloc[0]['Predicted_Power'], demand=demand_curve[0], battery_soc=60.0, reliability_score=pred_df.iloc[0]['Reliability_Score'], user_role='⚡ Grid Operator')

print(f"Location: {meta['location']} | Timezone: {meta['timezone']}")
print(f"Solar Generation: {pred_df.iloc[0]['Predicted_Power']:.1f} W | Demand: {demand_curve[0]:.1f} W | Surplus: {pred_df.iloc[0]['Surplus']:+.1f} W")
print(f"Opportunity Window: {opp['window_text']} | Net Clean Wh: {opp['total_surplus_wh']:.1f} Wh")
print(f"Risk Level: {risk['level']} | Mitigation: {risk['mitigation']}")
print(f"Action: {rec['action']} | Priority: {rec['priority']} | Reason: {rec['reason']}")

print("\n=== ALL REGRESSION & ACCURACY TESTS PASSED 100% ===")
