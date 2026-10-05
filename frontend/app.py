import os
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocess import FEATURE_COLUMNS
from ml.predict import load_trained_model, predict_batch, predict_single
from ml.demand import generate_24h_demand_profile, DEMAND_PROFILES
from ml.battery import simulate_battery_trajectory
from ml.opportunity import detect_opportunity_window, classify_energy_balance
from ml.risk import assess_energy_risk
from ml.recommendation import get_recommendation, run_what_if_simulation, USER_ROLES
from backend.weather_service import (
    get_live_24h_forecast_dataframe,
    geocode_location,
    reverse_geocode_coordinates,
    get_auto_detected_location
)

# Page Configuration
st.set_page_config(
    page_title="RenewAI - Live Renewable Operations Platform",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

PRESET_LOCATIONS = {
    "Chennai, Tamil Nadu, India": {"lat": 13.0827, "lon": 80.2707, "city": "Chennai", "state": "Tamil Nadu", "country": "India", "tz": "Asia/Kolkata"},
    "Bengaluru, Karnataka, India": {"lat": 12.9716, "lon": 77.5946, "city": "Bengaluru", "state": "Karnataka", "country": "India", "tz": "Asia/Kolkata"},
    "Kochi, Kerala, India": {"lat": 9.9312, "lon": 76.2673, "city": "Kochi", "state": "Kerala", "country": "India", "tz": "Asia/Kolkata"},
    "Mumbai, Maharashtra, India": {"lat": 19.0760, "lon": 72.8777, "city": "Mumbai", "state": "Maharashtra", "country": "India", "tz": "Asia/Kolkata"},
    "Delhi, India": {"lat": 28.6139, "lon": 77.2090, "city": "Delhi", "state": "Delhi", "country": "India", "tz": "Asia/Kolkata"},
    "London, England, UK": {"lat": 51.5074, "lon": -0.1278, "city": "London", "state": "England", "country": "UK", "tz": "Europe/London"},
    "Dubai, UAE": {"lat": 25.2048, "lon": 55.2708, "city": "Dubai", "state": "Dubai", "country": "UAE", "tz": "Asia/Dubai"},
    "New York, USA": {"lat": 40.7128, "lon": -74.0060, "city": "New York", "state": "New York", "country": "USA", "tz": "America/New_York"},
    "Tokyo, Japan": {"lat": 35.6762, "lon": 139.6503, "city": "Tokyo", "state": "Tokyo", "country": "Japan", "tz": "Asia/Tokyo"},
    "Sydney, NSW, Australia": {"lat": -33.8688, "lon": 151.2093, "city": "Sydney", "state": "New South Wales", "country": "Australia", "tz": "Australia/Sydney"}
}


@st.cache_resource(show_spinner=False)
def get_model_bundle():
    """Load and cache the trained XGBoost model bundle."""
    return load_trained_model()


def main():
    # Load ML Model Bundle
    try:
        bundle = get_model_bundle()
    except Exception as e:
        st.error(f"⚠️ Error loading ML model: {e}")
        st.info("Please run `python ml/train.py` first.")
        return

    # Check for Geolocation query parameters (from Browser Geolocation API)
    query_params = st.query_params
    if "lat" in query_params and "lon" in query_params:
        try:
            p_lat = float(query_params["lat"])
            p_lon = float(query_params["lon"])
            rev_loc = reverse_geocode_coordinates(p_lat, p_lon)
            st.session_state["selected_location"] = rev_loc
        except Exception:
            pass

    # Initialize Single Authoritative Location State
    if "selected_location" not in st.session_state:
        st.session_state["selected_location"] = get_auto_detected_location()

    # ====================================================================
    # 1. SIDEBAR: OPERATIONAL CONTROL CENTER
    # ====================================================================
    st.sidebar.title("☀️ RenewAI")
    st.sidebar.caption("Live Renewable Operations Center")
    st.sidebar.markdown("---")

    # 📍 Location Configuration
    st.sidebar.subheader("📍 Location Settings")
    loc_mode = st.sidebar.radio(
        "Location Method:",
        options=["🔍 Search City / State Worldwide", "📍 Preset Global Hubs", "🌐 Auto-Detect IP", "🎯 Custom Coordinates"],
        index=0
    )

    current_loc = st.session_state["selected_location"]
    selected_lat = current_loc.get("latitude", 12.9716)
    selected_lon = current_loc.get("longitude", 77.5946)
    selected_city = current_loc.get("city", "Bengaluru")
    selected_state = current_loc.get("state", "Karnataka")
    selected_country = current_loc.get("country", "India")
    selected_tz = current_loc.get("timezone", "auto")

    if loc_mode == "🔍 Search City / State Worldwide":
        search_query = st.sidebar.text_input("Enter City, State, or Region:", placeholder="e.g. Kerala, Chennai, Texas, Paris", key="loc_search_input")
        if search_query:
            results = geocode_location(search_query, limit=8)
            if results:
                options_map = {r["display_name"]: r for r in results}
                chosen_search = st.sidebar.selectbox("Select Verified Location:", list(options_map.keys()), key="loc_search_select")
                sel_res = options_map[chosen_search]
                st.session_state["selected_location"] = sel_res
                selected_lat = sel_res["latitude"]
                selected_lon = sel_res["longitude"]
                selected_city = sel_res["city"]
                selected_state = sel_res["state"]
                selected_country = sel_res["country"]
                selected_tz = sel_res["timezone"]
            else:
                st.sidebar.warning(f"No locations found for '{search_query}'. Please verify spelling.")

    elif loc_mode == "📍 Preset Global Hubs":
        chosen_preset = st.sidebar.selectbox("Select Operational Hub:", list(PRESET_LOCATIONS.keys()), index=0)
        p_info = PRESET_LOCATIONS[chosen_preset]
        sel_res = {
            "city": p_info["city"],
            "state": p_info["state"],
            "country": p_info["country"],
            "country_code": "",
            "latitude": p_info["lat"],
            "longitude": p_info["lon"],
            "timezone": p_info["tz"],
            "display_name": chosen_preset
        }
        st.session_state["selected_location"] = sel_res
        selected_lat, selected_lon, selected_city, selected_state, selected_country, selected_tz = p_info["lat"], p_info["lon"], p_info["city"], p_info["state"], p_info["country"], p_info["tz"]

    elif loc_mode == "🌐 Auto-Detect IP":
        if st.sidebar.button("📡 Re-Detect Current Location", use_container_width=True):
            auto_loc = get_auto_detected_location()
            st.session_state["selected_location"] = auto_loc
            selected_lat = auto_loc["latitude"]
            selected_lon = auto_loc["longitude"]
            selected_city = auto_loc["city"]
            selected_state = auto_loc["state"]
            selected_country = auto_loc["country"]
            selected_tz = auto_loc["timezone"]
            st.sidebar.success(f"Detected: {auto_loc['display_name']}")

    elif loc_mode == "🎯 Custom Coordinates":
        c_lat = st.sidebar.number_input("Latitude (°)", value=selected_lat, format="%.4f")
        c_lon = st.sidebar.number_input("Longitude (°)", value=selected_lon, format="%.4f")
        if st.sidebar.button("Resolve Coordinates"):
            rev_loc = reverse_geocode_coordinates(c_lat, c_lon)
            st.session_state["selected_location"] = rev_loc
            selected_lat, selected_lon, selected_city, selected_state, selected_country = rev_loc["latitude"], rev_loc["longitude"], rev_loc["city"], rev_loc["state"], rev_loc["country"]

    # Active Location Pill in Sidebar
    active_loc_name = st.session_state["selected_location"].get("display_name", f"{selected_city}, {selected_country}")
    st.sidebar.info(f"📍 **Active Site:**\n`{active_loc_name}`")

    st.sidebar.markdown("---")

    # 👤 Operating Facility Role
    st.sidebar.subheader("👤 Operating Facility Role")
    selected_role = st.sidebar.selectbox(
        "Select User Role:",
        options=USER_ROLES,
        index=0,
        help="Adapts all decision recommendations, risk mitigation, and dispatch priorities to your facility type."
    )

    st.sidebar.markdown("---")

    # ⚡ Dynamic Demand Configuration
    st.sidebar.subheader("⚡ Dynamic Load Demand")
    demand_profile_type = st.sidebar.selectbox(
        "Demand Profile Type:",
        options=list(DEMAND_PROFILES.keys()),
        index=0,
        help="Applies realistic time-of-day diurnal load variations."
    )

    baseline_demand_val = st.sidebar.slider(
        "Baseline Load Demand (Watts)",
        min_value=10.0,
        max_value=300.0,
        value=60.0,
        step=5.0,
        help="Base nominal load to be scaled by the diurnal curve [USER-DEFINED]."
    )

    st.sidebar.markdown("---")

    # 🔋 Dynamic Battery Storage Configuration
    st.sidebar.subheader("🔋 Battery Energy Storage")
    battery_soc_val = st.sidebar.slider(
        "Initial Battery SOC (%)",
        min_value=5.0,
        max_value=100.0,
        value=50.0,
        step=5.0,
        help="Current starting battery State of Charge."
    )

    battery_cap_val = st.sidebar.slider(
        "Battery Capacity (Wh)",
        min_value=100.0,
        max_value=2000.0,
        value=500.0,
        step=50.0,
        help="Total storage capacity of connected battery system."
    )

    st.sidebar.markdown("---")
    force_refresh = st.sidebar.button("🔄 Refresh Live Forecast Now", use_container_width=True)

    # ====================================================================
    # 2. FETCH LIVE DATA & GENERATE 24-HOUR ROLLING FORECAST
    # ====================================================================
    try:
        df_24, meta = get_live_24h_forecast_dataframe(
            lat=selected_lat,
            lon=selected_lon,
            city_name=selected_city,
            state_name=selected_state,
            country_name=selected_country,
            tz=selected_tz,
            force_refresh=force_refresh
        )
        pred_df = predict_batch(df_24, bundle=bundle)
    except Exception as e:
        st.error(f"⚠️ Live Weather Service Unavailable: {e}")
        st.warning("Last successfully retrieved forecast or fallback cache will be attempted.")
        if st.button("🔄 Retry Live Connection"):
            st.rerun()
        return

    # 1. Dynamic 24-Hour Demand Curve
    demand_curve = generate_24h_demand_profile(
        timestamps=pred_df["timestamp"],
        baseline_demand_w=baseline_demand_val,
        profile_type=demand_profile_type
    )
    pred_df["Demand"] = demand_curve
    pred_df["Surplus"] = pred_df["Predicted_Power"] - pred_df["Demand"]
    pred_df["Surplus_Category"] = pred_df["Surplus"].apply(classify_energy_balance)

    # 2. Dynamic 24-Hour Battery Simulation
    batt_sim = simulate_battery_trajectory(
        initial_soc_pct=battery_soc_val,
        capacity_wh=battery_cap_val,
        surplus_series=pred_df["Surplus"].values
    )
    pred_df["Projected_Battery_SOC"] = batt_sim["soc_trajectory_pct"]
    pred_df["Battery_Power_Flow"] = batt_sim["battery_power_flow_w"]

    # 3. Dynamic Opportunity Window & Energy Balance
    opp = detect_opportunity_window(pred_df, demand=demand_curve)

    # Current Hour Values
    curr_row = pred_df.iloc[0]
    curr_pred = float(curr_row["Predicted_Power"])
    curr_demand = float(curr_row["Demand"])
    curr_surplus = float(curr_row["Surplus"])
    curr_rel = float(curr_row["Reliability_Score"])
    curr_weather = meta.get("current_weather", {})
    conn_status = meta.get("connection_status", "LIVE")

    # Current Role Recommendation
    curr_rec = get_recommendation(
        predicted_power=curr_pred,
        demand=curr_demand,
        battery_soc=battery_soc_val,
        reliability_score=curr_rel,
        user_role=selected_role,
        battery_capacity_wh=battery_cap_val
    )

    # Energy Risk Assessment
    curr_risk = assess_energy_risk(
        predicted_power=curr_pred,
        demand=curr_demand,
        battery_soc=battery_soc_val,
        reliability_score=curr_rel,
        deficit_hours_in_horizon=opp.get("deficit_hours", 0),
        min_projected_soc=batt_sim.get("min_soc_pct")
    )

    # Next Hour & Peak
    next_row = pred_df.iloc[1] if len(pred_df) > 1 else curr_row
    next_pred = float(next_row["Predicted_Power"])
    peak_pred = float(pred_df["Predicted_Power"].max())
    peak_idx = pred_df["Predicted_Power"].idxmax()
    peak_time_str = pred_df.loc[peak_idx, "timestamp"].strftime("%I:%M %p (%a)")

    # Schedulers / Clock
    now_dt = datetime.now()
    last_updated_str = meta.get("last_updated", now_dt.strftime("%d %b %Y, %I:%M %p"))
    next_update_str = (now_dt + timedelta(minutes=10)).strftime("%I:%M %p")

    # ====================================================================
    # 3. TOP OPERATIONAL HEADER
    # ====================================================================
    c_head1, c_head2 = st.columns([3, 2])
    with c_head1:
        st.title("☀️ RenewAI")
        st.markdown(f"**AI-Powered Renewable Energy Operations & Decision Support Platform**  \n📍 **Location:** `{active_loc_name}` &bull; **Mode:** `{selected_role}`")
    with c_head2:
        st.info(f"● **{conn_status} 24-HOUR OPERATIONS**  \n**Updated:** {last_updated_str} | **Next Refresh:** {next_update_str}  \n**Live Weather:** {curr_weather.get('temperature_c', 25):.1f}°C, {curr_weather.get('cloud_cover_pct', 0)}% Cloud Cover")

    # Model Transparency Notice
    st.caption("🏷️ `[MODEL-DERIVED LIVE FORECAST]` &bull; *Generated using live Open-Meteo meteorological telemetry. Prediction confidence adapts to physical atmospheric clear-sky limits.*")

    with st.expander("ℹ️ Location & Weather Sensor Telemetry", expanded=False):
        c_l1, c_l2, c_l3, c_l4 = st.columns(4)
        c_l1.write(f"**Coordinates:** `{selected_lat:.4f}° N, {selected_lon:.4f}° E`")
        c_l2.write(f"**Timezone:** `{meta.get('timezone', 'auto')}`")
        c_l3.write(f"**Global Horizontal (GHI):** `{curr_weather.get('shortwave_radiation_wm2', 0.0)} W/m²`")
        c_l4.write(f"**Direct Beam (DNI):** `{curr_weather.get('direct_normal_irradiance_wm2', 0.0)} W/m²`")

    st.divider()

    # ====================================================================
    # 4. FIVE MAIN KPI CARDS (TRANSPARENT DATA CATEGORIZATION)
    # ====================================================================
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        with st.container(border=True):
            st.metric(
                label="Solar Generation",
                value=f"{curr_pred:.1f} W",
                delta=f"Peak: {peak_pred:.1f} W",
                help="AI Model-derived current solar power output based on live meteorological data."
            )
            st.caption("🏷️ `[MODEL-DERIVED]`")
    with m2:
        with st.container(border=True):
            st.metric(
                label="Current Demand",
                value=f"{curr_demand:.1f} W",
                delta=f"{curr_demand - baseline_demand_val:+.1f} W vs Base",
                help="Dynamic time-of-day load demand forecasted from facility operating curve."
            )
            st.caption("🏷️ `[FORECASTED DEMAND]`")
    with m3:
        with st.container(border=True):
            st.metric(
                label="Net Energy Balance",
                value=f"{curr_surplus:+.1f} W",
                delta="Surplus" if curr_surplus >= 0 else "Deficit",
                delta_color="normal" if curr_surplus >= 0 else "inverse",
                help="Net energy generation minus load demand."
            )
            st.caption("🏷️ `[ENERGY BALANCE]`")
    with m4:
        with st.container(border=True):
            st.metric(
                label="Battery Reserve (SOC)",
                value=f"{battery_soc_val:.0f}%",
                delta=f"Min 24h: {batt_sim['min_soc_pct']:.0f}%",
                help="Current battery State of Charge and 24-hour projected minimum reserve."
            )
            st.caption("🏷️ `[BATTERY MODEL]`")
    with m5:
        with st.container(border=True):
            st.metric(
                label="Forecast Confidence",
                value=f"{curr_rel:.0f}%",
                delta="High Reliability" if curr_rel >= 85 else "Moderate Confidence",
                help="Multi-factor forecast reliability index based on physical clear-sky radiation limits."
            )
            st.caption("🏷️ `[CONFIDENCE ENGINE]`")

    st.write("")

    # ====================================================================
    # 5. RECOMMENDED OPERATIONAL ACTION HERO CARD
    # ====================================================================
    with st.container(border=True):
        col_act1, col_act2 = st.columns([3, 1])
        with col_act1:
            st.subheader(f"🎯 Recommended Operational Action ({selected_role})")
        with col_act2:
            pri = curr_rec["priority"]
            st.warning(f"**PRIORITY: {pri}**")

        action_display = f"### {curr_rec['icon']} {curr_rec['action']}"
        if curr_rec["badge"] == "success":
            st.success(action_display)
        elif curr_rec["badge"] == "primary":
            st.info(action_display)
        elif curr_rec["badge"] == "warning":
            st.warning(action_display)
        else:
            st.error(action_display)

        st.markdown(f"**Operational Rationale:** {curr_rec['reason']}")
        st.markdown(f"⚡ **Expected System Impact:** `{curr_rec.get('expected_impact', 'Maintains system stability.')}`")

        # Target Dispatch Flow
        flow = curr_rec.get("suggested_flow_w", 0.0)
        flow_str = f"Target Dispatch: `+{flow:.0f} W (Charge/Export)`" if flow > 0 else (f"Target Dispatch: `{flow:.0f} W (Discharge/Import)`" if flow < 0 else "Target Dispatch: `0 W (Holding Reserve)`")
        st.caption(f"{flow_str} &bull; Profile: **{demand_profile_type}** &bull; Confidence: **{curr_rel}%**")

        # "Why This Action?" Mathematical Deduction Panel
        with st.expander("💡 Why This Action? (Mathematical Deduction Breakdown)", expanded=False):
            w = curr_rec["why"]
            cw1, cw2, cw3, cw4 = st.columns(4)
            cw1.metric("Predicted Solar Power", w["predicted_generation"])
            cw2.metric("Expected Load Demand", w["expected_demand"])
            cw3.metric("Net Surplus / Deficit", w["expected_surplus"])
            cw4.metric("Battery Reserve (SOC)", w["battery_soc"])

            st.markdown(f"""
            - **Evaluated Operational Rule:** `{w['rule_applied']}`
            - **Mathematical Deduction:** `{w['logic_formula']}` $\\longrightarrow$ **`{w['action']}`**
            """)

    st.write("")

    # ====================================================================
    # 6. INTERACTIVE 24-HOUR SOLAR, DEMAND & BATTERY SOC FORECAST CHART
    # ====================================================================
    st.subheader("📈 Rolling 24-Hour Energy Balance & Battery SOC Forecast")
    st.caption(f"Continuous 24-hour horizon showing AI Solar generation, dynamic load demand, and projected battery trajectory for **{active_loc_name}**.")

    if PLOTLY_AVAILABLE:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Solar Generation vs. Dynamic Load Demand (Watts)", "Projected Battery Storage State of Charge (%)"),
            row_heights=[0.65, 0.35]
        )

        # Solar Power Forecast
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=pred_df["Predicted_Power"],
            name="AI Solar Forecast (W)",
            line=dict(color="#F59E0B", width=3),
            mode="lines+markers",
            marker=dict(size=4)
        ), row=1, col=1)

        # Dynamic Demand Curve
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=pred_df["Demand"],
            name="Dynamic Demand (W)",
            line=dict(color="#10B981", width=2, dash="dash"),
            mode="lines"
        ), row=1, col=1)

        # Battery SOC Trajectory
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=pred_df["Projected_Battery_SOC"],
            name="Projected Battery SOC (%)",
            line=dict(color="#3B82F6", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.15)",
            mode="lines"
        ), row=2, col=1)

        # Minimum Safe Reserve Threshold Line (20%)
        fig.add_trace(go.Scatter(
            x=pred_df["timestamp"],
            y=[20.0] * len(pred_df),
            name="Min Reserve Guard (20%)",
            line=dict(color="#EF4444", width=1.5, dash="dot"),
            mode="lines"
        ), row=2, col=1)

        fig.update_layout(
            height=480,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig.update_yaxes(title_text="Power (Watts)", row=1, col=1, showgrid=True)
        fig.update_yaxes(title_text="SOC (%)", range=[0, 105], row=2, col=1, showgrid=True)
        fig.update_xaxes(title_text="Rolling Horizon Time (Local Time)", row=2, col=1, showgrid=True)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(pred_df[["Predicted_Power", "Demand"]])

    # ====================================================================
    # 7. ENERGY RISK & RENEWABLE OPPORTUNITY PANELS (2 COLUMNS)
    # ====================================================================
    c_risk, c_opp = st.columns(2)

    with c_risk:
        with st.container(border=True):
            st.subheader("⚠️ Energy Risk Assessment")
            r_level = curr_risk["level"]
            if curr_risk["badge"] == "danger":
                st.error(f"### {curr_risk['icon']} {r_level}")
            elif curr_risk["badge"] == "warning":
                st.warning(f"### {curr_risk['icon']} {r_level}")
            else:
                st.success(f"### {curr_risk['icon']} {r_level}")

            st.write(f"**Risk Cause:** {curr_risk['reason']}")
            st.info(f"🛡️ **Mitigation Strategy:** {curr_risk['mitigation']}")
            st.caption(f"Risk Score: **{curr_risk['score']}/100** &bull; Deficit Duration: **{opp.get('deficit_hours', 0)} hours** &bull; Min Projected SOC: **{batt_sim['min_soc_pct']:.0f}%**")

    with c_opp:
        with st.container(border=True):
            st.subheader("✨ Renewable Opportunity Window")
            if opp["has_opportunity"]:
                st.markdown(f"### 🕒 {opp['window_text']}")
                st.write(f"- **Peak Clean Surplus:** `+{opp['peak_surplus']} W` at **{opp['peak_time_str']}**")
                st.write(f"- **Total Net Clean Excess:** `{opp['total_surplus_wh']} Wh` over **{opp['surplus_hours']} hours**")
                st.write(f"- **Total Deficit Energy:** `{opp['total_deficit_wh']} Wh` over **{opp['deficit_hours']} hours**")
                st.write(f"- **Longest Continuous Surplus:** `{opp['longest_surplus_streak']} continuous hours`")
                if opp.get("recommended_uses"):
                    tags = " &bull; ".join([f"`{u}`" for u in opp["recommended_uses"]])
                    st.markdown(f"**Recommended Applications:** {tags}")
            else:
                st.markdown("### ⏳ No Net Surplus Window")
                st.write(f"Peak generation ({opp['peak_generation']} W) remains below demand curve.")
                st.caption(f"Total deficit duration: {opp['deficit_hours']}h ({opp['total_deficit_wh']} Wh). Utilize battery reserves or schedule grid import.")

    st.write("")

    # ====================================================================
    # 8. 24-HOUR OPERATIONAL TIMELINE TABLE
    # ====================================================================
    st.subheader("📋 24-Hour Operational Timeline Table")
    
    table_rows = []
    for _, row in pred_df.iterrows():
        p_val = float(row["Predicted_Power"])
        d_val = float(row["Demand"])
        surp = float(row["Surplus"])
        soc_val = float(row["Projected_Battery_SOC"])
        rel_val = float(row["Reliability_Score"])

        rec_h = get_recommendation(
            predicted_power=p_val,
            demand=d_val,
            battery_soc=soc_val,
            reliability_score=rel_val,
            user_role=selected_role,
            battery_capacity_wh=battery_cap_val
        )

        risk_h = assess_energy_risk(
            predicted_power=p_val,
            demand=d_val,
            battery_soc=soc_val,
            reliability_score=rel_val
        )

        # Weather icon proxy
        c_cover = float(row["raw_cloud_pct"])
        w_icon = "☀️" if p_val > 50 and c_cover < 25 else ("⛅" if p_val > 10 else ("🌙" if p_val <= 0.5 else "🌧️"))

        table_rows.append({
            "Time": row["timestamp"].strftime("%a %I:%M %p"),
            "Weather": w_icon,
            "Temp (°C)": f"{float(row['Temperature']):.1f}",
            "Solar Forecast": f"{p_val:.1f} W",
            "Demand": f"{d_val:.1f} W",
            "Surplus / Deficit": f"{surp:+.1f} W",
            "Tier": row["Surplus_Category"],
            "Battery SOC": f"{soc_val:.0f}%",
            "Risk": risk_h["level"].replace(" RISK", ""),
            "Recommended Action": f"{rec_h['icon']} {rec_h['action']}",
            "Priority": rec_h["priority"]
        })

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.write("")

    # ====================================================================
    # 9. LIVE WHAT-IF SCENARIO SIMULATOR
    # ====================================================================
    st.divider()
    st.subheader("🧪 Operational What-If Scenario Simulator")
    st.caption(f"Simulate weather and load stress tests against the live baseline for **{selected_role}**.")

    base_features = {
        "Temperature": float(curr_row["Temperature"]),
        "Prectotland": float(curr_row["Prectotland"]),
        "Rhoa": float(curr_row["Rhoa"]),
        "Irradiance (G)": float(curr_row["Irradiance (G)"]),
        "Irradiance (A)": float(curr_row["Irradiance (A)"]),
        "Cloud": float(curr_row["Cloud"]),
        "Hour": int(curr_row["Hour"]),
        "Day": int(curr_row["Day"]),
        "Month": int(curr_row["Month"]),
        "DayOfWeek": int(curr_row["DayOfWeek"]),
        "Demand": curr_demand,
        "Battery_SOC": battery_soc_val,
        "raw_pressure_hpa": float(curr_row.get("raw_pressure_hpa", 980.0))
    }

    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        sim_cloud = st.slider(
            "Simulated Cloud Cover (%)",
            min_value=0,
            max_value=100,
            value=int(float(curr_row["Cloud"]) * 100),
            step=5,
            key="sim_cloud_slider"
        )
    with w_col2:
        sim_temp = st.slider(
            "Simulated Temp (°C)",
            min_value=5.0,
            max_value=45.0,
            value=float(curr_row["Temperature"]),
            step=1.0,
            key="sim_temp_slider"
        )
    with w_col3:
        sim_demand = st.slider(
            "Simulated Demand (W)",
            min_value=10.0,
            max_value=300.0,
            value=curr_demand,
            step=5.0,
            key="sim_demand_slider"
        )
    with w_col4:
        sim_battery = st.slider(
            "Simulated Battery SOC (%)",
            min_value=5.0,
            max_value=100.0,
            value=battery_soc_val,
            step=5.0,
            key="sim_battery_slider"
        )

    sim_res = run_what_if_simulation(
        base_features=base_features,
        new_cloud_pct=float(sim_cloud),
        new_demand_w=float(sim_demand),
        new_battery_soc_pct=float(sim_battery),
        new_temperature_c=float(sim_temp),
        user_role=selected_role,
        bundle=bundle
    )

    s1, s2 = st.columns(2)
    with s1:
        with st.container(border=True):
            st.markdown("#### 📌 Live Baseline Scenario")
            st.write(f"- **Cloud Cover:** {sim_res['base_scenario']['cloud_pct']}% | **Temp:** {base_features['Temperature']:.1f}°C")
            st.write(f"- **Predicted Solar:** {sim_res['base_scenario']['predicted_power']} W")
            st.write(f"- **Surplus / Deficit:** {sim_res['base_scenario']['surplus_w']:+.1f} W")
            st.info(f"**Action:** {sim_res['base_scenario']['recommendation']['icon']} {sim_res['base_scenario']['recommendation']['action']}")

    with s2:
        with st.container(border=True):
            st.markdown("#### ⚡ Simulated Scenario")
            st.write(f"- **Cloud Cover:** {sim_res['simulated_scenario']['cloud_pct']}% | **Temp:** {sim_temp:.1f}°C")
            st.write(f"- **Predicted Solar:** {sim_res['simulated_scenario']['predicted_power']} W ({sim_res['comparison']['power_change_w']:+.1f} W)")
            st.write(f"- **Surplus / Deficit:** {sim_res['simulated_scenario']['surplus_w']:+.1f} W")
            if sim_res['comparison']['action_changed']:
                st.warning(f"**New Action:** {sim_res['simulated_scenario']['recommendation']['icon']} {sim_res['simulated_scenario']['recommendation']['action']}")
            else:
                st.success(f"**Maintained Action:** {sim_res['simulated_scenario']['recommendation']['icon']} {sim_res['simulated_scenario']['recommendation']['action']}")

    st.markdown(f"**Why the decision transitioned:** {sim_res['comparison']['why_change']}")

    # ====================================================================
    # 10. MODEL & TECHNICAL DETAILS (ISOLATED AT BOTTOM)
    # ====================================================================
    st.divider()
    with st.expander("📚 Model Architecture & Historical Validation Benchmark", expanded=False):
        st.markdown("""
        > [!NOTE]
        > **Academic Separation:** The metrics below represent offline holdout benchmark evaluation of the trained **XGBoost Regressor** on the 2014 single-panel solar dataset (1,745 rows). These reflect historical test-set accuracy and are completely separated from the live rolling forecast above.
        """)

        metrics = bundle.get("metrics", {})
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Holdout R² Score", f"{metrics.get('test_r2', 0.7418):.4f}")
        col_m1.caption("Explains 74.2% of solar variance")
        col_m2.metric("Holdout MAE", f"{metrics.get('test_mae', 23.19):.2f} W")
        col_m2.caption("Mean absolute error")
        col_m3.metric("Holdout RMSE", f"{metrics.get('test_rmse', 49.49):.2f} W")
        col_m3.caption("Root mean squared error")
        col_m4.metric("Benchmark Dataset", "1,745 Rows (80/20 Split)")
        col_m4.caption("Lahore Single Panel Dataset")

        c_img1, c_img2 = st.columns(2)
        img_p1 = os.path.join(BASE_DIR, "outputs", "actual_vs_predicted.png")
        img_p2 = os.path.join(BASE_DIR, "outputs", "feature_importance.png")
        with c_img1:
            if os.path.exists(img_p1):
                st.image(img_p1, caption="Holdout Test Set: Actual vs Predicted Solar Generation (Oct-Dec 2014)")
        with c_img2:
            if os.path.exists(img_p2):
                st.image(img_p2, caption="Relative Feature Importance in Power Forecasting")


if __name__ == "__main__":
    main()
