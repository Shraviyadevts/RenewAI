"""RenewAI - Multi-Role Live Backend API Service
================================================
RESTful API for live global renewable energy forecasting, dynamic demand profiling,
24-hour battery SOC simulation, energy risk assessment, and role-based decision recommendations.
"""

import os
import sys
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocess import FEATURE_COLUMNS
from ml.predict import load_trained_model, predict_single, predict_batch
from ml.demand import generate_24h_demand_profile, DEMAND_PROFILES
from ml.battery import simulate_battery_trajectory
from ml.opportunity import detect_opportunity_window
from ml.risk import assess_energy_risk
from ml.recommendation import get_recommendation, run_what_if_simulation, USER_ROLES
from backend.weather_service import (
    get_live_24h_forecast_dataframe,
    fetch_open_meteo_raw,
    search_cities,
    get_auto_detected_location
)

app = FastAPI(
    title="RenewAI Operations API",
    description="AI-Powered Renewable Energy Forecasting & Decision Support API with Dynamic Geolocation & Battery Simulation",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====================================================================
# Request & Response Schemas
# ====================================================================

class PredictRequest(BaseModel):
    Temperature: float = Field(25.0, description="Ambient temperature in °C")
    Prectotland: float = Field(0.0, description="Precipitation in mm")
    Rhoa: float = Field(1.18, description="Air density (kg/m³)")
    Irradiance_G: float = Field(450.0, alias="Irradiance (G)", description="Global horizontal irradiance (W/m²)")
    Irradiance_A: float = Field(700.0, alias="Irradiance (A)", description="Direct normal irradiance (W/m²)")
    Cloud: float = Field(0.15, description="Cloud index fraction (0.0 clear to 1.0 overcast)")
    Hour: int = Field(12, ge=0, le=23, description="Hour of day (0-23)")
    Day: int = Field(21, ge=1, le=31, description="Day of month")
    Month: int = Field(8, ge=1, le=12, description="Month of year")
    DayOfWeek: int = Field(4, ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")

    class Config:
        populate_by_name = True


class RecommendationRequest(BaseModel):
    predicted_power: float = Field(..., ge=0.0, description="Forecasted solar power in Watts")
    demand: float = Field(60.0, ge=0.0, description="Expected load demand in Watts")
    battery_soc: float = Field(50.0, ge=0.0, le=100.0, description="Battery State of Charge (%)")
    reliability_score: float = Field(90.0, ge=0.0, le=100.0, description="Forecast reliability score (%)")
    user_role: str = Field("⚡ Grid Operator", description="Target facility operating role")


class WhatIfRequest(BaseModel):
    base_features: Dict[str, Any] = Field(..., description="Baseline meteorological and operating parameters")
    new_cloud_pct: float = Field(50.0, ge=0.0, le=100.0, description="Simulated cloud cover percentage (0-100%)")
    new_demand_w: float = Field(80.0, ge=0.0, description="Simulated load demand in Watts")
    new_battery_soc_pct: float = Field(30.0, ge=0.0, le=100.0, description="Simulated battery SOC percentage (0-100%)")
    new_temperature_c: Optional[float] = Field(None, description="Optional simulated temperature in °C")
    user_role: str = Field("⚡ Grid Operator", description="Target operational role")


# ====================================================================
# Endpoints
# ====================================================================

@app.get("/", tags=["General"])
def root():
    """System overview, version status, supported roles, and demand profiles."""
    return {
        "system": "RenewAI – Production Renewable Energy Operations Platform",
        "status": "online",
        "version": "3.0.0",
        "supported_roles": USER_ROLES,
        "demand_profiles": list(DEMAND_PROFILES.keys()),
        "endpoints": {
            "GET /health": "Health check, model & live weather connectivity",
            "GET /location": "Worldwide city geocoding & IP auto-detection",
            "GET /weather/current": "Current weather & AI solar estimate for dynamic location",
            "GET /forecast": "Exact 24h rolling forecast with demand curve & battery SOC simulation",
            "GET /opportunity": "Dynamic opportunity window detection for custom demand",
            "GET /recommendation": "Prescriptive role-specific decision advisory",
            "POST /predict": "Single-step solar prediction with reliability score",
            "POST /recommendation": "Prescriptive decision recommendation",
            "POST /what-if": "Dynamic live What-If simulation"
        }
    }


@app.get("/health", tags=["Health"])
def health():
    """Check service health, ML model bundle, and live weather status."""
    try:
        bundle = load_trained_model()
        model_ok = bundle is not None
        model_type = bundle.get("model_type", "Unknown")
        metrics = bundle.get("metrics", {})
    except Exception as e:
        model_ok = False
        model_type = f"Error: {e}"
        metrics = {}

    return {
        "status": "healthy" if model_ok else "degraded",
        "model_loaded": model_ok,
        "model_type": model_type,
        "historical_validation_metrics": metrics
    }


@app.get("/location", tags=["Geolocation"])
def get_location(
    query: Optional[str] = Query(None, description="Search query for global city"),
    auto_detect: bool = Query(False, description="Attempt auto IP detection")
):
    """Search worldwide cities via Open-Meteo Geocoding or auto-detect current location."""
    if auto_detect:
        return get_auto_detected_location()
    elif query:
        results = search_cities(query, limit=5)
        return {"query": query, "results": results}
    else:
        return get_auto_detected_location()


@app.get("/weather/current", tags=["Live Forecast"])
def get_current_weather_and_estimate(
    lat: float = Query(12.9716, description="Latitude"),
    lon: float = Query(77.5946, description="Longitude"),
    city: str = Query("Bengaluru", description="City name"),
    country: str = Query("India", description="Country name"),
    demand: float = Query(60.0, ge=0.0, description="Expected load demand in Watts"),
    battery_soc: float = Query(50.0, ge=0.0, le=100.0, description="Battery SOC (%)"),
    role: str = Query("⚡ Grid Operator", description="Operating role")
):
    """Retrieve current live weather conditions and AI solar estimate for any global coordinate."""
    try:
        bundle = load_trained_model()
        df_24, meta = get_live_24h_forecast_dataframe(lat=lat, lon=lon, city_name=city, country_name=country)
        pred_df = predict_batch(df_24, bundle=bundle)

        curr_row = pred_df.iloc[0]
        curr_pred = float(curr_row["Predicted_Power"])
        curr_rel = float(curr_row["Reliability_Score"])

        rec = get_recommendation(
            predicted_power=curr_pred,
            demand=demand,
            battery_soc=battery_soc,
            reliability_score=curr_rel,
            user_role=role
        )

        risk = assess_energy_risk(
            predicted_power=curr_pred,
            demand=demand,
            battery_soc=battery_soc,
            reliability_score=curr_rel
        )

        return {
            "location": meta["location"],
            "coordinates": meta["coordinates"],
            "timezone": meta["timezone"],
            "last_updated": meta["last_updated"],
            "connection_status": meta.get("connection_status", "LIVE"),
            "current_weather": meta["current_weather"],
            "model_derived_solar_estimate_w": curr_pred,
            "forecast_reliability": curr_rel,
            "energy_risk": risk,
            "recommended_action": rec
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving current estimate: {str(e)}"
        )


@app.get("/forecast", tags=["Live Forecast"])
def get_live_forecast(
    lat: float = Query(12.9716, description="Latitude"),
    lon: float = Query(77.5946, description="Longitude"),
    city: str = Query("Bengaluru", description="City name"),
    country: str = Query("India", description="Country name"),
    baseline_demand: float = Query(60.0, ge=0.0, description="Baseline load demand in Watts"),
    demand_profile: str = Query("⚡ Grid Feeder Curve", description="Demand profile type"),
    battery_soc: float = Query(50.0, ge=0.0, le=100.0, description="Initial Battery SOC (%)"),
    battery_capacity: float = Query(500.0, ge=10.0, description="Battery capacity in Wh"),
    role: str = Query("⚡ Grid Operator", description="Target facility role"),
    force_refresh: bool = Query(False, description="Bypass cache and force live API fetch")
):
    """Retrieve exact 24-hour rolling forecast, dynamic demand curve, battery SOC simulation, and decision recommendations."""
    try:
        bundle = load_trained_model()
        df_24, meta = get_live_24h_forecast_dataframe(
            lat=lat, lon=lon, city_name=city, country_name=country, force_refresh=force_refresh
        )
        pred_df = predict_batch(df_24, bundle=bundle)

        # 1. Generate 24-hour dynamic demand curve
        demand_series = generate_24h_demand_profile(
            timestamps=pred_df["timestamp"],
            baseline_demand_w=baseline_demand,
            profile_type=demand_profile
        )
        pred_df["Demand"] = demand_series
        pred_df["Surplus"] = pred_df["Predicted_Power"] - pred_df["Demand"]

        # 2. Simulate 24-hour battery SOC trajectory
        batt_sim = simulate_battery_trajectory(
            initial_soc_pct=battery_soc,
            capacity_wh=battery_capacity,
            surplus_series=pred_df["Surplus"].values
        )
        pred_df["Projected_Battery_SOC"] = batt_sim["soc_trajectory_pct"]
        pred_df["Battery_Power_Flow"] = batt_sim["battery_power_flow_w"]

        # 3. Opportunity Window & Energy Balance
        opp = detect_opportunity_window(pred_df, demand=demand_series)

        # 4. Build 24-Hour Timeline Rows
        timeline_rows = []
        for idx, row in pred_df.iterrows():
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
                user_role=role
            )

            risk_h = assess_energy_risk(
                predicted_power=p_val,
                demand=d_val,
                battery_soc=soc_val,
                reliability_score=rel_val
            )

            timeline_rows.append({
                "timestamp": str(row["timestamp"]),
                "temperature_c": float(row["Temperature"]),
                "cloud_cover_pct": float(row["raw_cloud_pct"]),
                "solar_radiation_wm2": float(row["Irradiance (G)"]),
                "predicted_solar_w": p_val,
                "forecasted_demand_w": d_val,
                "surplus_w": round(surp, 1),
                "projected_battery_soc_pct": soc_val,
                "risk_level": risk_h["level"],
                "recommended_action": rec_h["action"],
                "priority": rec_h["priority"]
            })

        curr_row = pred_df.iloc[0]
        curr_pred = float(curr_row["Predicted_Power"])
        curr_demand = float(curr_row["Demand"])
        curr_rel = float(curr_row["Reliability_Score"])

        curr_rec = get_recommendation(
            predicted_power=curr_pred,
            demand=curr_demand,
            battery_soc=battery_soc,
            reliability_score=curr_rel,
            user_role=role
        )

        overall_risk = assess_energy_risk(
            predicted_power=curr_pred,
            demand=curr_demand,
            battery_soc=battery_soc,
            reliability_score=curr_rel,
            deficit_hours_in_horizon=opp.get("deficit_hours", 0),
            min_projected_soc=batt_sim.get("min_soc_pct")
        )

        return {
            "metadata": meta,
            "user_role": role,
            "demand_config": {
                "baseline_w": baseline_demand,
                "profile_type": demand_profile
            },
            "current_status": {
                "model_derived_solar_estimate_w": curr_pred,
                "current_demand_w": curr_demand,
                "current_surplus_w": round(curr_pred - curr_demand, 1),
                "battery_soc_pct": battery_soc,
                "forecast_reliability": curr_rel,
                "peak_solar_forecast_w": float(pred_df["Predicted_Power"].max())
            },
            "primary_recommendation": curr_rec,
            "energy_risk": overall_risk,
            "battery_simulation": batt_sim,
            "opportunity_window": opp,
            "timeline_table": timeline_rows
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Live forecasting error: {str(e)}"
        )


@app.get("/opportunity", tags=["Decision Support"])
def get_opportunity(
    lat: float = Query(12.9716, description="Latitude"),
    lon: float = Query(77.5946, description="Longitude"),
    demand: float = Query(60.0, ge=0.0, description="Expected load demand in Watts")
):
    """Dynamically compute opportunity window for custom location and demand."""
    try:
        bundle = load_trained_model()
        df_24, meta = get_live_24h_forecast_dataframe(lat=lat, lon=lon)
        pred_df = predict_batch(df_24, bundle=bundle)
        opp = detect_opportunity_window(pred_df, demand=demand)
        return {
            "location": meta["location"],
            "demand_w": demand,
            "opportunity_window": opp
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Opportunity calculation error: {str(e)}"
        )


@app.get("/recommendation", tags=["Decision Support"])
def get_recommendation_endpoint(
    predicted_power: float = Query(..., ge=0.0, description="Forecasted solar power in Watts"),
    demand: float = Query(60.0, ge=0.0, description="Expected demand in Watts"),
    battery_soc: float = Query(50.0, ge=0.0, le=100.0, description="Battery SOC (%)"),
    reliability_score: float = Query(90.0, ge=0.0, le=100.0, description="Forecast reliability (%)"),
    role: str = Query("⚡ Grid Operator", description="Operating role")
):
    """Retrieve prescriptive decision recommendation for arbitrary inputs."""
    rec = get_recommendation(
        predicted_power=predicted_power,
        demand=demand,
        battery_soc=battery_soc,
        reliability_score=reliability_score,
        user_role=role
    )
    return rec


@app.post("/predict", tags=["Forecasting"])
def predict(request: PredictRequest):
    """Predict solar power generation and calculate reliability score."""
    try:
        input_dict = request.model_dump(by_alias=True)
        result = predict_single(input_dict)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/recommendation", tags=["Decision Support"])
def recommend_post(request: RecommendationRequest):
    """Generate prescriptive decision recommendation with 'Why?' deduction via POST."""
    try:
        rec = get_recommendation(
            predicted_power=request.predicted_power,
            demand=request.demand,
            battery_soc=request.battery_soc,
            reliability_score=request.reliability_score,
            user_role=request.user_role
        )
        return rec
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Recommendation error: {str(e)}"
        )


@app.post("/what-if", tags=["Decision Support"])
def what_if_simulation(request: WhatIfRequest):
    """Run dynamic What-If simulation comparing baseline vs scenario for a given role."""
    try:
        sim_res = run_what_if_simulation(
            base_features=request.base_features,
            new_cloud_pct=request.new_cloud_pct,
            new_demand_w=request.new_demand_w,
            new_battery_soc_pct=request.new_battery_soc_pct,
            new_temperature_c=request.new_temperature_c,
            user_role=request.user_role
        )
        return sim_res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"What-If simulation error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
