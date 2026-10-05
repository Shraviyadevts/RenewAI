"""RenewAI - Dynamic Energy Demand Profiling & Forecasting Engine
==============================================================
Generates realistic 24-hour demand curves based on diurnal patterns,
facility operational profiles (Industrial, Commercial, Grid, EV, Plant),
or user-defined baselines.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


DEMAND_PROFILES = {
    "⚡ Grid Feeder Curve": {
        "description": "Standard regional distribution feeder with dual morning & evening peaks",
        "hourly_factors": [
            0.60, 0.55, 0.52, 0.50, 0.55, 0.68,  # 00:00 - 05:00
            0.85, 1.05, 1.15, 1.10, 1.05, 1.00,  # 06:00 - 11:00
            0.98, 0.95, 0.92, 0.95, 1.10, 1.25,  # 12:00 - 17:00
            1.30, 1.25, 1.15, 0.95, 0.80, 0.68   # 18:00 - 23:00
        ]
    },
    "🏭 Industrial Shift Curve": {
        "description": "Heavy daytime manufacturing load with constant baseline and shift ramps",
        "hourly_factors": [
            0.45, 0.45, 0.45, 0.45, 0.50, 0.70,  # 00:00 - 05:00
            1.10, 1.30, 1.35, 1.35, 1.30, 1.25,  # 06:00 - 11:00
            1.20, 1.30, 1.35, 1.30, 1.15, 0.85,  # 12:00 - 17:00
            0.65, 0.55, 0.50, 0.45, 0.45, 0.45   # 18:00 - 23:00
        ]
    },
    "🏢 Commercial Building Curve": {
        "description": "Office/Retail profile with sharp 8 AM - 6 PM occupancy & HVAC load",
        "hourly_factors": [
            0.30, 0.30, 0.30, 0.30, 0.30, 0.40,  # 00:00 - 05:00
            0.60, 0.90, 1.25, 1.35, 1.35, 1.30,  # 06:00 - 11:00
            1.25, 1.25, 1.30, 1.25, 1.15, 0.85,  # 12:00 - 17:00
            0.55, 0.40, 0.35, 0.30, 0.30, 0.30   # 18:00 - 23:00
        ]
    },
    "🚗 EV Charging Hub Curve": {
        "description": "Commuter charging spikes during morning arrival and evening return",
        "hourly_factors": [
            0.20, 0.20, 0.20, 0.20, 0.25, 0.40,  # 00:00 - 05:00
            0.80, 1.30, 1.40, 1.10, 0.90, 0.80,  # 06:00 - 11:00
            0.85, 0.90, 1.00, 1.15, 1.35, 1.45,  # 12:00 - 17:00
            1.30, 1.00, 0.70, 0.45, 0.30, 0.20   # 18:00 - 23:00
        ]
    },
    "📊 Constant Baseline Load": {
        "description": "Flat, unvarying load demand across all 24 hours",
        "hourly_factors": [1.00] * 24
    }
}


def generate_24h_demand_profile(
    timestamps: pd.Series,
    baseline_demand_w: float = 60.0,
    profile_type: str = "⚡ Grid Feeder Curve"
) -> np.ndarray:
    """Generate a realistic 24-hour demand series aligned with forecast timestamps."""
    if profile_type not in DEMAND_PROFILES:
        profile_type = "⚡ Grid Feeder Curve"

    factors_24 = DEMAND_PROFILES[profile_type]["hourly_factors"]
    demand_series = []

    for ts in timestamps:
        hour = ts.hour
        factor = factors_24[hour]
        demand_val = round(baseline_demand_w * factor, 1)
        demand_series.append(max(5.0, demand_val))

    return np.array(demand_series, dtype=float)
