"""RenewAI - Energy Balance & Renewable Opportunity Horizon Engine
===================================================================
Analyzes rolling 24-hour solar generation against dynamic load demand curves.
Classifies energy balance tiers, tracks surplus/deficit streaks, and detects
optimal renewable utilization windows.
"""

from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np


def classify_energy_balance(surplus_w: float) -> str:
    """Classify hourly energy balance into 5 operational tiers."""
    if surplus_w > 100.0:
        return "HIGH SURPLUS"
    elif surplus_w >= 20.0:
        return "SURPLUS"
    elif surplus_w >= -20.0:
        return "BALANCED"
    elif surplus_w >= -100.0:
        return "DEFICIT"
    else:
        return "SEVERE DEFICIT"


def format_window_range(start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> str:
    """Format time range clearly across midnight with date context."""
    if start_dt.date() == end_dt.date():
        return f"{start_dt.strftime('%I:%M %p')} – {end_dt.strftime('%I:%M %p')} (Today)"
    else:
        return f"{start_dt.strftime('%I:%M %p (%a)')} – {end_dt.strftime('%I:%M %p (%a)')}"


def detect_opportunity_window(
    forecast_df: pd.DataFrame,
    demand: Union[float, np.ndarray, List[float]] = 60.0,
    min_surplus_threshold: float = 10.0
) -> Dict[str, Any]:
    """Analyze 24-hour forecasted generation vs dynamic demand to compute full energy balance metrics."""
    if forecast_df.empty or "Predicted_Power" not in forecast_df.columns:
        return {
            "has_opportunity": False,
            "window_text": "No forecast data available",
            "start_time": None,
            "end_time": None,
            "peak_surplus": 0.0,
            "peak_deficit": 0.0,
            "peak_generation": 0.0,
            "peak_time_str": "N/A",
            "total_surplus_wh": 0.0,
            "total_deficit_wh": 0.0,
            "avg_surplus": 0.0,
            "surplus_hours": 0,
            "deficit_hours": 0,
            "longest_surplus_streak": 0,
            "longest_deficit_streak": 0,
            "recommended_uses": [],
            "hourly_categories": []
        }

    df = forecast_df.copy()
    time_col = "timestamp" if "timestamp" in df.columns else ("TimeStamp" if "TimeStamp" in df.columns else None)

    # Handle scalar or array demand
    if isinstance(demand, (np.ndarray, list)):
        df["Demand"] = np.array(demand)[:len(df)]
    else:
        df["Demand"] = float(demand)

    df["Surplus"] = df["Predicted_Power"] - df["Demand"]
    df["Surplus_Category"] = df["Surplus"].apply(classify_energy_balance)
    df["Is_Opportunity"] = df["Surplus"] >= min_surplus_threshold

    opp_rows = df[df["Is_Opportunity"]]
    deficit_rows = df[df["Surplus"] < 0.0]

    max_gen = float(df["Predicted_Power"].max())
    max_gen_idx = df["Predicted_Power"].idxmax()
    peak_row_global = df.loc[max_gen_idx]

    if time_col and pd.api.types.is_datetime64_any_dtype(df[time_col]):
        peak_time_str = peak_row_global[time_col].strftime("%I:%M %p (%a)")
    else:
        peak_time_str = "Midday Peak"

    # Calculate longest continuous streaks
    is_opp_arr = df["Is_Opportunity"].values
    is_def_arr = (df["Surplus"].values < 0.0)

    def get_longest_streak(arr):
        max_s = curr_s = 0
        for val in arr:
            if val:
                curr_s += 1
                max_s = max(max_s, curr_s)
            else:
                curr_s = 0
        return max_s

    longest_surplus_streak = get_longest_streak(is_opp_arr)
    longest_deficit_streak = get_longest_streak(is_def_arr)

    # Total surplus & deficit energies
    total_surplus_wh = float(df[df["Surplus"] > 0]["Surplus"].sum())
    total_deficit_wh = float(abs(df[df["Surplus"] < 0]["Surplus"].sum()))
    peak_deficit_w = float(abs(df["Surplus"].min())) if not deficit_rows.empty else 0.0

    if opp_rows.empty:
        return {
            "has_opportunity": False,
            "window_text": "No Net Surplus Window (Generation Below Demand)",
            "start_time": None,
            "end_time": None,
            "peak_surplus": round(df["Surplus"].max(), 1),
            "peak_deficit": round(peak_deficit_w, 1),
            "peak_generation": round(max_gen, 1),
            "peak_time_str": peak_time_str,
            "total_surplus_wh": 0.0,
            "total_deficit_wh": round(total_deficit_wh, 1),
            "avg_surplus": 0.0,
            "surplus_hours": 0,
            "deficit_hours": int(len(deficit_rows)),
            "longest_surplus_streak": 0,
            "longest_deficit_streak": longest_deficit_streak,
            "recommended_uses": ["CONSERVE BATTERY", "SCHEDULE GRID IMPORT", "PEAK SHAVING"],
            "summary": f"Forecasted generation ({max_gen:.1f} W peak) remains below demand curve.",
            "hourly_categories": df["Surplus_Category"].tolist()
        }

    # Find peak surplus row in the opportunity window
    peak_idx = opp_rows["Surplus"].idxmax()
    peak_row = opp_rows.loc[peak_idx]
    peak_surplus = float(peak_row["Surplus"])
    peak_gen = float(peak_row["Predicted_Power"])

    # Determine dynamic window range with date/day clarity
    if time_col and pd.api.types.is_datetime64_any_dtype(opp_rows[time_col]):
        start_dt = opp_rows[time_col].min()
        end_dt = opp_rows[time_col].max()
        window_text = format_window_range(start_dt, end_dt)
        start_str = start_dt.strftime("%I:%M %p (%a)")
        end_str = end_dt.strftime("%I:%M %p (%a)")
        peak_time_str = peak_row[time_col].strftime("%I:%M %p (%a)")
    else:
        window_text = f"Hours {opp_rows['Hour'].min()}:00 – {opp_rows['Hour'].max()}:00"
        start_str = "Start"
        end_str = "End"

    avg_surplus = float(opp_rows["Surplus"].mean())
    surplus_hours = int(len(opp_rows))
    deficit_hours = int(len(deficit_rows))

    # Recommended uses based on magnitude of surplus
    recommended_uses = ["CHARGE STORAGE BATTERY"]
    if peak_surplus > 100.0:
        recommended_uses.append("RUN FLEXIBLE INDUSTRIAL LOADS")
        recommended_uses.append("EXPORT SURPLUS POWER TO GRID")
    elif peak_surplus > 40.0:
        recommended_uses.append("POWER COOLING / HVAC PRE-COOL")

    return {
        "has_opportunity": True,
        "window_text": window_text,
        "start_time": start_str,
        "end_time": end_str,
        "peak_surplus": round(peak_surplus, 1),
        "peak_deficit": round(peak_deficit_w, 1),
        "peak_generation": round(peak_gen, 1),
        "peak_time_str": peak_time_str,
        "total_surplus_wh": round(total_surplus_wh, 1),
        "total_deficit_wh": round(total_deficit_wh, 1),
        "avg_surplus": round(avg_surplus, 1),
        "surplus_hours": surplus_hours,
        "deficit_hours": deficit_hours,
        "longest_surplus_streak": longest_surplus_streak,
        "longest_deficit_streak": longest_deficit_streak,
        "recommended_uses": recommended_uses,
        "summary": (
            f"Best Renewable Window: {window_text} with peak surplus of +{peak_surplus:.1f} W "
            f"at {peak_time_str}, clean excess of {total_surplus_wh:.1f} Wh, and total deficit of {total_deficit_wh:.1f} Wh."
        ),
        "hourly_categories": df["Surplus_Category"].tolist()
    }
