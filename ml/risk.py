"""RenewAI - Energy Risk Assessment Engine
==========================================
Computes dynamic operational energy risk based on net generation deficit,
battery reserve level (SOC %), battery trajectory minima, and forecast reliability.
"""

from typing import Dict, Any
import numpy as np


def assess_energy_risk(
    predicted_power: float,
    demand: float,
    battery_soc: float,
    reliability_score: float,
    deficit_hours_in_horizon: int = 0,
    min_projected_soc: float = None
) -> Dict[str, Any]:
    """Assess energy risk level and recommend operational mitigation strategy.
    
    Risk Tiers:
    - CRITICAL RISK: Severe deficit + battery SOC < 25% (or projected battery depletion)
    - HIGH RISK: Solar deficit during peak load + battery reserve < 40%
    - MODERATE RISK: Mild deficit manageable via battery discharge (40-65% SOC)
    - LOW RISK: Generation surplus or full battery storage (> 65% SOC)
    """
    surplus = predicted_power - demand
    deficit = max(0.0, -surplus)
    deficit_ratio = deficit / max(10.0, demand)

    if min_projected_soc is None:
        effective_soc = battery_soc
    else:
        effective_soc = min(battery_soc, min_projected_soc)

    # Base risk score (0 to 100)
    risk_score = 0.0

    # Deficit impact (up to 50 points)
    risk_score += min(50.0, deficit_ratio * 50.0)

    # Battery vulnerability impact (up to 35 points for low SOC)
    battery_penalty = max(0.0, (70.0 - effective_soc) / 70.0) * 35.0
    risk_score += battery_penalty

    # Deficit streak impact (up to 10 points for prolonged deficits)
    if deficit_hours_in_horizon > 6:
        risk_score += min(10.0, (deficit_hours_in_horizon - 6) * 1.5)

    # Forecast uncertainty impact (up to 10 points)
    if reliability_score < 75.0:
        risk_score += (75.0 - reliability_score) * 0.4

    risk_score = float(np.clip(risk_score, 5.0, 98.0))

    if risk_score >= 70.0 or (deficit > 50.0 and effective_soc < 25.0):
        level = "CRITICAL RISK"
        badge = "danger"
        icon = "🔴"
        reason = (
            f"Severe generation deficit (-{deficit:.1f} W) with battery reserve near depletion ({effective_soc:.0f}% SOC). "
            f"High risk of unserved load without immediate grid import or load shedding."
        )
        mitigation = "Activate emergency grid import and curtail non-essential flexible loads immediately."
    elif risk_score >= 45.0 or (deficit > 0.0 and effective_soc < 40.0):
        level = "HIGH RISK"
        badge = "danger"
        icon = "🟠"
        reason = (
            f"Solar deficit (-{deficit:.1f} W) expected during load peak with limited battery reserve ({effective_soc:.0f}% SOC). "
            f"Forecast deficit horizon spans ~{deficit_hours_in_horizon} hours."
        )
        mitigation = "Pre-schedule transmission grid capacity and defer discretionary energy processes."
    elif risk_score >= 25.0 or deficit > 0.0:
        level = "MODERATE RISK"
        badge = "warning"
        icon = "🟡"
        reason = (
            f"Mild generation deficit (-{deficit:.1f} W) manageable via battery discharge ({effective_soc:.0f}% SOC available). "
            f"Monitor cloud variance."
        )
        mitigation = "Utilize stored battery reserves and optimize cooling/HVAC setpoints."
    else:
        level = "LOW RISK"
        badge = "success"
        icon = "🟢"
        reason = (
            f"Generation covers demand (+{surplus:.1f} W surplus) with healthy battery reserve ({effective_soc:.0f}% SOC). "
            f"Energy supply is highly secure."
        )
        mitigation = "Maximize self-consumption, charge storage, or export clean surplus."

    return {
        "level": level,
        "badge": badge,
        "icon": icon,
        "score": round(risk_score, 1),
        "reason": reason,
        "mitigation": mitigation
    }
