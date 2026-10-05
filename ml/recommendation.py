"""RenewAI - Role-Specific Operational Decision Support Engine
============================================================
Prescriptive energy advisory engine providing role-specific actions for:
- ⚡ Grid Operator
- 🔋 Battery Storage Operator
- 🏭 Industrial Operator
- ☀️ Solar Plant Operator
- 🚗 EV Charging Station Operator
"""

import os
import sys
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.predict import predict_single, load_trained_model

USER_ROLES = [
    "⚡ Grid Operator",
    "🔋 Battery Storage Operator",
    "🏭 Industrial Operator",
    "☀️ Solar Plant Operator",
    "🚗 EV Charging Station Operator"
]


def get_recommendation(
    predicted_power: float,
    demand: float = 60.0,
    battery_soc: float = 50.0,
    reliability_score: float = 90.0,
    user_role: str = "⚡ Grid Operator",
    battery_capacity_wh: float = 500.0,
    is_peak_period: bool = False
) -> dict:
    """Generate role-specific operational decision recommendation."""
    surplus = predicted_power - demand
    is_surplus = surplus > 0.0
    is_balanced = abs(surplus) <= 20.0
    deficit = max(0.0, -surplus)
    is_night = predicted_power <= 0.5

    role_lower = user_role.lower()

    action = "MAINTAIN GRID BALANCE"
    priority = "LOW"
    badge = "info"
    icon = "ℹ️"
    reason = ""
    rule_name = ""
    logic_formula = ""
    suggested_flow_w = 0.0
    expected_impact = ""

    # ====================================================================
    # 1. GRID OPERATOR
    # ====================================================================
    if "grid" in role_lower:
        if is_surplus and surplus > 100.0 and battery_soc >= 85.0:
            action = "EXPORT SURPLUS TO GRID"
            priority = "HIGH"
            badge = "success"
            icon = "📤"
            rule_name = "Bulk Surplus Export"
            logic_formula = f"Surplus (+{surplus:.1f} W) > 100 W & Storage Full ({battery_soc:.0f}%) => Export clean energy to regional grid"
            reason = f"Renewable generation creates a +{surplus:.1f} W surplus while storage is at {battery_soc:.0f}% SOC. Export power to support regional grid balance."
            suggested_flow_w = surplus
            expected_impact = f"+{surplus:.1f} W exported to regional grid; reduces thermal generation."

        elif is_surplus and surplus > 150.0 and battery_soc >= 98.0:
            action = "CURTAIL EXCESS GENERATION"
            priority = "HIGH"
            badge = "warning"
            icon = "✂️"
            rule_name = "Over-generation Protection"
            logic_formula = f"Excess Surplus (+{surplus:.1f} W) & Interconnect Congestion => Curtail solar inverters"
            reason = f"Generation exceeds local demand by +{surplus:.1f} W and storage is 100% full. Signal inverter throttling to avoid bus over-voltage."
            suggested_flow_w = 0.0
            expected_impact = "Throttles inverter output to maintain distribution voltage stability."

        elif is_surplus:
            action = "DISPATCH BATTERY CHARGING"
            priority = "MEDIUM"
            badge = "success"
            icon = "🔋"
            rule_name = "Substation Balancing Charge"
            logic_formula = f"Surplus (+{surplus:.1f} W) with available storage buffer => Absorb surplus locally"
            reason = f"Renewable surplus of +{surplus:.1f} W available. Charge substation storage to absorb local generation."
            suggested_flow_w = min(surplus, 150.0)
            expected_impact = f"+{min(surplus, 150.0):.0f} W stored in grid battery; prevents reverse power flow."

        elif is_balanced:
            action = "MAINTAIN GRID BALANCE"
            priority = "LOW"
            badge = "info"
            icon = "⚖️"
            rule_name = "Feeder Equilibrium"
            logic_formula = f"Generation ({predicted_power:.1f} W) ~ Demand ({demand:.1f} W) => Steady-state operation"
            reason = f"Local generation ({predicted_power:.1f} W) closely tracks demand ({demand:.1f} W). Maintain normal balance without intervention."
            suggested_flow_w = 0.0
            expected_impact = "Grid feeder operates at zero net interchange."

        elif deficit > 80.0 and battery_soc < 30.0:
            action = "IMPORT FROM GRID & REDUCE GRID STRESS"
            priority = "CRITICAL"
            badge = "danger"
            icon = "🔌"
            rule_name = "Emergency Grid Import Requisition"
            logic_formula = f"Deficit (-{deficit:.1f} W) & Depleted Storage ({battery_soc:.0f}%) => Request bulk transmission infeed"
            reason = f"Severe generation deficit (-{deficit:.1f} W) and battery reserve depleted ({battery_soc:.0f}% SOC). Requisition external transmission import to prevent local brownouts."
            suggested_flow_w = -deficit
            expected_impact = f"-{deficit:.1f} W imported from transmission interconnect."

        else:
            action = "DISPATCH BATTERY"
            priority = "HIGH"
            badge = "warning"
            icon = "⚡"
            rule_name = "Storage Deficit Offset"
            logic_formula = f"Deficit (-{deficit:.1f} W) with Battery {battery_soc:.0f}% available => Discharge storage to offset deficit"
            reason = f"Feeder deficit of -{deficit:.1f} W. Dispatch substation battery to support load and avoid transmission draw."
            suggested_flow_w = -min(deficit, 150.0)
            expected_impact = f"-{min(deficit, 150.0):.0f} W supplied by battery; zero grid strain."

    # ====================================================================
    # 2. BATTERY STORAGE OPERATOR
    # ====================================================================
    elif "battery" in role_lower or "storage" in role_lower:
        if is_surplus and battery_soc < 90.0:
            action = "CHARGE BATTERY"
            priority = "HIGH" if surplus > 80 else "MEDIUM"
            badge = "success"
            icon = "🔋"
            rule_name = "Bulk Renewable Absorption"
            logic_formula = f"Surplus (+{surplus:.1f} W) & SOC {battery_soc:.0f}% < 90% => Charge storage bank"
            reason = f"Solar surplus of +{surplus:.1f} W available. Charge battery at max allowable rate to capture free clean energy."
            suggested_flow_w = min(surplus, 180.0)
            expected_impact = f"+{min(surplus, 180.0):.0f} W charging rate; increases SOC by ~{min(surplus, 180.0)/battery_capacity_wh*100:.1f}%/hr."

        elif is_surplus and battery_soc >= 90.0:
            action = "OPTIMIZE CHARGE/DISCHARGE & HOLD"
            priority = "LOW"
            badge = "primary"
            icon = "⏸️"
            rule_name = "Full Bank Trickle Hold"
            logic_formula = f"Surplus with Battery Full ({battery_soc:.0f}%) => Maintain trickle float charge"
            reason = f"Battery is nearly full ({battery_soc:.0f}% SOC). Hold charge and divert remaining generation to secondary storage."
            suggested_flow_w = 0.0
            expected_impact = "Battery preserved at full capacity with minimal degradation."

        elif not is_surplus and battery_soc >= 35.0:
            action = "DISCHARGE BATTERY"
            priority = "HIGH"
            badge = "warning"
            icon = "⚡"
            rule_name = "Load Support Discharge"
            logic_formula = f"Deficit (-{deficit:.1f} W) & SOC {battery_soc:.0f}% >= 35% => Discharge to power connected loads"
            reason = f"Deficit of -{deficit:.1f} W occurring. Discharge battery at {min(deficit, 150.0):.0f} W to serve load."
            suggested_flow_w = -min(deficit, 150.0)
            expected_impact = f"-{min(deficit, 150.0):.0f} W discharge power; avoids costly peak grid energy."

        elif not is_surplus and battery_soc < 35.0:
            action = "HOLD BATTERY RESERVE & PREPARE FOR DEFICIT"
            priority = "CRITICAL"
            badge = "danger"
            icon = "🛡️"
            rule_name = "Deep Discharge Lockout"
            logic_formula = f"Deficit with Critical SOC ({battery_soc:.0f}%) => Lock discharge to protect battery health"
            reason = f"Battery reserve is critical ({battery_soc:.0f}% SOC). Lock discharge to prevent cell damage; switch facility to grid backup."
            suggested_flow_w = 0.0
            expected_impact = "Prevents deep cycle degradation; preserves emergency reserve."

        else:
            action = "OPTIMIZE CHARGE/DISCHARGE"
            priority = "LOW"
            badge = "info"
            icon = "⚖️"
            rule_name = "Storage Standby"
            logic_formula = "Energy balanced => Maintain storage standby"
            reason = "Generation matches load. Maintain storage in optimal standby state."
            suggested_flow_w = 0.0
            expected_impact = "Zero storage throughput."

    # ====================================================================
    # 3. INDUSTRIAL OPERATOR
    # ====================================================================
    elif "industrial" in role_lower:
        if is_surplus and surplus > 80.0:
            action = "RUN FLEXIBLE INDUSTRIAL LOADS"
            priority = "HIGH"
            badge = "success"
            icon = "🏭"
            rule_name = "Solar Window Process Activation"
            logic_formula = f"High clean surplus (+{surplus:.1f} W) => Trigger batch processing & heavy machinery"
            reason = f"Peak solar surplus (+{surplus:.1f} W) available. Ideal window to run furnaces, heavy pumps, and batch manufacturing."
            suggested_flow_w = surplus
            expected_impact = f"Consumes +{surplus:.0f} W of zero-marginal-cost renewable energy for industrial output."

        elif is_surplus:
            action = "USE RENEWABLE POWER & CHARGE STORAGE"
            priority = "MEDIUM"
            badge = "success"
            icon = "⏱️"
            rule_name = "Industrial Self-Consumption"
            logic_formula = f"Moderate surplus (+{surplus:.1f} W) => Power base plant + charge onsite battery"
            reason = f"Moderate clean surplus (+{surplus:.1f} W). Power factory baseline and charge onsite storage."
            suggested_flow_w = min(surplus, 100.0)
            expected_impact = "Maximizes on-site clean energy utilization."

        elif not is_surplus and battery_soc >= 40.0:
            action = "SHIFT FLEXIBLE LOAD & USE STORED ENERGY"
            priority = "HIGH"
            badge = "warning"
            icon = "⚡"
            rule_name = "Industrial Peak-Shaving"
            logic_formula = f"Deficit (-{deficit:.1f} W) with Battery {battery_soc:.0f}% => Discharge onsite storage & defer flexible tasks"
            reason = f"Solar deficit (-{deficit:.1f} W). Discharge industrial battery and shift non-urgent batch tasks to avoid peak demand tariffs."
            suggested_flow_w = -min(deficit, 120.0)
            expected_impact = f"Shaves -{min(deficit, 120.0):.0f} W off factory peak demand."

        else:
            action = "REDUCE NON-CRITICAL LOAD & IMPORT FROM GRID"
            priority = "HIGH"
            badge = "danger"
            icon = "⚠️"
            rule_name = "Industrial Demand Curtailment"
            logic_formula = f"Deficit (-{deficit:.1f} W) & Low Storage ({battery_soc:.0f}%) => Curtail auxiliary machinery"
            reason = f"Solar output is low and battery reserve is {battery_soc:.0f}%. Pause auxiliary compressors and lighting to reduce peak grid draw."
            suggested_flow_w = -deficit
            expected_impact = f"Reduces factory grid infeed by shedding auxiliary loads."

    # ====================================================================
    # 4. SOLAR PLANT OPERATOR
    # ====================================================================
    elif "solar" in role_lower or "plant" in role_lower:
        if is_surplus and battery_soc >= 90.0:
            action = "EXPORT SURPLUS"
            priority = "HIGH"
            badge = "success"
            icon = "☀️"
            rule_name = "Full Solar PPA Export"
            logic_formula = f"Plant Output (+{predicted_power:.1f} W) & Storage Full => Dispatch 100% capacity to grid interconnect"
            reason = f"Solar output is optimal ({predicted_power:.1f} W) and collocated storage is full ({battery_soc:.0f}%). Export full generation to the interconnect."
            suggested_flow_w = predicted_power
            expected_impact = f"Delivers {predicted_power:.1f} W clean power under PPA agreement."

        elif is_surplus and battery_soc < 90.0:
            action = "MAXIMIZE SELF-CONSUMPTION & STORE ENERGY"
            priority = "MEDIUM"
            badge = "success"
            icon = "🔋"
            rule_name = "Solar-Plus-Storage Inflow"
            logic_formula = f"Generation (+{predicted_power:.1f} W) exceeds off-take => Store excess in plant BESS"
            reason = f"Plant generation (+{predicted_power:.1f} W) exceeds immediate off-take. Store surplus (+{surplus:.1f} W) into plant BESS."
            suggested_flow_w = min(surplus, 180.0)
            expected_impact = f"+{min(surplus, 180.0):.0f} W stored for firm evening delivery."

        elif not is_surplus and battery_soc >= 40.0:
            action = "MAINTAIN GENERATION USING BATTERY"
            priority = "MEDIUM"
            badge = "warning"
            icon = "⚡"
            rule_name = "Plant Capacity Firming"
            logic_formula = f"Low irradiance & Storage Available => Inject battery power to meet scheduled commitment"
            reason = f"Solar generation dropped ({predicted_power:.1f} W). Discharge plant battery to firm output and meet scheduled commitments."
            suggested_flow_w = -min(deficit, 150.0)
            expected_impact = f"Firms plant output by injecting +{min(deficit, 150.0):.0f} W."

        elif reliability_score < 75.0:
            action = "PREPARE FOR WEATHER REDUCTION"
            priority = "HIGH"
            badge = "warning"
            icon = "⛅"
            rule_name = "Weather Ramp Warning"
            logic_formula = f"High cloud variance (Reliability: {reliability_score:.1f}%) => Alert dispatch of upcoming solar drop"
            reason = f"Weather forecast indicates incoming cloud cover with high uncertainty ({reliability_score:.1f}% confidence). Prepare ramping alerts."
            suggested_flow_w = 0.0
            expected_impact = "Notifies grid interconnect of potential ramp-down."

        else:
            action = "CURTAIL EXCESS / STANDBY"
            priority = "LOW"
            badge = "info"
            icon = "📊"
            rule_name = "Plant Standby"
            logic_formula = "Night / Low solar with low reserves => Standard plant standby mode"
            reason = f"Low solar irradiance period ({predicted_power:.1f} W). Maintain plant in standby state."
            suggested_flow_w = 0.0
            expected_impact = "Plant operates in auxiliary standby mode."

    # ====================================================================
    # 5. EV CHARGING STATION OPERATOR
    # ====================================================================
    else:
        if is_surplus and surplus > 100.0:
            action = "START EV CHARGING & CHARGE DURING SURPLUS"
            priority = "HIGH"
            badge = "success"
            icon = "🚗"
            rule_name = "Solar EV Fast-Charging"
            logic_formula = f"High clean surplus (+{surplus:.1f} W) => Boost EV charging terminal throughput"
            reason = f"High solar surplus (+{surplus:.1f} W) detected. Optimal time to offer maximum charging power at lowest renewable cost."
            suggested_flow_w = surplus
            expected_impact = f"+{surplus:.0f} W routed to connected EV vehicle batteries."

        elif is_surplus:
            action = "USE RENEWABLE POWER FOR EV CHARGING"
            priority = "MEDIUM"
            badge = "success"
            icon = "⚡"
            rule_name = "Moderate EV Solar Charging"
            logic_formula = f"Surplus (+{surplus:.1f} W) => Power active EV dispensers"
            reason = f"Clean solar surplus (+{surplus:.1f} W) available. Power active EV dispensers directly from onsite solar canopy."
            suggested_flow_w = surplus
            expected_impact = "Powers EV charging dispensers purely on clean solar."

        elif not is_surplus and battery_soc >= 45.0:
            action = "USE STORED ENERGY FOR CHARGING"
            priority = "HIGH"
            badge = "warning"
            icon = "🔋"
            rule_name = "EV Buffer Battery Discharge"
            logic_formula = f"Deficit (-{deficit:.1f} W) with Battery {battery_soc:.0f}% => Discharge buffer to maintain EV charge rate"
            reason = f"Solar output dropped. Discharge onsite buffer battery to maintain vehicle charging speeds without expensive grid demand peaks."
            suggested_flow_w = -min(deficit, 140.0)
            expected_impact = f"Buffers -{min(deficit, 140.0):.0f} W to support EV charging."

        elif not is_surplus and battery_soc < 30.0:
            action = "REDUCE CHARGING RATE / DELAY EV CHARGING"
            priority = "CRITICAL"
            badge = "danger"
            icon = "⏱️"
            rule_name = "EV Demand Throttling"
            logic_formula = f"Deficit (-{deficit:.1f} W) & Battery Depleted => Throttle EV dispenser power"
            reason = f"Zero solar generation and buffer battery depleted ({battery_soc:.0f}% SOC). Throttle EV charging speeds to prevent commercial grid peak surge."
            suggested_flow_w = 0.0
            expected_impact = "Throttles dispenser output to avoid severe utility demand charges."

        else:
            action = "DELAY EV CHARGING / USE GRID"
            priority = "MEDIUM"
            badge = "info"
            icon = "🔌"
            rule_name = "Standard Grid EV Charging"
            logic_formula = "Low solar & moderate battery => Draw from standard grid"
            reason = "Moderate deficit. Provide standard EV charging using baseline grid connection."
            suggested_flow_w = -deficit
            expected_impact = "Supplies baseline EV charging from utility grid."

    why_data = {
        "predicted_generation": f"{predicted_power:.1f} W",
        "expected_demand": f"{demand:.1f} W",
        "expected_surplus": f"{'+' if surplus >= 0 else ''}{surplus:.1f} W",
        "battery_soc": f"{battery_soc:.0f}%",
        "reliability": f"{reliability_score:.1f}%",
        "user_role": user_role,
        "rule_applied": rule_name,
        "logic_formula": logic_formula,
        "action": action
    }

    return {
        "action": action,
        "priority": priority,
        "badge": badge,
        "icon": icon,
        "reason": reason,
        "expected_impact": expected_impact,
        "predicted_power": round(predicted_power, 1),
        "demand": round(demand, 1),
        "surplus": round(surplus, 1),
        "battery_soc": round(battery_soc, 1),
        "reliability_score": round(reliability_score, 1),
        "suggested_flow_w": round(suggested_flow_w, 1),
        "user_role": user_role,
        "why": why_data
    }


def run_what_if_simulation(
    base_features: dict,
    new_cloud_pct: float,
    new_demand_w: float,
    new_battery_soc_pct: float,
    new_temperature_c: float = None,
    user_role: str = "⚡ Grid Operator",
    bundle: dict = None
) -> dict:
    """Run live What-If simulation comparing baseline vs simulated scenario for a given role."""
    if bundle is None:
        bundle = load_trained_model()

    base_demand = float(base_features.get("Demand", 60.0))
    base_soc = float(base_features.get("Battery_SOC", 50.0))

    # 1. Base prediction
    base_pred_res = predict_single(base_features, bundle=bundle)
    base_power = base_pred_res["predicted_power"]
    base_rel = base_pred_res["reliability"]["score"]
    base_rec = get_recommendation(
        predicted_power=base_power,
        demand=base_demand,
        battery_soc=base_soc,
        reliability_score=base_rel,
        user_role=user_role
    )

    # 2. Simulated scenario
    sim_features = dict(base_features)
    sim_cloud_frac = np.clip(new_cloud_pct / 100.0, 0.0, 1.0)
    sim_features["Cloud"] = sim_cloud_frac

    if new_temperature_c is not None:
        sim_features["Temperature"] = float(new_temperature_c)
        pres = float(base_features.get("raw_pressure_hpa", 980.0))
        sim_features["Rhoa"] = (pres * 100.0) / (287.058 * (float(new_temperature_c) + 273.15))

    base_cloud = float(base_features.get("Cloud", 0.15))
    base_irr_g = float(base_features.get("Irradiance (G)", 400.0))
    base_irr_a = float(base_features.get("Irradiance (A)", 600.0))

    cloud_factor = (1.0 - 0.75 * sim_cloud_frac) / max(0.01, (1.0 - 0.75 * base_cloud))
    sim_features["Irradiance (G)"] = float(np.clip(base_irr_g * cloud_factor, 0.0, 1100.0))
    sim_features["Irradiance (A)"] = float(np.clip(base_irr_a * cloud_factor, 0.0, 1200.0))

    sim_pred_res = predict_single(sim_features, bundle=bundle)
    sim_power = sim_pred_res["predicted_power"]
    sim_rel = sim_pred_res["reliability"]["score"]

    sim_rec = get_recommendation(
        predicted_power=sim_power,
        demand=new_demand_w,
        battery_soc=new_battery_soc_pct,
        reliability_score=sim_rel,
        user_role=user_role
    )

    power_delta = sim_power - base_power
    surplus_base = base_power - base_demand
    surplus_sim = sim_power - new_demand_w

    return {
        "base_scenario": {
            "cloud_pct": round(base_cloud * 100.0, 1),
            "demand_w": round(base_demand, 1),
            "battery_soc_pct": round(base_soc, 1),
            "predicted_power": base_power,
            "surplus_w": round(surplus_base, 1),
            "reliability_score": base_rel,
            "recommendation": base_rec
        },
        "simulated_scenario": {
            "cloud_pct": round(new_cloud_pct, 1),
            "demand_w": round(new_demand_w, 1),
            "battery_soc_pct": round(new_battery_soc_pct, 1),
            "predicted_power": sim_power,
            "surplus_w": round(surplus_sim, 1),
            "reliability_score": sim_rel,
            "recommendation": sim_rec
        },
        "comparison": {
            "power_change_w": round(power_delta, 1),
            "action_changed": base_rec["action"] != sim_rec["action"],
            "previous_action": base_rec["action"],
            "new_action": sim_rec["action"],
            "why_change": (
                f"Predicted generation shifted by {power_delta:+.1f} W (Surplus: {surplus_base:+.1f} W -> {surplus_sim:+.1f} W). "
                f"Operational action for {user_role} transitioned from '{base_rec['action']}' to '{sim_rec['action']}'."
                if base_rec["action"] != sim_rec["action"] else
                f"Generation changed by {power_delta:+.1f} W, but conditions remain within the '{base_rec['action']}' operational rule."
            )
        }
    }
