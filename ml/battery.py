"""RenewAI - Dynamic Battery Energy Storage & 24-Hour SOC Simulation Engine
==========================================================================
Simulates the physical battery charge/discharge trajectory over the 24-hour horizon,
accounting for capacity constraints, charge/discharge limits, and reserve thresholds.
"""

from typing import Dict, Any, List
import numpy as np


def simulate_battery_trajectory(
    initial_soc_pct: float = 50.0,
    capacity_wh: float = 500.0,
    surplus_series: np.ndarray = None,
    max_charge_w: float = 150.0,
    max_discharge_w: float = 150.0,
    min_reserve_pct: float = 20.0,
    charge_eff: float = 0.92,
    discharge_eff: float = 0.92
) -> Dict[str, Any]:
    """Simulate 24-hour battery State of Charge (SOC) trajectory from hourly energy balance.
    
    Returns hourly SOC trajectory, power flows, total throughput, and reserve health.
    """
    if surplus_series is None or len(surplus_series) == 0:
        surplus_series = np.zeros(24)

    n_hours = len(surplus_series)
    soc_trajectory = []
    battery_power_flow = []  # (+) charging, (-) discharging

    current_energy_wh = (initial_soc_pct / 100.0) * capacity_wh
    min_energy_reserve_wh = (min_reserve_pct / 100.0) * capacity_wh

    total_stored_wh = 0.0
    total_discharged_wh = 0.0
    reserve_breached = False

    for net_power in surplus_series:
        current_soc = (current_energy_wh / capacity_wh) * 100.0
        soc_trajectory.append(round(current_soc, 1))

        if net_power > 0.0:  # SURPLUS -> CHARGE
            available_headroom_wh = capacity_wh - current_energy_wh
            max_possible_charge_wh = min(net_power, max_charge_w) * 1.0  # 1 hour dt
            actual_charge_wh = min(available_headroom_wh, max_possible_charge_wh * charge_eff)
            current_energy_wh += actual_charge_wh
            total_stored_wh += actual_charge_wh
            battery_power_flow.append(round(actual_charge_wh / charge_eff, 1))

        elif net_power < 0.0:  # DEFICIT -> DISCHARGE
            deficit_wh = abs(net_power) * 1.0
            available_usable_wh = max(0.0, current_energy_wh - min_energy_reserve_wh)
            max_possible_discharge_wh = min(deficit_wh, max_discharge_w) * 1.0
            actual_discharge_wh = min(available_usable_wh, max_possible_discharge_wh)
            current_energy_wh -= (actual_discharge_wh / discharge_eff)
            total_discharged_wh += actual_discharge_wh
            battery_power_flow.append(-round(actual_discharge_wh, 1))

            if current_energy_wh < min_energy_reserve_wh + 0.1:
                reserve_breached = True
        else:
            battery_power_flow.append(0.0)

        # Enforce physical bounds [0, capacity]
        current_energy_wh = float(np.clip(current_energy_wh, 0.0, capacity_wh))

    final_soc = (current_energy_wh / capacity_wh) * 100.0
    min_soc = float(min(soc_trajectory))
    max_soc = float(max(soc_trajectory))

    return {
        "soc_trajectory_pct": soc_trajectory,
        "battery_power_flow_w": battery_power_flow,
        "initial_soc_pct": round(initial_soc_pct, 1),
        "final_soc_pct": round(final_soc, 1),
        "min_soc_pct": round(min_soc, 1),
        "max_soc_pct": round(max_soc, 1),
        "capacity_wh": capacity_wh,
        "total_stored_wh": round(total_stored_wh, 1),
        "total_discharged_wh": round(total_discharged_wh, 1),
        "reserve_breached": reserve_breached,
        "reserve_status": "Reserve Depleted" if reserve_breached else "Nominal Reserve Maintained"
    }
