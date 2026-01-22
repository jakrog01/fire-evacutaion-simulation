import numpy as np
from numba import njit

from src.config.constants import SimulationConstants
from src.config.fire_constants import (
    get_fuel_capacity_param,
    get_heat_rate_param,
    get_ignition_temperature_param,
)
from src.physics.fire_kernel import (
    P_EX_MAX,
    P_EX_MIN,
    P_FUEL_POWER_MAX,
    P_FUEL_POWER_MIN,
    P_HEAT_NONLINEARITY_EXPONENT,
    P_HRR_SCALE,
    P_IGNITION_SHIFT,
)


@njit(cache=True, fastmath=True)
def compute_fire_active_mask(
    temp,
    fuel_power,
    fuel_remaining,
    nav,
    params,
    ign_table,
    heat_table,
    cap_table,
    wall_code=SimulationConstants.WALL_NAV_CODE,
    void_code=SimulationConstants.VOID_NAV_CODE,
):
    H, W = temp.shape
    out = np.zeros((H, W), dtype=np.uint8)

    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    ign_shift = params[P_IGNITION_SHIFT]
    heat_nl = params[P_HEAT_NONLINEARITY_EXPONENT]
    ex_min = params[P_EX_MIN]
    ex_max = params[P_EX_MAX]
    hrr_scale = params[P_HRR_SCALE]

    for y in range(H):
        for x in range(W):
            if nav[y, x] == void_code:
                continue

            fp = fuel_power[y, x]
            if fp <= 0.0:
                continue

            ignition_temp = get_ignition_temperature_param(
                fp, ign_table, fp_min, fp_max, ign_shift
            )
            cap = get_fuel_capacity_param(fp, cap_table, fp_min, fp_max)

            rem = fuel_remaining[y, x]
            burned_already = rem < cap

            hold_drop = 60.0 + 6.0 * fp
            hold_temp = ignition_temp - hold_drop
            threshold = hold_temp if burned_already else ignition_temp

            T = temp[y, x]
            if T < threshold:
                continue

            heat_rate = get_heat_rate_param(
                fp, heat_table, fp_min, fp_max, heat_nl, hrr_scale
            )

            if T < ignition_temp:
                if burned_already:
                    span = ignition_temp - hold_temp
                    if span <= 1e-6:
                        excess_factor = 1.0
                    else:
                        t = (T - hold_temp) / span
                        if t < 0.0:
                            t = 0.0
                        if t > 1.0:
                            t = 1.0
                        excess_factor = ex_min if t < ex_min else t
                else:
                    excess_factor = ex_min
            else:
                val = (T - ignition_temp) / 100.0
                if val < ex_min:
                    val = ex_min
                if val > ex_max:
                    val = ex_max
                excess_factor = val

            if excess_factor < ex_min:
                excess_factor = ex_min

            heat_gen = heat_rate * excess_factor
            if heat_gen > 0.0:
                out[y, x] = 1

    return out
