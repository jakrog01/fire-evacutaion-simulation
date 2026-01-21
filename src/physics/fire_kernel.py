from math import exp

import numpy as np
from numba import float32, njit

from src.config.constants import (
    COOLING_C2,
    COOLING_TAU,
    VENT_C2,
    VENT_TAU,
    SimulationConstants,
)
from src.config.fire_constants import (
    CONVECTION_RATE,
    ENERGY_TO_TEMPERATURE,
    FUEL_CONSUMPTION_RATIO,
    FUEL_POWER_MAX,
    FUEL_POWER_MIN,
    HEAT_NONLINEARITY_EXPONENT,
    RADIATION_DOWN_RATE,
    REACH_FALLOFF_EXPONENT,
    VERTICAL_TRANSFER_THRESHOLD,
    WALL_SYNC_RATE,
    compute_reach_falloff_param,
    get_after_burn_param,
    get_diffusion_rate_param,
    get_fuel_capacity_param,
    get_heat_rate_param,
    get_ignition_temperature_param,
    get_reach_radius_param,
    make_default_tables,
)

P_ENERGY_TO_TEMPERATURE = 0
P_FUEL_POWER_MIN = 1
P_FUEL_POWER_MAX = 2
P_VERTICAL_TRANSFER_THRESHOLD = 3
P_WALL_SYNC_RATE = 4
P_CONVECTION_RATE = 5
P_RADIATION_DOWN_RATE = 6
P_FUEL_CONSUMPTION_RATIO = 7
P_HEAT_NONLINEARITY_EXPONENT = 8
P_REACH_FALLOFF_EXPONENT = 9
P_EX_MIN = 10
P_EX_MAX = 11
P_HRR_SCALE = 12
P_IGNITION_SHIFT = 13
P_REACH_SCALE = 14
P_DIFFUSION_SCALE = 15


def make_default_params(ex_min: float = 0.2, ex_max: float = 2.0) -> np.ndarray:
    return np.array(
        [
            ENERGY_TO_TEMPERATURE,
            FUEL_POWER_MIN,
            FUEL_POWER_MAX,
            VERTICAL_TRANSFER_THRESHOLD,
            WALL_SYNC_RATE,
            CONVECTION_RATE,
            RADIATION_DOWN_RATE,
            FUEL_CONSUMPTION_RATIO,
            HEAT_NONLINEARITY_EXPONENT,
            REACH_FALLOFF_EXPONENT,
            ex_min,
            ex_max,
            1.0,
            0.0,
            1.0,
            1.0,
        ],
        dtype=np.float32,
    )


@njit(fastmath=True, cache=True)
def is_void_cell(nav_val, void_code):
    return nav_val == void_code


@njit(fastmath=True, cache=True)
def is_radiation_blocker(nav_val, fuel_power_val, wall_code, void_code):
    if nav_val == void_code:
        return True
    if nav_val == wall_code and fuel_power_val > 0.0:
        return True
    return False


@njit(fastmath=True, cache=True)
def is_line_blocked(y0, x0, y1, x1, nav, fuel_power_map, H, W, wall_code, void_code):
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy

    y = y0
    x = x0

    while True:
        if y == y1 and x == x1:
            break

        e2 = err + err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy

        if y == y1 and x == x1:
            break

        if y < 0 or y >= H or x < 0 or x >= W:
            return True

        if is_radiation_blocker(nav[y, x], fuel_power_map[y, x], wall_code, void_code):
            return True

    return False


@njit(fastmath=True, cache=True)
def apply_preheat_damage(
    temperature,
    fuel_power,
    fuel_remaining,
    dt,
    params,
    ign_table,
    cap_table,
):
    if fuel_power <= 0.0:
        return fuel_remaining

    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    ign_shift = params[P_IGNITION_SHIFT]

    ignition_temp = get_ignition_temperature_param(
        fuel_power, ign_table, fp_min, fp_max, ign_shift
    )
    cap = get_fuel_capacity_param(fuel_power, cap_table, fp_min, fp_max)

    if fuel_remaining < cap:
        return fuel_remaining

    hold_drop = 60.0 + 6.0 * fuel_power
    hold_temp = ignition_temp - hold_drop

    if temperature <= hold_temp:
        return fuel_remaining
    if temperature >= ignition_temp:
        return fuel_remaining

    span = ignition_temp - hold_temp
    if span <= 1e-6:
        return fuel_remaining

    t = (temperature - hold_temp) / span
    if t < 0.0:
        t = 0.0
    if t > 1.0:
        t = 1.0

    rate = 0.15 * (fuel_power / fp_max) * (fuel_power / fp_max)
    dmg = cap * rate * t * dt

    new_rem = fuel_remaining - dmg
    if new_rem < 0.0:
        new_rem = 0.0
    return new_rem


@njit(fastmath=True, cache=True)
def compute_cell_heat_generation(
    temperature,
    fuel_power,
    fuel_remaining,
    dt,
    params,
    ign_table,
    heat_table,
    cap_table,
):
    if fuel_power <= 0.0:
        return 0.0

    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    ign_shift = params[P_IGNITION_SHIFT]
    heat_nl = params[P_HEAT_NONLINEARITY_EXPONENT]
    ex_min = params[P_EX_MIN]
    ex_max = params[P_EX_MAX]
    hrr_scale = params[P_HRR_SCALE]

    ignition_temp = get_ignition_temperature_param(
        fuel_power, ign_table, fp_min, fp_max, ign_shift
    )
    cap = get_fuel_capacity_param(fuel_power, cap_table, fp_min, fp_max)

    burned_already = fuel_remaining < cap

    hold_drop = 60.0 + 6.0 * fuel_power
    hold_temp = ignition_temp - hold_drop
    threshold = hold_temp if burned_already else ignition_temp

    if temperature < threshold:
        return 0.0

    heat_rate = get_heat_rate_param(
        fuel_power, heat_table, fp_min, fp_max, heat_nl, hrr_scale
    )

    if temperature < ignition_temp:
        if burned_already:
            span = ignition_temp - hold_temp
            if span <= 1e-6:
                excess_factor = 1.0
            else:
                t = (temperature - hold_temp) / span
                if t < 0.0:
                    t = 0.0
                if t > 1.0:
                    t = 1.0
                excess_factor = ex_min if t < ex_min else t
        else:
            excess_factor = ex_min
    else:
        val = (temperature - ignition_temp) / 100.0
        if val < ex_min:
            val = ex_min
        if val > ex_max:
            val = ex_max
        excess_factor = val

    if excess_factor < ex_min:
        excess_factor = ex_min

    return heat_rate * excess_factor * dt


@njit(fastmath=True, cache=True)
def apply_radiative_heat(
    y,
    x,
    heat_generated,
    fuel_power,
    fuel_power_map,
    nav,
    heat_delta,
    H,
    W,
    wall_code,
    void_code,
    params,
    reach_table,
):
    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    reach_scale = params[P_REACH_SCALE]
    falloff_exp = params[P_REACH_FALLOFF_EXPONENT]

    reach = get_reach_radius_param(fuel_power, reach_table, fp_min, fp_max, reach_scale)

    heat_delta[y, x] += heat_generated

    distributed_heat = heat_generated * 0.35
    total_weight = 0.0

    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            if is_void_cell(nav[ny, nx], void_code):
                continue

            if is_line_blocked(
                y, x, ny, nx, nav, fuel_power_map, H, W, wall_code, void_code
            ):
                continue

            dist = (dy * dy + dx * dx) ** 0.5
            if dist > reach:
                continue

            falloff = compute_reach_falloff_param(dist, reach, falloff_exp)
            inv_sq = 1.0 / (1.0 + dist * dist)
            weight = falloff * inv_sq
            total_weight += weight

    if total_weight <= 0.0:
        return

    heat_delta[y, x] -= distributed_heat

    for dy in range(-reach, reach + 1):
        for dx in range(-reach, reach + 1):
            if dy == 0 and dx == 0:
                continue

            ny = y + dy
            nx = x + dx

            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue

            if is_void_cell(nav[ny, nx], void_code):
                continue

            if is_line_blocked(
                y, x, ny, nx, nav, fuel_power_map, H, W, wall_code, void_code
            ):
                continue

            dist = (dy * dy + dx * dx) ** 0.5
            if dist > reach:
                continue

            falloff = compute_reach_falloff_param(dist, reach, falloff_exp)
            inv_sq = 1.0 / (1.0 + dist * dist)
            weight = falloff * inv_sq
            neighbor_heat = distributed_heat * (weight / total_weight)
            heat_delta[ny, nx] += neighbor_heat


@njit(fastmath=True, cache=True)
def compute_diffusion_heat(
    y,
    x,
    temperature,
    fuel_power,
    nav,
    H,
    W,
    dt,
    wall_code,
    void_code,
    params,
    diff_table,
):
    if is_void_cell(nav[y, x], void_code):
        return 0.0

    center_temp = temperature[y, x]
    center_fp = fuel_power[y, x]

    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    diff_scale = params[P_DIFFUSION_SCALE]
    energy_to_temp = params[P_ENERGY_TO_TEMPERATURE]

    total_delta = float32(0.0)

    neighbors = ((0, 1), (0, -1), (1, 0), (-1, 0))

    for dy, dx in neighbors:
        ny = y + dy
        nx = x + dx

        if ny < 0 or ny >= H or nx < 0 or nx >= W:
            continue

        if is_void_cell(nav[ny, nx], void_code):
            continue

        neighbor_temp = temperature[ny, nx]
        neighbor_fp = fuel_power[ny, nx]
        temp_diff = neighbor_temp - center_temp

        dr0 = get_diffusion_rate_param(
            center_fp, diff_table, fp_min, fp_max, diff_scale
        )
        dr1 = get_diffusion_rate_param(
            neighbor_fp, diff_table, fp_min, fp_max, diff_scale
        )
        diffusion_rate = 0.5 * (dr0 + dr1)

        heat_flow = temp_diff * diffusion_rate * dt
        energy_flow = heat_flow / energy_to_temp
        total_delta += energy_flow

    return total_delta


@njit(fastmath=True, cache=True)
def has_void_neighbor(nav, y, x, H, W, void_code):
    if y > 0 and nav[y - 1, x] == void_code:
        return True
    if y + 1 < H and nav[y + 1, x] == void_code:
        return True
    if x > 0 and nav[y, x - 1] == void_code:
        return True
    if x + 1 < W and nav[y, x + 1] == void_code:
        return True
    return False


@njit(fastmath=True, cache=True)
def _consume_fuel_cell(
    fuel_power_grid,
    fuel_remaining_grid,
    y,
    x,
    heat_gen,
    params,
    after_burn_discrete,
    cap_table,
):
    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    consume_ratio = params[P_FUEL_CONSUMPTION_RATIO]

    fp = fuel_power_grid[y, x]
    consumed = heat_gen * consume_ratio * (1.0 + fp / fp_max)
    rem = fuel_remaining_grid[y, x] - consumed

    if rem <= 0.0:
        new_fp = get_after_burn_param(fp, after_burn_discrete, fp_min, fp_max)
        if new_fp > fp:
            new_fp = fp

        fuel_power_grid[y, x] = new_fp
        cap = get_fuel_capacity_param(new_fp, cap_table, fp_min, fp_max)
        new_rem = cap + rem

        if new_fp <= 0.0 or new_rem <= 0.0:
            fuel_remaining_grid[y, x] = 0.0
        else:
            if new_rem > cap:
                new_rem = cap
            fuel_remaining_grid[y, x] = new_rem
    else:
        fuel_remaining_grid[y, x] = rem


@njit(fastmath=True, cache=True)
def _process_layer_cell(
    y,
    x,
    temp_grid,
    fuel_power_grid,
    fuel_remaining_grid,
    heat_delta_grid,
    burn_energy_grid,
    nav,
    H,
    W,
    dt,
    wall_code,
    void_code,
    params,
    ign_table,
    heat_table,
    cap_table,
    reach_table,
    after_burn_discrete,
):
    fuel_remaining_grid[y, x] = apply_preheat_damage(
        temp_grid[y, x],
        fuel_power_grid[y, x],
        fuel_remaining_grid[y, x],
        dt,
        params,
        ign_table,
        cap_table,
    )

    heat_gen = compute_cell_heat_generation(
        temp_grid[y, x],
        fuel_power_grid[y, x],
        fuel_remaining_grid[y, x],
        dt,
        params,
        ign_table,
        heat_table,
        cap_table,
    )

    if heat_gen > 0.0:
        burn_energy_grid[y, x] += heat_gen
        apply_radiative_heat(
            y,
            x,
            heat_gen,
            fuel_power_grid[y, x],
            fuel_power_grid,
            nav,
            heat_delta_grid,
            H,
            W,
            wall_code,
            void_code,
            params,
            reach_table,
        )

    _consume_fuel_cell(
        fuel_power_grid,
        fuel_remaining_grid,
        y,
        x,
        heat_gen,
        params,
        after_burn_discrete,
        cap_table,
    )


@njit(fastmath=True, cache=True)
def compute_vertical_transfer(
    y,
    x,
    temp_floor,
    temp_ceil,
    fuel_floor,
    fuel_ceil,
    nav,
    dt,
    wall_code,
    void_code,
    params,
):
    nav_val = nav[y, x]
    floor_temp = temp_floor[y, x]
    ceil_temp = temp_ceil[y, x]
    temp_diff = abs(floor_temp - ceil_temp)

    wall_active = (nav_val == wall_code) and (
        (fuel_floor[y, x] > 0.0) or (fuel_ceil[y, x] > 0.0)
    )

    energy_to_temp = params[P_ENERGY_TO_TEMPERATURE]
    wall_sync = params[P_WALL_SYNC_RATE]
    vertical_thr = params[P_VERTICAL_TRANSFER_THRESHOLD]
    conv_rate = params[P_CONVECTION_RATE]
    rad_down = params[P_RADIATION_DOWN_RATE]

    if wall_active:
        avg_temp = (floor_temp + ceil_temp) * 0.5
        sync_amount = wall_sync * dt

        d_floor = (avg_temp - floor_temp) * sync_amount
        d_ceil = (avg_temp - ceil_temp) * sync_amount

        return (d_floor / energy_to_temp, d_ceil / energy_to_temp)

    k = temp_diff / (temp_diff + vertical_thr)

    if floor_temp > ceil_temp:
        transfer = (floor_temp - ceil_temp) * conv_rate * dt * k
        d_floor = -transfer
        d_ceil = transfer
    else:
        transfer = (ceil_temp - floor_temp) * rad_down * dt * k
        d_floor = transfer
        d_ceil = -transfer

    return (d_floor / energy_to_temp, d_ceil / energy_to_temp)


@njit(fastmath=True, cache=True)
def update_fire_kernel(
    temp_floor,
    temp_ceil,
    fuel_floor,
    fuel_ceil,
    fuel_remaining_floor,
    fuel_remaining_ceil,
    heat_delta_floor,
    heat_delta_ceil,
    burn_energy_floor,
    burn_energy_ceil,
    nav,
    params,
    tables,
    dt=SimulationConstants.DT,
    burn_tau=0.0,
    wall_code=SimulationConstants.WALL_NAV_CODE,
    void_code=SimulationConstants.VOID_NAV_CODE,
    ambient_temp=SimulationConstants.AMBIENT_TEMPERATURE,
    max_temp=SimulationConstants.MAX_TEMPERATURE,
) -> tuple:
    ign_table, heat_table, diff_table, reach_table, cap_table, after_burn_discrete = (
        tables
    )

    H, W = temp_floor.shape

    decay = 1.0
    if burn_tau > 0.0:
        decay = exp(-dt / burn_tau)

    heat_delta_floor[:] = 0.0
    heat_delta_ceil[:] = 0.0

    for y in range(H):
        for x in range(W):
            if is_void_cell(nav[y, x], void_code):
                burn_energy_floor[y, x] = 0.0
                continue
            burn_energy_floor[y, x] *= decay
            _process_layer_cell(
                y,
                x,
                temp_floor,
                fuel_floor,
                fuel_remaining_floor,
                heat_delta_floor,
                burn_energy_floor,
                nav,
                H,
                W,
                dt,
                wall_code,
                void_code,
                params,
                ign_table,
                heat_table,
                cap_table,
                reach_table,
                after_burn_discrete,
            )

    for y in range(H):
        for x in range(W):
            if is_void_cell(nav[y, x], void_code):
                burn_energy_ceil[y, x] = 0.0
                continue
            burn_energy_ceil[y, x] *= decay
            _process_layer_cell(
                y,
                x,
                temp_ceil,
                fuel_ceil,
                fuel_remaining_ceil,
                heat_delta_ceil,
                burn_energy_ceil,
                nav,
                H,
                W,
                dt,
                wall_code,
                void_code,
                params,
                ign_table,
                heat_table,
                cap_table,
                reach_table,
                after_burn_discrete,
            )

    for y in range(H):
        for x in range(W):
            if is_void_cell(nav[y, x], void_code):
                continue

            diff_floor = compute_diffusion_heat(
                y,
                x,
                temp_floor,
                fuel_floor,
                nav,
                H,
                W,
                dt,
                wall_code,
                void_code,
                params,
                diff_table,
            )
            heat_delta_floor[y, x] += diff_floor

            diff_ceil = compute_diffusion_heat(
                y,
                x,
                temp_ceil,
                fuel_ceil,
                nav,
                H,
                W,
                dt,
                wall_code,
                void_code,
                params,
                diff_table,
            )
            heat_delta_ceil[y, x] += diff_ceil

    for y in range(H):
        for x in range(W):
            if is_void_cell(nav[y, x], void_code):
                continue

            dE_floor, dE_ceil = compute_vertical_transfer(
                y,
                x,
                temp_floor,
                temp_ceil,
                fuel_floor,
                fuel_ceil,
                nav,
                dt,
                wall_code,
                void_code,
                params,
            )
            heat_delta_floor[y, x] += dE_floor
            heat_delta_ceil[y, x] += dE_ceil

    fp_min = params[P_FUEL_POWER_MIN]
    fp_max = params[P_FUEL_POWER_MAX]
    energy_to_temp = params[P_ENERGY_TO_TEMPERATURE]
    heat_nl = params[P_HEAT_NONLINEARITY_EXPONENT]
    hrr_scale = params[P_HRR_SCALE]

    for y in range(H):
        for x in range(W):
            if is_void_cell(nav[y, x], void_code):
                continue

            fp_f = fuel_floor[y, x]
            hr_f = get_heat_rate_param(
                fp_f, heat_table, fp_min, fp_max, heat_nl, hrr_scale
            )
            cap_dt_floor = max(1.0, hr_f) * 2.0 * dt

            dT_floor = heat_delta_floor[y, x] * energy_to_temp
            if dT_floor > cap_dt_floor:
                dT_floor = cap_dt_floor
            elif dT_floor < -cap_dt_floor:
                dT_floor = -cap_dt_floor

            t_floor = temp_floor[y, x] + dT_floor

            fp_c = fuel_ceil[y, x]
            hr_c = get_heat_rate_param(
                fp_c, heat_table, fp_min, fp_max, heat_nl, hrr_scale
            )
            cap_dt_ceil = max(1.0, hr_c) * 2.0 * dt

            dT_ceil = heat_delta_ceil[y, x] * energy_to_temp
            if dT_ceil > cap_dt_ceil:
                dT_ceil = cap_dt_ceil
            elif dT_ceil < -cap_dt_ceil:
                dT_ceil = -cap_dt_ceil

            t_ceil = temp_ceil[y, x] + dT_ceil

            tau = (
                VENT_TAU
                if has_void_neighbor(nav, y, x, H, W, void_code)
                else COOLING_TAU
            )
            c2 = VENT_C2 if tau == VENT_TAU else COOLING_C2

            t_floor += (ambient_temp - t_floor) * (dt / tau)
            t_ceil += (ambient_temp - t_ceil) * (dt / tau)

            df = t_floor - ambient_temp
            dc = t_ceil - ambient_temp

            t_floor -= (df * abs(df)) * c2 * dt
            t_ceil -= (dc * abs(dc)) * c2 * dt

            if t_floor < ambient_temp:
                t_floor = ambient_temp
            if t_ceil < ambient_temp:
                t_ceil = ambient_temp
            if t_floor > max_temp:
                t_floor = max_temp
            if t_ceil > max_temp:
                t_ceil = max_temp

            temp_floor[y, x] = t_floor
            temp_ceil[y, x] = t_ceil

    heat_delta_floor[:] = 0.0
    heat_delta_ceil[:] = 0.0

    return (
        temp_floor,
        temp_ceil,
        fuel_floor,
        fuel_ceil,
        fuel_remaining_floor,
        fuel_remaining_ceil,
    )


def update_fire_kernel_default(
    *args, ex_min: float = 0.2, ex_max: float = 2.0, **kwargs
):
    params = make_default_params(ex_min=ex_min, ex_max=ex_max)
    tables = make_default_tables()
    return update_fire_kernel(*args, params=params, tables=tables, **kwargs)
