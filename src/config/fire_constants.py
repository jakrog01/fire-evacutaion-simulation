import numpy as np
from numba import njit, prange

FUEL_POWER_MIN = 0.0
FUEL_POWER_MAX = 11.0
TABLE_RESOLUTION = int(FUEL_POWER_MAX - FUEL_POWER_MIN + 1)
HEAT_NONLINEARITY_EXPONENT = 2.0
REACH_FALLOFF_EXPONENT = 1
FUEL_CONSUMPTION_RATIO = 1.0

VERTICAL_TRANSFER_THRESHOLD = 40.0
WALL_SYNC_RATE = 200
CONVECTION_RATE = 0.08
RADIATION_DOWN_RATE = 0.008

ENERGY_TO_TEMPERATURE = 0.2


IGNITION_MAP = {
    0: 2000.0,
    4: 1200.0,
    6: 240,
    8: 320.0,
    10: 150.0,
    11: 130.0,
}

HRR_MAP = {
    0: 0.000,
    3: 10.0,
    4: 20.0,
    6: 40.0,
    8: 70.0,
    9: 150.0,
    10: 420.0,
    11: 50.0,
}

DIFFUSION_MAP = {
    0: 0.02,
    3: 0.10,
    6: 0.18,
    8: 0.25,
    10: 0.4,
    11: 0.01,
}

AFTER_BURN_MAP = {
    0: 0.0,
    3: 2.0,
    4: 2.0,
    6: 3.0,
    8: 6.0,
    10: 6.0,
    11: 10.0,
}

REACH_RADIUS_MAP = {
    0: 1,
    3: 2,
    6: 3,
    8: 5,
    10: 3,
    11: 1,
}

FUEL_CAPACITY_MAP = {
    0: 0.0,
    3: 60000.0,
    4: 120000.0,
    6: 350000.0,
    8: 2200000.0,
    10: 450000.0,
    11: 1200.0,
}


def _generate_lookup_table(control_map):
    x_points = np.array(sorted(control_map.keys()), dtype=np.float64)
    y_points = np.array([control_map[k] for k in x_points], dtype=np.float64)

    table = np.empty(TABLE_RESOLUTION, dtype=np.float32)

    for i in range(TABLE_RESOLUTION):
        fp = float(i + FUEL_POWER_MIN)
        val = np.interp(fp, x_points, y_points)
        table[i] = np.float32(val)

    for k, v in control_map.items():
        idx = int(k - FUEL_POWER_MIN)
        if 0 <= idx < TABLE_RESOLUTION:
            table[idx] = np.float32(v)

    return table


_IGNITION_TABLE = _generate_lookup_table(IGNITION_MAP)
_HEAT_TABLE = _generate_lookup_table(HRR_MAP)
_DIFFUSION_TABLE = _generate_lookup_table(DIFFUSION_MAP)
_REACH_TABLE = _generate_lookup_table(REACH_RADIUS_MAP)
_FUEL_CAPACITY_TABLE = _generate_lookup_table(FUEL_CAPACITY_MAP)


@njit(fastmath=True, cache=True)
def _lookup(fuel_power, table):
    if fuel_power <= FUEL_POWER_MIN:
        return table[0]
    if fuel_power >= FUEL_POWER_MAX:
        return table[-1]

    idx = int(fuel_power - FUEL_POWER_MIN)
    max_idx = len(table) - 1
    if idx >= max_idx:
        return table[max_idx]

    base = idx + FUEL_POWER_MIN
    t = fuel_power - base

    val_a = table[idx]
    val_b = table[idx + 1]
    return val_a + (val_b - val_a) * t


_AFTER_BURN_DISCRETE = np.zeros(TABLE_RESOLUTION, dtype=np.float32)

_keys = sorted(AFTER_BURN_MAP.keys())
_min_k = _keys[0]
_max_k = _keys[-1]

for i in range(TABLE_RESOLUTION):
    fp_i = i + int(FUEL_POWER_MIN)
    k = fp_i
    if k > _max_k:
        k = _max_k
    while k not in AFTER_BURN_MAP and k > _min_k:
        k -= 1
    v = float(AFTER_BURN_MAP.get(k, 0.0))
    if v < 0.0:
        v = 0.0
    if v > fp_i:
        v = float(fp_i)
    _AFTER_BURN_DISCRETE[i] = np.float32(v)


@njit(fastmath=True, cache=True)
def get_after_burn(fuel_power):
    fp = int(fuel_power)
    if fp <= int(FUEL_POWER_MIN):
        return np.float32(0.0)
    if fp >= int(FUEL_POWER_MAX):
        fp = int(FUEL_POWER_MAX)
    return _AFTER_BURN_DISCRETE[fp - int(FUEL_POWER_MIN)]


@njit(fastmath=True, cache=True)
def get_ignition_temperature(fuel_power):
    return _lookup(fuel_power, _IGNITION_TABLE)


@njit(fastmath=True, cache=True)
def get_fuel_capacity(fuel_power):
    return _lookup(fuel_power, _FUEL_CAPACITY_TABLE)


@njit(fastmath=True, cache=True)
def get_heat_rate(fuel_power):
    base_rate = _lookup(fuel_power, _HEAT_TABLE)

    if fuel_power > 5.0:
        boost = ((fuel_power - 5.0) / 5.0) ** HEAT_NONLINEARITY_EXPONENT
        return base_rate * (1.0 + boost)

    return base_rate


@njit(fastmath=True, cache=True)
def get_reach_radius(fuel_power):
    val = _lookup(fuel_power, _REACH_TABLE)
    radius = int(val + 0.5)
    return max(1, radius)


@njit(fastmath=True, cache=True)
def get_diffusion_rate(fuel_power):
    return _lookup(fuel_power, _DIFFUSION_TABLE)


@njit(fastmath=True, cache=True)
def compute_reach_falloff(distance, max_radius):
    if max_radius <= 0.0 or distance >= max_radius:
        return 0.0
    if distance <= 0.0:
        return 1.0

    normalized = distance / max_radius
    val = (1.0 - normalized) ** REACH_FALLOFF_EXPONENT

    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


@njit(parallel=True, fastmath=True, cache=True)
def map_fuel_capacity_grid(fuel_power):
    H, W = fuel_power.shape
    out = np.empty((H, W), dtype=np.float32)
    for y in prange(H):
        for x in range(W):
            out[y, x] = get_fuel_capacity(fuel_power[y, x])
    return out


@njit(fastmath=True, cache=True)
def _lookup_param(fuel_power, table, fp_min, fp_max):
    if fuel_power <= fp_min:
        return table[0]
    if fuel_power >= fp_max:
        return table[-1]

    idx = int(fuel_power - fp_min)
    max_idx = len(table) - 1
    if idx >= max_idx:
        return table[max_idx]

    base = idx + fp_min
    t = fuel_power - base
    val_a = table[idx]
    val_b = table[idx + 1]
    return val_a + (val_b - val_a) * t


@njit(fastmath=True, cache=True)
def get_ignition_temperature_param(
    fuel_power, ign_table, fp_min, fp_max, ignition_shift
):
    return _lookup_param(fuel_power, ign_table, fp_min, fp_max) + ignition_shift


@njit(fastmath=True, cache=True)
def get_fuel_capacity_param(fuel_power, cap_table, fp_min, fp_max):
    return _lookup_param(fuel_power, cap_table, fp_min, fp_max)


@njit(fastmath=True, cache=True)
def get_heat_rate_param(
    fuel_power, heat_table, fp_min, fp_max, heat_nonlinearity_exp, hrr_scale
):
    base = _lookup_param(fuel_power, heat_table, fp_min, fp_max) * hrr_scale
    if fuel_power > 5.0:
        boost = ((fuel_power - 5.0) / 5.0) ** heat_nonlinearity_exp
        return base * (1.0 + boost)
    return base


@njit(fastmath=True, cache=True)
def get_reach_radius_param(fuel_power, reach_table, fp_min, fp_max, reach_scale):
    val = _lookup_param(fuel_power, reach_table, fp_min, fp_max)
    r = int(val * reach_scale + 0.5)
    if r < 1:
        r = 1
    return r


@njit(fastmath=True, cache=True)
def get_diffusion_rate_param(fuel_power, diff_table, fp_min, fp_max, diffusion_scale):
    return _lookup_param(fuel_power, diff_table, fp_min, fp_max) * diffusion_scale


@njit(fastmath=True, cache=True)
def get_after_burn_param(fuel_power, after_burn_discrete, fp_min, fp_max):
    fp = int(fuel_power)
    if fp <= int(fp_min):
        return np.float32(0.0)
    if fp >= int(fp_max):
        fp = int(fp_max)
    return after_burn_discrete[fp - int(fp_min)]


@njit(fastmath=True, cache=True)
def compute_reach_falloff_param(distance, max_radius, reach_falloff_exponent):
    if max_radius <= 0.0 or distance >= max_radius:
        return 0.0
    if distance <= 0.0:
        return 1.0

    normalized = distance / max_radius
    val = (1.0 - normalized) ** reach_falloff_exponent

    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def make_default_tables():
    return (
        _IGNITION_TABLE,
        _HEAT_TABLE,
        _DIFFUSION_TABLE,
        _REACH_TABLE,
        _FUEL_CAPACITY_TABLE,
        _AFTER_BURN_DISCRETE,
    )


def map_fuel_capacity_grid_numpy(fuel_power: np.ndarray) -> np.ndarray:
    fp = fuel_power.astype(np.float32, copy=False)
    fp_clipped = np.clip(fp, np.float32(FUEL_POWER_MIN), np.float32(FUEL_POWER_MAX))
    idx0 = np.floor(fp_clipped - np.float32(FUEL_POWER_MIN)).astype(np.int32)
    idx1 = np.minimum(idx0 + 1, np.int32(TABLE_RESOLUTION - 1))
    t = fp_clipped - (idx0.astype(np.float32) + np.float32(FUEL_POWER_MIN))
    a = _FUEL_CAPACITY_TABLE[idx0]
    b = _FUEL_CAPACITY_TABLE[idx1]
    return a + (b - a) * t
