"""Synthetic electricity, water, and gas panel generator."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .config import BUILDING_PARAMS, BUILDING_TYPES, DataConfig, GENERATOR_REGIMES, PER_PERSON
logger = logging.getLogger(__name__)
__all__ = ['Panel', 'generate_panel']

@dataclass(frozen=True)
class Panel:
    """Public class: Panel."""
    meta: pd.DataFrame
    monthly: pd.DataFrame
    population: np.ndarray
    events: pd.DataFrame
    regime: str
    seed: int

def _seasonal_factor(month_of_year: int, amplitude: float) -> float:
    """Public function: _seasonal_factor."""
    if month_of_year in (12, 1, 2):
        return 1.0 + amplitude
    if month_of_year in (6, 7, 8):
        return 1.0 + amplitude * 0.7
    return 1.0 - amplitude * 0.3

def _vacation_factor(month_of_year: int, htype: str, drop: float) -> float:
    """Public function: _vacation_factor."""
    if month_of_year in (1, 2):
        if htype == 'residential':
            return 1.0 - drop
        if htype == 'commercial':
            return 1.0 + drop
        return 1.0
    if month_of_year in (7, 8) and htype == 'residential':
        return 1.0 - drop * 0.5
    return 1.0

def _assign_types(cfg: DataConfig, rng: np.random.RandomState) -> List[str]:
    types: List[str] = []
    for htype in BUILDING_TYPES:
        n = int(round(cfg.n_households * BUILDING_PARAMS[htype]['ratio']))
        types.extend([htype] * n)
    while len(types) < cfg.n_households:
        types.append('residential')
    types = types[:cfg.n_households]
    rng.shuffle(types)
    return types

def _population_series(htype: str, cfg: DataConfig, rng: np.random.RandomState) -> Tuple[np.ndarray, List[dict]]:
    """Public function: _population_series."""
    params = BUILDING_PARAMS[htype]
    (lo, hi) = params['pop_range']
    if hi == 0:
        return (np.zeros(cfg.n_months, dtype=int), [])
    base_pop = int(rng.randint(lo, hi + 1))
    series = np.full(cfg.n_months, base_pop, dtype=int)
    (ev_lo, ev_hi) = params['n_events']
    n_events = int(rng.randint(ev_lo, ev_hi + 1)) if ev_hi > 0 else 0
    if n_events == 0:
        return (series, [])
    latest_transition = cfg.n_months - 1 - cfg.event_followup_months
    first_start = max(1, cfg.event_start_month_min)
    candidates = list(range(first_start, latest_transition - 3))
    rng.shuffle(candidates)
    episodes: List[Tuple[int, int, int]] = []
    for start in candidates:
        if len(episodes) >= n_events:
            break
        max_duration = min(12, latest_transition - start)
        if max_duration < 4:
            continue
        duration = int(rng.randint(4, max_duration + 1))
        end = start + duration
        if any((not (end < other_start or other_end < start) for (other_start, other_end, _) in episodes)):
            continue
        possible = [int(np.clip(base_pop + d, max(lo, 0), hi + 4)) for d in (-2, -1, 1, 2)]
        possible = sorted(set((value for value in possible if value != base_pop)))
        if not possible:
            continue
        new_pop = int(rng.choice(possible))
        episodes.append((start, end, new_pop))
    for (start, end, new_pop) in sorted(episodes):
        series[start:end] = new_pop
    transitions = np.flatnonzero(np.diff(series) != 0) + 1
    events: List[dict] = []
    for t in transitions:
        delta = int(series[t] - series[t - 1])
        events.append({'start_month': int(t), 'delta': delta, 'direction': int(np.sign(delta)), 'event_kind': 'reversion' if series[t] == base_pop else 'onset'})
    return (series, events)

def generate_panel(cfg: Optional[DataConfig]=None) -> Panel:
    """Public function: generate_panel."""
    cfg = cfg or DataConfig()
    if cfg.regime not in GENERATOR_REGIMES:
        raise KeyError('unknown generator regime %r; known regimes: %s' % (cfg.regime, sorted(GENERATOR_REGIMES)))
    regime = GENERATOR_REGIMES[cfg.regime]
    latent_rng = np.random.RandomState(cfg.seed)
    signal_rng = np.random.RandomState(cfg.seed + 1000001)
    shared_rng = np.random.RandomState(cfg.seed + 1000003)
    missing_rng = np.random.RandomState(cfg.seed + 1000033)
    types = _assign_types(cfg, latent_rng)
    (pe, pw, pg) = PER_PERSON
    exponent = regime['exponent']
    spread = regime['coef_spread']
    noise_scale = regime['noise_scale']
    noise_corr = regime['noise_corr']
    population = np.zeros((cfg.n_households, cfg.n_months), dtype=int)
    records: List[dict] = []
    event_rows: List[dict] = []
    meta_rows: List[dict] = []
    n_shared = int(round(cfg.n_households * regime['shared_meter_rate']))
    shared_ids = sorted((int(h) for h in shared_rng.choice(cfg.n_households, n_shared, replace=False))) if n_shared else []
    shared_map: Dict[int, Tuple[int, float]] = {}
    for hid in shared_ids:
        candidates = np.delete(np.arange(cfg.n_households), hid)
        neighbour = int(shared_rng.choice(candidates))
        shared_map[int(hid)] = (neighbour, float(shared_rng.uniform(0.2, 0.6)))
    for hid in range(cfg.n_households):
        htype = types[hid]
        params = BUILDING_PARAMS[htype]
        (pop_series, events) = _population_series(htype, cfg, latent_rng)
        population[hid] = pop_series
        for ev in events:
            event_rows.append(dict(household_id=hid, true_type=htype, **ev))
        e_base = signal_rng.uniform(*params['elec_base'])
        w_base = signal_rng.uniform(*params['water_base'])
        g_base = signal_rng.uniform(*params['gas_base'])
        k_e = pe * (1.0 + spread * signal_rng.uniform(-1, 1))
        k_w = pw * (1.0 + spread * signal_rng.uniform(-1, 1))
        k_g = pg * (1.0 + spread * signal_rng.uniform(-1, 1))
        for m in range(cfg.n_months):
            pop = int(pop_series[m])
            moy = m % 12 + 1
            sf = _seasonal_factor(moy, params['seasonal_amp'])
            vf = _vacation_factor(moy, htype, params['vacation_drop'])
            occ = pop ** exponent if pop > 0 else 0.0
            e_mean = (e_base + k_e * occ) * sf * vf
            w_mean = (w_base + k_w * occ) * sf * vf
            g_mean = (g_base + k_g * occ) * sf * vf
            z_common = signal_rng.normal(0.0, 1.0)
            zs = [np.sqrt(noise_corr) * z_common + np.sqrt(max(0.0, 1.0 - noise_corr)) * signal_rng.normal(0.0, 1.0) for _ in range(3)]
            sd = np.array([0.08, 0.1, 0.12]) * noise_scale
            elec = max(0.5, e_mean * (1.0 + sd[0] * zs[0]))
            water = max(0.0, w_mean * (1.0 + sd[1] * zs[1]))
            gas = max(0.0, g_mean * (1.0 + sd[2] * zs[2]))
            records.append({'household_id': hid, 'month': m, 'year': cfg.start_year + m // 12, 'month_of_year': moy, 'electricity_kwh': round(elec, 2) if np.isfinite(elec) else np.nan, 'water_ton': round(water, 2) if np.isfinite(water) else np.nan, 'gas_m3': round(gas, 2) if np.isfinite(gas) else np.nan, 'true_population': pop, 'true_type': htype})
        meta_rows.append({'household_id': hid, 'true_type': htype, 'base_population': int(pop_series[0]), 'mean_population': float(pop_series.mean()), 'n_events': len(events), 'shared_meter': hid in shared_map, 'shared_with_household_id': shared_map[hid][0] if hid in shared_map else -1, 'shared_load_fraction': shared_map[hid][1] if hid in shared_map else 0.0})
    monthly = pd.DataFrame(records)
    utility_cols = ['electricity_kwh', 'water_ton', 'gas_m3']
    unmixed = monthly[['household_id', *utility_cols]].copy()
    for hid in sorted(shared_map):
        (neighbour, fraction) = shared_map[hid]
        target_idx = monthly.index[monthly['household_id'] == hid]
        target_values = unmixed.loc[unmixed['household_id'] == hid, utility_cols].to_numpy(copy=True)
        neighbour_values = unmixed.loc[unmixed['household_id'] == neighbour, utility_cols].to_numpy(copy=True)
        monthly.loc[target_idx, utility_cols] = target_values + fraction * neighbour_values
    if regime['missing_rate']:
        missing = missing_rng.uniform(size=len(monthly)) < regime['missing_rate']
        monthly.loc[missing, utility_cols] = np.nan
    monthly[utility_cols] = monthly[utility_cols].round(2)
    meta = pd.DataFrame(meta_rows)
    events_df = pd.DataFrame(event_rows) if event_rows else pd.DataFrame(columns=['household_id', 'true_type', 'start_month', 'delta', 'direction', 'event_kind'])
    logger.info('generated panel: regime=%s seed=%d households=%d months=%d records=%d events=%d', cfg.regime, cfg.seed, cfg.n_households, cfg.n_months, len(monthly), len(events_df))
    return Panel(meta=meta, monthly=monthly, population=population, events=events_df, regime=cfg.regime, seed=cfg.seed)
