"""Shared feature construction for screening and population estimation."""
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from .config import FEATURE_NAMES, TaskBConfig
logger = logging.getLogger(__name__)
__all__ = ['household_features', 'window_features', 'feature_vector']
_EPS = 0.01

def _nan_safe(series: np.ndarray) -> np.ndarray:
    """Public function: _nan_safe."""
    x = np.asarray(series, dtype=float)
    if not np.isnan(x).any():
        return x
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])

def feature_vector(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, month_of_year: np.ndarray) -> np.ndarray:
    """Public function: feature_vector."""
    e = _nan_safe(elec)
    w = _nan_safe(water)
    g = _nan_safe(gas)
    moy = np.asarray(month_of_year, dtype=int)
    n = len(e)
    half = max(1, min(6, n // 3))
    by_month: Dict[int, float] = {}
    for m in np.unique(moy):
        by_month[int(m)] = float(np.mean(e[moy == m]))
    seasonal_vals = np.array(list(by_month.values())) if by_month else np.array([0.0])
    winter = by_month.get(1, 0.0) + by_month.get(12, 0.0)
    summer = by_month.get(7, 0.0) + by_month.get(8, 0.0)
    return np.array([np.mean(e), np.mean(w), np.mean(g), np.std(e), np.std(w), np.std(g), np.mean(np.diff(e) / (e[:-1] + _EPS)) if n > 1 else 0.0, np.mean(np.diff(w) / (w[:-1] + _EPS)) if n > 1 else 0.0, np.std(e) / (np.mean(e) + _EPS), np.std(w) / (np.mean(w) + _EPS), np.std(g) / (np.mean(g) + _EPS), (np.mean(e[-half:]) - np.mean(e[:half])) / (np.mean(e[:half]) + _EPS), float(seasonal_vals.max() - seasonal_vals.min()), winter / (summer + _EPS), np.mean(e) + 0.5 * np.mean(w) + 2 * np.mean(g)], dtype=float)

def household_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Public function: household_features."""
    rows: List[dict] = []
    for (hid, grp) in monthly.groupby('household_id', sort=True):
        grp = grp.sort_values('month')
        vec = feature_vector(grp['electricity_kwh'].values, grp['water_ton'].values, grp['gas_m3'].values, grp['month_of_year'].values)
        rows.append({'household_id': int(hid), **dict(zip(FEATURE_NAMES, vec))})
    out = pd.DataFrame(rows)
    logger.info('household features: %d x %d', len(out), len(FEATURE_NAMES))
    return out

def window_features(monthly: pd.DataFrame, cfg: TaskBConfig) -> pd.DataFrame:
    """Public function: window_features."""
    w = cfg.window_months
    rows: List[dict] = []
    for (hid, grp) in monthly.groupby('household_id', sort=True):
        grp = grp.sort_values('month')
        e = grp['electricity_kwh'].values
        wa = grp['water_ton'].values
        g = grp['gas_m3'].values
        moy = grp['month_of_year'].values
        pop = grp['true_population'].values
        for start in range(0, len(grp) - w + 1, cfg.window_stride):
            sl = slice(start, start + w)
            vals = np.unique(pop[sl])
            rows.append({'household_id': int(hid), 'window_start': int(start), **dict(zip(FEATURE_NAMES, feature_vector(e[sl], wa[sl], g[sl], moy[sl]))), 'population': int(pop[start + w - 1]), 'population_stable': bool(len(vals) == 1)})
    out = pd.DataFrame(rows)
    logger.info('window features: %d windows from %d households', len(out), monthly['household_id'].nunique())
    return out
