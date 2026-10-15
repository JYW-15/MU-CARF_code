"""End-to-end screening, estimation, and monitoring cascade."""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from .config import FEATURE_NAMES, PipelineConfig
from .features import window_features
from .task_a_screening import TaskAResult
from .task_b_regression import TaskBResult, WeightedEnsembleRegressor, select_omega
logger = logging.getLogger(__name__)
__all__ = ['CascadeResult', 'run_cascade', 'population_trajectories']

@dataclass
class CascadeResult:
    """Public class: CascadeResult."""
    coverage: pd.DataFrame
    regression: pd.DataFrame
    error_propagation: pd.DataFrame
    trajectories: Dict[int, np.ndarray]
    omega_selected: float
    trajectory_train_households: np.ndarray
    evaluation_households: np.ndarray
    screened_in_households: np.ndarray

def population_trajectories(monthly: pd.DataFrame, model, cfg: PipelineConfig, household_ids: np.ndarray) -> Dict[int, np.ndarray]:
    """Public function: population_trajectories."""
    w = cfg.task_b.window_months
    out: Dict[int, np.ndarray] = {}
    sub = monthly[monthly['household_id'].isin(set((int(h) for h in household_ids)))]
    from .features import feature_vector
    for (hid, grp) in sub.groupby('household_id'):
        grp = grp.sort_values('month')
        e = grp['electricity_kwh'].values
        wa = grp['water_ton'].values
        g = grp['gas_m3'].values
        moy = grp['month_of_year'].values
        n = len(grp)
        traj = np.full(n, np.nan)
        (rows, idx) = ([], [])
        for t in range(w - 1, n):
            sl = slice(t - w + 1, t + 1)
            rows.append(feature_vector(e[sl], wa[sl], g[sl], moy[sl]))
            idx.append(t)
        if rows:
            traj[idx] = model.predict(np.array(rows))
        out[int(hid)] = traj
    return out

def run_cascade(panel, windows: pd.DataFrame, task_a: TaskAResult, task_b: TaskBResult, cfg: PipelineConfig) -> CascadeResult:
    """Public function: run_cascade."""
    occupied_types = {'residential', 'mixed'}
    test_hids = task_a.is_test[task_a.is_test].index.values
    true_type = panel.meta.set_index('household_id')['true_type']
    truly_occupied = np.array([true_type[h] in occupied_types for h in test_hids])
    pred_occupied = np.array([task_a.predicted_type[h] in occupied_types for h in test_hids])
    n_true = int(truly_occupied.sum())
    n_pred = int(pred_occupied.sum())
    missed = test_hids[truly_occupied & ~pred_occupied]
    spurious = test_hids[~truly_occupied & pred_occupied]
    coverage = pd.DataFrame([{'n_test_households': len(test_hids), 'n_truly_occupied': n_true, 'n_screened_in': n_pred, 'coverage_of_occupied': n_true and float((truly_occupied & pred_occupied).sum() / n_true), 'n_occupied_wrongly_excluded': int(len(missed)), 'n_unoccupied_wrongly_included': int(len(spurious)), 'screen_precision': n_pred and float((truly_occupied & pred_occupied).sum() / n_pred)}])
    train_hids = set(task_a.is_test[~task_a.is_test].index.values)
    train_occ = [h for h in train_hids if true_type[h] in occupied_types]
    tr = windows[windows['household_id'].isin(train_occ)]
    tr_X = tr[list(FEATURE_NAMES)].values
    tr_y = tr['population'].values.astype(float)
    tr_g = tr['household_id'].values
    (cascade_omega, _) = select_omega(tr_X, tr_y, tr_g, cfg.task_b)
    model = WeightedEnsembleRegressor(omega_mlp=cascade_omega, hidden=cfg.task_b.mlp_hidden, random_state=cfg.task_b.seed).fit(tr_X, tr_y, groups=tr_g)

    def _score_subset(hid_list: np.ndarray, label: str, use_regressor: bool=True) -> dict:
        sub = windows[windows['household_id'].isin(set((int(h) for h in hid_list)))]
        if sub.empty:
            return {'set': label, 'n_households': 0, 'n_windows': 0, 'MAE': float('nan'), 'within_1.0': float('nan')}
        y = sub['population'].values.astype(float)
        p = model.predict(sub[list(FEATURE_NAMES)].values) if use_regressor else np.zeros(len(sub), dtype=float)
        return {'set': label, 'n_households': int(sub['household_id'].nunique()), 'n_windows': int(len(sub)), 'MAE': float(mean_absolute_error(y, p)), 'within_1.0': float(np.mean(np.abs(y - p) < 1.0))}
    oracle_set = test_hids[truly_occupied]
    predicted_set = test_hids[pred_occupied]

    def _score_screen(active_hids: np.ndarray, label: str) -> dict:
        """Public function: _score_screen."""
        sub = windows[windows['household_id'].isin(set((int(h) for h in test_hids)))]
        y = sub['population'].values.astype(float)
        active = sub['household_id'].isin(set((int(h) for h in active_hids))).values
        p = np.zeros(len(sub), dtype=float)
        if active.any():
            p[active] = model.predict(sub.loc[active, list(FEATURE_NAMES)].values)
        return {'set': label, 'n_households': int(sub['household_id'].nunique()), 'n_windows': int(len(sub)), 'MAE': float(mean_absolute_error(y, p)), 'within_1.0': float(np.mean(np.abs(y - p) < 1.0))}
    regression = pd.DataFrame([_score_screen(oracle_set, 'Oracle screen (all test households)'), _score_screen(predicted_set, 'Task A screen (all test households)'), _score_subset(predicted_set, 'Screened-in subset (conditional MAE)')])
    rows: List[dict] = []
    for (label, ids, use_regressor) in [('Occupied, screened in', test_hids[truly_occupied & pred_occupied], True), ('Occupied, wrongly excluded', missed, False), ('Unoccupied, wrongly included', spurious, True)]:
        s = _score_subset(ids, label, use_regressor=use_regressor)
        true_pop = [float(panel.population[int(h)].mean()) for h in ids] if len(ids) else []
        rows.append({**s, 'mean_true_population': float(np.mean(true_pop)) if true_pop else float('nan')})
    error_propagation = pd.DataFrame(rows)
    traj = population_trajectories(panel.monthly, model, cfg, predicted_set)
    logger.info('Cascade: omega=%.2f, %d/%d occupied covered, %d unoccupied wrongly included, oracle MAE=%.4f vs screened MAE=%.4f', cascade_omega, int((truly_occupied & pred_occupied).sum()), n_true, len(spurious), regression.loc[0, 'MAE'], regression.loc[1, 'MAE'])
    return CascadeResult(coverage=coverage, regression=regression, error_propagation=error_propagation, trajectories=traj, omega_selected=cascade_omega, trajectory_train_households=np.array(sorted(train_occ), dtype=int), evaluation_households=np.array(sorted(test_hids), dtype=int), screened_in_households=np.array(sorted(predicted_set), dtype=int))
