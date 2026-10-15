"""Household-grouped population regression."""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from scipy import stats
from .config import FEATURE_GROUPS, FEATURE_NAMES, TaskBConfig
logger = logging.getLogger(__name__)
__all__ = ['BLEND_NAME', 'HouseholdGroupedMLPRegressor', 'WeightedEnsembleRegressor', 'TaskBResult', 'run_task_b', 'base_model_zoo', 'select_omega']
BLEND_NAME = 'One-parameter blend'

class HouseholdGroupedMLPRegressor(BaseEstimator, RegressorMixin):
    """Public class: HouseholdGroupedMLPRegressor."""

    def __init__(self, hidden: Tuple[int, ...]=(64, 32), alpha: float=0.001, batch_size: int=32, learning_rate_init: float=0.001, max_epochs: int=1000, validation_fraction: float=0.1, n_iter_no_change: int=10, tol: float=0.0001, random_state: int=42):
        self.hidden = hidden
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.random_state = random_state

    def _new_model(self) -> MLPRegressor:
        return MLPRegressor(hidden_layer_sizes=tuple(self.hidden), activation='relu', solver='adam', alpha=self.alpha, batch_size=self.batch_size, learning_rate_init=self.learning_rate_init, max_iter=1, early_stopping=False, random_state=self.random_state)

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]=None) -> 'HouseholdGroupedMLPRegressor':
        if groups is None:
            raise ValueError('household groups are required for grouped MLP stopping')
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        groups = np.asarray(groups)
        if len(X) != len(y) or len(X) != len(groups):
            raise ValueError('X, y, and groups must contain the same number of rows')
        if len(np.unique(groups)) < 2:
            raise ValueError('grouped MLP stopping requires at least two households')
        split = GroupShuffleSplit(n_splits=1, test_size=self.validation_fraction, random_state=self.random_state)
        (i_fit, i_val) = next(split.split(X, y, groups=groups))
        fit_households = np.unique(groups[i_fit])
        validation_households = np.unique(groups[i_val])
        if set(fit_households) & set(validation_households):
            raise AssertionError('household leaked into MLP stopping validation')
        selection_scaler = StandardScaler().fit(X[i_fit])
        X_fit = selection_scaler.transform(X[i_fit])
        X_val = selection_scaler.transform(X[i_val])
        selector = self._new_model()
        best_mae = float('inf')
        best_epoch = 1
        stale = 0
        validation_curve: List[float] = []
        for epoch in range(1, self.max_epochs + 1):
            selector.partial_fit(X_fit, y[i_fit])
            val_mae = float(mean_absolute_error(y[i_val], selector.predict(X_val)))
            validation_curve.append(val_mae)
            if best_mae - val_mae > self.tol:
                best_mae = val_mae
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= self.n_iter_no_change:
                break
        self.selected_epoch_ = int(best_epoch)
        self.selection_validation_mae_ = float(best_mae)
        self.validation_curve_ = np.asarray(validation_curve, dtype=float)
        self.fit_households_ = fit_households
        self.validation_households_ = validation_households
        self.scaler_ = StandardScaler().fit(X)
        X_all = self.scaler_.transform(X)
        self.model_ = self._new_model()
        for _ in range(self.selected_epoch_):
            self.model_.partial_fit(X_all, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model_.predict(self.scaler_.transform(np.asarray(X, dtype=float)))

class WeightedEnsembleRegressor(BaseEstimator, RegressorMixin):
    """Public class: WeightedEnsembleRegressor."""

    def __init__(self, omega_mlp: float=0.3, hidden: Tuple[int, ...]=(64, 32), random_state: int=42):
        self.omega_mlp = omega_mlp
        self.hidden = hidden
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray, groups: Optional[np.ndarray]=None) -> 'WeightedEnsembleRegressor':
        if groups is None:
            raise ValueError('household groups are required to fit the one-parameter blend')
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        self.mlp_ = HouseholdGroupedMLPRegressor(hidden=tuple(self.hidden), random_state=self.random_state).fit(X, y, groups=groups)
        self.gbrt_ = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=self.random_state).fit(Xs, y)
        self.svr_ = SVR(kernel='rbf', C=10, gamma='scale').fit(Xs, y)
        self.rf_ = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=self.random_state).fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(X)
        others = (self.gbrt_.predict(Xs) + self.svr_.predict(Xs) + self.rf_.predict(Xs)) / 3.0
        return self.omega_mlp * self.mlp_.predict(X) + (1.0 - self.omega_mlp) * others

    def component_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Public function: component_predictions."""
        Xs = self.scaler_.transform(X)
        return {'MLP': self.mlp_.predict(X), 'GBRT': self.gbrt_.predict(Xs), 'SVR': self.svr_.predict(Xs), 'RF': self.rf_.predict(Xs)}

def base_model_zoo(cfg: TaskBConfig) -> Dict[str, BaseEstimator]:
    """Public function: base_model_zoo."""
    return {'Linear Regression': LinearRegression(), 'Ridge': Ridge(alpha=1.0), 'SVR (RBF)': SVR(kernel='rbf', C=10, gamma='scale'), 'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=cfg.seed), 'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=cfg.seed), 'MLP (64-32)': HouseholdGroupedMLPRegressor(hidden=(64, 32), random_state=cfg.seed), 'MLP (128-64-32)': HouseholdGroupedMLPRegressor(hidden=(128, 64, 32), random_state=cfg.seed)}

@dataclass
class TaskBResult:
    """Public class: TaskBResult."""
    comparison: pd.DataFrame
    omega_selected: float
    omega_per_fold: List[float]
    omega_grid_dev: pd.DataFrame
    cv_fold_mae: Dict[str, List[float]]
    significance: pd.DataFrame
    ablation: pd.DataFrame
    bootstrap_ci: Dict[str, Tuple[float, float]]
    errors: np.ndarray
    y_test: np.ndarray
    pred_test: np.ndarray
    groups_train: np.ndarray
    groups_test: np.ndarray
    by_population: pd.DataFrame
    n_train: int
    n_test: int
    n_train_households: int
    n_test_households: int
    mlp_selected_epochs: Dict[str, int]

def _metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return {'MAE': float(mean_absolute_error(y, p)), 'RMSE': float(np.sqrt(mean_squared_error(y, p))), 'R2': float(r2_score(y, p)), 'MAPE': float(np.mean(np.abs((y - p) / np.maximum(y, 1))) * 100), 'within_0.5': float(np.mean(np.abs(y - p) < 0.5)), 'within_1.0': float(np.mean(np.abs(y - p) < 1.0))}

def select_omega(X_tr: np.ndarray, y_tr: np.ndarray, g_tr: np.ndarray, cfg: TaskBConfig) -> Tuple[float, pd.DataFrame]:
    """Public function: select_omega."""
    inner = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg.seed)
    (i_tr, i_va) = next(inner.split(X_tr, y_tr, groups=g_tr))
    ens = WeightedEnsembleRegressor(omega_mlp=0.5, hidden=cfg.mlp_hidden, random_state=cfg.seed).fit(X_tr[i_tr], y_tr[i_tr], groups=g_tr[i_tr])
    comp = ens.component_predictions(X_tr[i_va])
    others = (comp['GBRT'] + comp['SVR'] + comp['RF']) / 3.0
    rows = []
    for om in cfg.omega_grid:
        pred = om * comp['MLP'] + (1.0 - om) * others
        rows.append({'omega': om, 'val_MAE': float(mean_absolute_error(y_tr[i_va], pred))})
    grid = pd.DataFrame(rows)
    best = float(grid.loc[grid['val_MAE'].idxmin(), 'omega'])
    return (best, grid)

def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Public function: _holm_adjust."""
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if not len(valid):
        return adjusted
    ordered = valid[np.argsort(p[valid])]
    running = 0.0
    m = len(ordered)
    for (rank, idx) in enumerate(ordered):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted

def _paired_household_comparisons(y: np.ndarray, predictions: Dict[str, np.ndarray], groups: np.ndarray) -> pd.DataFrame:
    """Public function: _paired_household_comparisons."""
    ensemble_name = BLEND_NAME
    hids = np.unique(groups)
    ensemble_mae = np.array([np.mean(np.abs(y[groups == hid] - predictions[ensemble_name][groups == hid])) for hid in hids])
    rows: List[dict] = []
    n_comparisons = max(1, len(predictions) - 1)
    for (name, pred) in predictions.items():
        if name == ensemble_name:
            continue
        competitor_mae = np.array([np.mean(np.abs(y[groups == hid] - pred[groups == hid])) for hid in hids])
        diff = ensemble_mae - competitor_mae
        mean_diff = float(np.mean(diff))
        if len(diff) > 1 and (not np.allclose(np.std(diff, ddof=1), 0.0)):
            se = float(stats.sem(diff))
            critical = stats.t.ppf(1.0 - 0.05 / (2 * n_comparisons), len(diff) - 1)
            half = float(critical * se)
            (_, t_p) = stats.ttest_1samp(diff, popmean=0.0)
        else:
            half = 0.0
            t_p = float('nan')
        try:
            (_, w_p) = stats.wilcoxon(diff) if not np.allclose(diff, 0.0) else (0.0, np.nan)
        except ValueError:
            w_p = np.nan
        rows.append({'competitor': name, 'n_households': int(len(hids)), 'mean_diff': mean_diff, 'mean_diff_ci_low': mean_diff - half, 'mean_diff_ci_high': mean_diff + half, 'median_diff': float(np.median(diff)), 't_p': float(t_p) if np.isfinite(t_p) else np.nan, 'wilcoxon_p': float(w_p) if np.isfinite(w_p) else np.nan})
    out = pd.DataFrame(rows)
    out['t_p_holm'] = _holm_adjust(out['t_p'].values)
    out['wilcoxon_p_holm'] = _holm_adjust(out['wilcoxon_p'].values)
    return out

def _cluster_bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_boot: int, seed: int, alpha: float=0.05) -> Tuple[float, float]:
    """Public function: _cluster_bootstrap_ci."""
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    index_by_group = {gv: np.where(groups == gv)[0] for gv in uniq}
    stats_ = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index_by_group[gv] for gv in pick])
        stats_.append(mean_absolute_error(y[idx], p[idx]))
    (lo, hi) = np.percentile(stats_, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))

def run_task_b(windows: pd.DataFrame, eligible_households: Optional[np.ndarray]=None, cfg: Optional[TaskBConfig]=None) -> TaskBResult:
    """Public function: run_task_b."""
    cfg = cfg or TaskBConfig()
    df = windows
    if eligible_households is not None:
        df = df[df['household_id'].isin(set((int(h) for h in eligible_households)))]
    df = df.reset_index(drop=True)
    X = df[list(FEATURE_NAMES)].values
    y = df['population'].values.astype(float)
    groups = df['household_id'].values
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    (idx_tr, idx_te) = next(gss.split(X, y, groups=groups))
    (X_tr, X_te) = (X[idx_tr], X[idx_te])
    (y_tr, y_te) = (y[idx_tr], y[idx_te])
    (g_tr, g_te) = (groups[idx_tr], groups[idx_te])
    assert not set(g_tr) & set(g_te), 'household leaked across the Task B split'
    (omega, omega_grid) = select_omega(X_tr, y_tr, g_tr, cfg)
    logger.info('Task B omega_MLP selected on inner validation split: %.2f', omega)
    scaler = StandardScaler().fit(X_tr)
    (X_tr_s, X_te_s) = (scaler.transform(X_tr), scaler.transform(X_te))
    rows: List[dict] = []
    preds: Dict[str, np.ndarray] = {}
    mlp_selected_epochs: Dict[str, int] = {}
    for (name, model) in base_model_zoo(cfg).items():
        if isinstance(model, HouseholdGroupedMLPRegressor):
            m = clone(model).fit(X_tr, y_tr, groups=g_tr)
            p = m.predict(X_te)
            mlp_selected_epochs[name] = int(m.selected_epoch_)
        else:
            m = clone(model).fit(X_tr_s, y_tr)
            p = m.predict(X_te_s)
        preds[name] = p
        rows.append({'model': name, **_metrics(y_te, p)})
    ens = WeightedEnsembleRegressor(omega_mlp=omega, hidden=cfg.mlp_hidden, random_state=cfg.seed).fit(X_tr, y_tr, groups=g_tr)
    p_ens = ens.predict(X_te)
    preds[BLEND_NAME] = p_ens
    rows.append({'model': BLEND_NAME, **_metrics(y_te, p_ens)})
    mlp_selected_epochs[BLEND_NAME] = int(ens.mlp_.selected_epoch_)
    comparison = pd.DataFrame(rows)
    cv_mae: Dict[str, List[float]] = {n: [] for n in list(base_model_zoo(cfg)) + [BLEND_NAME]}
    omega_per_fold: List[float] = []
    gkf = GroupKFold(n_splits=cfg.n_cv_splits)
    for (f, (i_tr, i_te)) in enumerate(gkf.split(X_tr, y_tr, groups=g_tr)):
        sc = StandardScaler().fit(X_tr[i_tr])
        (A, B) = (sc.transform(X_tr[i_tr]), sc.transform(X_tr[i_te]))
        for (name, model) in base_model_zoo(cfg).items():
            if isinstance(model, HouseholdGroupedMLPRegressor):
                m = clone(model).fit(X_tr[i_tr], y_tr[i_tr], groups=g_tr[i_tr])
                fold_pred = m.predict(X_tr[i_te])
            else:
                m = clone(model).fit(A, y_tr[i_tr])
                fold_pred = m.predict(B)
            cv_mae[name].append(float(mean_absolute_error(y_tr[i_te], fold_pred)))
        (om_f, _) = select_omega(X_tr[i_tr], y_tr[i_tr], g_tr[i_tr], cfg)
        omega_per_fold.append(om_f)
        e = WeightedEnsembleRegressor(omega_mlp=om_f, hidden=cfg.mlp_hidden, random_state=cfg.seed).fit(X_tr[i_tr], y_tr[i_tr], groups=g_tr[i_tr])
        cv_mae[BLEND_NAME].append(float(mean_absolute_error(y_tr[i_te], e.predict(X_tr[i_te]))))
        logger.info('Task B CV fold %d: omega=%.2f ensemble MAE=%.4f', f, om_f, cv_mae[BLEND_NAME][-1])
    significance = _paired_household_comparisons(y_te, preds, g_te)
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=cfg.seed + 1)
    (a_tr, a_va) = next(inner.split(X_tr, y_tr, groups=g_tr))
    abl_rows: List[dict] = []
    all_idx = list(range(len(FEATURE_NAMES)))
    for (label, drop) in [('All features', ())] + [(f'w/o {k}', v) for (k, v) in FEATURE_GROUPS.items()]:
        keep = [i for i in all_idx if i not in set(drop)]
        (omega_ablation, _) = select_omega(X_tr[a_tr][:, keep], y_tr[a_tr], g_tr[a_tr], cfg)
        m = WeightedEnsembleRegressor(omega_mlp=omega_ablation, hidden=cfg.mlp_hidden, random_state=cfg.seed).fit(X_tr[a_tr][:, keep], y_tr[a_tr], groups=g_tr[a_tr])
        p = m.predict(X_tr[a_va][:, keep])
        abl_rows.append({'variant': label, 'n_features': len(keep), 'omega_selected': omega_ablation, 'val_MAE': float(mean_absolute_error(y_tr[a_va], p)), 'val_R2': float(r2_score(y_tr[a_va], p))})
    ablation = pd.DataFrame(abl_rows)
    base_mae = float(ablation.loc[ablation['variant'] == 'All features', 'val_MAE'].iloc[0])
    ablation['delta_MAE'] = ablation['val_MAE'] - base_mae
    ci = {name: _cluster_bootstrap_ci(y_te, p, g_te, cfg.n_bootstrap, cfg.seed) for (name, p) in preds.items()}
    by_pop_rows: List[dict] = []
    for pop in sorted(set(y_te.astype(int))):
        m = y_te == pop
        by_pop_rows.append({'population': int(pop), 'n': int(m.sum()), 'MAE': float(mean_absolute_error(y_te[m], p_ens[m])), 'rounding_accuracy': float(np.mean(np.clip(np.round(p_ens[m]), 0, None) == y_te[m]))})
    result = TaskBResult(comparison=comparison, omega_selected=omega, omega_per_fold=omega_per_fold, omega_grid_dev=omega_grid, cv_fold_mae=cv_mae, significance=significance, ablation=ablation, bootstrap_ci=ci, errors=p_ens - y_te, y_test=y_te, pred_test=p_ens, groups_train=g_tr, groups_test=g_te, by_population=pd.DataFrame(by_pop_rows), n_train=len(idx_tr), n_test=len(idx_te), n_train_households=len(set(g_tr)), n_test_households=len(set(g_te)), mlp_selected_epochs=mlp_selected_epochs)
    logger.info('Task B locked test: ensemble MAE=%.4f R2=%.4f (n=%d windows, %d households)', result.comparison.loc[result.comparison.model == BLEND_NAME, 'MAE'].iloc[0], result.comparison.loc[result.comparison.model == BLEND_NAME, 'R2'].iloc[0], result.n_test, result.n_test_households)
    return result
