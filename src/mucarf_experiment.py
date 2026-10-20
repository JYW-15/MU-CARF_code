"""Leakage-controlled MU-CARF experiment wrapper."""
from dataclasses import dataclass, replace
from typing import Dict, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .config import TaskCConfig
from .mucarf import MUCarfModel, build_banded_transition, detect_absolute_hmm, detect_covariance_adjusted, detect_mucarf, fit_absolute_hmm, fit_mucarf
from .task_a_screening import TaskAResult
from .task_c_detection import CAL_NAME, _bootstrap_event_ci, _random_timing_reference, match_events, run_task_c
MUCARF_NAME = 'MU-CARF relative-state filter'
MUCARF_RANDOM_NAME = 'Random timing (MU-CARF matched)'
SOFT_GATE_NAME = 'MU-CARF, soft gate'
COVARIANCE_NAME = 'Covariance-aware first difference'
COVARIANCE_RANDOM_NAME = 'Random timing (covariance-aware matched)'
DIAGONAL_NAME = 'MU-CARF, diagonal covariance'
HMM_NAME = 'Absolute-state HMM control'

@dataclass
class MUCarfExperimentResult:
    summary: pd.DataFrame
    dev_selection: pd.DataFrame
    tolerance_sensitivity: pd.DataFrame
    selected_tau: float
    selected_soft_gate_tau: float
    selected_covariance_threshold: float
    selected_diagonal_tau: float
    selected_hmm_tau: float
    baseline_selected_theta: Dict[str, float]
    model: MUCarfModel
    dev_households: np.ndarray
    test_households: np.ndarray
    training_households: np.ndarray
    active_households: np.ndarray
    test_alarms: Dict[int, np.ndarray]
    paired_comparisons: pd.DataFrame

def _paired_bootstrap_f1_difference(reference_alarms: Dict[int, np.ndarray], competitor_alarms: Dict[int, np.ndarray], events: pd.DataFrame, n_months: int, tolerance: int, n_bootstrap: int, seed: int, monitoring_start: int) -> Tuple[float, float, float]:
    """Public function: _paired_bootstrap_f1_difference."""
    if set(reference_alarms) != set(competitor_alarms):
        raise ValueError('paired algorithms must cover the same households')
    point_reference = match_events(reference_alarms, events, n_months, tolerance, monitoring_start)['event_f1']
    point_competitor = match_events(competitor_alarms, events, n_months, tolerance, monitoring_start)['event_f1']
    point = float(point_reference - point_competitor)
    rng = np.random.RandomState(seed)
    household_ids = np.array(sorted(reference_alarms), dtype=int)
    differences = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(household_ids, len(household_ids), replace=True)
        sampled_reference: Dict[int, np.ndarray] = {}
        sampled_competitor: Dict[int, np.ndarray] = {}
        sampled_events = []
        for (new_id, household_id) in enumerate(sampled):
            sampled_reference[new_id] = reference_alarms[int(household_id)]
            sampled_competitor[new_id] = competitor_alarms[int(household_id)]
            household_events = events[events['household_id'] == int(household_id)]
            for (_, event) in household_events.iterrows():
                sampled_events.append({'household_id': new_id, 'start_month': event['start_month'], 'direction': event['direction']})
        event_frame = pd.DataFrame(sampled_events) if sampled_events else pd.DataFrame(columns=['household_id', 'start_month', 'direction'])
        reference_f1 = match_events(sampled_reference, event_frame, n_months, tolerance, monitoring_start)['event_f1']
        competitor_f1 = match_events(sampled_competitor, event_frame, n_months, tolerance, monitoring_start)['event_f1']
        if np.isfinite(reference_f1) and np.isfinite(competitor_f1):
            differences.append(float(reference_f1 - competitor_f1))
    if not differences:
        return (point, float('nan'), float('nan'))
    (low, high) = np.percentile(differences, [2.5, 97.5])
    return (point, float(low), float(high))

def _series_by_household(panel, household_ids: Sequence[int]):
    requested = set((int(h) for h in household_ids))
    out = {}
    subset = panel.monthly[panel.monthly['household_id'].isin(requested)]
    for (hid, grp) in subset.groupby('household_id', sort=True):
        grp = grp.sort_values('month')
        out[int(hid)] = (grp['electricity_kwh'].to_numpy(dtype=float), grp['water_ton'].to_numpy(dtype=float), grp['gas_m3'].to_numpy(dtype=float))
    if set(out) != requested:
        raise ValueError('monthly utility series are missing for requested households')
    return out

def _detect_for_households(series, household_ids: Sequence[int], active_households: set, model: MUCarfModel, tau: float, n_months: int) -> Dict[int, np.ndarray]:
    return {int(h): detect_mucarf(*series[int(h)], model=model, tau=tau).alarms if int(h) in active_households else np.zeros(n_months, dtype=int) for h in household_ids}

def _detect_hmm_for_households(series, household_ids: Sequence[int], active_households: set, model, tau: float, n_months: int) -> Dict[int, np.ndarray]:
    """Public function: _detect_hmm_for_households."""
    return {int(h): detect_absolute_hmm(*series[int(h)], model=model, tau=tau) if int(h) in active_households else np.zeros(n_months, dtype=int) for h in household_ids}

def _detect_soft_for_households(series, household_ids: Sequence[int], occupied_probability: Dict[int, float], model: MUCarfModel, tau: float) -> Dict[int, np.ndarray]:
    alarms: Dict[int, np.ndarray] = {}
    for household_id in household_ids:
        hid = int(household_id)
        probability = float(np.clip(occupied_probability[hid], 0.0, 1.0))
        transition = build_banded_transition(model.states, p_stay=model.p_stay, max_step=model.max_step, occupancy_probability=probability)
        household_model = replace(model, transition=transition)
        alarms[hid] = detect_mucarf(*series[hid], model=household_model, tau=tau).alarms
    return alarms

def run_mucarf_experiment(panel, task_a: TaskAResult, task_c_cfg: Optional[TaskCConfig]=None, tau_grid: Sequence[float]=tuple(np.linspace(0.05, 0.95, 19)), covariance_threshold_grid: Sequence[float]=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0), paired_bootstrap: Optional[int]=None, state_radius: int=3, max_step: int=2) -> MUCarfExperimentResult:
    """MU-CARF against every declared comparator on one development/test split.

    Threshold rules follow the declared protocol and are deliberately *not* identical
    across detectors:

    * The baselines (CABL, ratio, multi-signal, CUSUM) keep ``cfg.tolerance_months`` as
      their operating point. Their thresholds are selected on the development households
      at that tolerance and then held fixed, so ``baseline.summary`` is evaluated at that
      tolerance.
    * MU-CARF's posterior-jump threshold is selected at the one-month endpoint, which is
      the primary endpoint of the study, so every MU-CARF, diagonal-covariance, soft-gate
      and covariance-aware row here is scored with ``tolerance=1``.

    Use ``tolerance_sensitivity`` for like-for-like comparisons across delays; do not
    compare a baseline row with a MU-CARF row without checking the tolerance column.
    """
    cfg = task_c_cfg or TaskCConfig()
    grid = tuple((float(value) for value in tau_grid))
    if not grid or any((value < 0.0 or value > 1.0 for value in grid)):
        raise ValueError('tau_grid must contain probabilities in [0, 1]')
    covariance_grid = tuple((float(value) for value in covariance_threshold_grid))
    if not covariance_grid or any((value <= 0.0 for value in covariance_grid)):
        raise ValueError('covariance_threshold_grid must be strictly positive')
    primary_paired_bootstrap = int(cfg.n_bootstrap if paired_bootstrap is None else paired_bootstrap)
    if primary_paired_bootstrap < 1:
        raise ValueError('paired_bootstrap must be at least one')
    evaluation_hids = np.array(sorted((int(h) for h in task_a.is_test[task_a.is_test].index)), dtype=int)
    training_universe = np.array(sorted((int(h) for h in task_a.is_test[~task_a.is_test].index)), dtype=int)
    meta = panel.meta.set_index('household_id')
    training_hids = np.array([int(h) for h in training_universe if str(meta.loc[int(h), 'true_type']) in {'residential', 'mixed'}], dtype=int)
    active_hids = np.array([int(h) for h in evaluation_hids if str(task_a.predicted_type.loc[int(h)]) in {'residential', 'mixed'}], dtype=int)
    occupied_probability = {int(h): float(task_a.occupied_probability.loc[int(h)]) for h in evaluation_hids}
    if set(training_hids) & set(evaluation_hids):
        raise ValueError('MU-CARF training/evaluation household overlap')
    baseline = run_task_c(panel, pop_estimates=None, cfg=cfg, eligible_households=evaluation_hids, active_households=active_hids, trajectory_train_households=training_hids)
    dev_hids = baseline.dev_households
    test_hids = baseline.test_households
    if set(training_hids) & (set(dev_hids) | set(test_hids)):
        raise ValueError('MU-CARF training household entered development/test')
    model = fit_mucarf(panel, training_hids, baseline_end=14, state_radius=state_radius, max_step=max_step)
    diagonal_model = replace(model, covariance=np.diag(np.diag(model.covariance)))
    hmm_model = fit_absolute_hmm(panel, training_hids, baseline_end=14, n_states=2 * state_radius + 1, max_step=max_step)
    n_months = panel.population.shape[1]
    series = _series_by_household(panel, evaluation_hids)
    active_set = set((int(h) for h in active_hids))
    events = panel.events[panel.events['start_month'] >= cfg.monitoring_start_month].copy()
    dev_events = events[events['household_id'].isin(set(dev_hids))]
    test_events = events[events['household_id'].isin(set(test_hids))]
    dev_rows = []
    for tau in grid:
        soft_alarms = _detect_soft_for_households(series, dev_hids, occupied_probability, model, tau)
        soft_metrics = match_events(soft_alarms, dev_events, n_months, 1, cfg.monitoring_start_month)
        dev_rows.append({'detector': SOFT_GATE_NAME, 'theta': tau, **soft_metrics})
        alarms = _detect_for_households(series, dev_hids, active_set, model, tau, n_months)
        metrics = match_events(alarms, dev_events, n_months, 1, cfg.monitoring_start_month)
        dev_rows.append({'detector': MUCARF_NAME, 'theta': tau, **metrics})
        diagonal_alarms = _detect_for_households(series, dev_hids, active_set, diagonal_model, tau, n_months)
        diagonal_metrics = match_events(diagonal_alarms, dev_events, n_months, 1, cfg.monitoring_start_month)
        dev_rows.append({'detector': DIAGONAL_NAME, 'theta': tau, **diagonal_metrics})
        hmm_alarms = _detect_hmm_for_households(series, dev_hids, active_set, hmm_model, tau, n_months)
        hmm_metrics = match_events(hmm_alarms, dev_events, n_months, 1, cfg.monitoring_start_month)
        dev_rows.append({'detector': HMM_NAME, 'theta': tau, **hmm_metrics})
    for threshold in covariance_grid:
        alarms = {int(h): detect_covariance_adjusted(*series[int(h)], model=model, threshold=threshold) if int(h) in active_set else np.zeros(n_months, dtype=int) for h in dev_hids}
        metrics = match_events(alarms, dev_events, n_months, 1, cfg.monitoring_start_month)
        dev_rows.append({'detector': COVARIANCE_NAME, 'theta': threshold, **metrics})
    dev_selection = pd.DataFrame(dev_rows)
    mucarf_dev = dev_selection[dev_selection['detector'] == MUCARF_NAME]
    soft_dev = dev_selection[dev_selection['detector'] == SOFT_GATE_NAME]
    covariance_dev = dev_selection[dev_selection['detector'] == COVARIANCE_NAME]
    diagonal_dev = dev_selection[dev_selection['detector'] == DIAGONAL_NAME]
    hmm_dev = dev_selection[dev_selection['detector'] == HMM_NAME]
    selected_tau = float(mucarf_dev.loc[mucarf_dev['event_f1'].idxmax(), 'theta'])
    selected_soft_gate_tau = float(soft_dev.loc[soft_dev['event_f1'].idxmax(), 'theta'])
    selected_covariance_threshold = float(covariance_dev.loc[covariance_dev['event_f1'].idxmax(), 'theta'])
    selected_diagonal_tau = float(diagonal_dev.loc[diagonal_dev['event_f1'].idxmax(), 'theta'])
    selected_hmm_tau = float(hmm_dev.loc[hmm_dev['event_f1'].idxmax(), 'theta'])
    test_alarms = _detect_for_households(series, test_hids, active_set, model, selected_tau, n_months)
    test_metrics = match_events(test_alarms, test_events, n_months, 1, cfg.monitoring_start_month)
    (ci_low, ci_high) = _bootstrap_event_ci(test_alarms, test_events, n_months, 1, 'event_f1', cfg.n_bootstrap, cfg.seed + 30000, cfg.monitoring_start_month)
    mucarf_row = {'detector': MUCARF_NAME, 'online': True, **test_metrics, 'event_f1_ci_low': ci_low, 'event_f1_ci_high': ci_high}
    soft_gate_alarms = _detect_soft_for_households(series, test_hids, occupied_probability, model, selected_soft_gate_tau)
    soft_gate_metrics = match_events(soft_gate_alarms, test_events, n_months, 1, cfg.monitoring_start_month)
    (soft_gate_low, soft_gate_high) = _bootstrap_event_ci(soft_gate_alarms, test_events, n_months, 1, 'event_f1', cfg.n_bootstrap, cfg.seed + 33000, cfg.monitoring_start_month)
    soft_gate_row = {'detector': SOFT_GATE_NAME, 'online': True, **soft_gate_metrics, 'event_f1_ci_low': soft_gate_low, 'event_f1_ci_high': soft_gate_high}
    diagonal_alarms = _detect_for_households(series, test_hids, active_set, diagonal_model, selected_diagonal_tau, n_months)
    diagonal_metrics = match_events(diagonal_alarms, test_events, n_months, 1, cfg.monitoring_start_month)
    (diagonal_low, diagonal_high) = _bootstrap_event_ci(diagonal_alarms, test_events, n_months, 1, 'event_f1', cfg.n_bootstrap, cfg.seed + 32000, cfg.monitoring_start_month)
    diagonal_row = {'detector': DIAGONAL_NAME, 'online': True, **diagonal_metrics, 'event_f1_ci_low': diagonal_low, 'event_f1_ci_high': diagonal_high}
    hmm_alarms = _detect_hmm_for_households(series, test_hids, active_set, hmm_model, selected_hmm_tau, n_months)
    hmm_metrics = match_events(hmm_alarms, test_events, n_months, 1, cfg.monitoring_start_month)
    (hmm_low, hmm_high) = _bootstrap_event_ci(hmm_alarms, test_events, n_months, 1, 'event_f1', cfg.n_bootstrap, cfg.seed + 34000, cfg.monitoring_start_month)
    hmm_row = {'detector': HMM_NAME, 'online': True, **hmm_metrics, 'event_f1_ci_low': hmm_low, 'event_f1_ci_high': hmm_high}
    covariance_alarms = {int(h): detect_covariance_adjusted(*series[int(h)], model=model, threshold=selected_covariance_threshold) if int(h) in active_set else np.zeros(n_months, dtype=int) for h in test_hids}
    covariance_metrics = match_events(covariance_alarms, test_events, n_months, 1, cfg.monitoring_start_month)
    (covariance_low, covariance_high) = _bootstrap_event_ci(covariance_alarms, test_events, n_months, 1, 'event_f1', cfg.n_bootstrap, cfg.seed + 31000, cfg.monitoring_start_month)
    covariance_row = {'detector': COVARIANCE_NAME, 'online': True, **covariance_metrics, 'event_f1_ci_low': covariance_low, 'event_f1_ci_high': covariance_high}
    eligible_masks = {int(h): np.array([bool(int(h) in active_set and t >= model.baseline_end) for t in range(n_months)], dtype=bool) for h in test_hids}
    random_primary = _random_timing_reference(test_alarms, eligible_masks, test_events, n_months, 1, cfg.n_random_baseline, cfg.seed + 40000, cfg.monitoring_start_month)
    random_row = {'detector': MUCARF_RANDOM_NAME, 'online': True, **random_primary}
    covariance_random_primary = _random_timing_reference(covariance_alarms, eligible_masks, test_events, n_months, 1, cfg.n_random_baseline, cfg.seed + 41000, cfg.monitoring_start_month)
    covariance_random_row = {'detector': COVARIANCE_RANDOM_NAME, 'online': True, **covariance_random_primary}
    extra_rows = pd.DataFrame([covariance_row, covariance_random_row, diagonal_row, hmm_row, soft_gate_row, mucarf_row, random_row]).assign(tolerance_months=1)
    baseline_rows = baseline.summary.assign(tolerance_months=cfg.tolerance_months)
    summary = pd.concat([baseline_rows, extra_rows], ignore_index=True, sort=False)
    sensitivity_rows = []
    for tolerance in sorted(set(cfg.tolerance_sensitivity_months + (1,))):
        observed = match_events(test_alarms, test_events, n_months, tolerance, cfg.monitoring_start_month)
        (low, high) = _bootstrap_event_ci(test_alarms, test_events, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed + 30000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': MUCARF_NAME, 'tolerance_months': tolerance, **observed, 'event_f1_ci_low': low, 'event_f1_ci_high': high})
        randomized = _random_timing_reference(test_alarms, eligible_masks, test_events, n_months, tolerance, cfg.n_random_baseline, cfg.seed + 40000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': MUCARF_RANDOM_NAME, 'tolerance_months': tolerance, **randomized})
        covariance_observed = match_events(covariance_alarms, test_events, n_months, tolerance, cfg.monitoring_start_month)
        (covariance_ci_low, covariance_ci_high) = _bootstrap_event_ci(covariance_alarms, test_events, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed + 31000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': COVARIANCE_NAME, 'tolerance_months': tolerance, **covariance_observed, 'event_f1_ci_low': covariance_ci_low, 'event_f1_ci_high': covariance_ci_high})
        covariance_randomized = _random_timing_reference(covariance_alarms, eligible_masks, test_events, n_months, tolerance, cfg.n_random_baseline, cfg.seed + 41000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': COVARIANCE_RANDOM_NAME, 'tolerance_months': tolerance, **covariance_randomized})
        diagonal_observed = match_events(diagonal_alarms, test_events, n_months, tolerance, cfg.monitoring_start_month)
        (diagonal_ci_low, diagonal_ci_high) = _bootstrap_event_ci(diagonal_alarms, test_events, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed + 32000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': DIAGONAL_NAME, 'tolerance_months': tolerance, **diagonal_observed, 'event_f1_ci_low': diagonal_ci_low, 'event_f1_ci_high': diagonal_ci_high})
        soft_observed = match_events(soft_gate_alarms, test_events, n_months, tolerance, cfg.monitoring_start_month)
        (soft_ci_low, soft_ci_high) = _bootstrap_event_ci(soft_gate_alarms, test_events, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed + 33000 + tolerance, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': SOFT_GATE_NAME, 'tolerance_months': tolerance, **soft_observed, 'event_f1_ci_low': soft_ci_low, 'event_f1_ci_high': soft_ci_high})
    tolerance_sensitivity = pd.concat([baseline.tolerance_sensitivity, pd.DataFrame(sensitivity_rows)], ignore_index=True, sort=False)
    comparison_alarms = {CAL_NAME: baseline.test_alarms[CAL_NAME], COVARIANCE_NAME: covariance_alarms, DIAGONAL_NAME: diagonal_alarms, HMM_NAME: hmm_alarms, SOFT_GATE_NAME: soft_gate_alarms}
    paired_rows = []
    for (index, (name, alarms)) in enumerate(comparison_alarms.items()):
        repetitions = primary_paired_bootstrap if name == CAL_NAME else cfg.n_bootstrap
        (difference, difference_low, difference_high) = _paired_bootstrap_f1_difference(test_alarms, alarms, test_events, n_months, 1, repetitions, cfg.seed + 50000 + index, cfg.monitoring_start_month)
        paired_rows.append({'competitor': name, 'tolerance_months': 1, 'f1_difference': difference, 'f1_difference_ci_low': difference_low, 'f1_difference_ci_high': difference_high, 'n_households': len(test_hids), 'n_bootstrap': repetitions})
    paired_comparisons = pd.DataFrame(paired_rows)
    return MUCarfExperimentResult(summary=summary, dev_selection=dev_selection, tolerance_sensitivity=tolerance_sensitivity, selected_tau=selected_tau, selected_soft_gate_tau=selected_soft_gate_tau, selected_covariance_threshold=selected_covariance_threshold, selected_diagonal_tau=selected_diagonal_tau, selected_hmm_tau=selected_hmm_tau, baseline_selected_theta=dict(baseline.selected_theta), model=model, dev_households=np.asarray(dev_hids, dtype=int), test_households=np.asarray(test_hids, dtype=int), training_households=training_hids, active_households=active_hids, test_alarms=test_alarms, paired_comparisons=paired_comparisons)
