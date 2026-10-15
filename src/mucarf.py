"""Calendar-adjusted relative-state forward filter."""
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.covariance import LedoitWolf
__all__ = ['MUCarfModel', 'MUCarfResult', 'AbsoluteHMMModel', 'build_banded_transition', 'fit_mucarf', 'fit_absolute_hmm', 'forward_filter_residuals', 'forward_filter_absolute', 'detect_covariance_adjusted', 'detect_mucarf', 'detect_absolute_hmm']
_UTILITY_COLUMNS = ('electricity_kwh', 'water_ton', 'gas_m3')
_OCCUPIED_TYPES = {'residential', 'mixed'}
_EPS = 1e-09

@dataclass(frozen=True)
class MUCarfModel:
    """Public class: MUCarfModel."""
    calendar: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    states: np.ndarray
    transition: np.ndarray
    baseline_end: int = 14
    min_calibration_observations: int = 8
    training_households: Tuple[int, ...] = ()
    p_stay: float = 0.94
    max_step: int = 2

@dataclass(frozen=True)
class MUCarfResult:
    """Public class: MUCarfResult."""
    alarms: np.ndarray
    posterior: np.ndarray
    state_mean: np.ndarray
    jump_positive: np.ndarray
    jump_negative: np.ndarray
    abstained: np.ndarray

def build_banded_transition(states: Sequence[int], p_stay: float=0.94, max_step: int=2, occupancy_probability: float=1.0) -> np.ndarray:
    """Public function: build_banded_transition."""
    s = np.asarray(states, dtype=int)
    if s.ndim != 1 or len(s) < 1 or len(np.unique(s)) != len(s):
        raise ValueError('states must be a non-empty vector of unique integers')
    if not 0.0 <= p_stay <= 1.0:
        raise ValueError('p_stay must lie in [0, 1]')
    if max_step < 1:
        raise ValueError('max_step must be at least one')
    if not 0.0 <= occupancy_probability <= 1.0:
        raise ValueError('occupancy_probability must lie in [0, 1]')
    n = len(s)
    out = np.zeros((n, n), dtype=float)
    jump_mass = (1.0 - p_stay) * occupancy_probability
    for (i, source) in enumerate(s):
        candidates: List[int] = []
        weights: List[float] = []
        for (j, target) in enumerate(s):
            distance = abs(int(target - source))
            if 0 < distance <= max_step:
                candidates.append(j)
                weights.append(1.0 / float(distance))
        if not candidates or jump_mass == 0.0:
            out[i, i] = 1.0
            continue
        weights_arr = np.asarray(weights, dtype=float)
        weights_arr /= weights_arr.sum()
        out[i, candidates] = jump_mass * weights_arr
        out[i, i] = 1.0 - jump_mass
    return out

def _training_log_series(panel, train_hids: Sequence[int]) -> Dict[int, np.ndarray]:
    meta = panel.meta.set_index('household_id')
    allowed = set((int(h) for h in train_hids))
    series: Dict[int, np.ndarray] = {}
    subset = panel.monthly[panel.monthly['household_id'].isin(allowed)]
    for (hid, grp) in subset.groupby('household_id', sort=True):
        if str(meta.loc[int(hid), 'true_type']) not in _OCCUPIED_TYPES:
            continue
        values = grp.sort_values('month')[list(_UTILITY_COLUMNS)].to_numpy(dtype=float)
        logged = np.full_like(values, np.nan, dtype=float)
        finite = np.isfinite(values) & (values > 0.0)
        logged[finite] = np.log(np.clip(values[finite], _EPS, None))
        series[int(hid)] = logged
    return series

def _fit_robust_calendar(log_series: Dict[int, np.ndarray], baseline_end: int, n_updates: int=3) -> np.ndarray:
    """Public function: _fit_robust_calendar."""
    calendar = np.zeros((3, 12), dtype=float)
    for _ in range(n_updates):
        buckets: List[List[List[float]]] = [[[] for _ in range(12)] for _ in range(3)]
        for values in log_series.values():
            segment = values[:baseline_end]
            adjusted = segment - calendar[:, np.arange(len(segment)) % 12].T
            level = np.nanmedian(adjusted, axis=0)
            for t in range(len(segment)):
                month = t % 12
                for utility in range(3):
                    value = segment[t, utility] - level[utility]
                    if np.isfinite(value):
                        buckets[utility][month].append(float(value))
        for utility in range(3):
            for month in range(12):
                values = buckets[utility][month]
                calendar[utility, month] = float(np.median(values)) if values else 0.0
            calendar[utility] -= float(np.mean(calendar[utility]))
    return calendar

def _household_residuals(values: np.ndarray, calendar: np.ndarray, baseline_end: int, min_calibration_observations: int) -> Tuple[np.ndarray, np.ndarray]:
    n_months = len(values)
    adjusted = values - calendar[:, np.arange(n_months) % 12].T
    baseline = np.full(3, np.nan, dtype=float)
    for utility in range(3):
        calibration = adjusted[:baseline_end, utility]
        finite = calibration[np.isfinite(calibration)]
        if len(finite) >= min_calibration_observations:
            baseline[utility] = float(np.median(finite))
    return (adjusted - baseline, baseline)

def fit_mucarf(panel, train_hids: Sequence[int], baseline_end: int=14, state_radius: int=3, max_step: int=2, min_calibration_observations: int=8, occupancy_probability: float=1.0) -> MUCarfModel:
    """Public function: fit_mucarf."""
    train_tuple = tuple((int(h) for h in train_hids))
    if not train_tuple:
        raise ValueError('at least one training household is required')
    if len(set(train_tuple)) != len(train_tuple):
        raise ValueError('training household ids must be unique')
    if baseline_end < 2 or baseline_end >= panel.population.shape[1]:
        raise ValueError('baseline_end must lie inside the observed panel')
    if state_radius < 1:
        raise ValueError('state_radius must be at least one')
    log_series = _training_log_series(panel, train_tuple)
    if not log_series:
        raise ValueError('no occupied training households were supplied')
    calendar = _fit_robust_calendar(log_series, baseline_end)
    ratios: List[List[float]] = [[], [], []]
    residual_records: List[np.ndarray] = []
    changes: List[int] = []
    for (hid, values) in log_series.items():
        (residuals, _) = _household_residuals(values, calendar, baseline_end, min_calibration_observations)
        baseline_population = int(panel.population[hid, baseline_end - 1])
        relative = panel.population[hid].astype(float) - baseline_population
        for t in range(baseline_end, len(relative)):
            q = float(relative[t])
            if q != 0.0:
                for utility in range(3):
                    value = residuals[t, utility]
                    if np.isfinite(value):
                        ratios[utility].append(float(value / q))
        changes.extend((int(v) for v in np.diff(panel.population[hid, baseline_end - 1:])))
    beta = np.array([max(0.001, float(np.median(values))) if values else 0.001 for values in ratios], dtype=float)
    for (hid, values) in log_series.items():
        (residuals, _) = _household_residuals(values, calendar, baseline_end, min_calibration_observations)
        baseline_population = int(panel.population[hid, baseline_end - 1])
        relative = panel.population[hid].astype(float) - baseline_population
        errors = residuals - relative[:, None] * beta[None, :]
        for row in errors:
            if np.all(np.isfinite(row)):
                residual_records.append(row.astype(float))
    if len(residual_records) < 4:
        raise ValueError('insufficient complete training residuals for covariance')
    covariance = LedoitWolf().fit(np.vstack(residual_records)).covariance_
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(3) * 1e-06
    if changes:
        p_stay = float(np.mean(np.asarray(changes) == 0))
    else:
        p_stay = 0.94
    p_stay = float(np.clip(p_stay, 0.85, 0.995))
    states = np.arange(-state_radius, state_radius + 1, dtype=int)
    transition = build_banded_transition(states, p_stay=p_stay, max_step=max_step, occupancy_probability=occupancy_probability)
    return MUCarfModel(calendar=calendar, beta=beta, covariance=covariance, states=states, transition=transition, baseline_end=int(baseline_end), min_calibration_observations=int(min_calibration_observations), training_households=train_tuple, p_stay=p_stay, max_step=int(max_step))

def _log_gaussian_by_state(observation: np.ndarray, observed: np.ndarray, model: MUCarfModel) -> np.ndarray:
    beta = model.beta[observed]
    covariance = model.covariance[np.ix_(observed, observed)]
    covariance = covariance + np.eye(len(beta)) * 1e-08
    (sign, logdet) = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError('observed covariance must be positive definite')
    inverse = np.linalg.inv(covariance)
    means = model.states[:, None].astype(float) * beta[None, :]
    diff = observation[observed][None, :] - means
    quadratic = np.einsum('si,ij,sj->s', diff, inverse, diff)
    return -0.5 * (quadratic + logdet + len(beta) * np.log(2.0 * np.pi))

def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    maximum = float(np.max(log_weights))
    weights = np.exp(log_weights - maximum)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError('posterior normalization failed')
    return weights / total

def forward_filter_residuals(residuals: np.ndarray, model: MUCarfModel, tau: float) -> MUCarfResult:
    """Public function: forward_filter_residuals."""
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('residuals must have shape (months, 3)')
    if not 0.0 <= tau <= 1.0:
        raise ValueError('tau must lie in [0, 1]')
    n_months = len(values)
    n_states = len(model.states)
    if model.transition.shape != (n_states, n_states):
        raise ValueError('transition shape does not match the state vector')
    zero_candidates = np.where(model.states == 0)[0]
    if len(zero_candidates) != 1:
        raise ValueError('states must contain zero exactly once')
    zero_index = int(zero_candidates[0])
    posterior = np.zeros((n_months, n_states), dtype=float)
    prior = np.zeros(n_states, dtype=float)
    prior[zero_index] = 1.0
    alarms = np.zeros(n_months, dtype=int)
    jump_positive = np.zeros(n_months, dtype=float)
    jump_negative = np.zeros(n_months, dtype=float)
    abstained = np.zeros(n_months, dtype=bool)
    for t in range(n_months):
        if t < model.baseline_end:
            posterior[t] = prior
            continue
        previous = posterior[t - 1]
        observed = np.isfinite(values[t])
        if not observed.any():
            posterior[t] = previous
            abstained[t] = True
            continue
        likelihood = _log_gaussian_by_state(values[t], observed, model)
        predictive = previous @ model.transition
        log_posterior = np.log(np.clip(predictive, _EPS, None)) + likelihood
        current = _normalize_log_weights(log_posterior)
        posterior[t] = current
        log_pair = np.log(np.clip(previous[:, None], _EPS, None)) + np.log(np.clip(model.transition, _EPS, None)) + likelihood[None, :]
        pair = _normalize_log_weights(log_pair.ravel()).reshape(n_states, n_states)
        for (i, source) in enumerate(model.states):
            for (j, target) in enumerate(model.states):
                if target > source:
                    jump_positive[t] += pair[i, j]
                elif target < source:
                    jump_negative[t] += pair[i, j]
        if max(jump_positive[t], jump_negative[t]) > tau:
            alarms[t] = 1 if jump_positive[t] >= jump_negative[t] else -1
    state_mean = posterior @ model.states.astype(float)
    return MUCarfResult(alarms=alarms, posterior=posterior, state_mean=state_mean, jump_positive=jump_positive, jump_negative=jump_negative, abstained=abstained)

def _causal_forward_fill(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    last = np.nan
    for (index, value) in enumerate(out):
        if np.isfinite(value) and value > 0.0:
            last = value
        elif np.isfinite(last):
            out[index] = last
        else:
            out[index] = np.nan
    return out

def detect_mucarf(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, model: MUCarfModel, tau: float) -> MUCarfResult:
    """Public function: detect_mucarf."""
    raw = np.column_stack([_causal_forward_fill(elec), _causal_forward_fill(water), _causal_forward_fill(gas)])
    logged = np.full_like(raw, np.nan, dtype=float)
    finite = np.isfinite(raw) & (raw > 0.0)
    logged[finite] = np.log(np.clip(raw[finite], _EPS, None))
    (residuals, baseline) = _household_residuals(logged, model.calendar, model.baseline_end, model.min_calibration_observations)
    unavailable = ~np.isfinite(baseline)
    if unavailable.any():
        residuals[:, unavailable] = np.nan
    return forward_filter_residuals(residuals, model, tau)

@dataclass(frozen=True)
class AbsoluteHMMModel:
    """Public class: AbsoluteHMMModel."""
    calendar: np.ndarray
    intercept: np.ndarray
    beta: np.ndarray
    covariance: np.ndarray
    states: np.ndarray
    transition: np.ndarray
    baseline_end: int = 14
    training_households: Tuple[int, ...] = ()
    p_stay: float = 0.94
    max_step: int = 2

def fit_absolute_hmm(panel, train_hids: Sequence[int], baseline_end: int=14, n_states: int=7, max_step: int=2) -> AbsoluteHMMModel:
    """Public function: fit_absolute_hmm."""
    train_tuple = tuple((int(h) for h in train_hids))
    if not train_tuple:
        raise ValueError('at least one training household is required')
    if len(set(train_tuple)) != len(train_tuple):
        raise ValueError('training household ids must be unique')
    if n_states < 2:
        raise ValueError('n_states must be at least two')
    if baseline_end < 2 or baseline_end >= panel.population.shape[1]:
        raise ValueError('baseline_end must lie inside the observed panel')
    log_series = _training_log_series(panel, train_tuple)
    if not log_series:
        raise ValueError('no occupied training households were supplied')
    calendar = _fit_robust_calendar(log_series, baseline_end)
    levels: List[np.ndarray] = []
    populations: List[float] = []
    for (hid, values) in log_series.items():
        n_months = len(values)
        adjusted = values - calendar[:, np.arange(n_months) % 12].T
        for t in range(n_months):
            population = float(panel.population[hid, t])
            if population <= 0.0:
                continue
            levels.append(adjusted[t])
            populations.append(population)
    if len(levels) < 8:
        raise ValueError('insufficient occupied training observations')
    level_matrix = np.vstack(levels)
    population_vector = np.asarray(populations, dtype=float)
    intercept = np.zeros(3, dtype=float)
    beta = np.zeros(3, dtype=float)
    unique_populations = np.unique(population_vector)
    for utility in range(3):
        medians: List[float] = []
        support: List[float] = []
        for population in unique_populations:
            mask = (population_vector == population) & np.isfinite(level_matrix[:, utility])
            if mask.sum() >= 3:
                medians.append(float(np.median(level_matrix[mask, utility])))
                support.append(float(population))
        if len(medians) >= 2:
            (slope, offset) = np.polyfit(np.asarray(support), np.asarray(medians), 1)
        else:
            finite = np.isfinite(level_matrix[:, utility])
            (slope, offset) = (0.001, float(np.median(level_matrix[finite, utility])))
        beta[utility] = max(0.001, float(slope))
        intercept[utility] = float(offset)
    predicted = intercept[None, :] + population_vector[:, None] * beta[None, :]
    errors = level_matrix - predicted
    complete = np.all(np.isfinite(errors), axis=1)
    if complete.sum() < 4:
        raise ValueError('insufficient complete training residuals for covariance')
    covariance = LedoitWolf().fit(errors[complete]).covariance_
    covariance = 0.5 * (covariance + covariance.T)
    covariance += np.eye(3) * 1e-06
    changes: List[int] = []
    for hid in log_series:
        changes.extend((int(v) for v in np.diff(panel.population[hid, baseline_end - 1:])))
    p_stay = float(np.mean(np.asarray(changes) == 0)) if changes else 0.94
    p_stay = float(np.clip(p_stay, 0.85, 0.995))
    states = np.arange(1, n_states + 1, dtype=int)
    transition = build_banded_transition(states, p_stay=p_stay, max_step=max_step)
    return AbsoluteHMMModel(calendar=calendar, intercept=intercept, beta=beta, covariance=covariance, states=states, transition=transition, baseline_end=int(baseline_end), training_households=train_tuple, p_stay=p_stay, max_step=int(max_step))

def _log_gaussian_absolute(observation: np.ndarray, observed: np.ndarray, model: AbsoluteHMMModel) -> np.ndarray:
    beta = model.beta[observed]
    intercept = model.intercept[observed]
    covariance = model.covariance[np.ix_(observed, observed)]
    covariance = covariance + np.eye(len(beta)) * 1e-08
    (sign, logdet) = np.linalg.slogdet(covariance)
    if sign <= 0:
        raise ValueError('observed covariance must be positive definite')
    inverse = np.linalg.inv(covariance)
    means = intercept[None, :] + model.states[:, None].astype(float) * beta[None, :]
    diff = observation[observed][None, :] - means
    quadratic = np.einsum('si,ij,sj->s', diff, inverse, diff)
    return -0.5 * (quadratic + logdet + len(beta) * np.log(2.0 * np.pi))

def forward_filter_absolute(levels: np.ndarray, model: AbsoluteHMMModel, tau: float) -> np.ndarray:
    """Public function: forward_filter_absolute."""
    values = np.asarray(levels, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError('levels must have shape (months, 3)')
    if not 0.0 <= tau <= 1.0:
        raise ValueError('tau must lie in [0, 1]')
    n_months = len(values)
    n_states = len(model.states)
    if model.transition.shape != (n_states, n_states):
        raise ValueError('transition shape does not match the state vector')
    posterior = np.full(n_states, 1.0 / n_states, dtype=float)
    alarms = np.zeros(n_months, dtype=int)
    for t in range(n_months):
        observed = np.isfinite(values[t])
        if not observed.any():
            continue
        likelihood = _log_gaussian_absolute(values[t], observed, model)
        previous = posterior
        predictive = previous @ model.transition
        posterior = _normalize_log_weights(np.log(np.clip(predictive, _EPS, None)) + likelihood)
        if t < model.baseline_end:
            continue
        log_pair = np.log(np.clip(previous[:, None], _EPS, None)) + np.log(np.clip(model.transition, _EPS, None)) + likelihood[None, :]
        pair = _normalize_log_weights(log_pair.ravel()).reshape(n_states, n_states)
        upper = float(np.triu(pair, 1).sum())
        lower = float(np.tril(pair, -1).sum())
        if max(upper, lower) > tau:
            alarms[t] = 1 if upper >= lower else -1
    return alarms

def detect_absolute_hmm(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, model: AbsoluteHMMModel, tau: float) -> np.ndarray:
    """Public function: detect_absolute_hmm."""
    raw = np.column_stack([_causal_forward_fill(elec), _causal_forward_fill(water), _causal_forward_fill(gas)])
    logged = np.full_like(raw, np.nan, dtype=float)
    finite = np.isfinite(raw) & (raw > 0.0)
    logged[finite] = np.log(np.clip(raw[finite], _EPS, None))
    n_months = len(logged)
    levels = logged - model.calendar[:, np.arange(n_months) % 12].T
    return forward_filter_absolute(levels, model, tau)

def detect_covariance_adjusted(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, model: MUCarfModel, threshold: float) -> np.ndarray:
    """Public function: detect_covariance_adjusted."""
    if threshold <= 0.0:
        raise ValueError('threshold must be strictly positive')
    raw = np.column_stack([_causal_forward_fill(elec), _causal_forward_fill(water), _causal_forward_fill(gas)])
    logged = np.full_like(raw, np.nan, dtype=float)
    finite = np.isfinite(raw) & (raw > 0.0)
    logged[finite] = np.log(np.clip(raw[finite], _EPS, None))
    (residuals, baseline) = _household_residuals(logged, model.calendar, model.baseline_end, model.min_calibration_observations)
    unavailable = ~np.isfinite(baseline)
    if unavailable.any():
        residuals[:, unavailable] = np.nan
    score = np.full(len(residuals), np.nan, dtype=float)
    for t in range(model.baseline_end, len(residuals)):
        observed = np.isfinite(residuals[t])
        if not observed.any():
            continue
        beta = model.beta[observed]
        covariance = model.covariance[np.ix_(observed, observed)]
        covariance = covariance + np.eye(len(beta)) * 1e-08
        inverse = np.linalg.inv(covariance)
        scale = float(np.sqrt(max(_EPS, beta @ inverse @ beta)))
        score[t] = float(beta @ inverse @ residuals[t, observed] / scale)
    alarms = np.zeros(len(score), dtype=int)
    for t in range(model.baseline_end, len(score)):
        if np.isfinite(score[t]) and np.isfinite(score[t - 1]):
            change = float(score[t] - score[t - 1])
            if abs(change) > threshold:
                alarms[t] = 1 if change > 0.0 else -1
    return alarms
