"""Past-only population-change detection and event scoring."""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from .config import TaskCConfig
logger = logging.getLogger(__name__)
__all__ = ['detect_ratio_causal', 'detect_multisignal_causal', 'detect_cusum_causal', 'detect_estimate_shift', 'retrospective_verify', 'match_events', 'learn_calendar_profile', 'calendar_adjusted_scores', 'detect_calendar_adjusted', 'split_task_c_households', 'TaskCResult', 'run_task_c']

def _clean_causal(x: np.ndarray) -> np.ndarray:
    """Public function: _clean_causal."""
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x
    out = x.copy()
    last = np.nan
    for (i, value) in enumerate(out):
        if np.isfinite(value):
            last = float(value)
        elif np.isfinite(last):
            out[i] = last
    return out

def _clean_retrospective(x: np.ndarray) -> np.ndarray:
    """Public function: _clean_retrospective."""
    x = np.asarray(x, dtype=float)
    if not np.isnan(x).any():
        return x
    idx = np.arange(len(x))
    good = ~np.isnan(x)
    if not good.any():
        return np.zeros_like(x)
    return np.interp(idx, idx[good], x[good])

def detect_ratio_causal(elec: np.ndarray, window: int=3, theta: float=0.15) -> np.ndarray:
    """Public function: detect_ratio_causal."""
    e = _clean_causal(elec)
    out = np.zeros(len(e), dtype=int)
    for t in range(2 * window, len(e)):
        recent_values = e[t - window:t]
        prior_values = e[t - 2 * window:t - window]
        if not (np.isfinite(recent_values).all() and np.isfinite(prior_values).all()):
            continue
        recent = np.mean(recent_values)
        prior = np.mean(prior_values)
        delta = (recent - prior) / (prior + 0.01)
        if delta > theta:
            out[t] = 1
        elif delta < -theta:
            out[t] = -1
    return out

def detect_multisignal_causal(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, window: int=3, theta: float=0.15, min_votes: int=2) -> np.ndarray:
    """Public function: detect_multisignal_causal."""
    sigs = [_clean_causal(elec), _clean_causal(water), _clean_causal(gas)]
    n = len(sigs[0])
    out = np.zeros(n, dtype=int)
    for t in range(2 * window, n):
        votes = []
        for s in sigs:
            recent_values = s[t - window:t]
            prior_values = s[t - 2 * window:t - window]
            if not (np.isfinite(recent_values).all() and np.isfinite(prior_values).all()):
                votes.append(0)
                continue
            recent = np.mean(recent_values)
            prior = np.mean(prior_values)
            delta = (recent - prior) / (prior + 0.01)
            votes.append(1 if delta > theta else -1 if delta < -theta else 0)
        if votes.count(1) >= min_votes:
            out[t] = 1
        elif votes.count(-1) >= min_votes:
            out[t] = -1
    return out

def detect_cusum_causal(elec: np.ndarray, warmup: int=6, k: float=0.5, h: float=4.0) -> np.ndarray:
    """Past-only CUSUM reference detector on electricity.

    Its three parameters are fixed declared constants rather than development-selected
    thresholds: ``warmup`` months of history give the running mean and standard deviation,
    ``k`` is the reference-value slack in standard deviations, and ``h`` is the decision
    interval. Unlike the other comparators, this detector is therefore not tuned on the
    development households, and it is reported as a declared baseline rather than as a
    tuned operating point.
    """
    e = _clean_causal(elec)
    out = np.zeros(len(e), dtype=int)
    hi = lo = 0.0
    anchor = warmup
    for t in range(warmup, len(e)):
        hist = e[anchor - warmup:t] if t - anchor >= 2 else e[max(0, t - warmup):t]
        if not (np.isfinite(hist).all() and np.isfinite(e[t - 1])):
            continue
        (mu, sd) = (float(np.mean(hist)), float(np.std(hist)))
        if sd <= 1e-09:
            continue
        z = (e[t - 1] - mu) / sd
        hi = max(0.0, hi + z - k)
        lo = max(0.0, lo - z - k)
        if hi > h:
            out[t] = 1
            hi = lo = 0.0
            anchor = t
        elif lo > h:
            out[t] = -1
            hi = lo = 0.0
            anchor = t
    return out

def detect_estimate_shift(pop_estimate: np.ndarray, window: int=3, min_delta: float=0.5) -> np.ndarray:
    """Public function: detect_estimate_shift."""
    p = np.asarray(pop_estimate, dtype=float)
    out = np.zeros(len(p), dtype=int)
    for t in range(2 * window, len(p)):
        recent_values = p[t - window:t]
        prior_values = p[t - 2 * window:t - window]
        if not (np.isfinite(recent_values).all() and np.isfinite(prior_values).all()):
            continue
        recent = np.mean(recent_values)
        prior = np.mean(prior_values)
        d = recent - prior
        if d > min_delta:
            out[t] = 1
        elif d < -min_delta:
            out[t] = -1
    return out

def apply_persistence(alarms: np.ndarray, min_consecutive: int=2) -> np.ndarray:
    """Public function: apply_persistence."""
    out = np.zeros_like(alarms, dtype=int)
    i = 0
    while i < len(alarms):
        if alarms[i] != 0:
            j = i + 1
            while j < len(alarms) and alarms[j] == alarms[i]:
                j += 1
            if j - i >= min_consecutive:
                out[i + min_consecutive - 1] = alarms[i]
            i = j
        else:
            i += 1
    return out

def retrospective_verify(alarms: np.ndarray, elec: np.ndarray, water: np.ndarray, gas: np.ndarray, window: int=6, threshold: float=0.08) -> np.ndarray:
    """Retrospective composite verifier (never an online operating point).

    An alarm survives only if the fixed-weight composite level moved in the alarm
    direction. The confirmation reads the ``window`` months **after** the alarm and
    refers them to the ``window`` months before it; the look-ahead half is what makes the
    comparator retrospective, and it is reported separately from the online detectors.
    """
    out = alarms.copy()
    comp = _clean_retrospective(elec) + 0.5 * _clean_retrospective(water) + 2.0 * _clean_retrospective(gas)
    for t in range(len(alarms)):
        if alarms[t] == 0:
            continue
        before = comp[max(0, t - window):t]
        after = comp[t:min(len(comp), t + window)]
        if len(before) == 0 or len(after) == 0:
            out[t] = 0
            continue
        d = (after.mean() - before.mean()) / (before.mean() + 0.01)
        if alarms[t] == 1 and d < threshold:
            out[t] = 0
        elif alarms[t] == -1 and d > -threshold:
            out[t] = 0
    return out

def match_events(alarms_by_hh: Dict[int, np.ndarray], events: pd.DataFrame, n_months: int, tolerance: int, monitoring_start: int=0) -> Dict[str, float]:
    """Public function: match_events."""
    n_events = 0
    n_detected = 0
    delays: List[int] = []
    matched_alarms = 0
    total_alarms = 0
    ev_by_hh: Dict[int, List[Tuple[int, int]]] = {}
    for (_, r) in events.iterrows():
        ev_by_hh.setdefault(int(r['household_id']), []).append((int(r['start_month']), int(r['direction'])))
    for (hid, alarms) in alarms_by_hh.items():
        pos = [(t, int(alarms[t])) for t in range(monitoring_start, len(alarms)) if alarms[t] != 0]
        total_alarms += len(pos)
        used = set()
        for (t_ev, d) in sorted(ev_by_hh.get(hid, [])):
            n_events += 1
            cand = [(t, s) for (t, s) in pos if s == d and t_ev <= t <= t_ev + tolerance and (t not in used)]
            if cand:
                t_hit = min((t for (t, _) in cand))
                used.add(t_hit)
                n_detected += 1
                delays.append(t_hit - t_ev)
        matched_alarms += len(used)
    n_premises = max(1, len(alarms_by_hh))
    premise_years = n_premises * max(0, n_months - monitoring_start) / 12.0
    premise_years = max(premise_years, 1e-12)
    recall = n_detected / n_events if n_events else float('nan')
    precision = matched_alarms / total_alarms if total_alarms else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if n_events and total_alarms and (precision + recall > 0) else 0.0
    return {'n_events': n_events, 'n_detected': n_detected, 'n_alarms': total_alarms, 'n_matched_alarms': matched_alarms, 'event_recall': recall, 'event_precision': precision, 'event_f1': f1, 'mean_delay_months': float(np.mean(delays)) if delays else float('nan'), 'median_delay_months': float(np.median(delays)) if delays else float('nan'), 'false_alarms_per_premise_year': (total_alarms - matched_alarms) / premise_years}

def _bootstrap_event_ci(alarms_by_hh: Dict[int, np.ndarray], events: pd.DataFrame, n_months: int, tolerance: int, metric: str, n_boot: int, seed: int, monitoring_start: int=0) -> Tuple[float, float]:
    """Public function: _bootstrap_event_ci."""
    rng = np.random.RandomState(seed)
    hids = np.array(sorted(alarms_by_hh))
    vals: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(hids, len(hids), replace=True)
        sub_alarms = {}
        sub_events = []
        for (j, h) in enumerate(pick):
            sub_alarms[j] = alarms_by_hh[h]
            ev = events[events['household_id'] == h]
            for (_, r) in ev.iterrows():
                sub_events.append({'household_id': j, 'start_month': r['start_month'], 'direction': r['direction']})
        se = pd.DataFrame(sub_events) if sub_events else pd.DataFrame(columns=['household_id', 'start_month', 'direction'])
        m = match_events(sub_alarms, se, n_months, tolerance, monitoring_start)
        v = m[metric]
        if v == v:
            vals.append(v)
    if not vals:
        return (float('nan'), float('nan'))
    (lo, hi) = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi))

def _random_timing_reference(reference_alarms: Dict[int, np.ndarray], eligible_masks: Dict[int, np.ndarray], events: pd.DataFrame, n_months: int, tolerance: int, n_repetitions: int, seed: int, monitoring_start: int=0) -> Dict[str, float]:
    """Public function: _random_timing_reference."""
    rng = np.random.RandomState(seed)
    rows: List[Dict[str, float]] = []
    for _ in range(n_repetitions):
        randomized = _randomize_alarm_timing_once(reference_alarms, eligible_masks, rng, monitoring_start)
        rows.append(match_events(randomized, events, n_months, tolerance, monitoring_start))
    frame = pd.DataFrame(rows)
    out: Dict[str, float] = {}
    detected = frame['n_detected'].astype(float)
    total_detected = float(detected.sum())
    for column in frame.columns:
        if column == 'mean_delay_months':
            out[column] = float((frame[column] * detected).sum() / total_detected) if total_detected > 0 else float('nan')
        elif column == 'median_delay_months':
            out[column] = float('nan')
        else:
            out[column] = float(frame[column].mean())
    out['n_events'] = int(round(out['n_events']))
    out['event_f1_ci_low'] = float(frame['event_f1'].quantile(0.025))
    out['event_f1_ci_high'] = float(frame['event_f1'].quantile(0.975))
    return out

def _randomize_alarm_timing_once(reference_alarms: Dict[int, np.ndarray], eligible_masks: Dict[int, np.ndarray], rng: np.random.RandomState, monitoring_start: int=0) -> Dict[int, np.ndarray]:
    """Public function: _randomize_alarm_timing_once."""
    if set(reference_alarms) != set(eligible_masks):
        raise ValueError('eligible-mask households must match reference alarms')
    randomized: Dict[int, np.ndarray] = {}
    for (hid, alarms) in reference_alarms.items():
        alarms = np.asarray(alarms, dtype=int)
        eligible = np.asarray(eligible_masks[hid], dtype=bool).copy()
        if eligible.shape != alarms.shape:
            raise ValueError(f'eligible mask shape mismatch for household {hid}')
        eligible[:monitoring_start] = False
        if np.any((alarms != 0) & ~eligible):
            bad = np.flatnonzero((alarms != 0) & ~eligible).tolist()
            raise ValueError(f'reference alarms outside eligible support for household {hid}: {bad}')
        positions = np.flatnonzero(eligible)
        values = np.zeros_like(alarms)
        values[positions] = rng.permutation(alarms[positions])
        randomized[hid] = values
    return randomized

def _persistent_structural_eligibility(raw_eligible: np.ndarray, persistence: int, monitoring_start: int) -> np.ndarray:
    """Public function: _persistent_structural_eligibility."""
    raw = np.asarray(raw_eligible, dtype=bool)
    out = np.zeros(len(raw), dtype=bool)
    if persistence < 1:
        raise ValueError('persistence must be at least one')
    for t in range(persistence - 1, len(raw)):
        out[t] = raw[t - persistence + 1:t + 1].all()
    out[:monitoring_start] = False
    return out

def _estimate_shift_eligible_mask(pop_estimate: np.ndarray, window: int, persistence: int, monitoring_start: int) -> np.ndarray:
    """Public function: _estimate_shift_eligible_mask."""
    p = np.asarray(pop_estimate, dtype=float)
    raw = np.zeros(len(p), dtype=bool)
    for t in range(2 * window, len(p)):
        raw[t] = np.isfinite(p[t - window:t]).all() and np.isfinite(p[t - 2 * window:t - window]).all()
    return _persistent_structural_eligibility(raw, persistence, monitoring_start)

def _multisignal_eligible_mask(signals: Sequence[np.ndarray], window: int, persistence: int, monitoring_start: int, min_votes: int=2) -> np.ndarray:
    """Public function: _multisignal_eligible_mask."""
    cleaned = [_clean_causal(np.asarray(signal, dtype=float)) for signal in signals]
    n = len(cleaned[0])
    raw = np.zeros(n, dtype=bool)
    for t in range(2 * window, n):
        available = sum((np.isfinite(signal[t - window:t]).all() and np.isfinite(signal[t - 2 * window:t - window]).all() for signal in cleaned))
        raw[t] = available >= min_votes
    return _persistent_structural_eligibility(raw, persistence, monitoring_start)
CAL_NAME = 'Calendar-adjusted baseline-locked'
CAL_BASELINE_END = 14
CAL_AT_RISK_START = 14

def learn_calendar_profile(panel, train_hids):
    """Public function: learn_calendar_profile."""
    meta = panel.meta.set_index('household_id')
    log_series: Dict[int, np.ndarray] = {}
    for (hid, grp) in panel.monthly[panel.monthly['household_id'].isin(set((int(h) for h in train_hids)))].groupby('household_id'):
        if str(meta.loc[hid, 'true_type']) not in ('residential', 'mixed'):
            continue
        grp = grp.sort_values('month')
        log_series[int(hid)] = np.log(np.clip(grp[['electricity_kwh', 'water_ton', 'gas_m3']].values, 1e-09, None))
    acc = [[[] for _ in range(12)] for _ in range(3)]
    for L in log_series.values():
        level = np.nanmean(L[:CAL_BASELINE_END], axis=0)
        for t in range(CAL_BASELINE_END):
            m = t % 12
            for u in range(3):
                v = L[t, u] - level[u]
                if np.isfinite(v):
                    acc[u][m].append(v)
    g = np.zeros((3, 12))
    for u in range(3):
        for m in range(12):
            g[u, m] = float(np.mean(acc[u][m])) if acc[u][m] else 0.0
    per_u: List[List[float]] = [[], [], []]
    for L in log_series.values():
        d = L[:CAL_BASELINE_END] - g[:, np.arange(CAL_BASELINE_END) % 12].T
        level = np.nanmean(d, axis=0)
        for u in range(3):
            per_u[u].extend([float(v) for v in d[:, u] - level[u] if np.isfinite(v)])
    sd = np.array([float(np.std(per_u[u])) for u in range(3)])
    return (g, sd)

def calendar_adjusted_scores(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, g: np.ndarray, sd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Locked baseline and joint de-seasonalised score for the CABL detector.

    Reads are forward-filled causally only. Months before the first observed read stay
    missing and are therefore excluded from the locked household baseline instead of being
    backfilled from a later value, so no future information reaches the score
    (``src/checks.py`` verifies this). A utility without a finite calibration level is
    dropped from the joint score rather than propagated as a missing value.
    """
    L = np.column_stack([_clean_causal(np.asarray(elec, dtype=float)), _clean_causal(np.asarray(water, dtype=float)), _clean_causal(np.asarray(gas, dtype=float))])
    L = np.log(np.clip(L, 1e-09, None))
    n = L.shape[0]
    moy = np.arange(n) % 12 + 1
    d = L - g[:, moy - 1].T
    baseline = d[:CAL_BASELINE_END]
    available = np.isfinite(baseline).any(axis=0)
    b = np.full(3, np.nan, dtype=float)
    if available.any():
        b[available] = np.nanmean(baseline[:, available], axis=0)
    z = np.zeros(n, dtype=float)
    scored = np.zeros(n, dtype=bool)
    for u in range(3):
        if not available[u] or not np.isfinite(sd[u]) or sd[u] <= 0.0:
            continue
        contribution = (d[:, u] - b[u]) / sd[u]
        finite = np.isfinite(contribution)
        z[finite] += contribution[finite]
        scored |= finite
    z[~scored] = np.nan
    return (b, z)

def detect_calendar_adjusted(elec: np.ndarray, water: np.ndarray, gas: np.ndarray, g: np.ndarray, sd: np.ndarray, tau: float) -> np.ndarray:
    """Calendar-adjusted baseline-locked (CABL) detector.

    Scores the first difference of the joint de-seasonalised score against ``tau`` and
    leaves the months before month 14 silent. See ``calendar_adjusted_scores`` for the
    missing-read rule.
    """
    (_, z) = calendar_adjusted_scores(elec, water, gas, g, sd)
    n = len(z)
    dz = np.diff(z, prepend=z[0])
    usable = np.isfinite(dz)
    a = np.zeros(n, dtype=int)
    a[usable & (dz > tau)] = 1
    a[usable & (dz < -tau)] = -1
    a[:CAL_AT_RISK_START] = 0
    return a

def _calendar_adjusted_eligible_mask(n_months: int) -> np.ndarray:
    """Public function: _calendar_adjusted_eligible_mask."""
    m = np.zeros(n_months, dtype=bool)
    m[CAL_AT_RISK_START:] = True
    return m

def split_task_c_households(panel, cfg: Optional[TaskCConfig]=None, eligible_households: Optional[Sequence[int]]=None) -> Tuple[np.ndarray, np.ndarray]:
    """Split the evaluation universe into development and test households.

    ``eligible_households`` is required. Task C is defined on the households held out by
    Task A (the 300-household evaluation set in the study), not on every occupied
    household of the panel; defaulting to the whole panel would silently change the
    development/test counts and every threshold selected from them.
    """
    cfg = cfg or TaskCConfig()
    if eligible_households is None:
        raise ValueError('eligible_households is required: pass the households held out by Task A')
    known = set((int(h) for h in panel.meta['household_id'].values))
    eligible = np.array(sorted(set((int(h) for h in eligible_households)) & known), dtype=int)
    hids = np.array(sorted(set((int(h) for h in eligible))), dtype=int)
    if len(hids) < 2:
        raise ValueError('Task C requires at least two eligible households')
    rng = np.random.RandomState(cfg.seed)
    shuffled = hids.copy()
    rng.shuffle(shuffled)
    n_dev = int(round(len(shuffled) * cfg.dev_fraction))
    n_dev = min(max(1, n_dev), len(shuffled) - 1)
    return (np.sort(shuffled[:n_dev]), np.sort(shuffled[n_dev:]))

@dataclass
class TaskCResult:
    """Public class: TaskCResult."""
    summary: pd.DataFrame
    dev_selection: pd.DataFrame
    selected_theta: Dict[str, float]
    n_dev_households: int
    n_test_households: int
    n_test_events: int
    example: dict
    pointwise_reference: pd.DataFrame
    tolerance_sensitivity: pd.DataFrame
    dev_households: np.ndarray
    test_households: np.ndarray
    trajectory_train_households: np.ndarray
    n_random_baseline: int
    active_households: np.ndarray
    monitoring_start_month: int
    random_reference_detector: str
    test_alarms: Dict[str, Dict[int, np.ndarray]]

def run_task_c(panel, pop_estimates: Optional[Dict[int, np.ndarray]]=None, cfg: Optional[TaskCConfig]=None, eligible_households: Optional[Sequence[int]]=None, active_households: Optional[Sequence[int]]=None, trajectory_train_households: Optional[Sequence[int]]=None) -> TaskCResult:
    """Public function: run_task_c."""
    cfg = cfg or TaskCConfig()
    (dev_ids, test_ids) = split_task_c_households(panel, cfg, eligible_households)
    (dev_hids, test_hids) = (set(dev_ids), set(test_ids))
    hids = np.array(sorted(dev_hids | test_hids), dtype=int)
    active_hids = set((int(h) for h in active_households)) if active_households is not None else set(hids)
    if not active_hids <= set(hids):
        raise ValueError('active Task C households must be a subset of eligible households')
    train_ids = np.array([], dtype=int)
    if pop_estimates is not None:
        if trajectory_train_households is None:
            raise ValueError('trajectory_train_households is required when pop_estimates are supplied')
        train_ids = np.array(sorted(set((int(h) for h in trajectory_train_households))), dtype=int)
        overlap = set(train_ids) & set(hids)
        if overlap:
            raise ValueError('Task B trajectory training/evaluation household overlap: %s' % sorted(overlap)[:10])
        missing = active_hids - set((int(h) for h in pop_estimates))
        if missing:
            raise ValueError('population trajectories missing for Task C households: %s' % sorted(missing)[:10])
    series: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (hid, grp) in panel.monthly[panel.monthly['household_id'].isin(hids)].groupby('household_id'):
        grp = grp.sort_values('month')
        series[int(hid)] = (grp['electricity_kwh'].values, grp['water_ton'].values, grp['gas_m3'].values)
    n_months = panel.population.shape[1]
    if not 0 <= cfg.monitoring_start_month < n_months:
        raise ValueError('monitoring_start_month must lie inside the panel')
    events = panel.events[panel.events['start_month'] >= cfg.monitoring_start_month].copy()
    ev_dev = events[events['household_id'].isin(dev_hids)]
    ev_test = events[events['household_id'].isin(test_hids)]
    calendar = None
    if trajectory_train_households is not None:
        cal_train_ids = np.array(sorted(set((int(h) for h in trajectory_train_households))), dtype=int)
        if len(cal_train_ids):
            calendar = learn_calendar_profile(panel, cal_train_ids)

    def monitored(alarms: np.ndarray) -> np.ndarray:
        out = np.asarray(alarms, dtype=int).copy()
        out[:cfg.monitoring_start_month] = 0
        return out
    dev_rows: List[dict] = []
    for th in cfg.elec_theta_grid:
        al = {h: monitored(apply_persistence(detect_ratio_causal(series[h][0], cfg.window, th), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in dev_hids}
        m = match_events(al, ev_dev, n_months, cfg.tolerance_months, cfg.monitoring_start_month)
        dev_rows.append({'detector': 'Past-only ratio (electricity)', 'theta': th, **m})
    for th in cfg.elec_theta_grid:
        al = {h: monitored(apply_persistence(detect_multisignal_causal(*series[h], window=cfg.window, theta=th), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in dev_hids}
        m = match_events(al, ev_dev, n_months, cfg.tolerance_months, cfg.monitoring_start_month)
        dev_rows.append({'detector': 'Past-only multi-signal (2/3)', 'theta': th, **m})
    if pop_estimates is not None:
        for th in cfg.theta_grid:
            al = {h: monitored(apply_persistence(detect_estimate_shift(pop_estimates[h], cfg.window, th), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in dev_hids}
            m = match_events(al, ev_dev, n_months, cfg.tolerance_months, cfg.monitoring_start_month)
            dev_rows.append({'detector': 'Past-only estimate shift (persons)', 'theta': th, **m})
    if calendar is not None:
        for th in cfg.cal_tau_grid:
            al = {h: detect_calendar_adjusted(*series[h], calendar[0], calendar[1], th) if h in active_hids else np.zeros(n_months, dtype=int) for h in dev_hids}
            m = match_events(al, ev_dev, n_months, cfg.tolerance_months, cfg.monitoring_start_month)
            dev_rows.append({'detector': CAL_NAME, 'theta': th, **m})
    dev_selection = pd.DataFrame(dev_rows)
    selected: Dict[str, float] = {}
    for (det, grp) in dev_selection.groupby('detector'):
        selected[det] = float(grp.loc[grp['event_f1'].idxmax(), 'theta'])
    logger.info('Task C thresholds selected on %d development households: %s', len(dev_hids), {k: round(v, 3) for (k, v) in selected.items()})
    th_e = selected['Past-only ratio (electricity)']
    th_m = selected['Past-only multi-signal (2/3)']
    variants: Dict[str, Dict[int, np.ndarray]] = {}
    variants['Past-only ratio (electricity)'] = {h: monitored(detect_ratio_causal(series[h][0], cfg.window, th_e)) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    variants['Past-only ratio + persistence'] = {h: monitored(apply_persistence(detect_ratio_causal(series[h][0], cfg.window, th_e), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    variants['Past-only multi-signal (2/3) + persistence'] = {h: monitored(apply_persistence(detect_multisignal_causal(*series[h], window=cfg.window, theta=th_m), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    variants['Past-only CUSUM (electricity)'] = {h: monitored(detect_cusum_causal(series[h][0])) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    if pop_estimates is not None:
        th_p = selected['Past-only estimate shift (persons)']
        variants['Past-only estimate shift (cascade)'] = {h: monitored(apply_persistence(detect_estimate_shift(pop_estimates[h], cfg.window, th_p), cfg.persistence)) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    if calendar is not None:
        th_cal = selected[CAL_NAME]
        variants[CAL_NAME] = {h: detect_calendar_adjusted(*series[h], calendar[0], calendar[1], th_cal) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    variants['Retrospective composite verification'] = {h: monitored(retrospective_verify(apply_persistence(detect_multisignal_causal(*series[h], window=cfg.window, theta=th_m), cfg.persistence), *series[h], window=cfg.retrospective_window, threshold=cfg.retrospective_threshold)) if h in active_hids else np.zeros(n_months, dtype=int) for h in test_hids}
    rows: List[dict] = []
    for (name, al) in variants.items():
        ev = ev_test[ev_test['household_id'].isin(al)]
        m = match_events(al, ev, n_months, cfg.tolerance_months, cfg.monitoring_start_month)
        (lo, hi) = _bootstrap_event_ci(al, ev, n_months, cfg.tolerance_months, 'event_f1', cfg.n_bootstrap, cfg.seed, cfg.monitoring_start_month)
        rows.append({'detector': name, 'online': 'Retrospective' not in name, **m, 'event_f1_ci_low': lo, 'event_f1_ci_high': hi})
    rows.append({'detector': 'Never alarm (trivial)', 'online': True, **match_events({h: np.zeros(n_months, dtype=int) for h in test_hids}, ev_test, n_months, cfg.tolerance_months, cfg.monitoring_start_month), 'event_f1_ci_low': 0.0, 'event_f1_ci_high': 0.0})
    reference_name = 'Past-only estimate shift (cascade)' if 'Past-only estimate shift (cascade)' in variants else 'Past-only multi-signal (2/3) + persistence'
    random_label = 'Random timing (cascade-alarm matched)' if reference_name == 'Past-only estimate shift (cascade)' else 'Random timing (multi-signal-alarm matched)'
    if reference_name == 'Past-only estimate shift (cascade)':
        random_eligible_masks = {h: _estimate_shift_eligible_mask(pop_estimates[h], cfg.window, cfg.persistence, cfg.monitoring_start_month) if h in active_hids and h in pop_estimates else np.zeros(n_months, dtype=bool) for h in test_hids}
    else:
        random_eligible_masks = {h: _multisignal_eligible_mask(series[h], cfg.window, cfg.persistence, cfg.monitoring_start_month) for h in test_hids}
    cal_eligible_masks = None
    if CAL_NAME in variants:
        cal_eligible_masks = {h: _calendar_adjusted_eligible_mask(n_months) for h in test_hids}
    sensitivity_rows: List[dict] = []
    tolerances = sorted(set(cfg.tolerance_sensitivity_months + (cfg.tolerance_months,)))
    random_by_tolerance: Dict[int, Dict[str, float]] = {}
    for tolerance in tolerances:
        observed = match_events(variants[reference_name], ev_test, n_months, tolerance, cfg.monitoring_start_month)
        (lo, hi) = _bootstrap_event_ci(variants[reference_name], ev_test, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed, cfg.monitoring_start_month)
        sensitivity_rows.append({'detector': reference_name, 'tolerance_months': tolerance, **observed, 'event_f1_ci_low': lo, 'event_f1_ci_high': hi})
        randomized = _random_timing_reference(variants[reference_name], random_eligible_masks, ev_test, n_months, tolerance, cfg.n_random_baseline, cfg.seed + 10000, cfg.monitoring_start_month)
        random_by_tolerance[tolerance] = randomized
        sensitivity_rows.append({'detector': random_label, 'tolerance_months': tolerance, **randomized})
    if CAL_NAME in variants:
        for tolerance in tolerances:
            observed = match_events(variants[CAL_NAME], ev_test, n_months, tolerance, cfg.monitoring_start_month)
            (lo, hi) = _bootstrap_event_ci(variants[CAL_NAME], ev_test, n_months, tolerance, 'event_f1', cfg.n_bootstrap, cfg.seed, cfg.monitoring_start_month)
            sensitivity_rows.append({'detector': CAL_NAME, 'tolerance_months': tolerance, **observed, 'event_f1_ci_low': lo, 'event_f1_ci_high': hi})
            randomized = _random_timing_reference(variants[CAL_NAME], cal_eligible_masks, ev_test, n_months, tolerance, cfg.n_random_baseline, cfg.seed + 20000, cfg.monitoring_start_month)
            sensitivity_rows.append({'detector': 'Random timing (calendar-adjusted matched)', 'tolerance_months': tolerance, **randomized})
    tolerance_sensitivity = pd.DataFrame(sensitivity_rows)
    random_row = random_by_tolerance[cfg.tolerance_months]
    rows.append({'detector': random_label, 'online': True, **random_row})
    if CAL_NAME in variants:
        cal_random = _random_timing_reference(variants[CAL_NAME], cal_eligible_masks, ev_test, n_months, cfg.tolerance_months, cfg.n_random_baseline, cfg.seed + 20000, cfg.monitoring_start_month)
        rows.append({'detector': 'Random timing (calendar-adjusted matched)', 'online': True, **cal_random})
    summary = pd.DataFrame(rows)
    pw_rows: List[dict] = []
    for (name, al) in variants.items():
        (yt, yp) = ([], [])
        for (h, a) in al.items():
            pop = panel.population[h]
            true_ch = np.clip(np.diff(pop), -1, 1)
            start = max(0, cfg.monitoring_start_month - 1)
            yt.append(true_ch[start:])
            yp.append(a[1:][start:])
        yt = np.concatenate(yt)
        yp = np.concatenate(yp)
        from sklearn.metrics import precision_recall_fscore_support
        (_, _, f1w, _) = precision_recall_fscore_support(yt, yp, average='weighted', zero_division=0)
        pw_rows.append({'detector': name, 'pointwise_accuracy': float(np.mean(yt == yp)), 'pointwise_weighted_f1': float(f1w), 'trivial_accuracy': float(np.mean(yt == 0))})
    (_, _, f1w_triv, _) = precision_recall_fscore_support(yt, np.zeros_like(yt), average='weighted', zero_division=0)
    pw_rows.append({'detector': 'Never alarm (trivial)', 'pointwise_accuracy': float(np.mean(yt == 0)), 'pointwise_weighted_f1': float(f1w_triv), 'trivial_accuracy': float(np.mean(yt == 0))})
    pointwise = pd.DataFrame(pw_rows)
    eventful = sorted((h for h in test_hids if h in active_hids and (ev_test['household_id'] == h).sum() >= 1))
    active_test = sorted(test_hids & active_hids)
    ex_hid = int(eventful[0] if eventful else active_test[0] if active_test else sorted(test_hids)[0])
    (e, w, g) = series[ex_hid]
    example = {'household_id': ex_hid, 'electricity': e, 'water': w, 'gas': g, 'population': panel.population[ex_hid], 'alarms_ratio': variants['Past-only ratio + persistence'][ex_hid], 'alarms_multisig': variants['Past-only multi-signal (2/3) + persistence'][ex_hid], 'events': ev_test[ev_test['household_id'] == ex_hid].to_dict('records')}
    if pop_estimates is not None and ex_hid in pop_estimates:
        example['pop_estimate'] = pop_estimates[ex_hid]
        example['alarms_cascade'] = variants['Past-only estimate shift (cascade)'][ex_hid]
    logger.info('Task C test households=%d events=%d', len(test_hids), len(ev_test))
    return TaskCResult(summary=summary, dev_selection=dev_selection, selected_theta=selected, n_dev_households=len(dev_hids), n_test_households=len(test_hids), n_test_events=int(len(ev_test)), example=example, pointwise_reference=pointwise, tolerance_sensitivity=tolerance_sensitivity, dev_households=dev_ids, test_households=test_ids, trajectory_train_households=train_ids, n_random_baseline=cfg.n_random_baseline, active_households=np.array(sorted(active_hids), dtype=int), monitoring_start_month=cfg.monitoring_start_month, random_reference_detector=reference_name, test_alarms=variants)
