"""Executable checks for the non-anticipation claims of the evaluation protocol.

The main article's validity audit states that every online detector uses past-only
windows and causal forward filling, with no leading-value backfill, and that
prefix-stability and missing-read checks cover that rule. This module supplies those
checks so that the claim is executable rather than descriptive.

Every detector is passed as a callable ``detect(elec, water, gas) -> np.ndarray`` that
returns one signed monthly alarm per input month. Three properties are checked:

``prefix_stability_check``
    Truncating the series after month ``p`` must not change any alarm at or before
    ``p``. A violation means the detector used information from after ``p``.

``missing_read_check``
    Masking an interior block of reads from month ``g`` must not change any alarm
    before ``g``. A violation means the detector filled the gap from later reads.

``leading_gap_check``
    Masking the first ``L`` months must not produce an alarm before the first observed
    month and must leave a usable alarm vector. This is a structural guard: a detector
    that backfilled leading gaps from a later observation would be able to arm itself
    before any read existed.

None of the three proves the absence of every possible leakage path; together they
exercise the three routes named in the audit table.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ['prefix_stability_check', 'missing_read_check', 'leading_gap_check', 'baseline_exclusion_check', 'run_all_checks']


def _as_arrays(elec, water, gas):
    return (np.asarray(elec, dtype=float), np.asarray(water, dtype=float), np.asarray(gas, dtype=float))


def _as_alarms(values, expected_length: int, detector_name: str) -> np.ndarray:
    alarms = np.asarray(values, dtype=int)
    if alarms.ndim != 1 or len(alarms) != expected_length:
        raise ValueError('%s must return one alarm per input month' % detector_name)
    return alarms


def prefix_stability_check(detect, elec, water, gas, cutoffs: Optional[Sequence[int]]=None, detector_name: str='detector') -> Dict[str, object]:
    """Alarms up to month ``p`` must be unchanged when the series ends at ``p``."""
    (e, w, g) = _as_arrays(elec, water, gas)
    n_months = len(e)
    if not (len(w) == len(g) == n_months):
        raise ValueError('utility series must share one length')
    months = list(range(n_months)) if cutoffs is None else [int(value) for value in cutoffs]
    full = _as_alarms(detect(e, w, g), n_months, detector_name)
    mismatches: List[Dict[str, int]] = []
    for cutoff in months:
        if not 0 <= cutoff < n_months:
            raise ValueError('cutoff %d lies outside the series' % cutoff)
        truncated = _as_alarms(detect(e[:cutoff + 1], w[:cutoff + 1], g[:cutoff + 1]), cutoff + 1, detector_name)
        differing = np.flatnonzero(full[:cutoff + 1] != truncated)
        if differing.size:
            first = int(differing[0])
            mismatches.append({'cutoff': cutoff, 'month': first, 'full_series': int(full[first]), 'truncated': int(truncated[first])})
    return {'check': 'prefix_stability', 'detector': detector_name, 'n_checked': len(months), 'n_mismatch': len(mismatches), 'mismatches': mismatches, 'passed': not mismatches}


def missing_read_check(detect, elec, water, gas, gap_start: int, gap_length: int=3, detector_name: str='detector') -> Dict[str, object]:
    """Masking reads from ``gap_start`` must not change alarms before ``gap_start``."""
    (e, w, g) = _as_arrays(elec, water, gas)
    n_months = len(e)
    gap_start = int(gap_start)
    gap_stop = min(n_months, gap_start + int(gap_length))
    if not 0 < gap_start < n_months:
        raise ValueError('gap_start must lie strictly inside the series')
    full = _as_alarms(detect(e, w, g), n_months, detector_name)
    masked = (e.copy(), w.copy(), g.copy())
    for values in masked:
        values[gap_start:gap_stop] = np.nan
    gapped = _as_alarms(detect(*masked), n_months, detector_name)
    differing = np.flatnonzero(full[:gap_start] != gapped[:gap_start])
    mismatches = [{'month': int(month), 'full_series': int(full[month]), 'gap_masked': int(gapped[month])} for month in differing]
    return {'check': 'missing_read', 'detector': detector_name, 'n_checked': int(gap_start), 'gap': (gap_start, gap_stop), 'n_mismatch': len(mismatches), 'mismatches': mismatches, 'passed': not mismatches}


def leading_gap_check(detect, elec, water, gas, leading_months: int=3, detector_name: str='detector') -> Dict[str, object]:
    """A leading gap must not arm the detector before the first observed month."""
    (e, w, g) = _as_arrays(elec, water, gas)
    n_months = len(e)
    leading_months = int(leading_months)
    if not 1 <= leading_months < n_months:
        raise ValueError('leading_months must lie strictly inside the series')
    masked = (e.copy(), w.copy(), g.copy())
    for values in masked:
        values[:leading_months] = np.nan
    alarms = _as_alarms(detect(*masked), n_months, detector_name)
    early = np.flatnonzero(alarms[:leading_months] != 0)
    mismatches = [{'month': int(month), 'alarm': int(alarms[month])} for month in early]
    return {'check': 'leading_gap', 'detector': detector_name, 'n_checked': 1, 'leading_months': leading_months, 'n_mismatch': len(mismatches), 'mismatches': mismatches, 'passed': not mismatches}


def baseline_exclusion_check(score_fn, calendar: np.ndarray, elec, water, gas, leading_months: int=3, baseline_end: int=14, tolerance: float=1e-09, detector_name: str='CABL') -> Dict[str, object]:
    """A leading gap must be excluded from the locked household baseline.

    ``score_fn(elec, water, gas)`` must return ``(baseline, score)``. The expected
    baseline for each utility is the mean of the de-seasonalised log levels over the
    calibration months that were actually observed. A detector that backfilled the
    leading gap from a later read, or that ignored the gap in the mean, deviates from
    this reference and fails the check.
    """
    (e, w, g) = _as_arrays(elec, water, gas)
    n_months = len(e)
    leading_months = int(leading_months)
    if not 1 <= leading_months < baseline_end:
        raise ValueError('leading_months must leave at least one observed calibration month')
    masked = (e.copy(), w.copy(), g.copy())
    for values in masked:
        values[:leading_months] = np.nan
    logged = np.column_stack([np.log(np.clip(values, 1e-09, None)) for values in masked])
    month_of_year = np.arange(n_months) % 12
    de_seasonalised = logged - np.asarray(calendar, dtype=float)[:, month_of_year].T
    window = de_seasonalised[:baseline_end]
    available = np.isfinite(window).any(axis=0)
    expected = np.full(3, np.nan, dtype=float)
    if available.any():
        expected[available] = np.nanmean(window[:, available], axis=0)
    (observed, _) = score_fn(*masked)
    observed = np.asarray(observed, dtype=float)
    if observed.shape != expected.shape:
        raise ValueError('%s must return a three-utility baseline' % detector_name)
    deviation = np.abs(observed - expected)
    worst = float(np.nanmax(deviation)) if np.isfinite(deviation).any() else 0.0
    passed = bool(np.allclose(observed, expected, atol=tolerance, rtol=0.0, equal_nan=True))
    return {'check': 'baseline_exclusion', 'detector': detector_name, 'n_checked': 1, 'leading_months': leading_months, 'max_deviation': worst, 'n_mismatch': 0 if passed else 1, 'mismatches': [] if passed else [{'expected': expected.tolist(), 'observed': observed.tolist()}], 'passed': passed}


def run_all_checks(detect, elec, water, gas, cutoffs: Sequence[int], gap_start: int, leading_months: int=3, detector_name: str='detector') -> List[Dict[str, object]]:
    """Run the three checks for one detector and return their reports."""
    reports = [prefix_stability_check(detect, elec, water, gas, cutoffs=cutoffs, detector_name=detector_name)]
    reports.append(missing_read_check(detect, elec, water, gas, gap_start=gap_start, detector_name=detector_name))
    reports.append(leading_gap_check(detect, elec, water, gas, leading_months=leading_months, detector_name=detector_name))
    return reports
