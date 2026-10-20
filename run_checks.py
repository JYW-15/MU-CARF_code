"""Run the non-anticipation checks on a small synthetic panel.

The main article's audit table says that prefix-stability and missing-read checks cover
the past-only rule. This script executes them for the calendar-adjusted baseline-locked
(CABL) comparator, MU-CARF, and the past-only ratio detector, and exits non-zero if any
check fails.

    python run_checks.py
"""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.checks import baseline_exclusion_check, run_all_checks
from src.config import DataConfig, TaskAConfig
from src.data_generation import generate_panel
from src.features import household_features
from src.mucarf import detect_mucarf, fit_mucarf
from src.task_a_screening import run_task_a
from src.task_c_detection import calendar_adjusted_scores, detect_calendar_adjusted, detect_ratio_causal, learn_calendar_profile

MUCARF_TAU_GRID = (0.15, 0.25, 0.35, 0.5)
CAL_TAU_GRID = (2.0, 3.0, 4.0, 5.0, 6.0)
RATIO_THETA_GRID = (0.1, 0.15, 0.2, 0.3)
CHECK_START = 14
GAP_START = 20
LEADING_MONTHS = 3


def _pick_tau(build, grid, elec, water, gas):
    """Return the first threshold in the grid that produces at least one alarm."""
    for tau in grid:
        alarms = np.asarray(build(tau)(elec, water, gas), dtype=int)
        if np.count_nonzero(alarms[CHECK_START:]) > 0:
            return (tau, alarms)
    return (grid[-1], np.asarray(build(grid[-1])(elec, water, gas), dtype=int))


def main() -> int:
    cfg = DataConfig(n_households=120, n_months=36, seed=42, regime='R1_linear')
    panel = generate_panel(cfg)
    calibration = panel.monthly[panel.monthly['month'] < 12]
    screen = run_task_a(household_features(calibration), panel.meta, TaskAConfig(n_bootstrap=5, seed=42))
    meta = panel.meta.set_index('household_id')
    train_ids = np.array(sorted(int(h) for h in screen.is_test[~screen.is_test].index
                                if meta.loc[int(h), 'true_type'] in {'residential', 'mixed'}))
    test_ids = np.array(sorted(int(h) for h in screen.is_test[screen.is_test].index
                               if screen.predicted_type.loc[int(h)] in {'residential', 'mixed'}))
    if len(test_ids) < 3:
        print('not enough screened-in test households for the checks')
        return 1
    model = fit_mucarf(panel, train_ids, baseline_end=14, state_radius=3)
    (calendar_profile, calendar_sd) = learn_calendar_profile(panel, train_ids)

    series = {}
    for hid in test_ids:
        rows = panel.monthly[panel.monthly['household_id'] == hid].sort_values('month')
        series[int(hid)] = (rows['electricity_kwh'].to_numpy(dtype=float),
                            rows['water_ton'].to_numpy(dtype=float),
                            rows['gas_m3'].to_numpy(dtype=float))

    selected = [int(h) for h in test_ids[:3]]
    cutoffs = list(range(CHECK_START, cfg.n_months))
    reports = []
    for hid in selected:
        (elec, water, gas) = series[hid]
        (mucarf_tau, _) = _pick_tau(lambda tau: (lambda e, w, g: detect_mucarf(e, w, g, model, tau).alarms), MUCARF_TAU_GRID, elec, water, gas)
        (cal_tau, _) = _pick_tau(lambda tau: (lambda e, w, g: detect_calendar_adjusted(e, w, g, calendar_profile, calendar_sd, tau)), CAL_TAU_GRID, elec, water, gas)
        (ratio_theta, _) = _pick_tau(lambda theta: (lambda e, w, g: detect_ratio_causal(e, window=3, theta=theta)), RATIO_THETA_GRID, elec, water, gas)
        detectors = {
            'MU-CARF (tau=%.2f)' % mucarf_tau: lambda e, w, g: detect_mucarf(e, w, g, model, mucarf_tau).alarms,
            'CABL (tau=%.1f)' % cal_tau: lambda e, w, g: detect_calendar_adjusted(e, w, g, calendar_profile, calendar_sd, cal_tau),
            'Past-only ratio (theta=%.2f)' % ratio_theta: lambda e, w, g: detect_ratio_causal(e, window=3, theta=ratio_theta),
        }
        for (name, detect) in detectors.items():
            for report in run_all_checks(detect, elec, water, gas, cutoffs=cutoffs, gap_start=GAP_START, leading_months=LEADING_MONTHS, detector_name=name):
                report['household_id'] = hid
                reports.append(report)
        exclusion = baseline_exclusion_check(lambda e, w, g: calendar_adjusted_scores(e, w, g, calendar_profile, calendar_sd),
                                            calendar_profile, elec, water, gas,
                                            leading_months=LEADING_MONTHS, detector_name='CABL')
        exclusion['household_id'] = hid
        reports.append(exclusion)

    print('%-28s %-18s %-6s %s' % ('detector', 'check', 'result', 'detail'))
    failed = 0
    for report in reports:
        status = 'pass' if report['passed'] else 'FAIL'
        failed += 0 if report['passed'] else 1
        detail = '%d cases' % report['n_checked']
        if report['n_mismatch']:
            detail = '%d mismatches, first %s' % (report['n_mismatch'], report['mismatches'][0])
        print('%-28s %-18s %-6s %s' % (report['detector'], report['check'], status, detail))
    households = sorted({report['household_id'] for report in reports})
    total = len(reports)
    print('\n%d checks over %d households (prefix cutoffs %d-%d, gap at month %d, leading gap %d months)'
          % (total, len(households), cutoffs[0], cutoffs[-1], GAP_START, LEADING_MONTHS))
    print('%d passed, %d failed' % (total - failed, failed))
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
