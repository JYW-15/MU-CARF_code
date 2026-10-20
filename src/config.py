"""Configuration for the synthetic multi-utility experiments.

Defaults in this module are deliberately small, low-cost demonstration values so that
``run_demo.py`` finishes in a couple of minutes. They are *not* the declared study configuration.
The values that reproduce the published single-seed results are listed in README.md
("Reproducing the published values"), and the constants below record them where a
demo default differs: panel size 1,000 households, primary seed 42, threshold-selection
split seed 42, 500 paired household bootstraps for MU-CARF versus CABL, 200
support-matched randomizations, 2,000 household-clustered bootstrap replicates for the
Task B MAE intervals, and the seed sets (42--46), (2001--2005) and (3001--3010) for the
cascade replication, internal validation and final confirmation.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
__all__ = ['GENERATOR_REGIMES', 'BUILDING_TYPES', 'FEATURE_NAMES', 'FEATURE_GROUPS', 'DataConfig', 'TaskAConfig', 'TaskBConfig', 'TaskCConfig', 'RobustnessConfig', 'CascadeReplicationConfig', 'CalendarStressConfig', 'PipelineConfig']
BUILDING_TYPES: Tuple[str, ...] = ('residential', 'vacant', 'commercial', 'mixed')
BUILDING_PARAMS: Dict[str, dict] = {'residential': {'ratio': 0.65, 'pop_range': (1, 6), 'elec_base': (80, 400), 'water_base': (3, 25), 'gas_base': (10, 80), 'seasonal_amp': 0.3, 'vacation_drop': 0.25, 'n_events': (0, 3)}, 'vacant': {'ratio': 0.18, 'pop_range': (0, 0), 'elec_base': (2, 10), 'water_base': (0.0, 0.5), 'gas_base': (0.0, 1.0), 'seasonal_amp': 0.05, 'vacation_drop': 0.0, 'n_events': (0, 0)}, 'commercial': {'ratio': 0.1, 'pop_range': (0, 0), 'elec_base': (200, 2000), 'water_base': (5, 50), 'gas_base': (20, 200), 'seasonal_amp': 0.1, 'vacation_drop': -0.05, 'n_events': (0, 0)}, 'mixed': {'ratio': 0.07, 'pop_range': (3, 12), 'elec_base': (300, 1500), 'water_base': (10, 80), 'gas_base': (30, 150), 'seasonal_amp': 0.2, 'vacation_drop': 0.15, 'n_events': (0, 3)}}
GENERATOR_REGIMES: Dict[str, dict] = {'R1_linear': dict(exponent=1.0, coef_spread=0.0, noise_scale=1.0, noise_corr=0.0, missing_rate=0.0, shared_meter_rate=0.0), 'R2_sublinear': dict(exponent=0.75, coef_spread=0.0, noise_scale=1.0, noise_corr=0.0, missing_rate=0.0, shared_meter_rate=0.0), 'R3_heterogeneous': dict(exponent=1.0, coef_spread=0.4, noise_scale=1.0, noise_corr=0.0, missing_rate=0.0, shared_meter_rate=0.0), 'R4_correlated_noise': dict(exponent=1.0, coef_spread=0.0, noise_scale=2.0, noise_corr=0.8, missing_rate=0.0, shared_meter_rate=0.0), 'R5_stress': dict(exponent=0.75, coef_spread=0.4, noise_scale=2.0, noise_corr=0.8, missing_rate=0.08, shared_meter_rate=0.05)}
PER_PERSON = (35.0, 4.5, 8.0)
FEATURE_NAMES: Tuple[str, ...] = ('avg_elec', 'avg_water', 'avg_gas', 'std_elec', 'std_water', 'std_gas', 'mom_elec_mean', 'mom_water_mean', 'cv_elec', 'cv_water', 'cv_gas', 'trend_elec', 'seasonal_range', 'winter_summer_ratio', 'composite_index')
FEATURE_GROUPS: Dict[str, Tuple[int, ...]] = {'Basic means': (0, 1, 2), 'Dispersion': (3, 4, 5), 'Dynamic': (6, 7), 'Stability': (8, 9, 10), 'Temporal': (11, 12, 13), 'Composite': (14,)}

@dataclass(frozen=True)
class DataConfig:
    """Public class: DataConfig.

    Demo default: 240 households, seed 17. The declared study panel is
    ``DataConfig(n_households=1000, n_months=36, seed=42, regime='R1_linear')``.
    """

    n_households: int = 240
    n_months: int = 36
    start_year: int = 2019
    seed: int = 17
    regime: str = 'R1_linear'
    event_start_month_min: int = 14
    event_followup_months: int = 6

@dataclass(frozen=True)
class TaskAConfig:
    """Public class: TaskAConfig.

    ``seed`` drives the 70/30 stratified household split, the K-means initialisation and
    the bootstrap refits, so it changes every Task A value. The declared study split uses
    ``seed=42``; the demo default (17) gives a different split and is not the study result.
    """

    k_candidates: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8)
    k_selected: Optional[int] = None
    test_size: float = 0.3
    n_init: int = 10
    n_bootstrap: int = 50
    seed: int = 17

@dataclass(frozen=True)
class TaskBConfig:
    """Public class: TaskBConfig.

    ``n_bootstrap`` sizes the household-clustered MAE intervals. The study used 2,000
    replicates; the demo default (50) gives the same point estimates with noisier
    interval endpoints. ``seed`` drives the 80/20 grouped household split and the inner
    validation split used to select omega; the study uses ``seed=42``.
    """

    window_months: int = 12
    window_stride: int = 1
    test_size: float = 0.2
    n_cv_splits: int = 5
    omega_grid: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    mlp_hidden: Tuple[int, ...] = (64, 32)
    n_bootstrap: int = 50
    seed: int = 17

@dataclass(frozen=True)
class TaskCConfig:
    """Public class: TaskCConfig.

    ``tolerance_months`` is the declared operating point at which every detector's
    threshold is selected on the development households and at which the main comparison
    table is evaluated (six months in the study). ``tolerance_sensitivity_months`` only
    re-scores the fixed alarm vectors at other delays. MU-CARF's own threshold is selected
    separately at the one-month endpoint; see ``mucarf_experiment.run_mucarf_experiment``.

    ``seed`` drives the 40/60 development/test household split, so it must be given
    explicitly when reproducing a specific run (the study uses 42).
    """

    window: int = 3
    tolerance_months: int = 6
    tolerance_sensitivity_months: Tuple[int, ...] = (1, 3, 6)
    min_delta_persons: float = 0.5
    theta_grid: Tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)
    elec_theta_grid: Tuple[float, ...] = (0.1, 0.15, 0.2, 0.25, 0.3)
    cal_tau_grid: Tuple[float, ...] = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0)
    persistence: int = 2
    monitoring_start_month: int = 12
    dev_fraction: float = 0.4
    retrospective_window: int = 6
    retrospective_threshold: float = 0.08
    n_bootstrap: int = 30
    n_random_baseline: int = 30
    seed: int = 17

    def __post_init__(self) -> None:
        """Public function: __post_init__."""
        if self.window < 1:
            raise ValueError('Task C window must be at least one month')
        if self.persistence < 1:
            raise ValueError('Task C persistence must be at least one')
        if self.tolerance_months < 0:
            raise ValueError('Task C tolerance_months must be non-negative')
        if not 0.0 < self.dev_fraction < 1.0:
            raise ValueError('Task C dev_fraction must lie strictly inside (0, 1)')
        if self.n_bootstrap < 1:
            raise ValueError('Task C n_bootstrap must be at least one')
        if self.n_random_baseline < 1:
            raise ValueError('Task C n_random_baseline must be at least one')
        if not self.theta_grid or any((t <= 0 for t in self.theta_grid)):
            raise ValueError('Task C theta_grid must be non-empty and strictly positive')
        if not self.elec_theta_grid or any((t <= 0 for t in self.elec_theta_grid)):
            raise ValueError('Task C elec_theta_grid must be non-empty and strictly positive')
        if not self.cal_tau_grid or any((t <= 0 for t in self.cal_tau_grid)):
            raise ValueError('Task C cal_tau_grid must be non-empty and strictly positive')
        if self.retrospective_window < 1:
            raise ValueError('Task C retrospective_window must be at least one month')
        if not 0.0 <= self.retrospective_threshold < 1.0:
            raise ValueError('Task C retrospective_threshold must lie in [0, 1)')
        if self.monitoring_start_month < 0:
            raise ValueError('Task C monitoring_start_month must be non-negative')
        if not self.tolerance_sensitivity_months:
            raise ValueError('Task C tolerance_sensitivity_months cannot be empty')
        if any((t < 0 for t in self.tolerance_sensitivity_months)):
            raise ValueError('Task C tolerance_sensitivity_months must be non-negative')
        if self.min_delta_persons < 0:
            raise ValueError('Task C min_delta_persons must be non-negative')

@dataclass(frozen=True)
class RobustnessConfig:
    """Public class: RobustnessConfig.

    Demo default: two seeds. The study's five-regime component sweep uses
    ``seeds=(42, 43, 44, 45, 46)``; the internal-validation cells use (2001--2005).
    """

    seeds: Tuple[int, ...] = (17, 18)
    regimes: Tuple[str, ...] = tuple(GENERATOR_REGIMES)
    n_households: int = 240

@dataclass(frozen=True)
class CascadeReplicationConfig:
    """Public class: CascadeReplicationConfig.

    Demo default: two seeds. The complete-cascade replication in the study uses
    ``seeds=(42, 43, 44, 45, 46)`` with 1,000 R1 households per seed.
    """

    seeds: Tuple[int, ...] = (17, 18)
    n_households: int = 240
    regime: str = 'R1_linear'

@dataclass(frozen=True)
class CalendarStressConfig:
    """Public class: CalendarStressConfig.

    Demo default: two seeds. The calendar-adjusted regime sweep in the study re-fits the
    complete 1,000-household cascade at ``regime_seed=42`` in each of the five regimes.
    """

    seeds: Tuple[int, ...] = (17, 18)
    regimes: Tuple[str, ...] = tuple(GENERATOR_REGIMES)
    regime_seed: int = 17
    n_households: int = 240
    n_bootstrap: int = 20
    n_random_baseline: int = 20

@dataclass(frozen=True)
class PipelineConfig:
    """Public class: PipelineConfig."""
    data: DataConfig = field(default_factory=DataConfig)
    task_a: TaskAConfig = field(default_factory=TaskAConfig)
    task_b: TaskBConfig = field(default_factory=TaskBConfig)
    task_c: TaskCConfig = field(default_factory=TaskCConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    cascade_replication: CascadeReplicationConfig = field(default_factory=CascadeReplicationConfig)
    calendar_stress: CalendarStressConfig = field(default_factory=CalendarStressConfig)

    def __post_init__(self) -> None:
        earliest_fair_event = self.task_b.window_months - 1 + self.task_c.window
        if self.data.event_start_month_min < earliest_fair_event:
            raise ValueError('event_start_month_min must be at least %d for the Task B warm-up and Task C baseline window' % earliest_fair_event)
        if self.data.event_start_month_min < self.task_c.monitoring_start_month:
            raise ValueError('events cannot begin before Task C monitoring')
        if self.data.event_followup_months < self.task_c.tolerance_months:
            raise ValueError('event follow-up must cover the Task C tolerance window')
        if not self.task_c.tolerance_sensitivity_months:
            raise ValueError('Task C tolerance sensitivity grid cannot be empty')
        if min(self.task_c.tolerance_sensitivity_months) < 0:
            raise ValueError('Task C tolerance sensitivity values must be non-negative')
        if self.data.event_followup_months < max(self.task_c.tolerance_sensitivity_months):
            raise ValueError('event follow-up must cover the Task C sensitivity grid')
