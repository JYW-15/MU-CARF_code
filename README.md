# MU-CARF reference code

This repository contains the reference implementation accompanying the manuscript **MU-CARF: an auditable relative-population filter for multi-utility household monitoring**. It exposes the core synthetic-data workflow and the main algorithms needed to inspect the paper: building-type screening (Task A), household population estimation (Task B), past-only change detection (Task C), and the calendar-adjusted relative-state MU-CARF filter.

The repository provides the code modules, synthetic-data generator, example configuration, and runnable workflows needed to inspect and reproduce the analyses described in the manuscript.

## Requirements

- Python 3.9 or newer
- NumPy, pandas, SciPy, scikit-learn

Install the dependencies in a clean environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

The code can either generate a small synthetic panel for a smoke test or load the released synthetic panel deposited in the Zenodo repository at https://doi.org/10.5281/zenodo.22840493. 

The dataset provides monthly electricity, water, and gas readings for 1,000 premises over 36 months, together with household population trajectories, metadata, and realised population change events. It is the data source associated with the manuscript and is released under the licence stated in the repository.

## Run the smoke example

From this directory:

```bash
python run_demo.py
```

The command generates a 120-household, 24-month panel, fits the screening and regression components, fits MU-CARF on training households, applies the past-only filter to one held-out series, prints a summary, and writes `outputs/demo_summary.json`. This is a smoke test, not the manuscript-scale confirmation run.

To use the released data snapshot, place the four CSV files in one directory and run:

```bash
python run_demo.py --data-dir path/to/data
```

The loader expects the four CSV files supplied with the Zenodo dataset: 1,000 households, 36 zero-based months from 2019-01 through 2021-12, 36,000 monthly records, 1,516 event rows, and 1,000 population trajectories. It validates dates, field names, population labels, event signs, missing reads, and consistency between the monthly and wide trajectory tables before analysis.

## Reproducing the published values

The manuscript states that all reported values were produced by the same declared sequence of generation, model fitting, development-set threshold selection, and event scoring, and can be reproduced from the released code. This repository is that code. The settings that reproduce it are the ones below, not the demonstration defaults that `src/config.py` carries so that the smoke example finishes in a couple of minutes.

```python
from src.config import DataConfig, TaskAConfig, TaskBConfig, TaskCConfig
from src.data_generation import generate_panel

data = DataConfig(n_households=1000, n_months=36, seed=42, regime='R1_linear')
task_a = TaskAConfig(seed=42)                                    # split, K-means, 50 ARI refits
task_b = TaskBConfig(seed=42, n_bootstrap=2000)                  # household-clustered MAE intervals
task_c = TaskCConfig(seed=42, tolerance_months=6, n_bootstrap=500, n_random_baseline=200)
```

| Quantity | Declared study value | Where it is set |
| --- | --- | --- |
| Panel | 1,000 households, 36 months, seed 42, R1 | `DataConfig` |
| Task A split and refits | `seed=42`, 50 bootstrap refits | `TaskAConfig` |
| Task B | `seed=42`, 2,000 household-clustered bootstrap replicates | `TaskBConfig` |
| Task C | `seed=42`, 40/60 development/test split, 500 household bootstraps, 200 support-matched randomizations | `TaskCConfig` |
| Regime sweep, cascade replication | seeds 42--46, 1,000 households per seed | `RobustnessConfig`, `CascadeReplicationConfig` |
| Internal validation | seeds 2001--2005 in R1--R5 | `RobustnessConfig` |
| Final confirmation | seeds 3001--3010 in R1 | `RobustnessConfig` |

The same values are recorded as data in `examples/config_template.json`, together with the fixed CUSUM parameters and the seed sets.

Seeds are not cosmetic. `TaskAConfig.seed` drives the 70/30 household split, the K-means initialisation and the bootstrap refits: on the same seed-42 panel, `seed=17` returns macro-F1 0.9492 and a 214/2/2 hold-out screen instead of the published 0.9517 and 213/2/3. `TaskCConfig.seed` fixes which households are development and which are test, so a different seed also changes every Task C value.

With the configuration above, the released modules regenerate the single-seed values quoted in the text and in Supplementary Tables S1--S13 -- for example Task A macro-F1 0.9517 with a 213/2/3 hold-out screen, CABL event F1 0.599 at six months and 0.550 with 0.350 false alarms per premise-year at one month, and 0.604 for the past-only estimate-shift detector at six months. Multi-seed cells are obtained by looping the declared seed sets over these same calls; there is no separate driver script, and each cell re-estimates its training-side parameters and re-selects thresholds on its own development households.

The pre-computed result tables are not bundled in this release. They are outputs of the code path above, not inputs to it, and no reported number requires a module that is absent from this repository.

## Threshold rules and fixed-parameter comparators

Every detector's operating point is the tolerance window at which a same-signed alarm scores an event, and the code keeps that attribution explicit rather than assuming a single window for all rows.

- The baseline detectors in `src/task_c_detection.py` (CABL, past-only ratio, multi-signal, estimate shift) select their threshold on the development households at `TaskCConfig.tolerance_months`, which is six months in the study, and are scored at that tolerance.
- MU-CARF selects its posterior-jump threshold at the one-month endpoint, the primary endpoint of the study, and the MU-CARF, diagonal-covariance, soft-gate and covariance-aware rows are scored at one month.
- `run_mucarf_experiment().summary` therefore carries a `tolerance_months` column: 6 on baseline rows and 1 on MU-CARF-family rows. Compare rows only within one tolerance value, or use `tolerance_sensitivity`, which re-scores the fixed alarm vectors at one, three and six months.
- CUSUM is the one comparator whose threshold is not searched on the development households. `detect_cusum_causal` uses fixed `warmup=6`, `k=0.5`, `h=4.0`, declared in its docstring and here, because that comparator is reported as a fixed configuration rather than as a tuned operating point; its alarm vector is not part of the threshold search.

## Non-anticipation checks

```bash
python run_checks.py
```

runs the audit table's past-only claims as executable checks on three screened-in test households of a 120-household seed-42 panel, for MU-CARF, the calendar-adjusted baseline-locked comparator and the past-only ratio detector, and exits non-zero if any check fails. `src/checks.py` supplies the four checks so that they can also be applied to a new detector through the same callable interface:

- `prefix_stability_check`: truncating the series at month `p` must not change any alarm at or before `p`, so no alarm is revised once later readings arrive.
- `missing_read_check`: masking an interior block of reads must not change any alarm before the block, so a gap is not filled from later readings.
- `leading_gap_check`: masking the first months must not arm the detector before the first observed month, so no value is carried backwards.
- `baseline_exclusion_check`: with a leading gap present, the locked household baseline must equal the mean of the de-seasonalised log levels over the calibration months that were actually observed, so the gap is excluded from the baseline rather than backfilled.

None of the four proves the absence of every leakage path; together they exercise the routes named in the audit table, and they fail on detectors that do look ahead. The current run reports 30 checks over 3 households with no failures.

## Module map

- `src/data_generation.py`: reproducible synthetic multi-utility panel and event generator.
- `src/features.py`: shared 15-feature household and rolling-window descriptors.
- `src/task_a_screening.py`: training-only scaling, K-means screening, and label calibration.
- `src/task_b_regression.py`: grouped household splits and the one-parameter population blend.
- `src/task_c_detection.py`: causal detectors and event-level matching utilities. `run_task_c` and `split_task_c_households` require `eligible_households`, the households held out by Task A; they raise rather than silently monitoring every occupied household, because the monitorable universe is part of the protocol.
- `src/mucarf.py`: calendar profile, response direction, shrinkage covariance, banded state transition, and exact forward filter.
- `src/mucarf_experiment.py`: leakage-controlled MU-CARF evaluation helpers.
- `src/cascade.py`: end-to-end trajectory and screening integration.
- `src/checks.py`: executable non-anticipation checks for any detector exposing `detect(elec, water, gas)`.

The public API is designed for inspection and adaptation. The default settings are rounded, low-cost demonstration values: 240 households, seed 17, and 30--50 resamples, so that a full run takes minutes rather than hours. They are not the study configuration; `Reproducing the published values` lists what to pass instead. With that configuration the modules reproduce the reported values, and the pre-computed result tables are regenerated rather than shipped as files.

## Inputs and outputs

Core functions receive pandas DataFrames or NumPy arrays. `src.data_io.load_csv_panel()` maps the documented CSV files to the same `Panel` interface returned by the synthetic generator:

- `meta`: one row per household and its synthetic type;
- `monthly`: one row per household-month with electricity, water, gas, and true population;
- `population`: the latent population matrix;
- `events`: realised signed population transitions.

The monthly CSV additionally contains `date`; metadata contains shared-meter fields; and the wide trajectory contains `household_id` plus `month_0` ... `month_35`. Building type `mixed` is the code value corresponding to the README's “mixed-use” composition label.

Analysis functions return dataclasses or arrays. MU-CARF returns signed alarms (`-1`, `0`, `+1`), posterior state probabilities, state means, and abstention flags. See function signatures and type names in `src/` for the exact schema.

## Reproducibility and scope

Set `DataConfig.seed` and `DataConfig.regime` to obtain deterministic synthetic panels. The generator supports linear, sublinear, heterogeneous, correlated-noise, and combined-stress regimes. Results are conditional on this declared synthetic generator and should not be interpreted as field accuracy.

## License

Released under the MIT License; see `LICENSE`.
