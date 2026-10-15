# MU-CARF reference code

This repository contains the compact reference implementation accompanying the manuscript **MU-CARF: an auditable relative-population filter for multi-utility household monitoring**. It exposes the core synthetic-data workflow and the main algorithms needed to inspect the paper: building-type screening (Task A), household population estimation (Task B), past-only change detection (Task C), and the calendar-adjusted relative-state MU-CARF filter.

The release is intentionally small. It contains algorithmic modules, a synthetic generator, a runnable smoke example, an example configuration, and no manuscript figures, logs, cached bytecode, private records, exploratory probes, or precomputed result tables.

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

The code can either generate a small synthetic panel for a smoke test or load the released CSV snapshot described by `Data README.md`. The snapshot is synthetic only and contains no real meter or person records.

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

The loader expects exactly the files and schema documented in `Data README.md`: 1,000 households, 36 zero-based months from 2019-01 through 2021-12, 36,000 monthly records, 1,516 event rows, and 1,000 population trajectories. It validates dates, field names, population labels, event signs, missing reads, and consistency between the monthly and wide trajectory tables before analysis.

## Module map

- `src/data_generation.py`: reproducible synthetic multi-utility panel and event generator.
- `src/features.py`: shared 15-feature household and rolling-window descriptors.
- `src/task_a_screening.py`: training-only scaling, K-means screening, and label calibration.
- `src/task_b_regression.py`: grouped household splits and the one-parameter population blend.
- `src/task_c_detection.py`: causal detectors and event-level matching utilities.
- `src/mucarf.py`: calendar profile, response direction, shrinkage covariance, banded state transition, and exact forward filter.
- `src/mucarf_experiment.py`: leakage-controlled MU-CARF evaluation helpers.
- `src/cascade.py`: end-to-end trajectory and screening integration.

The public API is designed for inspection and adaptation. The default example settings are rounded, low-cost demonstration values. The paper's exact confirmation protocol and generated result files are intentionally not distributed in this compact release.

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
