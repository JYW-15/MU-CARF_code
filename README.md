# MU-CARF reference code

This repository contains the compact reference implementation accompanying the Scientific Reports manuscript **MU-CARF: an auditable relative-population filter for multi-utility household monitoring**. It exposes the core synthetic-data workflow and the main algorithms needed to inspect the paper: building-type screening (Task A), household population estimation (Task B), past-only change detection (Task C), and the calendar-adjusted relative-state MU-CARF filter.

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

The code uses synthetic electricity, water, and gas readings. No network connection, GPU, or external data file is required.

## Run the smoke example

From this directory:

```bash
python run_demo.py
```

The command generates a small panel, fits the screening and regression components, fits MU-CARF on training households, applies the past-only filter to one held-out series, prints a summary, and writes `outputs/demo_summary.json`. The smoke run is intended to complete in seconds; it is not the manuscript-scale confirmation run.

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

Core functions receive pandas DataFrames or NumPy arrays. The generator returns a `Panel` with:

- `meta`: one row per household and its synthetic type;
- `monthly`: one row per household-month with electricity, water, gas, and true population;
- `population`: the latent population matrix;
- `events`: realised signed population transitions.

Analysis functions return dataclasses or arrays. MU-CARF returns signed alarms (`-1`, `0`, `+1`), posterior state probabilities, state means, and abstention flags. See function signatures and type names in `src/` for the exact schema.

## Reproducibility and scope

Set `DataConfig.seed` and `DataConfig.regime` to obtain deterministic synthetic panels. The generator supports linear, sublinear, heterogeneous, correlated-noise, and combined-stress regimes. Results are conditional on this declared synthetic generator and should not be interpreted as field accuracy.

## License

Released under the MIT License; see `LICENSE`.
