# Synthetic input schema

`src.data_io.load_csv_panel()` reads the four CSV files released with the synthetic dataset and validates the declared R1 snapshot before returning the common `Panel` interface. The dataset is available from [Zenodo](https://doi.org/10.5281/zenodo.22840493).

## Required files

- `monthly_consumption.csv`: 36,000 rows and the fields `household_id`, zero-based `month` (0-35), `date` (`YYYY-MM`), `year`, one-based `month_of_year`, `electricity_kwh`, `water_ton`, `gas_m3`, `true_population`, and `true_type`.
- `household_metadata.csv`: 1,000 rows and the fields `household_id`, `true_type`, `base_population`, `mean_population`, `n_events`, `shared_meter`, `shared_with_household_id`, and `shared_load_fraction`.
- `population_events.csv`: 1,516 rows and the fields `household_id`, `true_type`, `start_month`, signed `delta`, `direction`, and `event_kind` (`onset` or `reversion`).
- `population_trajectory.csv`: 1,000 rows with `household_id` followed by `month_0` ... `month_35`.

The released snapshot uses `residential`, `vacant`, `commercial`, and `mixed` as `true_type` values. `mixed` is the code value for the mixed-use composition category. In this R1 snapshot all meter reads are present and all shared-meter fields indicate no sharing. Population labels and event rows are required for supervised evaluation; the online detector itself only consumes past utility readings after fitting.
