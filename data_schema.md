# Synthetic input schema

The public release generates its own input panel. A replacement panel should provide the same logical fields:

- `meta`: `household_id`, `true_type`, `base_population`, `mean_population`, `n_events`.
- `monthly`: `household_id`, `month`, `year`, `month_of_year`, `electricity_kwh`, `water_ton`, `gas_m3`, `true_population`, `true_type`.
- `events`: `household_id`, `start_month`, signed `delta`, `direction`, and `event_kind`.

Meter columns may contain `NaN`; online preprocessing uses only past observations for forward filling. Population labels and event rows are required for supervised evaluation, but are not used by the detector after training.
