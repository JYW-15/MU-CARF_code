"""Load and validate the released synthetic CSV panel."""
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from .data_generation import Panel

__all__ = ["load_csv_panel", "validate_csv_panel"]

_MONTHLY_COLUMNS = (
    "household_id", "month", "date", "year", "month_of_year",
    "electricity_kwh", "water_ton", "gas_m3", "true_population", "true_type",
)
_META_COLUMNS = (
    "household_id", "true_type", "base_population", "mean_population", "n_events",
    "shared_meter", "shared_with_household_id", "shared_load_fraction",
)
_EVENT_COLUMNS = (
    "household_id", "true_type", "start_month", "delta", "direction", "event_kind",
)


def _require_columns(frame: pd.DataFrame, expected: tuple[str, ...], name: str) -> None:
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def validate_csv_panel(monthly: pd.DataFrame, metadata: pd.DataFrame, events: pd.DataFrame, trajectory: pd.DataFrame) -> None:
    """Validate the four CSV tables against Data README.md."""
    _require_columns(monthly, _MONTHLY_COLUMNS, "monthly_consumption.csv")
    _require_columns(metadata, _META_COLUMNS, "household_metadata.csv")
    _require_columns(events, _EVENT_COLUMNS, "population_events.csv")
    _require_columns(trajectory, tuple(["household_id"] + [f"month_{i}" for i in range(36)]), "population_trajectory.csv")
    month_columns = [f"month_{i}" for i in range(36)]
    if set(metadata["household_id"].astype(int)) != set(range(1000)):
        raise ValueError("household_metadata.csv must contain household_id values 0-999")
    if len(monthly) != 36_000 or monthly["household_id"].nunique() != 1000:
        raise ValueError("monthly_consumption.csv must contain 36,000 rows for 1,000 households")
    if len(events) != 1_516 or len(trajectory) != 1_000:
        raise ValueError("the declared snapshot must contain 1,516 events and 1,000 trajectories")
    if set(monthly["month"].astype(int)) != set(range(36)):
        raise ValueError("monthly month must be zero-based 0-35")
    if set(monthly["month_of_year"].astype(int)) != set(range(1, 13)):
        raise ValueError("monthly month_of_year must be one-based 1-12")
    expected_dates = pd.to_datetime(monthly["year"].astype(str) + "-" + monthly["month_of_year"].astype(str).str.zfill(2), format="%Y-%m").dt.strftime("%Y-%m")
    if not expected_dates.equals(monthly["date"].astype(str)):
        raise ValueError("monthly date does not match year and month_of_year")
    allowed_types = {"residential", "vacant", "commercial", "mixed"}
    if not set(monthly["true_type"]) <= allowed_types or not set(metadata["true_type"]) <= allowed_types:
        raise ValueError("true_type contains an undocumented building type")
    if not set(events["event_kind"]) <= {"onset", "reversion"}:
        raise ValueError("event_kind must be onset or reversion")
    if not np.all(events["direction"].astype(int).to_numpy() == np.sign(events["delta"].astype(int))):
        raise ValueError("event direction must equal the sign of delta")
    if not events["start_month"].between(14, 29).all():
        raise ValueError("event start_month must leave at least six months of follow-up")
    if metadata["shared_meter"].astype(bool).any() or not np.allclose(metadata["shared_load_fraction"], 0.0):
        raise ValueError("the declared R1 snapshot must not contain shared meters")
    monthly_sorted = monthly.sort_values(["household_id", "month"]).reset_index(drop=True)
    if monthly_sorted[["electricity_kwh", "water_ton", "gas_m3"]].isna().any().any():
        raise ValueError("the declared R1 snapshot must not contain missing meter reads")
    trajectory_sorted = trajectory.sort_values("household_id").reset_index(drop=True)
    if not np.array_equal(monthly_sorted["true_population"].to_numpy(dtype=int), trajectory_sorted[month_columns].to_numpy(dtype=int).reshape(-1)):
        raise ValueError("population_trajectory.csv does not match monthly true_population")
    event_counts = events.groupby("household_id").size().reindex(metadata["household_id"], fill_value=0).to_numpy()
    if not np.array_equal(event_counts, metadata["n_events"].to_numpy(dtype=int)):
        raise ValueError("metadata n_events does not match population_events.csv")


def load_csv_panel(data_dir: Union[str, Path]) -> Panel:
    """Load the four files documented in Data README.md into the public Panel type."""
    root = Path(data_dir)
    monthly = pd.read_csv(root / "monthly_consumption.csv")
    metadata = pd.read_csv(root / "household_metadata.csv")
    events = pd.read_csv(root / "population_events.csv")
    trajectory = pd.read_csv(root / "population_trajectory.csv")
    validate_csv_panel(monthly, metadata, events, trajectory)
    month_columns = [f"month_{i}" for i in range(36)]
    return Panel(
        meta=metadata.sort_values("household_id").reset_index(drop=True),
        monthly=monthly.sort_values(["household_id", "month"]).reset_index(drop=True),
        population=trajectory.sort_values("household_id")[month_columns].to_numpy(dtype=int),
        events=events.sort_values(["household_id", "start_month"]).reset_index(drop=True),
        regime="R1_linear",
        seed=42,
    )
