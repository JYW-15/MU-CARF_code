"""Small end-to-end demonstration for the public release."""
from pathlib import Path
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import DataConfig, PipelineConfig, TaskAConfig, TaskBConfig
from src.data_generation import generate_panel
from src.features import household_features, window_features
from src.task_a_screening import run_task_a
from src.task_b_regression import run_task_b
from src.mucarf import detect_mucarf, fit_mucarf


def main() -> None:
    cfg = PipelineConfig(
        data=DataConfig(n_households=120, n_months=24, seed=17),
        task_a=TaskAConfig(n_bootstrap=5, seed=17),
        task_b=TaskBConfig(n_bootstrap=20, n_cv_splits=3, seed=17),
    )
    panel = generate_panel(cfg.data)
    calibration = panel.monthly[panel.monthly["month"] < 12]
    screen = run_task_a(household_features(calibration), panel.meta, cfg.task_a)
    windows = window_features(panel.monthly, cfg.task_b)
    regression = run_task_b(windows, cfg=cfg.task_b)
    train_ids = screen.is_test[~screen.is_test].index.to_numpy(dtype=int)
    occupied = panel.meta.set_index("household_id").loc[train_ids]
    train_ids = occupied.index[occupied["true_type"].isin({"residential", "mixed"})].to_numpy(dtype=int)
    model = fit_mucarf(panel, train_ids, baseline_end=12, state_radius=2)
    test_id = int(train_ids[0])
    rows = panel.monthly[panel.monthly["household_id"] == test_id].sort_values("month")
    detected = detect_mucarf(rows["electricity_kwh"].to_numpy(), rows["water_ton"].to_numpy(), rows["gas_m3"].to_numpy(), model, tau=0.75)
    out = {
        "households": int(cfg.data.n_households),
        "records": int(len(panel.monthly)),
        "screened_test_accuracy": float(screen.accuracy_test),
        "population_test_windows": int(regression.n_test),
        "mucarf_alarm_count": int(np.count_nonzero(detected.alarms)),
    }
    out_path = ROOT / "outputs" / "demo_summary.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()


