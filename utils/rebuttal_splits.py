"""Target-row boundaries shared by the ETT loaders and experiment planning."""
import csv
from datetime import datetime, timedelta

PAPER = "paper_online_split"
CONVENTIONAL = "conventional_ett_split"


def ett_bounds(minute, split_mode):
    if split_mode == PAPER:
        return 2880, 3600, 14400
    if split_mode != CONVENTIONAL:
        raise ValueError(f"unknown split mode: {split_mode}")
    factor = 4 if minute else 1
    return 8640 * factor, 11520 * factor, 14400 * factor


def verify_ett_calendar(path, dataset):
    if dataset not in {"ETTh2", "ETTm1"}:
        raise ValueError("C6 conventional split supports ETTh2 and ETTm1")
    minute = dataset == "ETTm1"
    bounds = ett_bounds(minute, CONVENTIONAL)
    cadence = timedelta(minutes=15 if minute else 60)
    previous = None
    edges = {}
    requested = {0, *(b-1 for b in bounds), *bounds[:-1]}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader):
            stamp = datetime.fromisoformat(row["date"])
            if previous is not None and stamp-previous != cadence:
                raise ValueError(f"{dataset}: unexpected cadence at data row {index}")
            if index in requested:
                edges[str(index)] = stamp.isoformat(sep=" ")
            if index == bounds[-1]-1:
                return edges
            previous = stamp
    raise ValueError(f"{dataset}: CSV is shorter than conventional test end")
