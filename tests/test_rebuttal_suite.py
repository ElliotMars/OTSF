import csv
from pathlib import Path
import shlex
import subprocess

import pytest

from scripts import c1_lr_sweep as c1, rebuttal_collect as collect, rebuttal_suite as suite
from utils.rebuttal_splits import CONVENTIONAL, PAPER, ett_bounds


@pytest.fixture
def registry():
    return c1.read_json(suite.CONFIG)


def plan(experiment, registry, *tokens):
    args = suite.parse(["plan", "--experiment", experiment, *tokens])
    return args, suite.manifest(experiment, args, registry)


def test_ofat_and_shared_pretraining(registry):
    _, m = plan("C2", registry, "--settings", "ETTm1:48")
    base = next(g["config"] for g in m["groups"].values() if g["variant"] == "full")
    for g in m["groups"].values():
        changed = {k for k, v in g["config"].items() if v != base[k]}
        expected = {"stable_buffer_size", "recovery_buffer_size"} if g["parameter"] == "memory_capacity" else {g["parameter"]}
        if g["parameter"] == "reference":
            assert not changed
        else:
            assert changed and changed <= expected
    assert len(suite.planned(m, "prepare")) == 3
    _, c3 = plan("C3", registry, "--settings", "ETTm1:48")
    _, c4 = plan("C4", registry, "--settings", "ETTm1:48")
    assert c3 == c4
    assert len(suite.eligible_tasks(c4, "C4", "test")) == 3
    assert {t["checkpoint_key"] for t in suite.planned(m, "prepare")} == {
        t["checkpoint_key"] for t in suite.planned(c3, "prepare")
    }


@pytest.mark.parametrize("dataset,minute", [("ETTh2", False), ("ETTm1", True)])
def test_conventional_calendar_and_checkpoint_separation(registry, dataset, minute):
    tokens = ["--methods", "multi_expert", "--settings", f"{dataset}:48"]
    _, old = plan("C1", registry, *tokens)
    _, new = plan("C6", registry, *tokens)
    g = next(iter(new["groups"].values()))
    train, val, test = ett_bounds(minute, CONVENTIONAL)
    assert g["bounds"] == [0, train, val, test]
    assert g["scaler_fit"] == [0, train]
    assert g["origins"]["test"] == test-val-47
    assert str(val) in g["timestamps"] and str(test-1) in g["timestamps"]
    assert next(iter(old["groups"].values()))["bounds"][1:] == list(ett_bounds(minute, PAPER))
    assert suite.planned(old, "prepare")[0]["checkpoint_key"] != suite.planned(new, "prepare")[0]["checkpoint_key"]


def test_exported_commands_and_smoke(registry, tmp_path):
    args, m = plan("C7", registry, "--scope", "smoke", "--export-root", str(tmp_path))
    assert m["max_online_steps"] > 24 and m["test_seeds"] == [0]
    suite.export(args, [("C7", m)])
    with (tmp_path / "smoke/GPU_JOB_MANIFEST.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == sum(len(suite.planned(m, phase)) for phase in suite.phases(m))
    for row in rows:
        tokens = shlex.split(row["command"])
        parsed = suite.parse([*tokens[2:], "--dry-run"])
        assert parsed.execute and parsed.dry_run and parsed.job == row["task_id"]
        assert row["completion_path"].endswith(row["task_id"] + ".json")
    for path in tmp_path.rglob("*.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_calibration_uses_latency_and_no_test_labels(registry, monkeypatch, tmp_path):
    _, m = plan("C5", registry, "--time-match")
    hardware = {"physical_gpu": "2", "host": "test", "device": "test"}
    times = {None: 100, 0: 50, 1: 95, 2: 150, 3: 200}
    monkeypatch.setattr(c1, "bind_checkpoint", lambda root, manifest, group, task: task)

    def complete(root, task):
        assert task["split"] == "val"
        budget = task["calibration_budget"]
        return {"hardware": hardware, "mean_origin_ms": times[budget], "mse": 100 if budget == 1 else 0}

    monkeypatch.setattr(c1, "read_completed", complete)
    result = suite.freeze_budget(tmp_path, m)
    assert result["source_split"] == "val"
    assert result["selections"]["ECL:48"]["index"] == 1
    c1.write_json(tmp_path / "frozen_time_budget.json", result)
    jobs = [suite.bind_task(tmp_path, m, "test", task) for task in suite.planned(m, "test")]
    assert all(job["required_hardware"] == hardware for job in jobs)


def test_failed_or_smoke_seed_never_contributes_to_complete_group():
    rows = [dict(phase="test", dataset="WTH", horizon=48, method="dsof", protocol_variant="dsof",
                 variant="default", seed=s, status="completed", smoke=False) for s in range(3)]
    assert len(collect.completed_groups(rows)) == 1
    assert not collect.completed_groups(rows[:-1])
    assert not collect.completed_groups([*rows[:-1], rows[2] | {"status": "failed"}])
    assert not collect.completed_groups([*rows[:-1], rows[2] | {"smoke": True}])


def test_proxy_metrics_counts_ties_and_excludes_threshold_equality():
    result = collect.proxy_metrics([(3, 1), (2, 1), (2, 0), (1, 0)], 2)
    assert result["roc_auc"] == .875
    assert result["average_precision"] == pytest.approx(5/6)
    assert result["precision"] == 1 and result["recall"] == .5


def test_cache_reuses_full_control_only_for_identical_verified_request(registry, monkeypatch, tmp_path):
    _, c2 = plan("C2", registry, "--settings", "ETTm1:48")
    _, c5 = plan("C5", registry, "--settings", "ETTm1:48")
    tasks = [next(t for t in suite.planned(m, "test") if t["variant"] == "full") for m in (c2, c5)]
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint")
    optimizer = checkpoint.with_name("optimizer.pth")
    optimizer.write_bytes(b"optimizer")
    files = {str(p): c1.file_hash(p) for p in (checkpoint, optimizer)}
    tasks = [t | {"checkpoint_path": str(checkpoint), "checkpoint_files": files,
                 "selection_hash": t["manifest_hash"]} for t in tasks]
    calls = []

    def run(root, task, gpu, checkpoint_root):
        calls.append(task["task_id"])
        result = task | {"request_hash": c1.digest(task), "status": "completed", "runtime": 1.,
                         "mse": .2, "mae": .3, "origins": task["expected_origins"],
                         "observed_online_lrs": {key: {} for key in task["all_effective_online_lrs"]}}
        c1.write_json(c1.completion_path(root, task), result)
        return True

    monkeypatch.setattr(c1, "run_one", run)
    cache = tmp_path / "cache"
    for experiment, task in zip(("C2", "C5"), tasks):
        assert suite.run_cached(tmp_path / experiment, cache, task, "0", True)
    assert len(calls) == 1
    completion = c1.read_completed(tmp_path / "C5", tasks[1])
    assert completion["reused_from"] and completion["reuse_reason"]
    for path in (c1.completion_path(tmp_path / "C5", tasks[1]), Path(completion["reused_from"])):
        path.write_text("{}")
    assert suite.run_cached(tmp_path / "C5", cache, tasks[1], "0", True)
    assert len(calls) == 2


def test_resume_rebuilds_corrupt_frozen_selection(registry, tmp_path, monkeypatch):
    args, m = plan("C7", registry, "--output-root", str(tmp_path))
    args.resume = True
    root = suite.root_for(args, m)
    root.mkdir(parents=True)
    path = c1.selection_path(root, m)
    path.write_text("[]")
    selection = {"source_split": "val", "manifest_hash": m["manifest_hash"]}
    monkeypatch.setattr(c1, "make_selection", lambda root, m: selection)
    assert suite.execute(root, m, "select", args, [])
    assert c1.read_json(path) == selection
