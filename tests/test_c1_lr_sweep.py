import argparse
import itertools
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from scripts import c1_lr_sweep as c1


@pytest.fixture
def profile():
    return c1.read_json(c1.PROFILE)


@pytest.fixture
def manifest(profile):
    groups = {}
    for method in ("dsof", "multi_expert"):
        config, _ = c1.native_settings(method, "ETTh2:24", profile)
        groups[c1.group_id(method, "ETTh2:24")] = {
            "method": method, "dataset": "ETTh2", "horizon": 24, "config": config,
            "feedback_protocol": c1.feedback_protocol(method, config),
            "origins": {"val": 697, "test": 10777},
        }
    return dict(groups=groups, datasets={"ETTh2": {"path": "data/ETTh2.csv", "rows": 17420}},
                source={"git_commit": "fixture", "source_sha256": "fixture"},
                manifest_hash="manifest", validation_seeds=[0, 1], scales=list(c1.DEFAULT_SCALES),
                smoke=False, max_online_steps=-1)


def fake_result(task, **values):
    result = {k: v for k, v in task.items() if k not in ("config", "data", "source")}
    if task["stage"] != "prepare":
        result.update(mse=0.5, mae=0.25, origins=task["expected_origins"],
                      observed_online_lrs={k: {} for k in task["all_effective_online_lrs"]})
    return result | dict(status="completed", runtime=0.1, request_hash=c1.digest(task)) | values


def prepare_fixture(root, manifest):
    for task in c1.prepare_tasks(manifest):
        folder = root / "fixture" / task["checkpoint_key"]
        folder.mkdir(parents=True)
        path = folder / "checkpoint.pth"
        path.write_bytes(task["checkpoint_key"].encode())
        files = {str(path): c1.file_hash(path)}
        if task["method"] == "multi_expert":
            opt = folder / "optimizer.pth"
            opt.write_bytes(b"optimizer")
            files[str(opt)] = c1.file_hash(opt)
        c1.write_json(c1.completion_path(root, task), fake_result(
            task, checkpoint_path=str(path), checkpoint_files=files))


def test_all_native_optimizer_mappings(profile):
    expected = {
        "multi_expert": {"online_lr_expert": 1e-4, "online_lr_router": 1e-5},
        "dsof": {"baseline_online_lr": .001, "dsof_student_lr": .001},
        "dyname": {"dyname_online_lr": .001},
        "er": {"baseline_online_lr": .001},
        "fsnet": {"baseline_online_lr": .001},
        "onenet": {"baseline_online_lr": .001},
        "proceed": {"proceed_online_lr": .001},
        "patchtst_dgrad": {"dgrad_online_lr": .001},
    }
    for method, setting in itertools.product(expected, c1.SETTINGS):
        config, log = c1.native_settings(method, setting, profile)
        base = expected[method].copy()
        if method == "multi_expert":
            base["online_lr_expert"] = {"ECL": 1e-5, "WTH": 5e-5}.get(config["data"], 1e-4)
        assert c1.online_rates(method, config, 1) == base
        assert c1.online_rates(method, config, 3) == {k: 3*v for k, v in base.items()}
        assert log.startswith("log/")
        assert config["learning_rate"] == (.003 if config["data"] == "ECL" else .001)
        assert c1.feedback_protocol(method, config) == (
            "progressive" if method == "multi_expert" else "legacy_delayed")


def test_offline_params_and_checkpoint_key_are_invariant_to_online_scale(manifest):
    g = manifest["groups"]["multi_expert__ETTh2__24"]
    base = c1.request(manifest, g, 0, "prepare")
    tuned = c1.request(manifest, g, 0, "sweep", 10)
    assert tuned["checkpoint_key"] == base["checkpoint_key"]
    assert tuned["config_hash"] != base["config_hash"]
    changed = {k for k, v in tuned["config"].items() if base["config"][k] != v}
    assert changed == {"online_lr_expert", "online_lr_router"}
    assert g["config"]["adaptive_controller"] == "dynamic"
    assert c1.request(manifest, g, 1, "prepare")["checkpoint_key"] != base["checkpoint_key"]
    for key, value in (("seq_len", 12), ("learning_rate", .005), ("pred_len", 48),
                       ("expert_composition", "fsnet"), ("data", "ETTm1"), ("delay_fb", False)):
        args = g["config"] | {key: value}
        assert c1.checkpoint_key(g["method"], args, 0, base["data"], base["source"]) != base["checkpoint_key"]


def test_every_method_gets_same_validation_grid_and_split(tmp_path, manifest):
    prepare_fixture(tmp_path, manifest)
    trials = c1.sweep_tasks(tmp_path, manifest)
    assert Counter(t["method"] for t in trials) == {"dsof": 10, "multi_expert": 10}
    assert all(t["split"] == "val" and t["expected_origins"] == 697 for t in trials)
    for method, seed in itertools.product(("dsof", "multi_expert"), (0, 1)):
        same = [t for t in trials if t["method"] == method and t["seed"] == seed]
        assert len({t["checkpoint_path"] for t in same}) == 1
        assert len({t["task_id"] for t in same}) == 5


def test_validation_selection_requires_all_trials_and_uses_seed_mean(tmp_path, manifest):
    prepare_fixture(tmp_path, manifest)
    tasks = c1.sweep_tasks(tmp_path, manifest)
    with pytest.raises(ValueError, match="validation trial"):
        c1.make_selection(tmp_path, manifest)
    for t in tasks:
        # Scale 0.1 has the best individual seed, but worse mean. 1 and 3 tie.
        mse = {0.1: (0.01, 4.0)[t["seed"]], 1.0: 0.7, 3.0: 0.7}.get(t["lr_scale"], 1.0)
        result = fake_result(t, mse=mse)
        c1.write_json(c1.completion_path(tmp_path, t), result)
    pick = c1.make_selection(tmp_path, manifest)
    assert all(v["selected_scale"] == 1 and v["validation_trials"] == 10 for v in pick["selections"].values())
    t = tasks[0]
    c1.write_json(c1.completion_path(tmp_path, t), fake_result(t, split="test", mse=0.0))
    with pytest.raises(ValueError, match="validation trial"):
        c1.make_selection(tmp_path, manifest)


def frozen_fixture(root, manifest):
    prepare_fixture(root, manifest)
    for t in c1.sweep_tasks(root, manifest):
        c1.write_json(c1.completion_path(root, t), fake_result(t, mse=.2 if t["lr_scale"] == 3 else .5))
    pick = c1.make_selection(root, manifest)
    c1.write_json(root / "selected_lrs.json", pick)
    return pick


def test_test_stage_uses_only_frozen_selection_and_fixed_seeds(tmp_path, manifest, monkeypatch):
    frozen_fixture(tmp_path, manifest)
    def forbidden(*args, **kwargs):
        pytest.fail("test accessed validation-selection logic")
    monkeypatch.setattr(c1, "sweep_tasks", forbidden)
    monkeypatch.setattr(c1, "make_selection", forbidden)
    selection = c1.frozen_selection(tmp_path, manifest)
    tasks = c1.test_tasks(tmp_path, manifest, selection)
    assert len(tasks) == 12 and {t["seed"] for t in tasks} == {0, 1, 2}
    assert {t["lr_scale"] for t in tasks if t["variant"] == "tuned"} == {3.0}
    assert {t["lr_scale"] for t in tasks if t["variant"] == "default"} == {1.0}
    assert all(t["split"] == "test" and t["expected_origins"] == 10777 for t in tasks)
    assert len({t["task_id"] for t in tasks}) == len(tasks)


def test_smoke_and_incomplete_trials_do_not_enter_official_report(tmp_path, manifest):
    selection = frozen_fixture(tmp_path, manifest)
    tasks = c1.test_tasks(tmp_path, manifest, selection)
    for t in tasks:
        c1.write_json(c1.completion_path(tmp_path, t), fake_result(t, mse=.4 if t["variant"] == "tuned" else .5))
    assert all(r["change_pct"] == pytest.approx(-20) for r in c1.comparison(tmp_path, manifest, selection))
    c1.write_json(c1.completion_path(tmp_path, tasks[0]), fake_result(tasks[0], status="failed"))
    with pytest.raises(ValueError, match="completed full test"):
        c1.comparison(tmp_path, manifest, selection)
    with pytest.raises(ValueError, match="capped"):
        c1.comparison(tmp_path, manifest | {"smoke": True}, selection)


def test_resume_rejects_invalid_metadata_and_modified_checkpoint(tmp_path, manifest):
    prepare_fixture(tmp_path, manifest)
    task = next(t for t in c1.prepare_tasks(manifest) if t["method"] == "multi_expert")
    original = c1.read_completed(tmp_path, task)
    assert original
    c1.write_json(c1.completion_path(tmp_path, task), original | {"config_hash": "wrong"})
    assert not c1.read_completed(tmp_path, task)
    c1.write_json(c1.completion_path(tmp_path, task), original)
    Path(original["checkpoint_path"]).write_bytes(b"modified")
    assert not c1.read_completed(tmp_path, task)


def test_frozen_selection_detects_edited_scale(tmp_path, manifest):
    selection = frozen_fixture(tmp_path, manifest)
    selection["selections"]["dsof__ETTh2__24"]["selected_scale"] = 10
    c1.write_json(tmp_path / "selected_lrs.json", selection)
    with pytest.raises(ValueError, match="frozen selection"):
        c1.frozen_selection(tmp_path, manifest)


def test_parallel_workers_have_separate_paths_and_publish_only_success(tmp_path, manifest, monkeypatch):
    seen = {}
    tasks = c1.prepare_tasks(manifest)[:3]
    def fake_subprocess(command, cwd, stdout, stderr, env):
        job_path = Path(command[-1])
        job = c1.read_json(job_path)
        t = job["task"]
        target = Path(job["offline_root"]) / "state" / "checkpoint.pth"
        target.parent.mkdir(parents=True)
        target.write_bytes(job_path.name.encode())
        seen[t["seed"]] = job_path, env["CUDA_VISIBLE_DEVICES"], target
        # Even an apparently complete result must not be published for a failed worker.
        c1.write_json(job_path.with_name("result.json"), fake_result(
            t, checkpoint_path=str(target), checkpoint_files={str(target): c1.file_hash(target)}))
        return subprocess.CompletedProcess(command, 7 if t["seed"] == 1 else 0)
    monkeypatch.setattr(c1.subprocess, "run", fake_subprocess)
    args = argparse.Namespace(gpus=("4", "6"), max_per_gpu=1, resume=False)
    assert c1.run_stage(tmp_path, "prepare", tasks, args) == 1
    assert len({v[0] for v in seen.values()}) == len({v[2] for v in seen.values()}) == 3
    assert all(v[1] in ("4", "6") for v in seen.values())
    assert c1.read_completed(tmp_path, tasks[0])
    assert not c1.read_completed(tmp_path, tasks[1])
    assert c1.read_completed(tmp_path, tasks[2])
    assert c1.failure_rows(tmp_path)[0]["exit_code"] == 7


def test_cli_dry_run_creates_no_run_directory(tmp_path, capsys):
    assert c1.main(["prepare", "--methods", "dsof", "--settings", "ETTh2:24",
                    "--output-root", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())
    assert '"validation_trials": 5' in capsys.readouterr().out


def test_smoke_selection_stays_separate_and_test_is_capped(tmp_path, manifest):
    smoke = manifest | {"smoke": True, "max_online_steps": 4}
    prepare_fixture(tmp_path, smoke)
    for task in c1.sweep_tasks(tmp_path, smoke):
        c1.write_json(c1.completion_path(tmp_path, task), fake_result(task))
    pick = c1.make_selection(tmp_path, smoke)
    path = c1.selection_path(tmp_path, smoke)
    assert path.name == "smoke_selected_lrs.json" and pick["smoke"]
    c1.write_json(path, pick)
    tasks = c1.test_tasks(tmp_path, smoke, c1.frozen_selection(tmp_path, smoke))
    assert all(t["expected_origins"] == 4 and t["smoke"] for t in tasks)
    assert not (tmp_path / "selected_lrs.json").exists()
    with pytest.raises(ValueError, match="capped"):
        c1.comparison(tmp_path, smoke, pick)
