#!/usr/bin/env python3
"""C1 orchestration. The parent uses only the standard library and never imports a model."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import queue
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "experiments/c1_lr_tuning/native_profiles.json"
AUDIT = PROFILE.with_name("C1_CODE_AUDIT.md")
DEFAULT_OUTPUT = PROFILE.parent / "runs"
TEST_SEEDS = (0, 1, 2)
DEFAULT_SCALES = (0.1, 0.3, 1.0, 3.0, 10.0)
PILOT_METHODS = ("multi_expert", "dsof", "onenet", "fsnet")
PILOT_SETTINGS = ("ETTh2:24", "WTH:24", "ETTm1:48")
DATASETS = ("ETTh2", "ETTm1", "WTH", "ECL")
SETTINGS = tuple(f"{d}:{h}" for d in DATASETS for h in (1, 24, 48))
LR_FIELDS = {
    "multi_expert": ("online_lr_expert", "online_lr_router"),
    "dsof": ("baseline_online_lr", "dsof_student_lr"),
    "dyname": ("dyname_online_lr",),
    "er": ("baseline_online_lr",),
    "fsnet": ("baseline_online_lr",),
    "onenet": ("baseline_online_lr",),
    "proceed": ("proceed_online_lr",),
    "patchtst_dgrad": ("dgrad_online_lr",),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def atomic_write(path, render):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8", newline="") as f:
            render(f)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def write_json(path, value):
    atomic_write(path, lambda f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False))


def read_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def write_csv(path, fields, rows):
    def render(f):
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})
    atomic_write(path, render)


def native_settings(method, setting, profile):
    entry = profile["methods"][method]
    specific = entry["settings"][setting]
    return profile["common"] | entry["arguments"] | specific["arguments"], specific["source_log"]


def online_rates(method, config, scale):
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("LR multiplier must be finite and positive")
    rates = {key: float(config[key]) * scale for key in LR_FIELDS[method]}
    if any(not math.isfinite(value) or value <= 0 for value in rates.values()):
        raise ValueError("scaled optimizer LR is not finite and positive")
    return rates


def checkpoint_key(method, config, seed, data_identity, source_identity):
    offline = {k: v for k, v in config.items() if k not in LR_FIELDS[method]}
    return digest({"method": method, "config": offline, "seed": seed,
                   "data": data_identity, "source": source_identity})


def feedback_protocol(method, config):
    if config["online_learning"] != "full" or not config["delay_fb"] or config["progressive_baseline_fb"]:
        raise ValueError(f"{method}: incompatible C1 feedback flags")
    if bool(config["progressive_fb"]) != (method == "multi_expert"):
        raise ValueError(f"{method}: C1 expects the archived main-table feedback protocol")
    return "progressive" if method == "multi_expert" else "legacy_delayed"


def target_bounds(dataset, rows):
    if dataset.startswith("ETT"):
        if rows < 3600:
            raise ValueError(f"{dataset}: fewer than 3600 data rows")
        return 2880, 3600, min(14400, rows)
    return int(rows * .2), rows - int(rows * .75), rows


def source_fingerprint():
    files = [REPO / "main.py", PROFILE, Path(__file__).resolve(), Path(__file__).with_name("c1_lr_worker.py")]
    for directory in ("exp", "data", "models", "utils"):
        files.extend((REPO / directory).rglob("*.py"))
    return digest({str(p.relative_to(REPO)): file_hash(p) for p in sorted(set(files))})


def data_fingerprint(path):
    # Counts are split metadata. No test data set, evaluation, or target statistics is used here.
    with path.open("rb") as f:
        sha = hashlib.sha256()
        count = 0
        for line in f:
            sha.update(line)
            count += bool(line.strip())
    return {"path": str(path), "sha256": sha.hexdigest(), "rows": count - 1}


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def make_manifest(args):
    profile = read_json(PROFILE)
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    source = {"git_commit": git_commit, "source_sha256": source_fingerprint()}
    data_root = args.data_root.resolve()
    datasets = {}
    groups = {}
    for method, setting in itertools.product(args.methods, args.settings):
        config, log = native_settings(method, setting, profile)
        dataset, horizon = setting.split(":")
        if not (config["method"] == method and config["data"] == dataset
                and config["pred_len"] == int(horizon) and config["test_bsz"] == 1):
            raise ValueError(f"incompatible profile for {method}/{setting}")
        if not (REPO / "exp" / f"exp_{method}.py").is_file():
            raise FileNotFoundError(f"missing experiment class for {method}")
        if dataset not in datasets:
            datasets[dataset] = data_fingerprint(data_root / config["data_path"])
        train_end, val_end, test_end = target_bounds(dataset, datasets[dataset]["rows"])
        origins = {"val": val_end - train_end - int(horizon) + 1,
                   "test": test_end - val_end - int(horizon) + 1}
        if min(origins.values()) <= 0:
            raise ValueError(f"{setting}: split does not contain a full target window")
        for scale in args.scales:
            online_rates(method, config, scale)
        groups[group_id(method, setting)] = dict(
            method=method, dataset=dataset, horizon=int(horizon), config=config,
            source_log=log, feedback_protocol=feedback_protocol(method, config),
            origins=origins, bounds=[0, train_end, val_end, test_end])
    result = dict(schema=1, run_id=args.run_id, validation_seeds=list(args.seeds),
                  test_seeds=list(TEST_SEEDS), scales=list(args.scales),
                  max_online_steps=args.max_online_steps, smoke=args.max_online_steps > 0,
                  source=source, datasets=datasets, groups=groups,
                  runtime_environment={"python": sys.version.split()[0], "torch": package_version("torch"),
                                       "numpy": package_version("numpy")})
    return result | {"manifest_hash": digest(result)}


def group_id(method, setting):
    return f"{method}__{setting.replace(':', '__')}"


def request(manifest, group, seed, stage, scale=1.0, variant=None, selection_hash=None):
    config = group.get("offline_config", group["config"]) if stage == "prepare" else group["config"]
    method = group["method"]
    dataset = group["dataset"]
    data = manifest["datasets"][dataset]
    offline_key = checkpoint_key(method, group.get("offline_config", config), seed, data, manifest["source"])
    rates = online_rates(method, config, scale)
    effective_config = config | rates
    config_hash = digest({"checkpoint_key": offline_key, "config": effective_config})
    split = {"prepare": "train", "sweep": "val", "test": "test"}[stage]
    cap = manifest["max_online_steps"] if stage != "prepare" else -1
    expected = group["origins"].get(split)
    if cap > 0:
        expected = min(expected, cap)
    base_group = group_id(method, f"{dataset}:{group['horizon']}")
    identity = base_group if stage == "prepare" else group.get("group_id", base_group)
    suffix = f"scale-{scale:.17g}" if stage == "sweep" else variant or stage
    return dict(
        task_id=f"{identity}__s{seed}__{suffix}", group_id=identity, stage=stage, split=split,
        method=method, dataset=dataset, horizon=group["horizon"], seed=seed, variant=variant,
        lr_scale=scale, native_online_lrs=online_rates(method, config, 1.0),
        all_effective_online_lrs=rates, config=effective_config,
        config_hash=config_hash, checkpoint_key=offline_key,
        feedback_protocol=group.get("offline_feedback_protocol", group["feedback_protocol"])
        if stage == "prepare" else group["feedback_protocol"], data=data, source=manifest["source"],
        manifest_hash=manifest["manifest_hash"], selection_hash=selection_hash, expected_origins=expected,
        max_online_steps=cap, smoke=manifest["smoke"] and stage != "prepare",
    )


def completion_path(root, task):
    return root / "tasks" / task["stage"] / (task["task_id"] + ".json")


@lru_cache(maxsize=512)
def cached_file_hash(path, size, mtime_ns, ctime_ns):
    return file_hash(path)


def valid_files(files):
    if not files:
        return False
    for name, sha in files.items():
        path = Path(name)
        if not path.is_file():
            return False
        stat = path.stat()
        if cached_file_hash(str(path), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != sha:
            return False
    return True


def valid_result(result, task):
    if result.get("status") != "completed" or result.get("request_hash") != digest(task):
        return False
    for field in ("stage", "split", "method", "dataset", "horizon", "seed", "variant",
                  "lr_scale", "config_hash", "checkpoint_key", "feedback_protocol", "smoke", "manifest_hash",
                  "all_effective_online_lrs", "selection_hash"):
        if field not in result or result[field] != task[field]:
            return False
    if not isinstance(result.get("runtime"), (int, float)) or not math.isfinite(result["runtime"]) or result["runtime"] < 0:
        return False
    if not result.get("checkpoint_path") or result["checkpoint_path"] not in result.get("checkpoint_files", {}):
        return False
    if task["method"] == "multi_expert" and str(Path(result["checkpoint_path"]).with_name("optimizer.pth")) not in result["checkpoint_files"]:
        return False
    if task["stage"] != "prepare":
        if (result["checkpoint_files"] != task.get("checkpoint_files")
                or result["checkpoint_path"] != task.get("checkpoint_path")
                or result.get("origins") != task["expected_origins"]):
            return False
        for name in ("mse", "mae"):
            value = result.get(name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                return False
        observed = result.get("observed_online_lrs")
        if not isinstance(observed, dict) or observed.keys() != task["all_effective_online_lrs"].keys():
            return False
    if task.get("config", {}).get("rebuttal_collect_capability_diagnostics"):
        artifacts = result.get("artifacts", {})
        expected = {"capability_unique.jsonl.gz", "capability_refresh_events.jsonl.gz", "capability_sample.json"}
        if {Path(p).name for p in artifacts} != expected or not valid_files(artifacts):
            return False
    return valid_files(result["checkpoint_files"])


def read_completed(root, task):
    path = completion_path(root, task)
    try:
        result = read_json(path)
        if valid_result(result, task):
            return result
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        pass
    return None


def bind_checkpoint(root, manifest, group, task):
    prepared = request(manifest, group, task["seed"], "prepare")
    result = read_completed(root, prepared)
    if result is None:
        raise ValueError(f"missing or incompatible checkpoint for {prepared['task_id']}; run prepare")
    return task | {key: result[key] for key in ("checkpoint_path", "checkpoint_files")}


def prepare_tasks(manifest):
    seeds = sorted(set(manifest["validation_seeds"]) | set(TEST_SEEDS))
    tasks = (request(manifest, group, seed, "prepare")
             for group, seed in itertools.product(manifest["groups"].values(), seeds))
    return list({task["task_id"]: task for task in tasks}.values())


def sweep_tasks(root, manifest):
    return [bind_checkpoint(root, manifest, group, request(manifest, group, seed, "sweep", scale))
            for group, seed, scale in itertools.product(manifest["groups"].values(),
                                                        manifest["validation_seeds"], manifest["scales"])]


def selection_path(root, manifest):
    return root / ("smoke_selected_lrs.json" if manifest["smoke"] else "selected_lrs.json")


def make_selection(root, manifest):
    trials = sweep_tasks(root, manifest)
    groups = {}
    for task in trials:
        result = read_completed(root, task)
        if result is None or result["split"] != "val":
            raise ValueError(f"missing completed validation trial: {task['task_id']}")
        name = task["group_id"]
        groups.setdefault(name, {}).setdefault(task["lr_scale"], []).append(result)
    selections = {}
    for name, scales in groups.items():
        means = {scale: statistics.mean(r["mse"] for r in rows) for scale, rows in scales.items()}
        minimum = min(means.values())
        tied = [scale for scale, value in means.items()
                if math.isclose(value, minimum, rel_tol=1e-9, abs_tol=1e-12)]
        scale = min(tied, key=lambda value: (abs(math.log(value)), value))
        checkpoint_files = {str(r["seed"]): r["checkpoint_files"] for r in scales[scale]}
        selections[name] = dict(selected_scale=scale, validation_mse=means[scale],
                                scale_mean_mse={str(s): v for s, v in means.items()},
                                default_validation_mse=means[1.0], checkpoint_files=checkpoint_files,
                                validation_trials=sum(map(len, scales.values())))
    value = dict(schema=1, source_split="val", smoke=manifest["smoke"], manifest_hash=manifest["manifest_hash"],
                 validation_seeds=manifest["validation_seeds"], selections=selections,
                 tie_breaker="within rel_tol=1e-9/abs_tol=1e-12, nearest to 1 in log space, then smaller scale")
    return value | {"selection_hash": digest(value)}


def frozen_selection(root, manifest):
    selection = read_json(selection_path(root, manifest))
    sha = selection.get("selection_hash")
    if (sha != digest({k: v for k, v in selection.items() if k != "selection_hash"})
            or selection.get("schema") != 1 or selection.get("source_split") != "val"
            or selection.get("smoke") != manifest["smoke"]
            or selection.get("manifest_hash") != manifest["manifest_hash"]
            or selection.get("validation_seeds") != manifest["validation_seeds"]
            or set(selection["selections"]) != set(manifest["groups"])):
        raise ValueError("frozen selection does not match this run")
    for pick in selection["selections"].values():
        if pick["selected_scale"] not in manifest["scales"]:
            raise ValueError("frozen scale is outside the registered grid")
    return selection


def test_tasks(root, manifest, selection, task_ids=None):
    tasks = []
    for name, group in manifest["groups"].items():
        pick = selection["selections"][name]
        for seed, variant in itertools.product(TEST_SEEDS, ("default", "tuned")):
            scale = 1.0 if variant == "default" else pick["selected_scale"]
            task = request(manifest, group, seed, "test", scale, variant, selection["selection_hash"])
            if task_ids is not None and task["task_id"] not in task_ids:
                continue
            task = bind_checkpoint(root, manifest, group, task)
            if str(seed) in pick["checkpoint_files"] and task["checkpoint_files"] != pick["checkpoint_files"][str(seed)]:
                raise ValueError(f"checkpoint changed after validation: {task['task_id']}")
            tasks.append(task)
    return tasks


TRIAL_FIELDS = ("method", "dataset", "horizon", "seed", "variant", "lr_scale", "all_effective_online_lrs",
                "observed_online_lrs", "validation_mse", "validation_mae", "test_mse", "test_mae", "status",
                "runtime", "checkpoint_path", "config_hash", "checkpoint_key", "feedback_protocol", "origins", "smoke", "log_path")
FAILURE_FIELDS = ("stage", "task_id", "method", "dataset", "horizon", "seed", "variant", "lr_scale",
                  "status", "exit_code", "error", "runtime", "log_path")


def failure_rows(root):
    return [read_json(p) for p in sorted((root / "failures").glob("*/*.json"))]


def write_trial_table(root, stage, tasks):
    recent = {r["task_id"]: r for r in failure_rows(root) if r["stage"] == stage}
    rows = []
    for task in tasks:
        row = dict(task, status="pending")
        result = read_completed(root, task)
        if result:
            label = "validation" if stage == "sweep" else "test"
            row.update(result, **{label + "_mse": result["mse"], label + "_mae": result["mae"]})
        elif task["task_id"] in recent:
            row.update(status="failed", runtime=recent[task["task_id"]]["runtime"],
                       log_path=recent[task["task_id"]]["log_path"])
        rows.append(row)
    filename = "validation_trials.csv" if stage == "sweep" else "test_trials.csv"
    write_csv(root / filename, TRIAL_FIELDS, rows)


def run_one(root, task, gpu, checkpoint_root=None):
    attempt_id = uuid.uuid4().hex
    attempt = root / "attempts" / task["stage"] / task["task_id"] / attempt_id
    attempt.mkdir(parents=True)
    job_path = attempt / "request.json"
    log_path = root / "logs" / task["stage"] / f"{task['task_id']}__{attempt_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    offline = (checkpoint_root or root / "offline_checkpoints") / task["checkpoint_key"] / attempt_id
    worker = Path(__file__).with_name("c1_lr_worker.py")
    command = [sys.executable, "-B", "-u", str(worker), "--execute", str(job_path)]
    write_json(job_path, {"task": task, "offline_root": str(offline), "physical_gpu": gpu})
    code = None
    start = time.perf_counter()
    try:
        with log_path.open("x", encoding="utf-8") as log:
            log.write(f"CUDA_VISIBLE_DEVICES={gpu} {shlex.join(command)}\n")
            log.flush()
            env = os.environ | {"CUDA_VISIBLE_DEVICES": gpu, "PYTHONDONTWRITEBYTECODE": "1"}
            code = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
        if code:
            raise RuntimeError(f"worker exited with {code}")
        result = read_json(attempt / "result.json")
        if not valid_result(result, task):
            raise ValueError("worker result lacks valid metadata, finite full metrics or compatible checkpoint")
        write_json(completion_path(root, task), result | {"log_path": str(log_path), "command": command})
        return True
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        failure = {k: task[k] for k in FAILURE_FIELDS if k in task}
        failure.update(status="failed", exit_code=code, error=str(exc),
                       runtime=time.perf_counter()-start, log_path=str(log_path))
        write_json(root / "failures" / task["stage"] / f"{attempt_id}.json", failure)
        print(f"FAILED {task['task_id']}: {exc}", file=sys.stderr)
        return False


def run_stage(root, stage, tasks, args):
    remaining = []
    for task in tasks:
        if read_completed(root, task):
            if not args.resume:
                raise ValueError(f"completed task exists: {task['task_id']}; use --resume")
        else:
            remaining.append(task)
    slots = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.max_per_gpu):
            slots.put(gpu)

    def run(task):
        gpu = slots.get()
        try:
            return run_one(root, task, gpu)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        results = list(executor.map(run, remaining))
    if stage != "prepare":
        write_trial_table(root, stage, tasks)
    write_csv(root / "failures.csv", FAILURE_FIELDS, failure_rows(root))
    print(f"{stage}: {len(remaining)-results.count(False)} succeeded, {results.count(False)} failed, "
          f"{len(tasks)-len(remaining)} resumed")
    return 1 if not all(results) else 0


def comparison(root, manifest, selection):
    if manifest["smoke"]:
        raise ValueError("capped runs cannot produce official comparison reports")
    results = {}
    for task in test_tasks(root, manifest, selection):
        result = read_completed(root, task)
        if result is None:
            raise ValueError(f"missing completed full test trial: {task['task_id']}")
        key = task["method"], task["dataset"], task["horizon"], task["variant"]
        results.setdefault(key, []).append(result)
    output = []
    for name, group in manifest["groups"].items():
        method, dataset, horizon = (group[k] for k in ("method", "dataset", "horizon"))
        pick = selection["selections"][name]
        row = dict(method=method, dataset=dataset, horizon=horizon,
                   feedback_protocol=group["feedback_protocol"],
                   original_lr=online_rates(method, group["config"], 1.0),
                   selected_lr=online_rates(method, group["config"], pick["selected_scale"]),
                   selected_scale=pick["selected_scale"], validation_mse=pick["validation_mse"],
                   default_validation_mse=pick["default_validation_mse"],
                   validation_trials=pick["validation_trials"], validation_seed_count=len(manifest["validation_seeds"]),
                   candidate_count=len(manifest["scales"]), test_seeds=list(TEST_SEEDS), std_ddof=0)
        for variant in ("default", "tuned"):
            trials = results[method, dataset, horizon, variant]
            for metric in ("mse", "mae"):
                values = [r[metric] for r in trials]
                row[f"{variant}_{metric}_mean"] = statistics.mean(values)
                row[f"{variant}_{metric}_std"] = statistics.pstdev(values)
            row[f"{variant}_observed_online_lrs"] = {str(r["seed"]): r["observed_online_lrs"] for r in trials}
        default = row["default_mse_mean"]
        row["change_pct"] = 100 * (row["tuned_mse_mean"] - default) / default if default else None
        output.append(row)
    return output


def report(root, manifest):
    selection = frozen_selection(root, manifest)
    rows = comparison(root, manifest, selection)
    write_csv(root / "comparison.csv", list(rows[0]), rows)
    write_json(root / "comparison.json", {"selection_hash": selection["selection_hash"], "comparison": rows})

    def render(f):
        f.write("# C1 online LR comparison\n\nSeeds 0, 1, 2. Mean ± population SD (ddof=0). "
                "Change = 100 × (tuned MSE − default MSE) / default MSE. Negative means lower MSE.\n\n"
                "| Dataset | H | Method | Feedback | Original LR | Selected LR | Val MSE | Val trials | Test default MSE | Test tuned MSE | Default MAE | Tuned MAE | Change (%) |\n"
                "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            show_rates = lambda key: ", ".join(f"{k}={v:g}" for k, v in row[key].items())
            cells = [row["dataset"], row["horizon"], row["method"], row["feedback_protocol"],
                     show_rates("original_lr"), show_rates("selected_lr"), f'{row["validation_mse"]:.6g}', row["validation_trials"]]
            cells.extend(f'{row[f"{arm}_{metric}_mean"]:.6g} ± {row[f"{arm}_{metric}_std"]:.6g}'
                         for metric in ("mse", "mae") for arm in ("default", "tuned"))
            cells.append("N/A" if row["change_pct"] is None else f'{row["change_pct"]:+.3f}')
            f.write("| " + " | ".join(map(str, cells)) + " |\n")
    atomic_write(root / "comparison.md", render)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inspect", "prepare", "sweep", "select", "test", "report"))
    parser.add_argument("--run-id", default="pilot", help="one directory name under output-root")
    parser.add_argument("--methods", default=",".join(PILOT_METHODS), help="comma-separated method IDs or all")
    parser.add_argument("--settings", default=",".join(PILOT_SETTINGS), help="comma-separated dataset:H settings or all")
    parser.add_argument("--seeds", default="0", help="validation seeds, comma-separated; test always uses 0,1,2")
    parser.add_argument("--scales", default=",".join(map(str, DEFAULT_SCALES)), help="fixed comma-separated positive LR multipliers including 1")
    parser.add_argument("--gpus", default="0,1", help="comma-separated physical GPU indices")
    parser.add_argument("--max-per-gpu", type=int, default=1, help="maximum simultaneous workers per listed GPU")
    parser.add_argument("--data-root", type=Path, default=REPO / "data", help="directory containing original CSVs")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT, help="parent of per-run output directories")
    parser.add_argument("--max-online-steps", type=int, default=-1, help="positive cap for a separate smoke run; -1 means complete streams")
    parser.add_argument("--execute", action="store_true", help="perform this stage; without it, only show the plan")
    parser.add_argument("--resume", action="store_true", help="skip only completed matching tasks with intact metadata and checkpoints")
    args = parser.parse_args(argv)

    def parse_list(value, cast):
        result = tuple(cast(item.strip()) for item in value.split(","))
        if len(set(result)) != len(result) or any(item == "" for item in result):
            raise ValueError("comma-separated list entries must be unique and nonempty")
        return result

    try:
        args.methods = tuple(LR_FIELDS) if args.methods == "all" else parse_list(args.methods, str)
        args.settings = SETTINGS if args.settings == "all" else parse_list(args.settings, str)
        args.seeds = tuple(sorted(parse_list(args.seeds, int)))
        args.scales = tuple(sorted(parse_list(args.scales, float)))
        args.gpus = parse_list(args.gpus, str)
        if not set(args.methods) <= LR_FIELDS.keys() or not set(args.settings) <= set(SETTINGS):
            raise ValueError("unknown method or dataset:horizon")
        if min(args.seeds) < 0 or max(args.seeds) >= 2**32:
            raise ValueError("seeds must be unsigned 32-bit integers")
        if 1.0 not in args.scales or any(s <= 0 or not math.isfinite(s) for s in args.scales):
            raise ValueError("scales must be positive, finite, and include 1.0")
        if any(not re.fullmatch(r"0|[1-9][0-9]*", g) for g in args.gpus) or args.max_per_gpu < 1:
            raise ValueError("GPU indices must be canonical nonnegative integers and max-per-gpu positive")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
            raise ValueError("run-id must be a single safe directory name")
        if args.max_online_steps != -1 and args.max_online_steps <= 0:
            raise ValueError("max-online-steps must be -1 or positive")
    except ValueError as exc:
        parser.error(str(exc))
    return args


def describe(root, manifest, args):
    n = len(manifest["groups"])
    counts = {"offline_checkpoints": len(prepare_tasks(manifest)),
              "validation_trials": n * len(manifest["validation_seeds"]) * len(manifest["scales"]),
              "test_trials": n * 2 * len(TEST_SEEDS)}
    print(json.dumps({"stage": args.stage, "run": str(root), "smoke": manifest["smoke"],
                      "gpu_slots": {g: args.max_per_gpu for g in args.gpus}, **counts}, sort_keys=True))
    if args.stage == "inspect":
        checkpoints = prepare_tasks(manifest)
        devices = None
        if shutil.which("nvidia-smi"):
            probe = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            devices = probe.stdout.splitlines() if not probe.returncode else None
        print(json.dumps({"environment": manifest["runtime_environment"], "gpu_devices": devices,
                          "ready_checkpoints": sum(read_completed(root, task) is not None for task in checkpoints),
                          "data": manifest["datasets"], "groups": {
                              name: {"feedback_protocol": g["feedback_protocol"], "origins": g["origins"],
                                     "native_source": g["source_log"], "online_rates": {
                                         str(s): online_rates(g["method"], g["config"], s) for s in manifest["scales"]}}
                              for name, g in manifest["groups"].items()}}, indent=2, sort_keys=True))
    elif not args.execute:
        print("Dry run. Add --execute to perform this stage.")


def execute(root, manifest, args):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another stage is already running for this run-id") from exc
        path = root / "manifest.json"
        if path.exists():
            if read_json(path) != manifest:
                raise ValueError("manifest changed; use the same configuration or a new run-id")
        elif args.stage == "prepare":
            atomic_write(root / "audit.md", lambda f: f.write(AUDIT.read_text(encoding="utf-8")))
            write_json(path, manifest)
        else:
            raise ValueError("prepare must create the run manifest first")
        if args.stage == "prepare":
            return run_stage(root, args.stage, prepare_tasks(manifest), args)
        if args.stage == "sweep":
            return run_stage(root, args.stage, sweep_tasks(root, manifest), args)
        if args.stage == "select":
            target = selection_path(root, manifest)
            if target.exists():
                if args.resume:
                    frozen_selection(root, manifest)
                    return 0
                raise ValueError("selection is frozen; use --resume to keep it")
            selection = make_selection(root, manifest)
            write_json(target, selection)
            print(f"Frozen {len(selection['selections'])} selections from validation only in {target.name}.")
        elif args.stage == "test":
            return run_stage(root, args.stage, test_tasks(root, manifest, frozen_selection(root, manifest)), args)
        elif args.stage == "report":
            report(root, manifest)
            print(f"Wrote {root / 'comparison.md'}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        manifest = make_manifest(args)
        root = args.output_root.resolve() / args.run_id
        describe(root, manifest, args)
        if args.execute and args.stage != "inspect":
            return execute(root, manifest, args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"C1: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
