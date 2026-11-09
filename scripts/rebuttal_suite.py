#!/usr/bin/env python3
"""Plan and run the PAVE rebuttal suite. GPU execution is opt-in."""
import argparse
import fcntl
from functools import lru_cache
import itertools
import json
import math
import os
from pathlib import Path
import queue
import re
import shlex
import statistics
from collections import defaultdict
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import c1_lr_sweep as c1
from utils.rebuttal_splits import CONVENTIONAL, PAPER, ett_bounds, verify_ett_calendar

HOME = c1.REPO / "experiments/rebuttal_2027"
CONFIG = HOME / "CONFIGS/suite.json"
EXPERIMENTS = tuple(f"C{i}" for i in range(1, 8))
TUNING = ("C1", "C7")
PRIORITIES = {"C1": "P0", "C2": "P0", "C3": "P0", "C4": "P0/P1", "C5": "P1", "C6": "P1", "C7": "P0"}
DEFAULTS = {
    "C1": (c1.PILOT_SETTINGS, c1.SETTINGS),
    "C2": (("ETTm1:48", "WTH:48"), ("ETTm1:48", "WTH:48", "ECL:48")),
    "C3": (("ETTm1:48", "WTH:48"), c1.SETTINGS),
    "C5": (("ECL:48",), ("ETTm1:48", "WTH:48", "ECL:48")),
    "C6": (("ETTh2:48", "ETTm1:48"), ("ETTh2:24", "ETTh2:48", "ETTm1:24", "ETTm1:48")),
    "C7": (("ETTh2:24",), ("ETTh2:24", "ETTm1:24", "WTH:48", "ECL:48")),
}
OFFLINE_IGNORED = "identical offline configuration, data, split, seed, runtime and source"


def ofat(config, grids):
    yield "full", {}, {"parameter": "reference", "candidate": None}
    for parameter, values in grids.items():
        keys = ("stable_buffer_size", "recovery_buffer_size") if parameter == "memory_capacity" else (parameter,)
        for value in values:
            changes = {key: value for key in keys if config[key] != value}
            if changes:
                yield f"{parameter}_{value}", changes, {"parameter": parameter, "candidate": value}


def moe_config(mode, budget):
    return {"rebuttal_moe_mode": mode, "disable_version_awareness": True, "disable_recovery": True,
            "expert_update_strategy": "plain", "disable_online_correction": mode == "uniform", **budget}


def variants(experiment, method, config, registered, args):
    if experiment == "C2":
        yield from ofat(config, registered["sensitivity"])
    elif experiment == "C3":
        for mode in registered["validity_modes"]:
            yield mode, {"rebuttal_validity_mode": mode, "rebuttal_collect_capability_diagnostics": True,
                         "rebuttal_parameter_proxy_size": args.parameter_proxy_size,
                         "rebuttal_full_rep_proxy": args.full_encoder_proxy}, {}
    elif experiment == "C5":
        yield "full", {}, {}
        budget = registered["prospective_moe_budget"]
        yield "prospective_matched_moe", moe_config("matched", budget), {}
        yield "prospective_uniform_moe", moe_config("uniform", budget), {}
        if args.time_match:
            yield "prospective_time_matched_moe", moe_config("matched", budget), {"time_matched": True}
    elif experiment == "C7" and method == "dsof":
        yield "original_adapted_dsof", {}, {}
        yield "dsof_partial_td", {"rebuttal_dsof_partial_td": True}, {}
    else:
        yield method, {}, {}


@lru_cache(maxsize=None)
def data_details(path, conventional, dataset):
    details = c1.data_fingerprint(path)
    if conventional:
        edges = verify_ett_calendar(path, dataset)
        bounds = ett_bounds(dataset == "ETTm1", CONVENTIONAL)
    else:
        edges = {}
        bounds = c1.target_bounds(dataset, details["rows"])
    return details, bounds, edges


def scope_methods(experiment, args):
    if args.methods:
        methods = tuple(c1.LR_FIELDS) if args.methods == "all" else args.methods.split(",")
    elif experiment in ("C2", "C3", "C5"):
        methods = ("multi_expert",)
    elif experiment == "C7":
        methods = ("dsof", "multi_expert")
    elif experiment == "C6" and args.scope == "smoke":
        methods = ("multi_expert",)
    elif experiment == "C1" and args.scope != "full":
        methods = c1.PILOT_METHODS
    else:
        methods = tuple(c1.LR_FIELDS)
    allowed = {"multi_expert"} if experiment in ("C2", "C3", "C5") else (
        {"dsof", "multi_expert"} if experiment == "C7" else set(c1.LR_FIELDS))
    if not set(methods) <= allowed or len(methods) != len(set(methods)):
        raise ValueError(f"{experiment}: unsupported or duplicate method")
    return tuple(sorted(methods))


def manifest(experiment, args, registered):
    experiment = "C3" if experiment == "C4" else experiment
    profile = c1.read_json(c1.PROFILE)
    settings = tuple(args.settings.split(",")) if args.settings else DEFAULTS[experiment][args.scope == "full"]
    if len(set(settings)) != len(settings) or not set(settings) <= set(c1.SETTINGS):
        raise ValueError("invalid or duplicate settings")
    conventional = experiment == "C6"
    if args.scope == "smoke" and not args.settings and not conventional:
        settings = settings[:1]
    source = source_identity()
    groups, datasets = {}, {}
    for method, setting in itertools.product(scope_methods(experiment, args), sorted(settings)):
        base, log = c1.native_settings(method, setting, profile)
        dataset, horizon_text = setting.split(":")
        horizon = int(horizon_text)
        data_path = args.data_root.resolve() / base["data_path"]
        data, bounds, timestamps = data_details(data_path, conventional, dataset)
        datasets[dataset] = data
        train_end, val_end, test_end = bounds
        mode = CONVENTIONAL if conventional else PAPER
        if conventional:
            base = base | {"split_mode": CONVENTIONAL}
        if args.scope == "smoke":
            base = base | {"train_epochs": 1, "patience": 1}
        protocol = c1.feedback_protocol(method, base)
        for variant, changes, labels in variants(experiment, method, base, registered, args):
            gid = c1.group_id(method, setting) + "__" + variant
            config = base | changes
            groups[gid] = dict(group_id=gid, method=method, dataset=dataset, horizon=horizon, variant=variant,
                               config=config, offline_config=base, offline_feedback_protocol=protocol,
                               feedback_protocol="dsof_partial_td_adaptation" if variant == "dsof_partial_td" else protocol,
                               source_log=log, split_mode=mode, bounds=[0, *bounds], timestamps=timestamps,
                               scaler_fit=[0, train_end],
                               origins={"train": train_end-base["seq_len"]-horizon+1,
                                        "val": val_end-train_end-horizon+1, "test": test_end-val_end-horizon+1},
                               **labels)
    cap = args.max_online_steps if args.max_online_steps is not None else (max(g["horizon"] for g in groups.values()) + 4 if args.scope == "smoke" else -1)
    seeds = [0] if args.scope == "smoke" else list(c1.TEST_SEEDS)
    value = dict(schema=1, experiment=experiment, scope=args.scope, run_id=args.run_id,
                 validation_seeds=list(args.seeds), test_seeds=seeds,
                 scales=[1.0] if args.scope == "smoke" else registered["lr_scales"],
                 smoke=cap > 0, max_online_steps=cap,
                 source=source, orchestration_sha256=c1.digest({
                     p.name: c1.file_hash(p) for p in (CONFIG, Path(__file__).resolve())}),
                 groups=groups, datasets=datasets,
                 time_budgets=registered["time_calibration_candidates"] if experiment == "C5" and args.time_match else [],
                 environment={"python": sys.version.split()[0], "torch": c1.package_version("torch"),
                              "numpy": c1.package_version("numpy")})
    return value | {"manifest_hash": c1.digest(value)}


@lru_cache(maxsize=1)
def source_identity():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=c1.REPO, text=True).strip()
    return {"git_commit": commit, "source_sha256": c1.source_fingerprint(),
            "runtime_versions": {name: c1.package_version(name) for name in ("torch", "numpy")}}


def phases(m):
    if m["experiment"] in TUNING:
        return ("prepare", "sweep", "select", "test")
    if m["time_budgets"]:
        return ("prepare", "calibrate", "freeze", "test")
    return ("prepare", "test")


def root_for(args, m):
    return args.output_root.resolve() / args.scope / args.run_id / m["experiment"]


def planned(m, phase):
    groups = m["groups"]
    if phase == "prepare":
        seeds = set(m["test_seeds"])
        if m["experiment"] in TUNING or m["time_budgets"]:
            seeds.update(m["validation_seeds"])
        by_key = {}
        for group, seed in itertools.product(groups.values(), sorted(seeds)):
            t = c1.request(m, group, seed, "prepare")
            by_key[t["task_id"]] = t
        return list(by_key.values())
    if phase == "sweep":
        return [c1.request(m, group, seed, "sweep", scale)
                for group, seed, scale in itertools.product(groups.values(), m["validation_seeds"], m["scales"])]
    if phase == "calibrate":
        tasks = []
        for group, seed in itertools.product(groups.values(), m["validation_seeds"]):
            if group["variant"] == "full":
                task = c1.request(m, group, seed, "sweep")
                tasks.append(task | {"calibration_budget": None})
            elif group.get("time_matched"):
                for index, budget in enumerate(m["time_budgets"]):
                    adjusted = group | {"config": group["config"] | budget}
                    task = c1.request(m, adjusted, seed, "sweep")
                    tasks.append(task | {"task_id": f"{task['task_id']}__budget{index}",
                                         "calibration_budget": index})
        return tasks
    if phase == "test":
        variants_ = ("default", "tuned") if m["experiment"] in TUNING else None
        return [c1.request(m, group, seed, "test", variant=arm)
                for group, seed in itertools.product(groups.values(), m["test_seeds"])
                for arm in (variants_ or (group["variant"],))]
    return []


def freeze_budget(root, m):
    by_setting = {}
    for task in planned(m, "calibrate"):
        group = m["groups"][task["group_id"]]
        bound = c1.bind_checkpoint(root, m, group, task)
        result = c1.read_completed(root, bound)
        if result is None:
            raise ValueError(f"missing calibration job: {task['task_id']}")
        setting = f"{task['dataset']}:{task['horizon']}"
        by_setting.setdefault(setting, {}).setdefault(task["calibration_budget"], []).append(result)
    picks = {}
    for setting, candidates in by_setting.items():
        hardware = {json.dumps(r["hardware"], sort_keys=True) for rows in candidates.values() for r in rows}
        if len(hardware) != 1:
            raise ValueError(f"{setting}: calibrate all time candidates on the same host and GPU")
        times = {i: statistics.mean(r["mean_origin_ms"] for r in rows) for i, rows in candidates.items()}
        reference = times.pop(None)
        if any(v <= 0 or not math.isfinite(v) for v in (reference, *times.values())):
            raise ValueError(f"{setting}: invalid measured validation timing")
        index = min(times, key=lambda i: (abs(math.log(times[i]/reference)), i))
        picks[setting] = dict(index=index, budget=m["time_budgets"][index], target_ms=reference,
                              measured_ms=times[index], hardware=json.loads(next(iter(hardware))),
                              candidate_ms={str(i): v for i, v in times.items()})
    record = dict(source_split="val", manifest_hash=m["manifest_hash"], selections=picks)
    return record | {"selection_hash": c1.digest(record)}


def load_budget(root, m):
    pick = c1.read_json(root / "frozen_time_budget.json")
    body = {k: v for k, v in pick.items() if k != "selection_hash"}
    if (pick["selection_hash"] != c1.digest(body) or pick["manifest_hash"] != m["manifest_hash"]
            or pick["source_split"] != "val"):
        raise ValueError("time budget is not frozen for this run")
    return pick


def bind_task(root, m, phase, task):
    if phase == "prepare":
        return task
    group = m["groups"][task["group_id"]]
    if phase in ("calibrate", "sweep"):
        return c1.bind_checkpoint(root, m, group, task)
    scale = 1.0
    frozen_hash = m["manifest_hash"]
    required = {}
    if m["experiment"] in TUNING:
        selection = c1.frozen_selection(root, m)
        picked = selection["selections"][task["group_id"]]
        scale = picked["selected_scale"] if task["variant"] == "tuned" else 1.0
        frozen_hash = selection["selection_hash"]
    if m["time_budgets"]:
        selection = load_budget(root, m)
        budget = selection["selections"][f"{task['dataset']}:{task['horizon']}"]
        required = {"required_hardware": budget["hardware"]}
        frozen_hash = selection["selection_hash"]
        if group.get("time_matched"):
            group = group | {"config": group["config"] | budget["budget"]}
    fresh = c1.request(m, group, task["seed"], "test", scale, task["variant"], frozen_hash) | required
    bound = c1.bind_checkpoint(root, m, group, fresh)
    if m["experiment"] in TUNING:
        expected = picked["checkpoint_files"].get(str(task["seed"]))
        if expected is not None and expected != bound["checkpoint_files"]:
            raise ValueError("checkpoint changed since validation selection")
    return bound


def run_cached(root, checkpoint_root, task, gpu, resume):
    completion = c1.completion_path(root, task)
    completion.parent.mkdir(parents=True, exist_ok=True)
    with completion.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if c1.read_completed(root, task):
            if not resume:
                raise ValueError(f"completed task exists: {task['task_id']}; use --resume")
            return True
        if task["stage"] == "prepare":
            entry = checkpoint_root / task["checkpoint_key"] / "cache.json"
            reason = OFFLINE_IGNORED
        elif (root.name in ("C2", "C5") and task["stage"] == "test" and task["variant"] == "full"
              and not task["smoke"] and "required_hardware" not in task):
            fields = ("method", "dataset", "horizon", "seed", "stage", "split", "feedback_protocol",
                      "config_hash", "checkpoint_files", "expected_origins", "source", "variant")
            key = c1.digest({field: task[field] for field in fields})
            entry = checkpoint_root / "_verified_online" / f"{key}.json"
            reason = "identical uncapped Full PAVE test with the same checkpoint and diagnostics configuration"
        else:
            return c1.run_one(root, task, gpu, checkpoint_root)
        entry.parent.mkdir(parents=True, exist_ok=True)
        with entry.with_suffix(".lock").open("a") as cache_lock:
            fcntl.flock(cache_lock, fcntl.LOCK_EX)
            try:
                cached = c1.read_json(entry)
            except (OSError, ValueError):
                cached = {}
            if not isinstance(cached, dict):
                cached = {}
            result = cached | {field: task[field] for field in ("manifest_hash", "selection_hash")} | {
                "request_hash": c1.digest(task), "reused_from": str(entry), "reuse_reason": reason}
            if c1.valid_result(result, task):
                c1.write_json(completion, result)
                return True
            if not c1.run_one(root, task, gpu, checkpoint_root):
                return False
            c1.write_json(entry, c1.read_json(completion))
            return True


def ensure_manifest(root, m):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    with (root / ".manifest.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            if c1.read_json(path) != m:
                raise ValueError(f"{root}: manifest differs; use the original options or a new run-id")
        else:
            c1.write_json(path, m)


def execute(root, m, phase, args, tasks):
    ensure_manifest(root, m)
    if phase in ("select", "freeze"):
        if args.job:
            raise ValueError("select and freeze require the complete validation grid")
        path = c1.selection_path(root, m) if phase == "select" else root / "frozen_time_budget.json"
        with path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if path.exists():
                if not args.resume:
                    raise ValueError("selection already frozen; use --resume")
                try:
                    if phase == "select":
                        c1.frozen_selection(root, m)
                    else:
                        load_budget(root, m)
                except (OSError, ValueError, KeyError, AttributeError, TypeError):
                    pass
                else:
                    return True
            pick = c1.make_selection(root, m) if phase == "select" else freeze_budget(root, m)
            c1.write_json(path, pick)
        return True
    if phase == "calibrate" and (len(args.gpus) != 1 or args.max_per_gpu != 1):
        raise ValueError("run calibration serially on one GPU with --max-per-gpu 1")
    tasks = [bind_task(root, m, phase, task) for task in tasks]
    checkpoint_root = args.checkpoint_root.resolve() if args.checkpoint_root else args.output_root.resolve() / "_checkpoints"
    pinned = defaultdict(list)
    flexible = []
    for task in tasks:
        hardware = task.get("required_hardware")
        if hardware:
            gpu = hardware["physical_gpu"]
            if gpu not in args.gpus:
                raise ValueError(f"test requires calibration GPU {gpu}; include it in --gpus")
            pinned[gpu].append(task)
        else:
            flexible.append(task)

    def serial(gpu, batch):
        return [run_cached(root, checkpoint_root, task, gpu, args.resume) for task in batch]

    results = []
    if pinned:
        with ThreadPoolExecutor(max_workers=len(pinned)) as executor:
            for batch in executor.map(serial, pinned, pinned.values()):
                results.extend(batch)
    slots = queue.Queue()
    for gpu in args.gpus:
        for _ in range(args.max_per_gpu):
            slots.put(gpu)

    def worker(task):
        gpu = slots.get()
        try:
            return run_cached(root, checkpoint_root, task, gpu, args.resume)
        finally:
            slots.put(gpu)

    with ThreadPoolExecutor(max_workers=slots.qsize()) as executor:
        results.extend(executor.map(worker, flexible))
    print(f"{m['experiment']} {phase}: {sum(results)}/{len(tasks)} complete")
    return all(results)


def option_tokens(args, experiment):
    tokens = ["--experiment", experiment, "--scope", args.scope, "--run-id", args.run_id,
              "--output-root", str(args.output_root.resolve()), "--data-root", str(args.data_root.resolve()),
              "--seeds", ",".join(map(str, args.seeds))]
    for flag, value in (("--methods", args.methods), ("--settings", args.settings),
                        ("--max-online-steps", args.max_online_steps),
                        ("--checkpoint-root", args.checkpoint_root.resolve() if args.checkpoint_root else None)):
        if value is not None:
            tokens.extend([flag, str(value)])
    if args.time_match:
        tokens.append("--time-match")
    if args.full_encoder_proxy:
        tokens.append("--full-encoder-proxy")
    tokens.extend(["--parameter-proxy-size", str(args.parameter_proxy_size)])
    return tokens


def command_for(args, experiment, phase, job=None):
    argv = ["python", str(Path(__file__).resolve()), "run", *option_tokens(args, experiment),
            "--phase", phase, "--resume"]
    if job is not None:
        argv.extend(["--job", job])
    return shlex.join(argv)


def eligible_tasks(m, experiment, phase):
    tasks = planned(m, phase)
    if experiment == "C4" and phase != "prepare":
        tasks = [t for t in tasks if m["groups"][t["group_id"]]["variant"] == "full"]
    if experiment == "C4" and phase == "prepare":
        # All C3 modes share the same offline checkpoint.
        return tasks
    return tasks


def job_rows(args, experiment, m):
    for phase in phases(m):
        for task in eligible_tasks(m, experiment, phase):
            g = m["groups"].get(task["group_id"], {})
            command = command_for(args, experiment, phase, task["task_id"])
            deps = {"prepare": "shared cache or offline training", "sweep": "prepare", "calibrate": "prepare",
                    "test": "prepare + select" if m["experiment"] in TUNING else (
                        "prepare + freeze" if m["time_budgets"] else "prepare")}[phase]
            root = root_for(args, m)
            yield dict(scope=args.scope, run_id=args.run_id, experiment=experiment, priority=PRIORITIES[experiment], phase=phase,
                       task_id=task["task_id"], method=task["method"], dataset=task["dataset"],
                       horizon=task["horizon"], seed=task["seed"], variant=task["variant"] or g.get("variant", ""),
                       split=task["split"], expected_origins=task["expected_origins"],
                       command=f'{command} --gpus 0 --execute', estimated_cost="unknown wall-time and peak memory",
                       log_directory=str(root / "logs" / task["stage"]),
                       completion_path=str(c1.completion_path(root, task)),
                       shared_task=("C3" if experiment == "C4" else ""), dependencies=deps, status="planned")


def export(args, plans):
    dest = args.export_root.resolve() / args.scope
    manifest_rows = []
    for experiment, m in plans:
        rows = list(job_rows(args, experiment, m))
        manifest_rows.extend(rows)
        exp_dir = dest / experiment
        c1.write_json(exp_dir / "planned_manifest.json", m)
        # Shell wrappers remain dry-run unless the user explicitly sets EXECUTE=1.
        env = 'gpu_ids="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}"\n'
        env += 'gpu_args=(--gpus "$gpu_ids" --max-per-gpu "${MAX_PER_GPU:-1}")\n'
        env += 'calibration_args=(--gpus "${CALIBRATION_GPU:-${gpu_ids%%,*}}" --max-per-gpu 1)\n'
        env += 'exec_args=()\nif [[ "${EXECUTE:-0}" == 1 ]]; then exec_args+=(--execute); fi\n'
        header = "#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(c1.REPO)) + "\n" + env

        def launch(phase, job=None):
            slots = "calibration_args" if phase == "calibrate" else "gpu_args"
            return command_for(args, experiment, phase, job) + f' "${{{slots}[@]}}" "${{exec_args[@]}}"\n'

        for row in rows:
            job_path = exp_dir / "jobs" / row["phase"] / (row["task_id"] + ".sh")
            content = header + launch(row["phase"], row["task_id"])
            c1.atomic_write(job_path, lambda f: f.write(content))
        lines = [header, *(launch(phase) for phase in phases(m))]
        c1.atomic_write(exp_dir / "run.sh", lambda f: f.writelines(lines))
        log_dir = root_for(args, m) / "launcher" / experiment
        launcher = ("#!/usr/bin/env bash\nset -euo pipefail\n"
                    'if [[ "${EXECUTE:-0}" != 1 ]]; then echo "Set EXECUTE=1 to launch."; exit 1; fi\n'
                    'runner=$(cd "$(dirname "$0")" && pwd)/run.sh\n'
                    f"log_dir={shlex.quote(str(log_dir))}\n"
                    'mkdir -p "$log_dir"\n'
                    'attempt=$(mktemp -d "$log_dir/attempt.XXXXXX")\n'
                    "nohup bash -c "
                    + shlex.quote('bash "$1"; rc=$?; printf "%s\\n" "$rc" > "$2"; exit "$rc"') + " "
                    '_ "$runner" "$attempt/exit_code" > "$attempt/run.log" 2>&1 < /dev/null &\n'
                    'printf "%s\\n" "$!" > "$attempt/pid"\nprintf "%s\\n" "$attempt"\n')
        c1.atomic_write(exp_dir / "nohup.sh", lambda f: f.write(launcher))
    if manifest_rows:
        c1.write_csv(dest / "GPU_JOB_MANIFEST.csv", list(manifest_rows[0]), manifest_rows)
    print(f"Exported {len(manifest_rows)} job commands to {dest}")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("inspect", "plan", "run", "collect", "report", "status"))
    p.add_argument("--experiment", default="all", choices=(*EXPERIMENTS, "all"))
    p.add_argument("--scope", choices=("smoke", "pilot", "full"), default="pilot")
    p.add_argument("--run-id", default="rebuttal", help="safe run directory name")
    p.add_argument("--phase", choices=("prepare", "sweep", "select", "calibrate", "freeze", "test"))
    p.add_argument("--job", help="exact exported task ID for individual execution")
    p.add_argument("--seed", type=int, help="run only jobs for one planned seed")
    p.add_argument("--variant", help="run one arm or protocol variant during online evaluation")
    p.add_argument("--methods", help="comma-separated subset or all")
    p.add_argument("--settings", help="comma-separated Dataset:H subset")
    p.add_argument("--seeds", default="0", help="validation seeds; formal test uses 0,1,2")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    default_gpu = visible if re.fullmatch(r"[0-9]+(?:,[0-9]+)*", visible) else "0"
    p.add_argument("--gpus", default=default_gpu, help="comma-separated physical GPU indices")
    p.add_argument("--max-per-gpu", type=int, default=1)
    p.add_argument("--data-root", type=Path, default=c1.REPO / "data")
    p.add_argument("--output-root", type=Path, default=HOME / "runs")
    p.add_argument("--checkpoint-root", type=Path, help="shared content-addressed offline cache")
    p.add_argument("--export-root", type=Path, default=HOME / "generated_commands")
    p.add_argument("--report-root", type=Path, default=HOME / "reports/final")
    p.add_argument("--c1-root", type=Path, help="optional existing standalone C1 run to collect")
    p.add_argument("--max-online-steps", type=int, help="positive smoke cap; default smoke cap is maximum horizon plus four")
    p.add_argument("--time-match", action="store_true", help="add C5 serial validation timing calibration")
    p.add_argument("--parameter-proxy-size", type=int, default=128, help="C3/C4 sampled parameters per expert (0–256)")
    p.add_argument("--collect-capability-diagnostics", action="store_true",
                   help="document explicit C3/C4 diagnostics intent; already enabled for those experiments")
    p.add_argument("--full-encoder-proxy", action="store_true", help="retain full paired encoder representations")
    p.add_argument("--execute", action="store_true", help="authorize GPU execution only for run")
    p.add_argument("--dry-run", action="store_true", help="display commands even if --execute is present")
    p.add_argument("--resume", action="store_true", help="reuse verified completed work")
    p.add_argument("--export", action="store_true", help="write plan manifests and individual .sh jobs")
    args = p.parse_args(argv)
    try:
        seeds = list(map(int, args.seeds.split(",")))
        if len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= 2**32:
            raise ValueError("validation seeds must be unique unsigned 32-bit values")
        args.seeds = tuple(sorted(seeds))
        args.gpus = args.gpus.split(",")
        if (len(set(args.gpus)) != len(args.gpus) or not all(re.fullmatch(r"0|[1-9][0-9]*", g) for g in args.gpus)
                or args.max_per_gpu < 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id)):
            raise ValueError("invalid GPU slots or run-id")
        if args.max_online_steps is not None and (args.scope != "smoke" or args.max_online_steps < 1):
            raise ValueError("positive --max-online-steps is only allowed with --scope smoke")
        if not 0 <= args.parameter_proxy_size <= 256:
            raise ValueError("parameter proxy size must be in [0, 256]")
        if args.collect_capability_diagnostics and args.experiment not in ("C3", "C4", "all"):
            raise ValueError("capability diagnostics belong to C3/C4")
        if (args.seed is not None or args.variant is not None) and (
                args.action != "run" or args.phase in ("select", "freeze")):
            raise ValueError("seed/variant filters require a GPU job phase")
        if args.variant and args.phase == "prepare":
            raise ValueError("prepare shares checkpoints across variants; use method, setting, seed or job")
        if args.action == "run" and (args.experiment == "all" or not args.phase):
            raise ValueError("run requires one experiment and an explicit phase")
    except ValueError as exc:
        p.error(str(exc))
    return args


def main(argv=None):
    args = parse(argv)
    try:
        if args.action in ("collect", "report", "status"):
            from scripts import rebuttal_collect
            return rebuttal_collect.collect(args)
        registered = c1.read_json(CONFIG)
        experiments = EXPERIMENTS if args.experiment == "all" else (args.experiment,)
        plans = [(exp, manifest(exp, args, registered)) for exp in experiments]
        if args.action == "run":
            experiment, m = plans[0]
            if args.phase not in phases(m):
                raise ValueError(f"{args.phase} is not a phase for {experiment}")
            tasks = eligible_tasks(m, experiment, args.phase)
            if args.job or args.seed is not None or args.variant:
                tasks = [t for t in tasks if (not args.job or t["task_id"] == args.job)
                         and (args.seed is None or t["seed"] == args.seed)
                         and (not args.variant or args.variant in (
                             t["variant"], m["groups"][t["group_id"]]["variant"]))]
                if not tasks:
                    raise ValueError("no job matches the selected experiment/phase/options")
            if args.execute and not args.dry_run:
                return 0 if execute(root_for(args, m), m, args.phase, args, tasks) else 1
            command = command_for(args, experiment, args.phase, args.job)
            for name in ("seed", "variant"):
                value = getattr(args, name)
                if value is not None:
                    command += f" --{name} {shlex.quote(str(value))}"
            print(command + f" --gpus {','.join(args.gpus)} --max-per-gpu {args.max_per_gpu}")
            print(f"Dry run: {len(tasks)} GPU jobs")
        else:
            for exp, m in plans:
                root = root_for(args, m)
                checkpoints = planned(m, "prepare")
                print(json.dumps({"experiment": exp, "root": str(root), "environment": m["environment"],
                                  "source": m["source"], "checkpoint_cache": str(args.checkpoint_root or args.output_root / "_checkpoints"),
                                  "ready_checkpoints": sum(c1.read_completed(root, t) is not None for t in checkpoints),
                                  "counts": {phase: len(eligible_tasks(m, exp, phase)) for phase in phases(m)},
                                  "split_audit": {g["dataset"]: {key: g[key] for key in
                                                  ("bounds", "timestamps", "scaler_fit")} for g in m["groups"].values()}
                                                  if exp == "C6" else {},
                                  "notes": {"C4": "shares C3 Full jobs",
                                            "C5": "prospective controls; Table 10 source unavailable",
                                            "C7": "archived adaptation plus local partial TD"}.get(exp, "")}))
            if args.action == "plan" and args.export:
                export(args, plans)
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"rebuttal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
