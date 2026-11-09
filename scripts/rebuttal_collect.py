#!/usr/bin/env python3
"""Collect verified rebuttal trials and render R1–R8 without running models."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import statistics
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import c1_lr_sweep as c1, rebuttal_suite as suite

TEST_SEEDS = set(c1.TEST_SEEDS)
RAW_FIELDS = (
    "experiment", "run_id", "scope", "phase", "task_id", "group_id", "stage", "split",
    "method", "variant", "protocol_variant", "parameter", "candidate", "dataset", "horizon", "seed", "split_mode",
    "feedback_protocol", "git_commit", "source_sha256", "manifest_hash", "selection_hash",
    "command", "checkpoint_path", "checkpoint_files", "checkpoint_key", "config_hash",
    "all_effective_online_lrs", "observed_online_lrs", "validation_mse", "validation_mae",
    "test_mse", "test_mae", "lr_scale", "calibration_budget", "runtime", "online_seconds",
    "mean_origin_ms", "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes", "parameters",
    "trainable_parameters", "replay_draws", "replay_records", "replay_buffer_records",
    "replay_buffer_tensor_bytes", "optimizer_steps", "mechanism",
    "memory_occupancy", "artifacts", "origins", "expected_origins", "hardware",
    "smoke", "status", "error", "log_path", "reused_from", "reuse_reason",
)
TITLES = {
    "R1": "Fair online learning rate tuning", "R2": "Hyperparameter sensitivity",
    "R3": "Loss-only validity and high-alignment guards", "R4": "High-alignment reliability",
    "R5": "Capability proxy comparison", "R6": "Capacity and compute",
    "R7": "Conventional ETT split", "R8": "DSOF feedback protocol",
}
PAPER_COLUMNS = {
    "R1": ("method", "dataset", "horizon", "selected_scale", "default_mse_mean", "default_mse_sd",
           "tuned_mse_mean", "tuned_mse_sd", "tuned_mse_change_pct"),
    "R2": ("parameter", "candidate", "dataset", "horizon", "mse_mean", "mse_sd", "relative_mse_change_pct"),
    "R3": ("dataset", "horizon", "variant", "mse_mean", "mse_sd", "relative_mse_change_pct"),
    "R4": ("dataset", "horizon", "seed", "stage", "stream_quartile", "denominator", "high_alignment",
           "degradation_gt_0pct_rate", "degradation_gt_10pct_rate", "degradation_gt_50pct_rate"),
    "R5": ("dataset", "horizon", "seed", "proxy", "status", "observations", "roc_auc",
           "average_precision", "recall", "pearson"),
    "R6": ("dataset", "horizon", "variant", "mse_mean", "mse_sd", "parameters_mean",
           "mean_origin_ms_mean", "replay_draws_mean", "peak_gpu_memory_bytes_mean"),
    "R7": ("dataset", "horizon", "method", "mse_mean", "mse_sd", "mae_mean", "mae_sd"),
    "R8": ("dataset", "horizon", "protocol_variant", "selected_scale",
           "default_mse_mean", "default_mse_sd", "tuned_mse_mean", "tuned_mse_sd"),
}
SEVERITY = (0, .05, .10, .20, .50)


def verified_manifest(path):
    manifest = c1.read_json(path)
    if c1.digest({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError(f"invalid manifest digest: {path}")
    return manifest


def trials(root, m, experiment, args, standalone=False):
    failures = c1.failure_rows(root)
    recent = {row["task_id"]: row for row in failures}
    phase_tasks = ([(phase, tasks) for phase, tasks in (
        ("prepare", c1.prepare_tasks(m)),
        ("sweep", [c1.request(m, g, s, "sweep", scale)
                   for g in m["groups"].values() for s in m["validation_seeds"] for scale in m["scales"]]),
        ("test", [c1.request(m, g, s, "test", variant=arm)
                  for g in m["groups"].values() for s in m["test_seeds"] for arm in ("default", "tuned")]),
    )] if standalone else [(phase, suite.eligible_tasks(m, experiment, phase)) for phase in suite.phases(m)])
    common = {"experiment": experiment, "run_id": m["run_id"], "scope": "standalone" if standalone else args.scope}
    rows = []
    for phase, tasks in phase_tasks:
        for t in tasks:
            group = m["groups"].get(t["group_id"], {})
            row = t | common | {
                "phase": phase, "protocol_variant": group.get("variant", t["method"]),
                "parameter": group.get("parameter"), "candidate": group.get("candidate"),
                "split_mode": group.get("split_mode", suite.PAPER),
                "git_commit": m["source"]["git_commit"], "source_sha256": m["source"]["source_sha256"],
                "status": "pending",
            }
            try:
                if standalone and phase == "test":
                    selection = c1.frozen_selection(root, m)
                    t = c1.test_tasks(root, m, selection, {t["task_id"]})[0]
                else:
                    t = suite.bind_task(root, m, phase, t)
            except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
                row.update(status="blocked_dependency", error=str(exc))
            else:
                row.update(t)
                result = c1.read_completed(root, t)
                if result:
                    row.update(result)
                    if phase != "prepare":
                        label = "test" if phase == "test" else "validation"
                        row.update({label + "_mse": result["mse"], label + "_mae": result["mae"]})
                        row["optimizer_steps"] = {
                            k: v["steps"] for k, v in result["observed_online_lrs"].items()
                        }
                elif c1.completion_path(root, t).exists():
                    row.update(status="invalid_completion", error="result or required artifact failed validation")
                elif t["task_id"] in recent:
                    row.update(recent[t["task_id"]])
            rows.append(row)
    task_ids = {row["task_id"] for row in rows}
    return rows, [common | row for row in failures if row["task_id"] in task_ids]


def completed_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        if row["phase"] == "test":
            key = (row["dataset"], row["horizon"], row["method"], row["protocol_variant"], row["variant"])
            groups[key].append(row)
    return {key: sorted(batch, key=lambda r: r["seed"]) for key, batch in groups.items()
            if len(batch) == len(TEST_SEEDS) and {r["seed"] for r in batch} == TEST_SEEDS
            and all(r["status"] == "completed" and not r["smoke"] for r in batch)}


def mean(batch, key):
    values = [r.get(key) for r in batch]
    return statistics.mean(values) if all(v is not None for v in values) else None


def metric(batch):
    return {f"{name}_{stat}": function([r["test_" + name] for r in batch])
            for name in ("mse", "mae") for stat, function in (("mean", statistics.mean), ("sd", statistics.stdev))}


def delta(value, reference):
    return 100 * (value/reference-1) if reference else None


def tuning(rows, m):
    groups = completed_groups(rows)
    val = {(r["group_id"], r["lr_scale"]): [] for r in rows if r["phase"] == "sweep"}
    for r in rows:
        if r["phase"] == "sweep" and r["status"] == "completed":
            val[r["group_id"], r["lr_scale"]].append(r["validation_mse"])
    result = []
    for (dataset, horizon, method, variant, arm), default in groups.items():
        if arm != "default":
            continue
        tuned = groups.get((dataset, horizon, method, variant, "tuned"))
        if not tuned:
            continue
        gid = default[0]["group_id"]
        scale = tuned[0]["lr_scale"]
        d = metric(default)
        t = metric(tuned)
        result.append(dict(dataset=dataset, horizon=horizon, method=method, protocol_variant=variant,
                           feedback_protocol=default[0]["feedback_protocol"],
                           default_lr=default[0]["all_effective_online_lrs"],
                           tuned_lr=tuned[0]["all_effective_online_lrs"], selected_scale=scale,
                           validation_seeds=m["validation_seeds"],
                           validation_mse_by_scale={str(s): statistics.mean(values)
                                                    for (group, s), values in val.items() if group == gid and values},
                           **{"default_" + k: v for k, v in d.items()}, **{"tuned_" + k: v for k, v in t.items()},
                           tuned_mse_change_pct=delta(t["mse_mean"], d["mse_mean"]),
                           tuned_mse_sd_change=t["mse_sd"]-d["mse_sd"],
                           tuned_online_steps_mean=step_means(tuned),
                           tuned_replay_draws_mean=mean(tuned, "replay_draws"),
                           tuned_replay_records_mean=mean(tuned, "replay_records")))
    return result


def step_means(batch):
    fields = sorted(set().union(*(r.get("optimizer_steps", {}) for r in batch)))
    return {key: statistics.mean(r["optimizer_steps"].get(key, 0) for r in batch) for key in fields}


def map_means(batch, name):
    fields = sorted(set().union(*(r.get(name, {}) for r in batch)))
    return {key: statistics.mean(r[name].get(key, 0) for r in batch) for key in fields}


def artifacts(row):
    return {Path(p).name: Path(p) for p in row["artifacts"]}


def diagnostic_rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def diagnosis(row):
    files = artifacts(row)
    metadata = c1.read_json(files["capability_sample.json"])
    sampled = {tuple(key) for key in metadata["sample_keys"]}
    counts = defaultdict(Counter)
    proxy_fields = {
        "sketch_32": ("alignment", -1), "parameter_sample_rms": ("parameter_sample_rms_distance", 1),
        "prediction_output_rmse": ("output_rmse", 1), "full_encoder": ("full_encoder_cosine", -1),
        "relative_loss_degradation_oracle": ("relative_degradation", 1),
    }
    observations = defaultdict(list)
    for case in diagnostic_rows(files["capability_unique.jsonl.gz"]):
        quartile = min(4, int(case["stream_position"] * 4) + 1)
        selected = (case["origin"], case["expert_id"]) in sampled
        stages = ["before_responsibility"]
        if case["eligible"]:
            stages.append("after_responsibility")
            for proxy, (field, direction) in proxy_fields.items():
                value = case.get(field)
                if value is not None:
                    observations[proxy].append((direction * value, case["relative_degradation"]))
        if case["candidate"]:
            stages.append("memory_candidate")
        if case["admitted"]:
            stages.append("admitted_" + case["admitted"])
        if selected:
            stages.append("new_hash_sample")
        for stage in stages:
            for position in (quartile, "all"):
                count = counts[stage, position]
                count["denominator"] += 1
                if case["high_alignment"]:
                    count["high_alignment"] += 1
                    for threshold in SEVERITY:
                        count[threshold] += case["relative_degradation"] > threshold
    identity = {key: row[key] for key in ("dataset", "horizon", "seed")}
    reliability = []
    stages = ("before_responsibility", "after_responsibility", "memory_candidate",
              "admitted_stable", "admitted_recovery", "new_hash_sample")
    for stage in stages:
        for position in ("all", 1, 2, 3, 4):
            count = counts[stage, position]
            high = count["high_alignment"]
            reliability.append(identity | dict(
                stream_quartile=position, stage=stage, denominator=count["denominator"], high_alignment=high,
                **{f"degradation_gt_{int(t*100)}pct_count": count[t] for t in SEVERITY},
                **{f"degradation_gt_{int(t*100)}pct_rate": count[t]/high if high else None for t in SEVERITY},
            ))
    refresh = Counter()
    events = reassessments = 0
    for case in diagnostic_rows(files["capability_refresh_events.jsonl.gz"]):
        if case["event"] == "owner_reassessment":
            reassessments += 1
            continue
        events += 1
        refresh.update({k: v for k, v in case.items()
                        if k not in ("dataset", "horizon", "seed", "event", "origin") and isinstance(v, int)})
    proxies = []
    for proxy in proxy_fields:
        pairs = observations[proxy]
        entry = identity | dict(proxy=proxy, observations=len(pairs), threshold=None,
                               positives=None, roc_auc=None, average_precision=None,
                               precision=None, recall=None, false_negative_rate=None, pearson=None,
                               recorder_seconds=metadata["diagnostic_seconds"],
                               recorder_fraction_of_online_time=(metadata["diagnostic_seconds"]/row["online_seconds"]),
                               parameter_snapshot_bytes=metadata["parameter_snapshot_bytes_per_pending_record"],
                               max_representation_bytes=metadata["max_representation_bytes_per_pending_record"])
        if proxy == "relative_loss_degradation_oracle":
            entry["status"] = "label_oracle"
        elif pairs:
            entry.update(proxy_metrics(pairs, -row["config"]["alignment_threshold"] if proxy == "sketch_32" else None),
                         status="available")
        else:
            entry["status"] = "not_available"
        proxies.append(entry)
    return reliability, proxies, identity | dict(refresh_ticks=events, owner_reassessments=reassessments, event_counts=dict(refresh),
                                                 unique_owner_rows=metadata["unique_owner_rows"])


def proxy_metrics(pairs, threshold):
    ordered = sorted(pairs, reverse=True)
    positives = sum(y > 0 for _, y in ordered)
    negatives = len(ordered) - positives
    tp = fp = ap = auc = 0
    start = 0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop][0] == ordered[start][0]:
            stop += 1
        group_tp = sum(y > 0 for _, y in ordered[start:stop])
        group_fp = stop - start - group_tp
        auc += group_tp * (negatives - fp - .5 * group_fp)
        tp += group_tp
        fp += group_fp
        ap += group_tp * tp/(tp + fp)
        start = stop
    predicted = [y for x, y in pairs if threshold is not None and x > threshold]
    hits = sum(y > 0 for y in predicted)
    try:
        pearson = statistics.correlation(*zip(*pairs))
    except statistics.StatisticsError:
        pearson = None
    return dict(positives=positives, roc_auc=auc/(positives*negatives) if positives and negatives else None,
                average_precision=ap/positives if positives else None, threshold=threshold,
                precision=hits/len(predicted) if predicted else None,
                recall=hits/positives if threshold is not None and positives else None,
                false_negative_rate=1-hits/positives if threshold is not None and positives else None,
                pearson=pearson)


def summarize(experiment, rows, m):
    if experiment in suite.TUNING:
        return tuning(rows, m)
    groups = completed_groups(rows)
    results = []
    for (dataset, horizon, method, variant, _), batch in groups.items():
        g = m["groups"][batch[0]["group_id"]]
        r = dict(dataset=dataset, horizon=horizon, method=method, variant=variant, **metric(batch))
        full = groups.get((dataset, horizon, method, "full", "full"))
        if full:
            r["relative_mse_change_pct"] = delta(r["mse_mean"], mean(full, "test_mse"))
        if experiment == "C2":
            r.update(parameter=g["parameter"], candidate=g["candidate"])
        elif experiment == "C3":
            admissions = Counter()
            per_seed = {}
            for trial in batch:
                counts = Counter()
                files = artifacts(trial)
                for case in diagnostic_rows(files["capability_unique.jsonl.gz"]):
                    if case["eligible"]:
                        counts["eligible"] += 1
                        counts[case["category"]] += 1
                        counts["high_alignment_degraded"] += case["high_alignment"] and case["degraded_by_gate_margin"]
                    if case["admitted"]:
                        counts["admitted_" + case["admitted"]] += 1
                admissions.update(counts)
                recovery = trial["mechanism"]
                attempts = recovery.get("recovery_attempts", 0)
                eligible = counts["eligible"]
                per_seed[trial["seed"]] = {
                    "counts": dict(counts),
                    "rates_per_eligible": {k: v/eligible for k, v in counts.items() if k != "eligible"} if eligible else {},
                    "recovery_attempts": attempts,
                    "recovery_success_rate": recovery.get("recovery_successes", 0)/attempts if attempts else None,
                    "recovery_failure_rate": recovery.get("recovery_failed", 0)/attempts if attempts else None,
                    "recorder_seconds": c1.read_json(files["capability_sample.json"])["diagnostic_seconds"],
                    "online_seconds": trial["online_seconds"],
                }
            r.update(mechanism_counts_mean=map_means(batch, "mechanism"),
                     first_maturity_counts_sum=dict(admissions), mechanism_by_seed=per_seed,
                     memory_occupancy_by_seed={x["seed"]: x.get("memory_occupancy") for x in batch},
                     optimizer_steps_mean=step_means(batch), replay_draws_mean=mean(batch, "replay_draws"))
        elif experiment == "C5":
            r.update(**{key + "_mean": mean(batch, key) for key in (
                "parameters", "trainable_parameters", "runtime", "online_seconds", "mean_origin_ms",
                "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes", "replay_draws", "replay_records",
                "replay_buffer_records", "replay_buffer_tensor_bytes")},
                optimizer_steps_mean=step_means(batch), n_inner=batch[0]["config"]["n_inner"],
                configured_moe_replay=batch[0]["config"].get("rebuttal_moe_replay"),
                hardware_by_seed={x["seed"]: x.get("hardware") for x in batch})
            if full and len({c1.digest(x["hardware"]) for x in full + batch}) == 1:
                r["relative_latency_change_pct"] = delta(r["mean_origin_ms_mean"], mean(full, "mean_origin_ms"))
        elif experiment == "C6":
            r.update(split_mode=g["split_mode"], bounds=g["bounds"], timestamps=g["timestamps"],
                     scaler_fit=g["scaler_fit"], origins=g["origins"])
        results.append(r)
    return results


def display(value):
    if value is None:
        return "NA"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return f"{value:.8g}" if isinstance(value, float) else str(value)


def emit(root, name, title, rows, fields=()):
    fields = tuple(dict.fromkeys([*fields, *(k for r in rows for k in r)]))
    c1.write_csv(root / f"{name}.csv", fields, rows)
    c1.write_json(root / f"{name}.json", rows)
    markdown = [f"# {title}", ""]
    tex = ["% " + title]
    if rows:
        def md(v):
            return display(v).replace("|", r"\|").replace("\n", " ")

        def latex(v):
            replacements = {"\\": r"\textbackslash{}", "_": r"\_", "%": r"\%", "&": r"\&",
                            "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}", "^": r"\textasciicircum{}",
                            "~": r"\textasciitilde{}"}
            return "".join(replacements.get(ch, ch) for ch in display(v))

        markdown += ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
        markdown += ["| " + " | ".join(md(r.get(k)) for k in fields) + " |" for r in rows]
        table_id = next((key for key, value in TITLES.items() if value == title), None)
        tex_fields = tuple(k for k in PAPER_COLUMNS.get(table_id, fields) if k in fields)
        tex += [r"\begin{tabular}{" + "l" * len(tex_fields) + "}", r"\hline",
                " & ".join(latex(k) for k in tex_fields) + r" \\", r"\hline"]
        tex += [" & ".join(latex(r.get(k)) for k in tex_fields) + r" \\" for r in rows]
        tex += [r"\hline", r"\end{tabular}"]
    else:
        markdown.append("No eligible completed results.")
        tex.append("% No eligible completed results.")
    for extension, lines in (("md", markdown), ("tex", tex)):
        c1.atomic_write(root / f"{name}.{extension}", lambda f: f.write("\n".join(lines) + "\n"))


def collect(args):
    requested = suite.EXPERIMENTS if args.experiment == "all" else (args.experiment,)
    registered = c1.read_json(suite.CONFIG)
    root = args.report_root.resolve()
    by_experiment = {}
    for experiment in requested:
        stored = args.output_root.resolve() / args.scope / args.run_id / ("C3" if experiment == "C4" else experiment)
        standalone = experiment == "C1" and not (stored / "manifest.json").exists() and args.c1_root
        if standalone:
            stored = args.c1_root.resolve()
        path = stored / "manifest.json"
        m = verified_manifest(path) if path.exists() else suite.manifest(experiment, args, registered)
        if standalone and m["smoke"]:
            raise ValueError("standalone C1 smoke cannot be collected as formal evidence")
        rows, failures = trials(stored, m, experiment, args, standalone)
        by_experiment[experiment] = (rows, m)
        print(f"{experiment} {dict(Counter(r['status'] for r in rows))} failures={len(failures)}")
        if args.action == "status":
            continue
        dest = root / experiment
        c1.write_csv(dest / "raw_trials.csv", RAW_FIELDS, rows)
        c1.write_csv(dest / "failure_attempts.csv", ("experiment", "run_id", "scope", *c1.FAILURE_FIELDS), failures)
        c1.write_csv(dest / "log_index.csv", ("task_id", "stage", "status", "log_path", "command"),
                     [r for r in rows + failures if r.get("log_path")])
        c1.write_json(dest / "parameters_and_provenance.json", m)
        selection_reader = None
        if experiment in suite.TUNING and c1.selection_path(stored, m).exists():
            selection_reader = c1.frozen_selection
            selection_file = "frozen_validation_lrs.json"
        elif experiment == "C5" and (stored / "frozen_time_budget.json").exists():
            selection_reader = suite.load_budget
            selection_file = "frozen_time_budget.json"
        if selection_reader:
            try:
                c1.write_json(dest / selection_file, selection_reader(stored, m))
            except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
                c1.write_json(dest / selection_file, {"status": "invalid_selection", "error": str(exc)})
        if experiment != "C4":
            number = {"C1": "R1", "C2": "R2", "C3": "R3", "C5": "R6", "C6": "R7", "C7": "R8"}[experiment]
            summary = summarize(experiment, rows, m)
            emit(dest, "summary", TITLES[number], summary)
            emit(root, number, TITLES[number], summary)
        else:
            r4, r5, refresh = [], [], []
            for row in rows:
                if (row["phase"] == "test" and row["status"] == "completed" and not row["smoke"]):
                    reliability, proxies, events = diagnosis(row)
                    r4.extend(reliability)
                    r5.extend(proxies)
                    refresh.append(events)
            for table, records in (("R4", r4), ("R5", r5)):
                emit(dest, table, TITLES[table], records)
                emit(root, table, TITLES[table], records)
            emit(dest, "refresh_event_summary", "Separate reassessment events", refresh)
    if args.action != "status":
        if {"C1", "C2"} <= by_experiment.keys():
            c1_rows, _ = by_experiment["C1"]
            lr_rows = [r for r in c1_rows if r["phase"] == "sweep" and r["status"] == "completed"
                       and not r["smoke"] and r["method"] == "multi_expert"]
            c1.write_csv(root / "C2" / "online_lr_sensitivity_from_C1.csv", RAW_FIELDS, lr_rows)
        if args.action == "report":
            notes = """# Rebuttal results

Formal predictive tables require verified uncapped test results for exactly seeds 0, 1, and 2 per arm. SD is sample standard deviation with denominator n−1. Negative relative MSE change means lower MSE. Raw rows retain dependency status, and failed attempts are listed separately.

R4 reports newly collected first-maturity expert-owner records. The denominator is stage-specific and severity rates use the high-alignment count in that stage. The hash sample contains up to 10,000 eligible high-alignment owners per stream. Refresh ticks have a separate event-level file. Historical Appendix E sample membership is unavailable.

R5 labels any relative loss increase as positive. ROC-AUC credits tied scores by one half. PR performance uses non-interpolated average precision with score ties grouped. The sketch decision uses its configured threshold, with equality assigned to high alignment. The observed loss degradation is a label oracle. Parameter distance uses bounded sampled parameters, and full encoder cosine is optional.

R6 reports measured optimizer calls, replay expert-record draws, replay records, wall time and memory. Cross-arm latency changes require identical recorded hardware. Prospective controls use the PAVE expert architecture. R7 uses separate conventional-split checkpoints. R8 names the local partial-TD adaptation separately from the archived DSOF.

C2 references C1 validation trials for online LR sensitivity. Empty tables are pending evidence.
"""
            c1.atomic_write(root / "README.md", lambda f: f.write(notes))
        print(f"Reports written to {root}")
    return 0


if __name__ == "__main__":
    sys.exit(suite.main(sys.argv[1:]))
