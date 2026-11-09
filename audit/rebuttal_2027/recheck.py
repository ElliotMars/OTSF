"""Read existing artifacts and write only reproducible audit tables."""
import argparse
import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
UNKNOWN = "UNKNOWN"
BASELINES = {"dsof", "dyname", "er", "fsnet", "onenet", "proceed", "patchtst_dgrad"}
BASELINE_LR = {m: "baseline_online_lr" for m in BASELINES}
BASELINE_LR.update(dyname="dyname_online_lr", proceed="proceed_online_lr", patchtst_dgrad="dgrad_online_lr")
ONLINE_OPT = dict.fromkeys(BASELINES, "AdamW")
ONLINE_OPT.update(dsof="teacher AdamW / student Adam", dyname="Adam (gate)", proceed="Adam (adapter)")
DATA = {name: sum(1 for _ in open(Path("data", name + ".csv"), "rb")) - 1
        for name in ("ETTh2", "ETTm1", "ECL", "WTH")}


def save_csv(name, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval=UNKNOWN)
        writer.writeheader()
        writer.writerows(rows)


def get_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def relative_result(value):
    match = re.search(r"(?:^|/)(result/results\d+/[^/\s]+)", value)
    return Path(match[1]) if match else None


def namespace(line):
    try:
        return {arg.arg: ast.literal_eval(arg.value)
                for arg in ast.parse(line.strip(), mode="eval").body.keywords}
    except (ValueError, SyntaxError, AttributeError):
        return {}


def collect_logs():
    links = defaultdict(list)
    result_by_setting = defaultdict(list)
    for metrics in Path("result").glob("results*/*/metrics.npy"):
        result_by_setting[metrics.parent.name].append(metrics.parent)
    standalone = []
    for log in sorted(Path("log").rglob("*.out")) + sorted(Path("../PAVE/log").rglob("*.out")):
        # Split carriage-return progress lines too. Read logs one at a time.
        contents = log.read_text(errors="replace")
        args_match = re.search(r"^Namespace\(.*\)$", contents, re.M)
        args = namespace(args_match[0]) if args_match else {}
        if args:
            args["log_failure_marker"] = next(iter(re.findall(
                r"OutOfMemoryError|FileNotFoundError|KeyboardInterrupt|Traceback \(most recent call last\)|(?m:^Killed$)",
                contents
            )), "NONE")
        protocol = re.search(r"^Causal feedback protocol:\s*(\w+)", contents, re.M)
        if protocol:
            args["causal_feedback_protocol"] = protocol[1]
        parent = None
        iterations = {}
        metrics = []
        for match in re.finditer(
            r"run config: ([^\s]+)|RESULT_DIR: ([^\s]+)|"
            r">>>>>>>(start training|testing) : ([^<>\s]+)|"
            r"mse:([\d.eE+-]+),\s*mae:([\d.eE+-]+)", contents
        ):
            config_path, result_path, stage, setting, mse, mae = match.groups()
            if config_path or result_path:
                if result_path:
                    args["logged_result_path"] = result_path
                parent = relative_result(config_path or result_path) or parent
                if config_path and parent:
                    itr = re.search(r"/itr_(\d+)/run_config\.json$", config_path)
                    if itr:
                        iterations.setdefault(int(itr[1]), {})
            elif setting:
                if not iterations:
                    iterations[0] = {}
                entry = iterations[max(iterations)]
                entry["setting"] = setting
                if stage == "start training":
                    entry["trained"] = True
            elif mse:
                metrics.append((float(mse), float(mae)))
        if parent is None and len(iterations) == 1:
            setting = iterations.get(0, {}).get("setting")
            exact = result_by_setting.get(setting, [])
            if len(exact) == 1:
                parent = exact[0]
        log_row = (log, args, iterations, metrics)
        if parent and parent.exists():
            links[parent].append(log_row)
        elif args:
            standalone.append(log_row)
    return links, standalone


def settings(path, config):
    match = re.search(r"(.+?)_(ETTh2|ETTm1|WTH|ECL)_pl(\d+)_", path.name)
    label, data, horizon = match.groups() if match else (UNKNOWN, UNKNOWN, UNKNOWN)
    method = config.get("method", "multi_expert" if label.startswith("multi_expert") else label.removesuffix("_progfb"))
    return method, config.get("data", data), config.get("pred_len", int(horizon) if match else UNKNOWN), label


def variant(config):
    if config.get("method") != "multi_expert" or not config.get("progressive_fb"):
        return UNKNOWN
    flags = []
    for key, name in (
        ("disable_online_correction", "wo_online_correction"),
        ("disable_version_awareness", "wo_version_awareness"),
        ("disable_directional_recovery", "wo_direction_awareness"),
        ("disable_recovery", "wo_recovery"),
        ("disable_credit_weighted_subspace", "unweighted_subspace"),
        ("disable_expert_online_update", "frozen_expert"),
    ):
        if config.get(key):
            flags.append(name)
    if config.get("local_credit_weight") == 0:
        flags.append("wo_local_credit")
    if config.get("effective_expert_update_strategy", config.get("expert_update_strategy")) == "plain":
        flags.append("wo_subspace")
    required = {"disable_online_correction", "disable_version_awareness", "disable_directional_recovery",
                "disable_recovery", "disable_credit_weighted_subspace", "local_credit_weight",
                "disable_expert_online_update", "expert_update_strategy"}
    return "+".join(flags) or ("full" if required <= config.keys() else UNKNOWN)


def arrays(directory, method, data, horizon):
    metrics_path = directory / "metrics.npy"
    if not metrics_path.exists():
        return {}
    raw = np.load(metrics_path, allow_pickle=False).squeeze()
    if raw.ndim != 1 or raw.size < 2:
        return {"array_status": "aggregate_or_unrecognized_metrics"}
    out = dict(metrics_mae=float(raw[0]), metrics_mse=float(raw[1]),
               metrics_nonfinite=int((~np.isfinite(raw)).sum()))
    if not (directory / "preds.npy").exists() or not (directory / "trues.npy").exists():
        return out | {"array_status": "missing_prediction_or_target"}
    p = np.load(directory / "preds.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(directory / "trues.npy", mmap_mode="r", allow_pickle=False)
    if p.shape != y.shape or p.ndim < 2 or not p.size:
        return out | {"array_status": "shape_mismatch_or_empty"}
    if p.ndim == 3 and p.shape[0] == 1:
        p, y = p[0], y[0]
    n = len(p)
    out.update(origins=n, values=p.size)
    if data in DATA and isinstance(horizon, int):
        test_targets = min(DATA[data], 14400) - 3600 if data.startswith("ETT") else int(DATA[data] * .75)
        out["expected_origins_head"] = test_targets - horizon + 1
    delta = p.astype(np.float64) - y
    per_origin_mse = (delta.reshape(n, -1)**2).mean(axis=1)
    out.update(recomputed_mae=float(np.abs(delta).mean()),
               recomputed_mse=float(per_origin_mse.mean()),
               origin_mse_p99=float(np.quantile(per_origin_mse, .99)),
               origin_mse_max=float(per_origin_mse.max()))
    if not np.isfinite(delta).all() or not np.isfinite(raw[:2]).all():
        out["array_status"] = "nonfinite_mse_mae_or_arrays"
    elif np.allclose(raw[:2], [out["recomputed_mae"], out["recomputed_mse"]], rtol=3e-5, atol=1e-7):
        out["array_status"] = "recomputed_agrees"
    else:
        out["array_status"] = "recomputed_differs"
    return out


def row_for(directory, parent, index, log_info, use_arrays=True, saved=None):
    config_path = directory / "run_config.json"
    config = get_json(config_path)
    log, logged, iterations, logged_metrics = log_info
    c = logged | config
    compare = ("method", "data", "pred_len", "pretrain_mode", "progressive_fb", "online_learning",
               "disable_online_correction", "disable_version_awareness", "disable_directional_recovery",
               "disable_recovery", "disable_credit_weighted_subspace", "disable_expert_online_update",
               "expert_update_strategy", "local_credit_weight", "learning_rate",
               "online_lr_expert", "online_lr_router")
    conflicts = [key for key in compare if key in config and key in logged and config[key] != logged[key]]
    method, data, horizon, label = settings(parent, c)
    seed = config.get("finetune_model_seed")
    if not isinstance(seed, int):
        seed = config.get("seed", logged.get("seed", UNKNOWN) if index == 0 else UNKNOWN)
    checkpoint = c.get("pretrained_checkpoint") or UNKNOWN
    entry = iterations.get(index, {})
    if entry.get("trained") and not c.get("pretrain_mode"):
        c["pretrain_mode"] = "retrain"
    if c.get("pretrain_mode") == "retrain":
        setting = entry.get("setting")
        if setting and c.get("checkpoints"):
            checkpoint = str(Path(c["checkpoints"], setting, "checkpoint.pth"))
    if c.get("pretrain_mode") == "none":
        checkpoint = "NONE"
    if saved is not None:
        status = {k: saved[k] for k in ARRAY_FIELDS if saved.get(k, UNKNOWN) != UNKNOWN}
        for k in ("origins", "expected_origins_head", "values", "metrics_nonfinite"):
            if k in status:
                status[k] = int(status[k])
    else:
        status = arrays(directory, method, data, horizon) if use_arrays else {}
    if logged_metrics and index < len(logged_metrics):
        status["log_mse"], status["log_mae"] = logged_metrics[index]
    if c.get("max_online_steps", -1) > 0:
        run_status = "capped_or_smoke"
    elif status.get("origins") == status.get("expected_origins_head") and status.get("origins"):
        run_status = "full_length_against_HEAD"
    elif status.get("origins"):
        run_status = "length_differs_from_HEAD_or_unknown"
    elif c.get("skip_test"):
        run_status = "pretraining_only"
    else:
        run_status = "console_metrics_only" if logged_metrics else "missing_metrics_or_incomplete"
    protocol = c.get("causal_feedback_protocol", UNKNOWN)
    if protocol == UNKNOWN and logged.get("delay_fb") and not logged.get("progressive_fb"):
        protocol = "delay_fb_flag_in_log"
    opt = UNKNOWN
    if config.get("git_commit") and method in BASELINES | {"multi_expert"}:
        opt = ONLINE_OPT.get(method, "AdamW (expert/router)")
    row = dict(method=method, dataset=data, horizon=horizon, seed=seed,
               variant="AMBIGUOUS_CONFIG_LOG" if conflicts else variant(c),
               variant_in_log=variant(logged), variant_in_config=variant(config),
               config_log_conflicts="|".join(conflicts) or ("NONE" if config and logged else UNKNOWN),
               config_path=str(config_path) if config else UNKNOWN, checkpoint_path=checkpoint,
               metrics_path=str(directory / "metrics.npy") if (directory / "metrics.npy").exists() else UNKNOWN,
               log_path=str(log) if log else UNKNOWN, commit=config.get("git_commit", UNKNOWN),
               run_status=run_status, evidence_status="PARTIALLY_VERIFIED", result_path=str(directory),
               folder_label=label, iteration=index, pretrain_mode=c.get("pretrain_mode", UNKNOWN),
               seed_in_config=config.get("seed", UNKNOWN), base_seed_in_log=logged.get("seed", UNKNOWN),
               logged_result_path=c.get("logged_result_path", UNKNOWN), logged_setting=entry.get("setting", UNKNOWN),
               log_failure_marker=c.get("log_failure_marker", UNKNOWN),
               feedback_protocol=protocol, optimizer=opt, online_learning=c.get("online_learning", UNKNOWN),
               online_learning_rate=c.get(BASELINE_LR.get(method, "online_lr_expert"), UNKNOWN),
               checkpoint_exists=Path(checkpoint).is_file() if checkpoint not in (UNKNOWN, "NONE") else UNKNOWN)
    for key in ("learning_rate", "learning_rate_expert", "learning_rate_router", "online_lr_router",
                "weight_decay", "batch_size", "test_bsz", "n_inner", "train_epochs", "patience",
                "expert_grad_clip", "router_grad_clip", "dgrad_grad_clip", "adaptive_controller",
                "dsof_student_lr", "dsof_td_weight", "dsof_student_width", "replay_batch_size",
                "replay_buffer_size", "stable_buffer_size", "recovery_buffer_size", "recovery_batch_size",
                "alignment_threshold", "responsibility_threshold", "local_credit_weight", "opt",
                "max_online_steps", "git_dirty", "effective_expert_update_strategy"):
        row[key] = c.get(key, UNKNOWN)
    row.update(status)
    row["result_evidence_status"] = (
        "VERIFIED_RESULT" if status.get("array_status") == "recomputed_agrees"
        else "CONTRADICTION" if status.get("array_status") == "recomputed_differs"
        else "PARTIALLY_VERIFIED" if status.get("log_mse") is not None
        else "MISSING_EVIDENCE"
    )
    if "metrics_mse" in status and "log_mse" in status:
        row["metric_matches_log"] = bool(np.allclose(
            [float(status["metrics_mse"]), float(status["metrics_mae"])],
            [status["log_mse"], status["log_mae"]], rtol=3e-5, atol=1e-7
        ))
    if conflicts or row.get("metric_matches_log") is False:
        row["evidence_status"] = "CONTRADICTION"
    return row


def wilson(k, n):
    if not n:
        return UNKNOWN, UNKNOWN
    z = 1.959963984540054
    p = k/n
    shift = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return ((p + z*z/(2*n) - shift)/(1+z*z/n), (p + z*z/(2*n) + shift)/(1+z*z/n))


ARRAY_FIELDS = ("metrics_mae", "metrics_mse", "metrics_nonfinite", "array_status",
                "origins", "values", "expected_origins_head", "recomputed_mae",
                "recomputed_mse", "origin_mse_p99", "origin_mse_max")


def credit_rows(rows):
    strata, bins = [], []
    edges = (0, .25, .5, .75, 1)
    for row in rows:
        d = Path(row["result_path"])
        p = d / "credit_diagnostics.npz"
        if not p.exists():
            continue
        with np.load(p, allow_pickle=False) as a:
            required = {"origin", "prediction_responsibility", "current_responsibility",
                        "prediction_expert_mse", "current_expert_mse", "capability_alignment"}
            if not required <= set(a.files):
                strata.append(dict(source=str(p), status="missing " + ",".join(sorted(required-set(a.files)))))
                continue
            origin = a["origin"]
            n = origin.size
            if not n:
                continue
            old_resp, now_resp = a["prediction_responsibility"], a["current_responsibility"]
            owner = old_resp.argmax(axis=1)
            ridx = np.arange(n)
            reversal = owner != now_resp.argmax(axis=1)
            alignment = a["capability_alignment"]
            old = a["prediction_expert_mse"][ridx, owner]
            now = a["current_expert_mse"][ridx, owner]
            valid = np.isfinite(old) & np.isfinite(now) & np.isfinite(alignment[ridx, owner])
            relative = (now - old) / (old + 1e-8)
            fields = {k: row[k] for k in ("dataset", "horizon", "seed", "variant", "commit", "origins")}
            summary = get_json(d / "online_diagnostics_summary.json")
            fields.update(source=str(p), first_origin=int(origin.min()), last_origin=int(origin.max()),
                          retained_records=n, unique_origins=int(np.unique(origin).size),
                          raw_reversal_count=int(reversal.sum()),
                          saved_reversal_disagreements=int(np.sum(reversal != a["ranking_reversal"])) if "ranking_reversal" in a else UNKNOWN,
                          completed_records=summary.get("iteration", {}).get("completed_record_count", UNKNOWN))
            threshold = row["alignment_threshold"] if isinstance(row["alignment_threshold"], (float, int)) else .8
            total = row["origins"] if isinstance(row["origins"], int) else int(origin.max()) + 1
            owner_threshold = row["responsibility_threshold"]
            eligible = old_resp[ridx, owner] >= owner_threshold if isinstance(owner_threshold, (int, float)) else None
            direction = a["capability_evolution_code"][ridx, owner] if "capability_evolution_code" in a else None
            for label, stage in [("all", np.ones(n, dtype=bool))] + [
                (str(i+1), (origin*3//total) == i) for i in range(3)
            ]:
                for alignment_group, cond in (("high", alignment[ridx, owner] >= threshold),
                                              ("low", alignment[ridx, owner] < threshold)):
                    mask = stage & cond & valid
                    count = int(mask.sum())
                    counts = dict(denominator=count)
                    if eligible is not None:
                        counts["top1_responsibility_eligible_count"] = int((mask & eligible).sum())
                    if direction is not None:
                        for code, category in enumerate(("retained", "harmful", "beneficial_neutral")):
                            counts[category + "_owner_count"] = int((mask & (direction == code)).sum())
                    for name, cutoff in (("any", 0), ("gt10", .1), ("gt20", .2), ("gt50", .5)):
                        k = int(((relative > cutoff) & mask).sum())
                        counts[name+"_count"] = k
                        counts[name+"_rate"] = k/count if count else UNKNOWN
                    strata.append(fields | dict(stage=label, alignment_group=alignment_group, threshold=threshold,
                                                stage_record_count=int(stage.sum()), invalid_count=int((stage & ~valid).sum())) | counts)
            for kind, values in (("historical_owner", alignment[ridx, owner]), ("mean_expert", alignment.mean(axis=1))):
                for lo, hi in [(0., 1.)] + list(zip(edges[:-1], edges[1:])):
                    mask = (values >= lo) & ((values <= hi) if hi == 1 else (values < hi))
                    k, m = int((reversal & mask).sum()), int(mask.sum())
                    low, high = wilson(k, m)
                    bins.append(fields | dict(alignment_kind=kind, lo=lo, hi=hi, denominator=m, reversals=k,
                                              rate=k/m if m else UNKNOWN, wilson_low=low, wilson_high=high))
    return strata, bins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-arrays", action="store_true", help="reuse local array results and refresh metadata")
    args = parser.parse_args()
    cached = {}
    if args.reuse_arrays:
        with (OUT / "EXPERIMENT_INVENTORY.csv").open(newline="", encoding="utf-8") as handle:
            cached = {r["result_path"]: r for r in csv.DictReader(handle) if r["result_path"] != UNKNOWN}
    links, standalone = collect_logs()
    parents = {p.parent.parent if p.parent.name.startswith("itr_") else p.parent
               for name in ("metrics.npy", "run_config.json") for p in Path("result").rglob(name)}
    rows = []
    duplicates = []
    empty = (None, {}, {}, [])
    for parent in sorted(parents, key=lambda p: (int(p.parts[1][7:]), str(p))):
        log_options = links.get(parent, [empty])
        log_info = log_options[0] if len(log_options) == 1 else empty
        iterations = sorted(d for d in parent.glob("itr_*") if d.is_dir())
        if iterations:
            aggregate = parent / "metrics.npy"
            matches = UNKNOWN
            if aggregate.exists() and all((d / "metrics.npy").exists() for d in iterations):
                original = np.load(aggregate, allow_pickle=False)
                child = np.array([np.load(d / "metrics.npy", allow_pickle=False) for d in iterations])
                matches = bool(original.shape == child.shape and np.allclose(original, child, equal_nan=True))
            duplicates.append(dict(path=str(aggregate), iterations="|".join(str(d) for d in iterations),
                                   has_aggregate=aggregate.exists(), matches_iter_metrics=matches))
        for d in iterations or [parent]:
            index = int(d.name[4:]) if iterations else 0
            row = row_for(d, parent, index, log_info, saved=cached.get(str(d)))
            if len(log_options) > 1:
                row["log_path"] = "|".join(str(option[0]) for option in log_options)
            rows.append(row)
    for option in standalone:
        log, c, iterations, metrics = option
        for index in sorted(iterations) or [0]:
            row = row_for(Path("UNKNOWN"), Path(log.stem), index, option, False)
            row["result_path"] = UNKNOWN
            rows.append(row)
    save_csv("EXPERIMENT_INVENTORY.csv", rows)
    save_csv("AGGREGATE_ARTIFACTS.csv", duplicates)
    credit, bins = credit_rows(rows)
    save_csv("CAPABILITY_RECHECK.csv", credit)
    save_csv("REVERSAL_RECHECK.csv", bins)
    print("Rows", len(rows), "parents", len(parents), "standalone logs", len(standalone),
          "aggregate roots", len(duplicates), "credit files", len({r["source"] for r in credit}))
    print("Array status", Counter(r.get("array_status", "missing") for r in rows))
    print("Run status", Counter(r["run_status"] for r in rows))
    groups = defaultdict(Counter)
    for r in rows:
        if r["metrics_path"] != UNKNOWN and r["horizon"] in (1, 24, 48):
            groups[(r["method"], r["variant"], r["commit"], str(r["log_path"]).split("/")[1] if r["log_path"] != UNKNOWN else UNKNOWN)][r["seed"]] += 1
    for group, counts in sorted(groups.items()):
        if group[2] != UNKNOWN or group[1] != UNKNOWN:
            print(*group, dict(counts))


if __name__ == "__main__":
    main()
