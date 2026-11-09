"""Build the expanded audit from existing assets. Run from the repository root."""
import ast
import csv
import gzip
import itertools
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent
PREVIOUS = OUT.parent
NA = "UNKNOWN"
AUTH = dict(author_confirmed="", author_correction="", author_note="")
METHODS = ("PAVE", "dsof", "dyname", "er", "fsnet", "onenet", "proceed", "patchtst_dgrad")
DATASETS = ("ETTh2", "ETTm1", "WTH", "ECL")
HORIZONS = ("1", "24", "48")
SEEDS = ("0", "1", "2")
ABLATIONS = ("wo_online_correction", "wo_local_credit", "wo_version_awareness",
             "wo_direction_awareness", "wo_subspace", "unweighted_subspace", "wo_recovery")
ANOMALIES = {
    ("dsof", "ETTh2", "24"): (3.552, .443),
    ("proceed", "ETTh2", "48"): (17.438, .831),
    ("onenet", "WTH", "24"): (1.362, 1.261),
    ("fsnet", "ETTm1", "48"): (1.416, .067),
}


def read(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def previous(name):
    return read(PREVIOUS / name)


def write(name, rows):
    rows = [r | AUTH for r in rows]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval=NA)
        writer.writeheader()
        writer.writerows(rows)
    print(name, len(rows))


def known(value):
    return value not in (NA, "", None)


def numeric(value):
    return float(value) if known(value) else NA


def log_namespace(path):
    if not known(path) or "|" in path:
        return {}
    with Path(path).open("rb") as handle:
        text = handle.read(65536).decode(errors="replace")
    match = re.search(r"^Namespace\(.*\)$", text, flags=re.M)
    if not match:
        return {}
    try:
        return {k.arg: ast.literal_eval(k.value)
                for k in ast.parse(match[0], mode="eval").body.keywords}
    except (ValueError, SyntaxError):
        return {}


def timestamp(row):
    for key in ("logged_setting", "folder_label", "result_path"):
        match = re.search(r"(20\d{2})_(\d\d)_(\d\d)_(\d\d)_(\d\d)_", row[key])
        if match:
            y, m, d, hh, mm = match.groups()
            return f"{y}-{m}-{d} {hh}:{mm}", "run setting name, timezone unknown"
    match = re.search(r"(20\d{2})(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)", row["log_path"])
    if match:
        y, m, d, hh, mm, ss = match.groups()
        return f"{y}-{m}-{d} {hh}:{mm}:{ss}", "campaign directory, timezone unknown"
    return NA, NA


def inventory():
    rows, by_log = [], {}
    for index, row in enumerate(previous("EXPERIMENT_INVENTORY.csv"), 1):
        time, time_source = timestamp(row)
        args = log_namespace(row["log_path"])
        status = row["run_status"]
        if row["evidence_status"] == "CONTRADICTION":
            status = "conflicting_artifacts_possible_overwrite"
        elif "smoke" in row["log_path"] or row["run_status"] == "capped_or_smoke":
            status = "smoke_or_capped"
        elif status == "length_differs_from_HEAD_or_unknown":
            status = "historical_stream_or_truncated"
        elif status == "full_length_against_HEAD":
            status = "full_length_paper_selection_unknown"
        elif status == "missing_metrics_or_incomplete" and row["metrics_nonfinite"] == "6":
            status = "empty_metrics_placeholder"
        elif status == "missing_metrics_or_incomplete" and row["log_failure_marker"] not in (NA, "NONE"):
            status = "incomplete_failure_marker"
        fields = dict(experiment_id=f"RUN-{index:04}", run_timestamp=time,
                      timestamp_source=time_source, command=NA,
                      command_evidence="logged Namespace in " + row["log_path"] if args else NA,
                      dirty_state=row["git_dirty"], status=status,
                      lradj=args.get("lradj", NA), recorded_loss_arg=args.get("loss", NA),
                      recorded_gpu=args.get("gpu", NA), diagnostic_buffer=args.get("credit_diagnostic_buffer_size", NA),
                      recorded_seed_origin="run config" if known(row["seed_in_config"]) else (
                          "logged Namespace" if known(row["base_seed_in_log"]) else NA),
                      paper_adoption=NA)
        fields["configuration_verification_status"] = (
            "CONTRADICTION" if row["evidence_status"] == "CONTRADICTION"
            else "VERIFIED_RUN" if args and known(row["config_path"]) and row["metric_matches_log"] == "True"
            else "PARTIALLY_VERIFIED" if args or known(row["config_path"])
            else "MISSING_EVIDENCE")
        record = fields | row
        rows.append(record)
        if known(row["log_path"]):
            by_log[(row["log_path"], row["seed"])] = record
    repeats = {r["result_path"] for r in previous("REPEATED_SETTINGS.csv")}
    for row in rows:
        row["repeated_setting"] = row["result_path"] in repeats
    write("EXPERIMENT_INVENTORY.csv", rows)
    return rows, by_log


def candidate_rows(by_log):
    rows = previous("CAMPAIGN_PER_SEED.csv")
    for r in rows:
        i = by_log[(r["source_log"], r["seed"])]
        r["experiment_id"] = i["experiment_id"]
        r["run_status"] = i["status"]
        r["configuration_status"] = i["configuration_verification_status"]
        r["online_lr"] = i["online_learning_rate"]
        r["config_path"] = i["config_path"]
        r["archive_path"] = r["source_array"]
        if known(r["source_array"]):
            raw = np.load(r["source_array"], allow_pickle=False).reshape(-1)
            r["archived_mae"], r["archived_mse"] = map(float, raw[:2])
        else:
            r["archived_mae"], r["archived_mse"] = NA, NA
    return rows


def hyperparameters(inventory_rows, candidates):
    actual = {r["experiment_id"]: r["cohort"] for r in candidates if r["cohort"] not in ABLATIONS}
    output = []
    fields = (
        "experiment_id", "method", "dataset", "horizon", "seed", "variant", "feedback_protocol",
        "config_path", "log_path", "checkpoint_path", "commit", "learning_rate", "learning_rate_expert",
        "learning_rate_router", "online_learning_rate", "online_lr_router", "optimizer", "opt",
        "weight_decay", "expert_grad_clip", "router_grad_clip", "dgrad_grad_clip", "batch_size",
        "test_bsz", "n_inner", "replay_buffer_size", "replay_batch_size", "stable_buffer_size",
        "recovery_buffer_size", "recovery_batch_size", "dsof_student_lr", "dsof_td_weight",
        "adaptive_controller", "lradj", "pretrain_mode", "train_epochs", "patience",
        "recorded_loss_arg", "configuration_verification_status",
    )
    for r in inventory_rows:
        if r["experiment_id"] not in actual:
            continue
        method = actual[r["experiment_id"]]
        is_pave = method == "PAVE_full_candidate"
        current_optimizer = {
            "PAVE_full_candidate": "AdamW expert + AdamW router", "dsof": "AdamW teacher + Adam student",
            "dyname": "Adam gate", "er": "AdamW", "fsnet": "AdamW", "onenet": "AdamW",
            "proceed": "Adam adapter", "patchtst_dgrad": "AdamW",
        }[method]
        result = {k: r[k] for k in fields}
        result.update(row_type="actual_run", method="PAVE" if is_pave else method,
                      offline_optimizer_code="AdamW", online_optimizer_code=current_optimizer,
                      actual_online_schedule_code="dynamic for full updates; partial router inherits current LR"
                      if is_pave else "fixed base LR for applicable optimizer",
                      online_validation_metric=NA,
                      offline_checkpoint_selection_code="validation MSE early stopping when retrained",
                      verification_status=r["configuration_verification_status"],
                      selection_status="Final value verified; selection history unverified.")
        output.append(result)
    searches = [
        ("expert/router offline LR", "expert 1e-4|3e-4|1e-3|3e-3 × router 1e-4|3e-4|1e-3",
         12, "scripts/hp_search/run_multi_expert_hplr.sh:81"),
        ("expert/router offline LR with diversity", "(2e-3,2e-3)|(1e-3,1e-3)",
         2, "scripts/hp_search/run_multi_expert_hplr_div.sh:80"),
    ]
    for parameter, values, budget, source in searches:
        output.append(dict(row_type="search_design", method="historical multi_expert",
                           dataset="ETTm1", horizon="24", parameter=parameter, candidate_values=values,
                           validation_results=NA, selected_value=NA, search_budget=budget,
                           budget_basis="launcher candidates per setting, execution unverified",
                           selection_rule=NA, source=source, verification_status="VERIFIED_CODE"))
    for method in METHODS:
        output.append(dict(row_type="search_history", method=method, parameter="online LR",
                           candidate_values=NA, validation_results=NA, selected_value=NA, search_budget=NA,
                           selection_rule=NA, verification_status="MISSING_EVIDENCE",
                           source="scripts/hp_search; log/pretrain_*; log/*/*_full.out; git history"))
    write("HYPERPARAMETER_AUDIT.csv", output)


def main_results(candidates):
    lookup = {}
    for r in candidates:
        name = "PAVE" if r["cohort"] == "PAVE_full_candidate" else r["cohort"]
        key = name, r["dataset"], r["horizon"], r["seed"]
        assert key not in lookup, key
        lookup[key] = r
    mae_ref = {r["method"].lower().replace("-", "_"): r for r in read("mae_results.csv")}
    output = []
    for method, dataset, horizon, metric in itertools.product(METHODS, DATASETS, HORIZONS, ("mse", "mae")):
        setting = method, dataset, horizon
        row = dict(audit="A02", method=method, dataset=dataset, horizon=horizon, metric=metric,
                   paper_location="Table 1" if metric == "mse" else "Table 6",
                   paper_value=NA, paper_std=NA, recomputed_value=NA, difference=NA)
        values = []
        for seed in SEEDS:
            found = lookup.get(setting + (seed,), {})
            v = numeric(found.get("console_" + metric, NA))
            row.update({
                f"seed_{seed}": v,
                f"seed_{seed}_metrics_npy": found.get("archived_" + metric, NA),
                f"seed_{seed}_recomputed_from_predictions": found.get("array_" + metric, NA),
                f"seed_{seed}_experiment_id": found.get("experiment_id", NA),
                f"seed_{seed}_log": found.get("source_log", NA),
                f"seed_{seed}_metrics_path": found.get("source_array", NA),
                f"seed_{seed}_checkpoint": found.get("checkpoint", NA),
                f"seed_{seed}_online_lr": found.get("online_lr", NA),
                f"seed_{seed}_run_status": found.get("run_status", NA),
            })
            if known(v):
                values.append(v)
        mean = statistics.mean(values) if values else NA
        row.update(n_seeds=len(values), available_seed_mean=mean,
                   std_ddof0=statistics.pstdev(values) if values else NA,
                   std_ddof1=statistics.stdev(values) if len(values) > 1 else NA,
                   aggregate_basis="logged console values for explicit candidate campaign",
                   verification_status="PARTIALLY_VERIFIED",
                   paper_mapping="MISSING_EVIDENCE")
        if metric == "mse" and setting in ANOMALIES:
            paper, sd = ANOMALIES[setting]
            row.update(paper_value=paper, paper_std=sd,
                       single_seed_minus_reported_mean=values[0]-paper if len(values) == 1 else NA)
        if metric == "mae":
            target = mae_ref["pace" if method == "PAVE" else method][f"{dataset}_{horizon}"]
            row.update(local_mae_csv=target, available_seed_mean_minus_local_csv=mean-float(target),
                       seed0_round3_matches_local_csv=f"{row['seed_0']:.3f}" == target)
        output.append(row)
    write("MAIN_RESULTS_RECHECK.csv", output)
    with gzip.open(OUT / "A02_ANOMALY_ORIGIN_MSE.csv.gz", "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dataset", "horizon", "seed", "origin", "mse", "source",
                         *AUTH])
        for setting in ANOMALIES:
            found = lookup[setting + ("0",)]
            root = Path(found["source_array"]).parent
            p, y = (np.load(root / f"{name}.npy", mmap_mode="r") for name in ("preds", "trues"))
            for start in range(0, len(p), 512):
                err = (p[start:start+512].astype("float64") - y[start:start+512]).reshape(-1, np.prod(p.shape[1:]))
                for j, value in enumerate(np.mean(err**2, axis=1)):
                    writer.writerow([*setting, 0, start+j, value, str(root), "", "", ""])


def interval(count, n):
    if not n:
        return NA, NA
    p, z = count/n, 1.959963984540054
    mid = p + z*z/(2*n)
    span = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (mid-span)/(1+z*z/n), (mid+span)/(1+z*z/n)


def mechanisms(inventory_rows, candidates):
    full_roots = {str(Path(r["source_array"]).parent) for r in candidates
                  if r["cohort"] == "PAVE_full_candidate" and r["seed"] == "0"}
    output, pooled = [], defaultdict(lambda: [0, 0, set()])
    for r in inventory_rows:
        root = r["result_path"]
        p = Path(root) / "credit_diagnostics.npz"
        if not p.is_file():
            continue
        key = {k: r[k] for k in ("experiment_id", "dataset", "horizon", "seed", "variant", "commit")}
        key.update(source=str(p), paper_value=NA, recomputed_value=NA,
                   verification_status="VERIFIED_RESULT", paper_mapping="MISSING_EVIDENCE")
        with np.load(p, allow_pickle=False) as a:
            names = ("origin", "prediction_responsibility", "current_responsibility",
                     "prediction_expert_mse", "current_expert_mse", "capability_alignment")
            missing = set(names) - set(a.files)
            if missing:
                output.append(key | dict(audit="A05/A06", row_type="unavailable_archive",
                                          reason="missing " + "|".join(sorted(missing)), verification_status="MISSING_EVIDENCE"))
                continue
            origin, old_r, new_r, old_l, new_l, alignment = (a[name] for name in names)
        if not origin.size:
            output.append(key | dict(audit="A05/A06", row_type="unavailable_archive",
                                      reason="empty", verification_status="MISSING_EVIDENCE"))
            continue
        owner = old_r.argmax(axis=1)
        i = np.arange(len(owner))
        old, new, own_alignment = old_l[i, owner], new_l[i, owner], alignment[i, owner]
        relative = (new.astype("float64") - old) / (old + 1e-8)
        reversal = owner != new_r.argmax(axis=1)
        valid_loss = np.isfinite(relative) & np.isfinite(own_alignment)
        valid_r = np.isfinite(old_r).all(axis=1) & np.isfinite(new_r).all(axis=1)
        eligible_threshold = numeric(r["responsibility_threshold"])
        threshold = numeric(r["alignment_threshold"])
        groups = [("all", np.ones(len(i), dtype=bool))]
        if known(r["origins"]):
            stage = 3 * origin // int(r["origins"])
            groups.extend((f"third_{j+1}", stage == j) for j in range(3))
        for stage_name, stage_mask in groups:
            for group, match in (("high", own_alignment >= (.8 if threshold == NA else threshold)),
                                 ("low", own_alignment < (.8 if threshold == NA else threshold))):
                mask = stage_mask & match & valid_loss
                n = int(mask.sum())
                detail = dict(audit="A05", row_type="owner_degradation", stage=stage_name, alignment_group=group,
                              threshold=.8 if threshold == NA else threshold,
                              threshold_source="analysis fallback" if threshold == NA else "run metadata",
                              denominator=n, archived_records=len(i), unique_origins=len(np.unique(origin)),
                              first_origin=int(origin.min()), last_origin=int(origin.max()),
                              full_candidate_seed0=root in full_roots,
                              top1_eligible=int((mask & (old_r[i, owner] >= eligible_threshold)).sum())
                              if eligible_threshold != NA else NA)
                for label, cutoff in (("any", 0), ("gt5", .05), ("gt10", .1), ("gt20", .2), ("gt50", .5)):
                    count = int((mask & (relative > cutoff)).sum())
                    detail[label+"_count"] = count
                    detail[label+"_pct"] = 100*count/n if n else NA
                detail["no_count"] = n-detail["any_count"]
                detail["no_pct"] = 100*detail["no_count"]/n if n else NA
                output.append(key | detail)
        for kind, score in (("historical_owner", own_alignment), ("mean_expert", alignment.mean(axis=1))):
            for lo, hi in ((0, 1), (0, .25), (.25, .5), (.5, .75), (.75, 1)):
                mask = valid_r & (score >= lo) & ((score <= hi) if hi == 1 else (score < hi))
                n, k = int(mask.sum()), int((mask & reversal).sum())
                low, high = interval(k, n)
                output.append(key | dict(audit="A06", row_type="reversal_bin", alignment_kind=kind,
                                         lo=lo, hi=hi, denominator=n, count=k, rate=k/n if n else NA,
                                         wilson_low=low, wilson_high=high))
                if root in full_roots:
                    for h in (r["horizon"], "all"):
                        pool = pooled[h, kind, lo, hi]
                        pool[0] += n
                        pool[1] += k
                        pool[2].add(str(p))
    for (h, kind, lo, hi), (n, k, sources) in sorted(pooled.items()):
        low, high = interval(k, n)
        output.append(dict(audit="A06", row_type="pooled_full_candidate_seed0", horizon=h, seed=0,
                           alignment_kind=kind, lo=lo, hi=hi, denominator=n, count=k, rate=k/n if n else NA,
                           wilson_low=low, wilson_high=high, source="|".join(sorted(sources)),
                           verification_status="VERIFIED_RESULT", paper_mapping="MISSING_EVIDENCE"))
    for r in previous("MEMORY_EVENT_COUNTS.csv"):
        output.append(r | dict(audit="A07", row_type="logged_memory_events", verification_status="VERIFIED_RUN",
                               stable_admission_count=NA, reassessment_count=NA,
                               recovery_admission_counter=r["directional_recovery_admission_count"],
                               replay_attempt_counter=r["recovery_attempts"],
                               rebase_event_counter=r["capability_rebase_count"]))
    paper = [
        ("A05", "Table 8", "N", 10000), ("A05", "Table 8", "no_pct", 95.8),
        ("A05", "Table 8", "any_pct", 4.2), ("A05", "Table 8", "gt10_pct", 1.8),
        ("A05", "Table 8", "gt20_pct", .84), ("A05", "Table 8", "gt50_pct", .13),
        ("A06", "Figure 3(a)(c)", "N", 2000), ("A06", "Figure 3(a)(c)", "overall_pct", 49.9),
        ("A06", "Figure 3(a)(c)", "H1_pct", 21.0), ("A06", "Figure 3(a)(c)", "H24_pct", 44.7),
        ("A06", "Figure 3(a)(c)", "H48_pct", 84.1), ("A07", "Figure 3(b)", "N", 120),
        ("A07", "Figure 3(b)", "before_median", 1.47), ("A07", "Figure 3(b)", "after_median", 1.12),
        ("A07", "Figure 3(b)", "improved_pct", 97.5),
    ]
    for aid, location, metric, v in paper:
        output.append(dict(audit=aid, row_type="paper_claim", paper_location=location,
                           metric=metric, paper_value=v, recomputed_value=NA,
                           verification_status="NOT_AUDITABLE", source="author-provided task excerpt"))
    write("MECHANISM_STATISTICS.csv", output)


def ablation_and_control(candidates):
    lookup = {(r["cohort"], r["dataset"], r["horizon"], r["seed"]): r for r in candidates}
    output = []
    for variant, dataset, horizon, seed in itertools.product(ABLATIONS, DATASETS, HORIZONS, SEEDS):
        found = lookup.get((variant, dataset, horizon, seed))
        output.append(dict(audit="A11", row_type="ablation_per_seed", variant=variant,
                           dataset=dataset, horizon=horizon, seed=seed, paper_value=NA,
                           recomputed_value=found["array_mse"] if found else NA,
                           verification_status="VERIFIED_RESULT" if found and known(found["array_mse"])
                           else "PARTIALLY_VERIFIED" if found else "MISSING_EVIDENCE")
                      | (found or {}))
    for filename, row_type in (("ABLATION_SEED0_RECHECK.csv", "seed0_pair"),
                               ("ABLATION_HORIZON_RECHECK.csv", "equal_dataset_relative_change")):
        output.extend(r | dict(audit="A11", row_type=row_type, paper_value=NA,
                               recomputed_value=r.get("relative_change", r.get("mean_relative_change")),
                               verification_status="PARTIALLY_VERIFIED") for r in previous(filename))
    for r in previous("CAMPAIGN_STATS.csv"):
        if r["cohort"] in ABLATIONS:
            output.append(r | dict(audit="A11", row_type="ablation_available_seed_stats",
                                   paper_value=NA, recomputed_value=NA, verification_status="PARTIALLY_VERIFIED"))
    protocol = {r["result_dir"]: r for r in previous("pf_smoke_recheck/protocol_summary.csv")}
    for r in previous("pf_smoke_recheck/summary.csv"):
        output.append(r | protocol[r["result_dir"]] | dict(audit="A03", row_type="pf_smoke",
                      paper_value=NA, recomputed_value=NA, verification_status="VERIFIED_RUN"))
    for method in ("dsof", "dyname", "er"):
        output.append(dict(audit="A03", row_type="table2_pf", method=method,
                           paper_value=NA, recomputed_value=NA, verification_status="MISSING_EVIDENCE"))
    for variant, seed in itertools.product(("uniform_moe", "matched_moe", "full_pave"), SEEDS):
        output.append(dict(audit="A08", row_type="table10_per_seed_unmapped", variant=variant,
                           dataset=NA, horizon=NA, seed=seed, paper_value=NA, recomputed_value=NA,
                           verification_status="MISSING_EVIDENCE"))
    for variant, seed in itertools.product(("full", "no_direction", "no_recovery"), SEEDS):
        for metric in ("B_early", "B_late", "A2_early", "reacquisition_time", "cumulative_excess_error"):
            output.append(dict(audit="A11", row_type="synthetic_per_seed", variant=variant,
                               seed=seed, metric=metric, paper_value=NA, recomputed_value=NA,
                               verification_status="MISSING_EVIDENCE"))
    for metric, v in (("A2_early_reduction_pct", 32.9), ("reacquisition_reduction_pct", 40.6),
                      ("excess_error_reduction_pct", 51.8)):
        output.append(dict(audit="A11", row_type="synthetic_paper_claim", metric=metric,
                           paper_value=v, recomputed_value=NA, verification_status="NOT_AUDITABLE"))
    for variant, value in (("full_pave", 165), ("matched_moe", 125)):
        for metric in ("time_ms_per_origin", "total_parameters", "trainable_parameters",
                       "peak_gpu_memory", "persistent_state"):
            output.append(dict(audit="A10", row_type="table11_profile", dataset="ECL", horizon=48,
                               variant=variant, metric=metric, paper_value=value if metric == "time_ms_per_origin" else NA,
                               recomputed_value=NA, verification_status="NOT_AUDITABLE"))
    for r in candidates:
        if r["dataset"] == "ECL" and r["horizon"] == "48" and known(r["source_array"]) and known(r["origins"]):
            raw = np.load(r["source_array"], allow_pickle=False).reshape(-1)
            if raw.size == 6 and np.isfinite(raw[-1]):
                output.append(dict(audit="A10", row_type="historical_elapsed_proxy", variant=r["cohort"],
                                   dataset="ECL", horizon=48, seed=r["seed"], metric="whole_test_ms_per_origin",
                                   elapsed_seconds=float(raw[-1]), origins=r["origins"],
                                   recomputed_value=1000*float(raw[-1])/int(r["origins"]),
                                   source=r["source_array"], verification_status="VERIFIED_RESULT",
                                   paper_mapping="MISSING_EVIDENCE"))
    write("ABLATION_AND_CONTROL_RECHECK.csv", output)


if __name__ == "__main__":
    records, logs = inventory()
    runs = candidate_rows(logs)
    hyperparameters(records, runs)
    main_results(runs)
    mechanisms(records, runs)
    ablation_and_control(runs)
