"""Summarize explicit historical campaigns and diagnostic event counters."""
import csv
import json
import re
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path

from recheck import OUT, UNKNOWN, save_csv

FULL = {
    "20260828_154332_progressive_subspace": 0,
    "20260914_203827_progressive_subspace": 1,
    "20260914_203833_progressive_subspace": 2,
}
BASELINE = {
    "dsof": "dsof_20260903_000847",
    "dyname": "dyname_20260903_000847",
    "er": "er_20260903_125908",
    "fsnet": "fsnet_20260902_133552",
    "onenet": "onenet_20260902_133552",
    "proceed": "proceed_20260903_125908",
    "patchtst_dgrad": "patchtst_dgrad_20260903_175412",
}
ABLATION = {
    "wo_online_correction": "20260919_161920_progressive_subspace",
    "wo_local_credit": "20260907_115256_progressive_subspace",
    "wo_version_awareness": "20260912_160858_progressive_subspace",
    "wo_direction_awareness": "20260907_115338_progressive_subspace",
    "wo_subspace": "20260914_151423_progressive_subspace",
    "unweighted_subspace": "20260914_151409_progressive_subspace",
    "wo_recovery": "20260919_161923_progressive_subspace",
}


def read(name):
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    rows = read("EXPERIMENT_INVENTORY.csv")
    groups = defaultdict(list)
    source_rows = []
    memory = []
    checkpoint_links = defaultdict(list)
    repeats = defaultdict(list)
    for r in rows:
        method, dataset, horizon, seed, variant = (r[k] for k in ("method", "dataset", "horizon", "seed", "variant"))
        path = Path(r["log_path"])
        batch = path.parent.name
        if batch in FULL and str(FULL[batch]) == seed and variant == "full":
            cohort = "PAVE_full_candidate"
        elif batch == BASELINE.get(method) and seed == "0":
            cohort = method
        elif batch == ABLATION.get(variant) and seed == "0":
            cohort = variant
        else:
            cohort = None
        if seed != UNKNOWN and r["metrics_mse"] != UNKNOWN:
            repeat_key = method, dataset, horizon, seed, variant, r["feedback_protocol"]
            repeats[repeat_key].append(r)
        if cohort and horizon in {"1", "24", "48"} and r["evidence_status"] != "CONTRADICTION":
            fields = dict(cohort=cohort, dataset=dataset, horizon=horizon, seed=seed, variant=variant,
                          commit=r["commit"], checkpoint=r["checkpoint_path"],
                          array_mse=r["recomputed_mse"], array_mae=r["recomputed_mae"],
                          console_mse=r["log_mse"], console_mae=r["log_mae"],
                          source_log=r["log_path"], source_array=r["metrics_path"],
                          logged_result_path=r["logged_result_path"], git_dirty=r["git_dirty"],
                          origins=r["origins"], array_status=r["array_status"])
            source_rows.append(fields)
            if r["log_mse"] != UNKNOWN:
                groups[cohort, dataset, horizon].append(fields)
        d = Path(r["result_path"])
        summary = d / "online_diagnostics_summary.json"
        if summary.is_file():
            raw = json.loads(summary.read_text())
            counters = raw.get("counters", {})
            if not counters:
                counters = raw.get("counts", {})
            keys = ("method", "dataset", "horizon", "seed", "variant", "commit", "origins")
            memory.append({k: r[k] for k in keys} | dict(
                source=str(summary), completed_record_count=raw.get("iteration", {}).get("completed_record_count", UNKNOWN),
                diagnostic_unique_count=UNKNOWN) | {k: v for k, v in counters.items() if isinstance(v, (int, float))})
        cp = r["checkpoint_path"]
        if cp not in (UNKNOWN, "NONE"):
            checkpoint_links[Path(cp).parent.name].append(r["result_path"] if r["result_path"] != UNKNOWN else r["log_path"])
    save_csv("CAMPAIGN_PER_SEED.csv", sorted(source_rows, key=lambda r: (r["cohort"], r["dataset"], r["horizon"], r["seed"])))
    save_csv("MEMORY_EVENT_COUNTS.csv", memory)
    repeat_rows = []
    for key, runs in sorted(repeats.items()):
        if len(runs) < 2:
            continue
        fields = dict(zip(("method", "dataset", "horizon", "seed", "variant", "feedback_protocol"), key))
        for r in runs:
            repeat_rows.append(fields | {k: r[k] for k in (
                "result_path", "log_path", "checkpoint_path", "commit", "origins",
                "metrics_mse", "metrics_mae", "run_status", "result_evidence_status")})
    save_csv("REPEATED_SETTINGS.csv", repeat_rows)
    checkpoint_rows = []
    for path in sorted(Path("checkpoints").rglob("*.pth")):
        matches = checkpoint_links.get(path.parent.name, [])
        checkpoint_rows.append(dict(path=str(path), size_bytes=path.stat().st_size,
                                    matching_run_or_log="|".join(dict.fromkeys(matches)) or UNKNOWN,
                                    role=path.stem))
    save_csv("CHECKPOINT_ARTIFACTS.csv", checkpoint_rows)

    with Path("mae_results.csv").open(newline="", encoding="utf-8") as handle:
        published_mae = {r.pop("method").lower().replace("patchtst-dgrad", "patchtst_dgrad"): r
                         for r in csv.DictReader(handle)}
    summaries = []
    for (cohort, dataset, horizon), runs in sorted(groups.items()):
        runs.sort(key=lambda r: r["seed"])
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            print("Ambiguous cohort, excluded from mean", cohort, dataset, horizon, seeds)
            continue
        n = len(seeds)
        row = dict(cohort=cohort, dataset=dataset, horizon=horizon, seeds="|".join(seeds), n=n,
                   local_array_n=sum(r["array_status"] == "recomputed_agrees" for r in runs))
        for metric in ("mse", "mae"):
            vals = [float(r["console_" + metric]) for r in runs]
            row.update({metric+"_values": "|".join(map(str, vals)), metric+"_mean": stats.mean(vals),
                        metric+"_std_ddof0": stats.pstdev(vals),
                        metric+"_std_ddof1": stats.stdev(vals) if n > 1 else UNKNOWN})
        report_key = "pace" if cohort == "PAVE_full_candidate" else cohort
        target = published_mae.get(report_key, {}).get(f"{dataset}_{horizon}")
        row.update(mae_csv=target if target is not None else UNKNOWN,
                   mae_round3_matches=(f'{row["mae_mean"]:.3f}' == target) if target is not None else UNKNOWN)
        summaries.append(row)
    save_csv("CAMPAIGN_STATS.csv", summaries)
    deltas = []
    for (cohort, dataset, horizon), runs in sorted(groups.items()):
        if cohort not in ABLATION:
            continue
        full = [r for r in groups.get(("PAVE_full_candidate", dataset, horizon), []) if r["seed"] == "0"]
        if len(runs) != 1 or len(full) != 1:
            continue
        variant_mse, full_mse = float(runs[0]["console_mse"]), float(full[0]["console_mse"])
        deltas.append(dict(variant=cohort, dataset=dataset, horizon=horizon, seed=0,
                           baseline_mse=full_mse, ablation_mse=variant_mse,
                           relative_change=(variant_mse-full_mse)/full_mse,
                           full_log=full[0]["source_log"], ablation_log=runs[0]["source_log"],
                           same_checkpoint_recorded_string=full[0]["checkpoint"] == runs[0]["checkpoint"],
                           same_local_checkpoint=(Path(full[0]["checkpoint"]).is_file()
                                                  and Path(runs[0]["checkpoint"]).is_file()
                                                  and Path(full[0]["checkpoint"]).samefile(runs[0]["checkpoint"])),
                           same_checkpoint_filename=Path(full[0]["checkpoint"]).parent.name == Path(runs[0]["checkpoint"]).parent.name,
                           same_known_commit=full[0]["commit"] != UNKNOWN and full[0]["commit"] == runs[0]["commit"]))
    save_csv("ABLATION_SEED0_RECHECK.csv", deltas)
    agg = defaultdict(list)
    for d in deltas:
        agg[d["variant"], d["horizon"]].append(d)
    save_csv("ABLATION_HORIZON_RECHECK.csv", [
        dict(variant=v, horizon=h, n_datasets=len(ds), datasets="|".join(d["dataset"] for d in ds),
             mean_relative_change=stats.mean(d["relative_change"] for d in ds),
             same_known_commit_count=sum(d["same_known_commit"] for d in ds),
             same_checkpoint_filename_count=sum(d["same_checkpoint_filename"] for d in ds),
             same_local_checkpoint_count=sum(d["same_local_checkpoint"] for d in ds))
        for (v, h), ds in sorted(agg.items())
    ])
    print("Campaigns", Counter((r["cohort"], r["seed"]) for r in source_rows))
    print("MAE matrix match", Counter((r["cohort"], r["mae_round3_matches"]) for r in summaries))
    print("Memory keys", list(memory[-1]) if memory else [], "memory runs", len(memory))
    print("Ablation same commit, checkpoint", Counter((r["same_known_commit"], r["same_checkpoint_filename"]) for r in deltas))


if __name__ == "__main__":
    main()
