#!/usr/bin/env python3
"""Run exactly one isolated C1 training or replay job."""
import argparse
from contextlib import nullcontext
import importlib
import math
import os
import socket
import sys
import time
from pathlib import Path

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.c1_lr_sweep import LR_FIELDS, digest, file_hash, read_json, valid_files, write_json

OPTIMIZER_ATTRS = {
    "multi_expert": ("opt_expert", "opt_router"),
    "dsof": ("opt", "opt_student"),
    "dyname": ("opt_gate",),
    "er": ("opt",),
    "fsnet": ("opt",),
    "onenet": ("opt",),
    "proceed": ("opt",),
    "patchtst_dgrad": ("opt",),
}


def experiment_args(task, offline_root):
    train = task["stage"] == "prepare"
    return argparse.Namespace(**(task["config"] | {
        "root_path": str(Path(task["data"]["path"]).parent),
        "checkpoints": str(offline_root),
        "seed": task["seed"],
        "finetune_model_seed": task["seed"],
        "itr": 1,
        "pretrain_mode": "retrain" if train else "load",
        "pretrained_checkpoint": "" if train else task["checkpoint_path"],
        "checkpoint_tag": "c1",
        "max_online_steps": task["max_online_steps"],
        "online_validation_source_rows": task["data"]["rows"],
        "gpu": 0,
        "devices": "0",
        "use_gpu": True,
        "use_multi_gpu": False,
        "preserve_cuda_visible_devices": True,
    }))


def evaluate(exp, task, output_directory=None):
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    exp.load_pretrained(task["checkpoint_path"])
    fields = dict(zip(OPTIMIZER_ATTRS[task["method"]], LR_FIELDS[task["method"]]))
    observed = {name: {"steps": 0, "min": None, "max": None, "first": None, "last": None}
                for name in fields.values()}

    def record_step(optimizer, _args, _kwargs):
        for attr, field in fields.items():
            if getattr(exp, attr, None) is optimizer:
                lrs = [float(group["lr"]) for group in optimizer.param_groups]
                stat = observed[field]
                if stat["steps"]:
                    stat["min"] = min(stat["min"], *lrs)
                    stat["max"] = max(stat["max"], *lrs)
                else:
                    stat.update(min=min(lrs), max=max(lrs), first=lrs)
                stat.update(steps=stat["steps"]+1, last=lrs)
                return
        raise RuntimeError("online optimizer step has no C1 LR mapping")

    recorder_context = nullcontext()
    diagnostics = task.get("config", {}).get("rebuttal_collect_capability_diagnostics", False)
    if diagnostics:
        from utils.rebuttal_diagnostics import CapabilityRecorder
        recorder_context = CapabilityRecorder(output_directory, exp.args, exp.model.experts,
                                              task["expected_origins"])
    handle = register_optimizer_step_pre_hook(record_step)
    start = time.perf_counter()
    try:
        with recorder_context as recorder:
            if recorder is not None:
                exp.rebuttal_recorder = recorder
            metrics, _, _, preds, truth = exp.test("c1", split=task["split"])
    finally:
        handle.remove()
        if hasattr(exp, "rebuttal_recorder") and diagnostics:
            del exp.rebuttal_recorder
    if getattr(exp, "device", None) is not None and str(exp.device).startswith("cuda"):
        import torch
        torch.cuda.synchronize(exp.device)
    online_seconds = time.perf_counter()-start
    if len(preds) != task["expected_origins"] or preds.shape != truth.shape:
        raise ValueError(f"unexpected online output shape: {preds.shape}, expected {task['expected_origins']} origins")
    mae, mse = map(float, metrics[:2])
    if not all(math.isfinite(v) and v >= 0 for v in (mse, mae)):
        raise ValueError(f"invalid online metrics: MSE={mse}, MAE={mae}")
    result = dict(mse=mse, mae=mae, origins=len(preds), observed_online_lrs=observed,
                  online_seconds=online_seconds, mean_origin_ms=1000*online_seconds/len(preds),
                  replay_draws=getattr(exp, "rebuttal_replay_draws", None),
                  replay_records=getattr(exp, "rebuttal_replay_records", None))
    history = getattr(exp, "rebuttal_moe_history", None)
    if history is not None:
        result["replay_buffer_records"] = len(history)
        result["replay_buffer_tensor_bytes"] = sum(t.numel() * t.element_size() for row in history for t in row)
    memory = getattr(exp, "memory_manager", None)
    if memory is not None:
        result["mechanism"] = dict(exp.diagnostics.counters)
        result["memory_occupancy"] = [list(v) for v in memory.buffer_sizes()]
    model = getattr(exp, "model", None)
    if model is not None:
        params = list(model.parameters())
        if getattr(exp, "student", None) is not None:
            params.extend(exp.student.parameters())
        result.update(parameters=sum(p.numel() for p in params),
                      trainable_parameters=sum(p.numel() for p in params if p.requires_grad))
    if diagnostics:
        result["artifacts"] = {str(p): file_hash(p) for p in output_directory.glob("capability_*")}
    return result


def run(job, output_directory=None):
    task = job["task"]
    stage = task["stage"]
    if task["split"] != {"prepare": "train", "sweep": "val", "test": "test"}[stage]:
        raise ValueError("job stage and split disagree")
    if stage == "test" and not task["selection_hash"]:
        raise ValueError("test requires a frozen selection")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != job["physical_gpu"]:
        raise ValueError("worker must start with its assigned CUDA_VISIBLE_DEVICES")
    import torch

    from main import init_dl_program
    from utils.run_config import validate_online_feedback_protocol

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C1 workers require exactly one visible CUDA GPU")
    hardware = {"host": socket.gethostname(), "device": torch.cuda.get_device_name(0),
                "physical_gpu": job["physical_gpu"]}
    if "required_hardware" in task and task["required_hardware"] != hardware:
        raise ValueError("time-matched test must use the calibration GPU and host")
    offline_root = Path(job["offline_root"])
    args = experiment_args(task, offline_root)
    if validate_online_feedback_protocol(args) != task["feedback_protocol"]:
        raise ValueError("configured feedback protocol differs from the C1 manifest")
    init_dl_program(args, seed=args.seed)
    cls = importlib.import_module(f"exp.exp_{args.method}").Exp_TS2VecSupervised
    exp = cls(args)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if stage == "prepare":
        exp.train("state")
        checkpoint = offline_root / "state" / "checkpoint.pth"
        files = [checkpoint]
        if args.method == "multi_expert":
            files.append(checkpoint.with_name("optimizer.pth"))
        result = {"checkpoint_path": str(checkpoint),
                  "checkpoint_files": {str(p): file_hash(p) for p in files}}
    else:
        if not valid_files(task["checkpoint_files"]):
            raise ValueError("checkpoint changed or is incomplete")
        result = evaluate(exp, task, output_directory)
        result.update(checkpoint_path=task["checkpoint_path"], checkpoint_files=task["checkpoint_files"])
    torch.cuda.synchronize()
    fields = ("stage", "split", "method", "dataset", "horizon", "seed", "variant", "lr_scale",
              "config_hash", "checkpoint_key", "feedback_protocol", "smoke", "manifest_hash",
              "all_effective_online_lrs", "selection_hash")
    return {k: task[k] for k in fields} | result | dict(
        status="completed", request_hash=digest(task), runtime=time.perf_counter()-start,
        physical_gpu=job["physical_gpu"], device=hardware["device"], hardware=hardware,
        peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
        peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        git_commit=task["source"].get("git_commit"),
        split_mode=task["config"].get("split_mode", "paper_online_split"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="request file generated by the C1 orchestrator")
    parser.add_argument("--execute", action="store_true", help="execute one worker job")
    args = parser.parse_args()
    if not args.execute:
        parser.error("worker execution requires --execute")
    write_json(args.request.with_name("result.json"), run(read_json(args.request), args.request.parent))
