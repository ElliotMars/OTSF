import csv
import importlib
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data import data_loader
from exp.exp_dyname import Exp_TS2VecSupervised as DynaME
from exp.exp_multi_expert import Exp_TS2VecSupervised as PAVE
from exp.exp_patchtst_dgrad import Exp_TS2VecSupervised as DGrad
from exp.exp_stream_baselines import ExpER, ExpStreamBaseline
from scripts import c1_lr_sweep as c1
from scripts import c1_lr_worker as worker


def wth_csv(tmp_path, corrupt_test=False):
    path = tmp_path / "WTH.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "WetBulbCelsius", *(f"x{i}" for i in range(11))])
        for i in range(360):
            # For WTH the validation end is 360 - int(0.75 * 360) = 90.
            date = datetime(2020, 1, 1) + timedelta(hours=i)
            if corrupt_test and i >= 90:
                writer.writerow(["invalid date"] + ["test-only"] * 12)
            else:
                writer.writerow([date.isoformat(sep=" "), *(i+j+(i%7)*.2 for j in range(12))])
    return path


def replay_args(path, method="er"):
    config, _ = c1.native_settings(method, "WTH:24", c1.read_json(c1.PROFILE))
    task = {
        "config": config | {"pred_len": 3}, "data": {"path": str(path), "rows": 360},
        "seed": 0, "stage": "sweep", "checkpoint_path": str(path.with_suffix(".pth")),
        "max_online_steps": 5,
    }
    args = worker.experiment_args(task, path.parent)
    args.use_gpu = False
    return task, args


@pytest.mark.parametrize("cls", [PAVE, ExpER, DynaME, DGrad])
def test_online_validation_uses_val_bounds_and_does_not_parse_test_rows(tmp_path, monkeypatch, cls):
    path = wth_csv(tmp_path, corrupt_test=True)
    _, args = replay_args(path)
    exp = cls.__new__(cls)
    exp.args = args
    csv_limits = []
    original = data_loader.pd.read_csv

    def read(*args, **kwargs):
        csv_limits.append(kwargs.get("nrows"))
        return original(*args, **kwargs)

    monkeypatch.setattr(data_loader.pd, "read_csv", read)
    dataset, loader = exp._get_online_data("val")
    assert csv_limits == [90]
    assert dataset.set_type == 1 and len(dataset) == 16
    assert loader.batch_size == 1 and not loader.drop_last
    x0, _, _, _ = dataset[0]
    x1, _, _, _ = dataset[1]
    np.testing.assert_array_equal(x0[1:], x1[:-1])


def test_legacy_replay_updates_only_after_horizon_has_matured():
    exp = ExpStreamBaseline.__new__(ExpStreamBaseline)
    exp.args = SimpleNamespace(pred_len=3)
    exp.model = torch.nn.Identity()
    exp.student = None
    exp._forward = lambda x, _: x
    seen = []
    exp._online_update = lambda x, y, _: seen.append((int(x.item()), y.flatten().tolist()))
    queue = deque()
    for origin in range(6):
        t = torch.tensor([[[origin]]])
        truth = torch.arange(origin+1, origin+4).view(1, 3, 1)
        exp._delayed_online_batch(queue, t, truth, t)
        assert len(seen) == max(0, origin-2)
        if seen:
            assert seen[-1][0] == origin-3
            assert seen[-1][1] == list(range(origin-2, origin+1))
    assert len(queue) == 3


def test_fresh_candidate_models_and_optimizers_load_same_initial_state(tmp_path):
    path = tmp_path / "initial.pth"
    torch.save({"value": torch.tensor([3.0])}, path)
    starts, opt_ids = [], []

    class TinyExperiment:
        def load_pretrained(self, source):
            self.value = torch.nn.Parameter(torch.load(source, weights_only=True)["value"])

        def test(self, setting, *, split):
            assert split == "val"
            starts.append(self.value.item())
            self.opt = torch.optim.AdamW([self.value], lr=self.lr)
            opt_ids.append(self.opt)
            assert not self.opt.state
            self.value.sum().backward()
            self.opt.step()
            p = np.array([[self.value.item()]])
            return [abs(p.item()), p.item()**2], None, None, p, np.zeros_like(p)

    output = []
    for lr in (.001, .01):
        exp = TinyExperiment()
        exp.lr = lr
        output.append(worker.evaluate(exp, {"checkpoint_path": str(path), "method": "er",
                                            "split": "val", "expected_origins": 1}))
    assert starts == [3, 3] and opt_ids[0] is not opt_ids[1]
    assert output[0]["observed_online_lrs"]["baseline_online_lr"]["first"] == [.001]
    assert output[1]["observed_online_lrs"]["baseline_online_lr"]["last"] == [.01]


@pytest.mark.parametrize("method", c1.LR_FIELDS)
def test_cpu_online_replay_loads_a_checkpoint_through_normal_entrypoint(tmp_path, method):
    cls = importlib.import_module(f"exp.exp_{method}").Exp_TS2VecSupervised
    path = wth_csv(tmp_path, corrupt_test=True)
    task, args = replay_args(path, method)
    torch.manual_seed(0)
    base = cls(args)
    if method == "multi_expert":
        base.prepare_without_pretraining()
        torch.save({"expert": base.opt_expert.state_dict(), "router": base.opt_router.state_dict()},
                   Path(task["checkpoint_path"]).with_name("optimizer.pth"))
    torch.save(base.model.state_dict(), task["checkpoint_path"])
    task.update(method=method, split="val", expected_origins=5)
    result = worker.evaluate(cls(args), task)
    assert result["origins"] == 5 and np.isfinite(result["mse"])
    rates = result["observed_online_lrs"]
    assert tuple(rates) == c1.LR_FIELDS[method]
    assert all(stat["steps"] > 0 for stat in rates.values())
    if method == "dsof":
        assert rates["dsof_student_lr"]["steps"] == 2 and rates["baseline_online_lr"]["steps"] == 1


def _pave_rebuttal_checkpoint(tmp_path, steps=8):
    path = wth_csv(tmp_path, corrupt_test=True)
    task, args = replay_args(path, "multi_expert")
    args.max_online_steps = steps
    args.memory_refresh_interval = 1
    torch.manual_seed(0)
    exp = PAVE(args)
    exp.prepare_without_pretraining()
    torch.save(exp.model.state_dict(), task["checkpoint_path"])
    torch.save({"expert": exp.opt_expert.state_dict(), "router": exp.opt_router.state_dict()},
               Path(task["checkpoint_path"]).with_name("optimizer.pth"))
    task.update(method="multi_expert", split="val", expected_origins=steps)
    return task, args


def test_rebuttal_c4_recorder_is_observational_and_separates_events(tmp_path):
    from scripts.rebuttal_collect import diagnosis, diagnostic_rows

    task, args = _pave_rebuttal_checkpoint(tmp_path)
    args.rebuttal_parameter_proxy_size = 16
    args.rebuttal_full_rep_proxy = True
    plain, observed = PAVE(args), PAVE(args)
    torch.manual_seed(11)
    baseline = worker.evaluate(plain, task)
    torch.manual_seed(11)
    diagnostic_config = task["config"] | {"rebuttal_collect_capability_diagnostics": True}
    result = worker.evaluate(observed, task | {"config": diagnostic_config}, tmp_path)
    assert (baseline["mse"], baseline["mae"]) == (result["mse"], result["mae"])
    assert baseline["observed_online_lrs"] == result["observed_online_lrs"]
    for key, tensor in plain.model.state_dict().items():
        torch.testing.assert_close(tensor, observed.model.state_dict()[key], rtol=0, atol=0)
    assert baseline["memory_occupancy"] == result["memory_occupancy"]
    for left, right in zip(plain.model.modules(), observed.model.modules()):
        for name, value in vars(left).items():
            if isinstance(value, torch.Tensor) and hasattr(right, name):
                torch.testing.assert_close(value, getattr(right, name), rtol=0, atol=0)
    cases = list(diagnostic_rows(tmp_path / "capability_unique.jsonl.gz"))
    assert len(cases) == (task["expected_origins"] - args.pred_len) * observed.model.num_experts
    assert len({(r["origin"], r["expert_id"]) for r in cases}) == len(cases)
    assert all(r["event"] == "first_maturity" and r["parameter_sample_rms_distance"] is not None for r in cases)
    refresh = list(diagnostic_rows(tmp_path / "capability_refresh_events.jsonl.gz"))
    assert {r["event"] for r in refresh} == {"refresh_tick", "owner_reassessment"}
    r4, r5, separate = diagnosis(result | diagnostic_config | {
        "config": diagnostic_config, "dataset": "WTH", "horizon": 3, "seed": 0,
    })
    assert r4[0]["denominator"] == len(cases)
    assert separate["refresh_ticks"] == sum(r["event"] == "refresh_tick" for r in refresh)
    assert any(r["proxy"] == "relative_loss_degradation_oracle" and r["status"] == "label_oracle" for r in r5)


@pytest.mark.parametrize("mode", ["matched", "uniform"])
def test_rebuttal_c5_counts_fifo_replay_and_router_steps(tmp_path, mode):
    from scripts.rebuttal_suite import moe_config

    task, args = _pave_rebuttal_checkpoint(tmp_path)
    config = moe_config(mode, {"n_inner": 1, "rebuttal_moe_replay": 2})
    vars(args).update(config)
    exp = PAVE(args)
    result = worker.evaluate(exp, task)
    mature = task["expected_origins"] - args.pred_len
    assert result["replay_records"] == 2*mature-3
    assert result["replay_draws"] == result["replay_records"] * exp.model.num_experts
    assert result["observed_online_lrs"]["online_lr_expert"]["steps"] == mature
    router_steps = result["observed_online_lrs"]["online_lr_router"]["steps"]
    assert (router_steps == 0) == (mode == "uniform")


def test_rebuttal_c7_partial_td_ignores_evaluation_targets():
    class Forecaster(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.factor = torch.nn.Parameter(torch.tensor(.2))

        def forward(self, x):
            return self.factor * x[:, -1:, :].expand(-1, 3, -1)

    def run(target):
        exp = ExpStreamBaseline.__new__(ExpStreamBaseline)
        exp.args = SimpleNamespace(features="M", c_out=1, pred_len=3, dsof_td_weight=.5)
        exp.model, exp.student = Forecaster(), Forecaster()
        exp.opt = torch.optim.SGD(exp.model.parameters(), lr=.001)
        exp.opt_student = torch.optim.SGD(exp.student.parameters(), lr=.001)
        exp.rebuttal_replay_draws = 0
        exp.buffer = []
        exp._sample_replay = lambda: exp.buffer[0] if exp.buffer else None
        x = torch.arange(7.).reshape(7, 1, 1)
        pred, _ = exp._dsof_partial_td_batch(deque(), x, target, x)
        assert exp.rebuttal_replay_draws == 3
        assert exp.buffer[0][1].flatten().tolist() == [1., 2., 3.]
        return pred, exp.model.factor, exp.student.factor

    first = run(torch.full((7, 3, 1), float("nan")))
    second = run(torch.full((7, 3, 1), 100.))
    for x, y in zip(first, second):
        torch.testing.assert_close(x, y)


@pytest.mark.parametrize("dataset,cls,step", [("ETTh2", data_loader.Dataset_ETT_hour, 60),
                                             ("ETTm1", data_loader.Dataset_ETT_minute, 15)])
def test_conventional_loader_target_overlap_and_train_only_scaler(tmp_path, dataset, cls, step):
    from utils.rebuttal_splits import CONVENTIONAL, ett_bounds

    train, val, end = ett_bounds(step == 15, CONVENTIONAL)
    start = datetime(2020, 1, 1)
    with (tmp_path / (dataset + ".csv")).open("w") as f:
        f.write("date,OT\n")
        for i in range(end):
            timestamp = start + timedelta(minutes=step*i)
            f.write(f"{timestamp},{i}\n")
    kwargs = dict(root_path=tmp_path, data_path=dataset + ".csv", size=[5, 0, 3], split_mode=CONVENTIONAL)
    val_ds = cls(**kwargs, flag="val", validation_source_rows=end)
    test_ds = cls(**kwargs, flag="test")
    assert val_ds.scaler.mean == test_ds.scaler.mean == (train-1)/2
    assert len(val_ds) == val-train-2 and len(test_ds) == end-val-2
    context, y, _, _ = test_ds[0]
    np.testing.assert_allclose(test_ds.inverse_transform(context).ravel(), np.arange(val-5, val))
    np.testing.assert_allclose(test_ds.inverse_transform(y).ravel(), np.arange(val, val+3))
    np.testing.assert_allclose(test_ds.inverse_transform(test_ds[len(test_ds)-1][1]).ravel(), np.arange(end-3, end))
