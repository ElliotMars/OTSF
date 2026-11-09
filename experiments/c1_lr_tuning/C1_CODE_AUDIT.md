# C1 公平在线学习率选择

## 配置来源

本地 `master` 起始 HEAD 为 `367d19d0f65f545734c849dfd6af6e6a3c0c99ec`。历史 seed 0 候选多为 `d6a0c14`，本次执行使用当前工作树，manifest 保存实际 HEAD 与参与执行的源码内容哈希。逐设置从 96 份历史 stdout 的 normalized Namespace 恢复配置，来源路径随每组一起冻结。`native_profiles.json` 可独立供 runner 使用。Table 1 正式运行的逐格映射仍缺作者确认。

交叉核对了 `scripts/run_multi_expert.sh`、`scripts/run_progressive_credit_subspace.sh`、`scripts/run_multi_expert_pretrain_only.sh`、`scripts/run_progressive_baseline_maintext.sh`、`scripts/run_{dsof,dyname,er,fsnet,onenet,proceed,patchtst_dgrad}.sh`。PAVE 主候选使用 progressive、horizon-channel router、mixed 四专家、rank 16 的 subspace 和 dynamic controller。较早的 multi-expert launcher 有不同 router/feedback 与 checkpoint tag。PF matched-control launcher 也属于独立实验。C1 保持主候选协议。

| 方法 | Online optimizer 和原始基础 LR | Feedback | 生效位置 |
|---|---|---|---|
| multi_expert | AdamW Expert `online_lr_expert`，ETTh2/ETTm1 1e-4，WTH 5e-5，ECL 1e-5；AdamW Router `online_lr_router`，均 1e-5 | progressive | `exp/exp_multi_expert.py` 的 `_adaptive_online_hparams`、`load_pretrained`、`_ol_one_batch`、partial Router 更新 |
| dsof | AdamW teacher `baseline_online_lr=1e-3`；Adam student `dsof_student_lr=1e-3` | legacy_delayed | `exp/exp_stream_baselines.py` 的 `test`、`_online_update` |
| dyname | Adam gate `dyname_online_lr=1e-3` | legacy_delayed | `exp/exp_dyname.py` 的 `test`、`_update_gate_from_cache` |
| er | AdamW `baseline_online_lr=1e-3`，当前和 replay 损失同一步 | legacy_delayed | `exp/exp_stream_baselines.py` 的 `test`、`_online_update` |
| fsnet | AdamW `baseline_online_lr=1e-3` | legacy_delayed | 同共享 baseline 入口 |
| onenet | AdamW `baseline_online_lr=1e-3` | legacy_delayed | 同共享 baseline 入口 |
| proceed | Adam adapter `proceed_online_lr=1e-3`，冻结 backbone | legacy_delayed | `exp/exp_stream_baselines.py` 的 `test`、`_online_update` |
| patchtst_dgrad | AdamW `dgrad_online_lr=1e-3` | legacy_delayed | `exp/exp_patchtst_dgrad.py` 的 `_new_optimizer`、`test` |

DSOF 归档版本的 student 也等待整窗标签，再做残差及 TD 更新。Teacher 从成熟历史 replay。这个协议与原生 DSOF 的 incomplete-feedback fast stream 有差异，C1 保留本地归档协议。DynaME 的 KRR 与 PROCEED 的 `observe` 并非额外优化器 LR。

`main.py` 中的 `learning_rate` 经归一化对应 offline Expert 的 `learning_rate_expert`，独立的 `learning_rate_router` 供 offline Router 初始化。offline scheduler 使用 `learning_rate`。候选记录的 offline 基础 LR 为 ECL 0.003、其他 0.001，15 epochs、patience 3、batch 32、`lradj=type1`。C1 只对表中实际 online optimizer 字段应用同倍率。PAVE 完整反馈时控制器按已成熟误差与 EMA 得到 `clip(1/sqrt(ratio), 0.1, 1.0)` 并乘配置的两个基础 LR，partial Router 继承当时的 optimizer LR。Correction LR 0.1、memory、subspace、architecture 和 feedback 参数保持归档配置。结果同时保存缩放后的基础 LR 与真实 optimizer step 的次数、首末 LR 和最小最大值。

## 边界与因果性

原始目标范围是半开区间，val/test 各自借用 60 个历史 context 观察。Forecast origin 只在目标窗完全位于 split 内时产生。

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| ETTh2 | [0,2880) | [2880,3600) | [3600,14400) |
| ETTm1 | [0,2880) | [2880,3600) | [3600,14400) |
| WTH（35064 行） | [0,7012) | [7012,8766) | [8766,35064) |
| ECL（26304 行） | [0,5260) | [5260,6576) | [6576,26304) |

以下时间为原 CSV 各段首末目标行，末时间含在对应段内。

| Dataset | Train 首末 | Validation 首末 | Test 首末 |
|---|---|---|---|
| ETTh2 | 2016-07-01 00:00 → 2016-10-28 23:00 | 2016-10-29 00:00 → 2016-11-27 23:00 | 2016-11-28 00:00 → 2018-02-20 23:00 |
| ETTm1 | 2016-07-01 00:00 → 2016-07-30 23:45 | 2016-07-31 00:00 → 2016-08-07 11:45 | 2016-08-07 12:00 → 2016-11-27 23:45 |
| WTH | 2010-01-01 00:00 → 2010-10-20 03:00 | 2010-10-20 04:00 → 2011-01-01 05:00 | 2011-01-01 06:00 → 2013-12-31 23:00 |
| ECL | 2012-01-01 00:00 → 2012-08-07 03:00 | 2012-08-07 04:00 → 2012-09-30 23:00 | 2012-10-01 00:00 → 2014-12-31 23:00 |

`data/data_loader.py` 保留已有划分和 train-only scaler。在线 validation 走原 `flag="val"`，用 test 的 batch size 1、顺序、频率及同一在线循环和 metric。CSV 只把行解析到原 validation 结束，源文件哈希和物理记录数用于身份及 Custom 划分边界。legacy 只从延迟 H 的 FIFO 放行到期整窗，PAVE 从当前 context 最后一条观测逐 horizon 释放，保留原 correction、expert、memory 和 Recovery。末尾未成熟项留在队列。Offline early stopping 仍使用同一 validation 划分的原离线评估，随后在线 LR 选择使用完整的因果 validation replay。

## Checkpoint 与执行隔离

每个 method/dataset/H/seed 的缓存键包含归档 offline/architecture/feedback 配置、CSV 身份、实际源码和 commit，排除缩放的 online optimizer 字段。PAVE 的 `checkpoint.pth` 保存模型及 registered buffers，并同时要求 `optimizer.pth`，后者保留两个优化器的最佳 offline moments。FSNet 的 `grads`、`f_grads`、`q_ema`、`trigger` 为注册缓冲。其他方法通过原 `load_pretrained` 恢复 model state，再按原 replay 路径创建在线优化器。DSOF offline 只训练 teacher，student 在 fresh 实例中以同 seed 初始化。不同在线 arm 都重新设 seed、实例化和加载。运行级 Python 队列及缓存从空状态建立，在线 RNG 从显式 seed 起步。

每个 job 使用自己的目录和单 GPU 子进程。只有退出码为零、checkpoint 哈希匹配、metadata 完整且在线 origin 数正确的结果才会发布。默认不导入没有匹配元数据的旧 checkpoint。各验证 seed 的全部候选完成后，以平均 validation MSE 选取，容差内优先 log-distance 到 1.0 最小的倍率。冻结结果带 manifest 和 checkpoint 哈希。test 只读冻结文件，固定 0/1/2 两个 arm，report 要求全部正式 test 完整。报告 SD 采用 ddof=0。

## 实验命令

在仓库根目录使用满足 `requirement.txt` 且匹配 CUDA 的 Python。各阶段重复相同实验选项。默认 pilot 是 4 方法 × 3 settings，validation seed 0。GPU slot 由 `--gpus 0,1 --max-per-gpu 1` 控制，每个子进程内部为逻辑 `cuda:0`。

```bash
python scripts/c1_lr_sweep.py inspect --run-id c1-pilot
python scripts/c1_lr_sweep.py prepare --run-id c1-pilot --execute
python scripts/c1_lr_sweep.py sweep --run-id c1-pilot --execute
python scripts/c1_lr_sweep.py select --run-id c1-pilot --execute
python scripts/c1_lr_sweep.py test --run-id c1-pilot --execute
python scripts/c1_lr_sweep.py report --run-id c1-pilot --execute
```

所有阶段省略 `--execute` 时只显示计划。中断后可在相同命令附加 `--resume`。短流使用独立 run ID 并在各阶段重复 `--max-online-steps N`。Smoke 选择单独写入 `smoke_selected_lrs.json`，其 test 结果带 smoke 标记，report 拒绝该 run。

Pilot 含 36 checkpoints、60 validation trials、72 test trials。全量含 288 checkpoints、480 validation trials、576 test trials。增至两个 validation seeds 后 validation trials 翻倍，test 始终 0/1/2。

```bash
for phase in inspect prepare sweep select test report; do
  python scripts/c1_lr_sweep.py "$phase" --run-id c1-full --methods all --settings all \
    --gpus 0,1 --max-per-gpu 1 --execute || break
done
```

## 本轮检查

Python 语法、六阶段 CLI dry-run、pilot/全量预算和 Git diff 检查通过。CPU-only pytest 全套 230 项通过，覆盖八方法的真实短流在线入口、验证读取边界、迟到标签、optimizer 步进 LR、冻结选择、状态隔离、失败/未完成过滤及短流报告隔离。CPU 测试使用现有 Python 3.11 / PyTorch 2.6 环境，缺失依赖仅放入临时目录。A100 上的正式运行由作者发起。
