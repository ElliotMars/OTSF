# PAVE / OTSF Rebuttal 证据审计

## 范围与版本

- 起始工作区为 `master`，HEAD `367d19d0f65f545734c849dfd6af6e6a3c0c99ec`，`git status --porcelain=v1 --untracked-files=all` 无输出。Git 忽略 `result/`、`log/`、`checkpoints/`、`*.csv`、`*.pdf`、本地 baseline launcher、HP 和绘图脚本。证据搜索包含这些文件。此次仅在本审计目录写入。
- 查阅了本地分支及变更历史。渐进 credit/subspace 自 `594b4b9` 加入，方向感知及 Recovery 后续多次修改，recurring 工具自 `d9f486c` 加入，渐进 baseline 自 `eb1a7dc` 加入，HEAD 更新 run metadata。旁边的 `../PAVE` 是单提交 `ae06390bfb82d23be8712ccea9af5f737cc7648b` 的发布候选副本，两个本地日志子目录为空。也查了 `../paper`、`../tmp`、`/tmp` 中的项目和产物名称，以及 `../DynaME`、`../OneNet` 对应项目目录。
- 论文表图与 reviewer 主张来自任务中给出的摘录。待发表版本的完整 PDF/TeX、OpenReview 提交材料和制表 manifest 未定位。精确的表图映射、版本和运行采纳由作者确认。远程 OTSF URL 未能取回，原始 DSOF 方法以作者论文和官方实现为依据。[DSOF 论文](https://openreview.net/pdf?id=I0n3EyogMi) · [DSOF 官方仓库](https://github.com/yyalau/iclr2025_dsof)
- `VERIFIED_CODE` 表示源码路径已核对。候选运行用 `PARTIALLY_VERIFIED` 保留命令、版本或制表映射缺口。数组独立复算的数值用 `VERIFIED_RESULT`，适用范围为表中明确列出的原始文件。实际打印的 Namespace 是运行参数证据，和真实 shell argv 分开。原始输入路径、运行 seed、产物及缺口见 [实验索引](EXPERIMENT_INVENTORY.csv) 和 [统计复核](STATISTICAL_RECHECK.md)。

## 关键数值发现

- 主索引共 597 条，其中 462 个本地结果条目。431 份有效预测与真值独立复算匹配各自的 MAE/MSE。重建的 PAVE 12 个三种子候选组合每组只有 seed 0 可查本地数组，seed 1/2 有各自 console 指标。逐层诊断、版本不一致、短流及空结果见统计报告。
- 本地 `mae_results.csv` 的 96 格全部与归档 seed 0 日志逐格吻合。已找到的 PAVE 三 seed 候选均值仅吻合 4/12。优先向作者确认 Table 6 来源和聚合，见 MAN-002。现有证据支持本地汇总的来源线索，对提交表的结论仍待绑定。
- PAVE ETTh2-24 的候选 seed 0/1/2 console MSE 为 2.463477、2.664164、2.797185，均值 2.641609，population SD 0.137166，sample SD 0.167993。seed 1/2 在远端 `/autodl-tmp/online/` 的对应数组未挂载，本地复核采用 stdout。不同候选中的版本、checkpoint 和配置需核对。
- 本地四项 baseline 异常条目的 seed 0 已定位，提交表所示的 SD 仍缺完整 per-seed 证据。配对 replay、recurring、Matched MoE 与 profiler 的独立复算需要 MAN-005/006/008/009 的产物。

## A · Baseline 参数、公平性与 DSOF

| 方法 | Launcher / 运行参数字段 | HEAD 线上实际入口 | 重要差异 |
|---|---|---|---|
| PAVE | `scripts/run_progressive_credit_subspace.sh:156–236`，`learning_rate_expert/router` 和 `online_lr_expert/router` | `exp/exp_multi_expert.py:1007–1017,1735–1764,3216–3293,3432–3466` | Expert/Router 分立 AdamW，配置 weight decay，典型 clip 1.0/0.5。局部 Router 路径与完整 Expert/Router 更新分开，Recovery 加入 Expert loss。 |
| DSOF | `scripts/run_dsof.sh:10–13`，`baseline_online_lr`、`dsof_student_lr` | `exp/exp_dsof.py:1` → `exp/exp_stream_baselines.py:116–176,241–274` | teacher AdamW、student Adam，teacher replay 容量 128、batch 16 的本地命令，student TD weight 0.2 的日志参数。 |
| DynaME | `scripts/run_dyname.sh:19–30`，`dyname_online_lr` | `exp/exp_dyname.py:78–105,125–177,249–259` | AdamW 预训 backbone，Adam 更新 gate，clip 1.0；周期 KRR / concept state 影响适应。 |
| ER | `scripts/run_er.sh:10–12`，`baseline_online_lr` | `exp/exp_er.py:1` → `exp/exp_stream_baselines.py:148–176,267–274` | AdamW，将当前成熟样本与 replay loss 相加后单步；容量 128、batch 16 为本地 launcher 参数。 |
| FSNet / OneNet | `scripts/run_fsnet.sh:10–11`、`scripts/run_onenet.sh:10–11`，`baseline_online_lr` | `exp/exp_fsnet.py:1`、`exp/exp_onenet.py:1` → `exp/exp_stream_baselines.py:153–176,267–274` | AdamW，共享 baseline 更新路径，FSNet 状态更新和 OneNet 模型机制仍影响实际适应。共享路径给这两者分配了 buffer，更新目标不采样该 buffer。 |
| PROCEED | `scripts/run_proceed.sh:10–13`，`proceed_online_lr` | `exp/exp_proceed.py` → `exp/exp_stream_baselines.py:116–134,166–168,267–274` | AdamW 预训 backbone；线上冻结 backbone，Adam 更新 adapter，再调用 observe。 |
| PatchTST-DGrad | `scripts/run_patchtst_dgrad.sh:19–29`，`dgrad_online_lr`、`dgrad_grad_clip` | `exp/exp_patchtst_dgrad.py:54–83,99–108,139` | AdamW，实际循环使用 `n_inner`，典型 clip 1.0；梯度适应由模型和成熟批次路径决定。 |

- **VERIFIED_CODE / PARTIALLY_VERIFIED（6htc、fvv、ZDG6）**。运行参数先由 launcher 传 CLI，`main.py:124–182,393–405` 解析和补齐数据参数，`main.py:410–434,484–503` 确立 pretrain 和迭代 seed，再进入上述类。日志的 `opt='adam'` 是通用标志，AdamW 的实际构造见类方法。七个 9 月 baseline 批次的 84 条日志均显示：线上基础 LR 为 0.001，offline LR 对 ECL 为 0.003、其他三数据集为 0.001，batch 32、test batch 1、epoch 上限 15、patience 3、`n_inner=1`、weight decay 参数 0.001。Adam gate/adapter/student 的构造未传这个 decay。共享 baseline 更新分支不循环 `n_inner`，也没有显式 clip。实际参数、路径和缺项逐运行记录在 CSV。Offline `EarlyStopping` 的验证 loss 与在线 LR 搜索是两条证据链。
- **PARTIALLY_VERIFIED（6htc）**。PAVE launcher 设 Expert 基础 online LR 为 ETTh2/ETTm1 1e-4、WTH 5e-5、ECL 1e-5，Router 基础 1e-5。Launcher 给 ECL 3e-3 的 offline LR，其余 1e-3，日志逐次可核查。PAVE 分支的 `dynamic` 在配置和日志可见。源码按前次已成熟线上 loss 与 EMA 的比值缩放，`s=clip(1/sqrt(max(prev/(ema+eps),eps)),0.1,1)`，Expert 和完整 Router 乘 `s`，EMA 系数 0.9，平滑启用时 TSB alpha 也调整。局部 Router 路径复用 optimizer 当时 LR，通常沿用前一次完整更新设置；非有限梯度会跳步。梯度裁剪、Replay 权重、subspace gamma、修正步长与优化器动量共同决定实际更新。见 `exp/exp_multi_expert.py:815–850,1696–1769,3432–3466`。
- **MISSING_EVIDENCE（6htc）**。本地有 `scripts/hp_search/run_multi_expert_hp*.sh`、`scripts/dataset_horizon/`、`log/pretrain_*/pretrain_vali_loss.csv`、多轮结果和 offline epoch 验证日志。HP 入口偏旧版 multi_expert，预训练 CSV 对比 checkpoint 的验证 loss。缺少将每个 baseline 和 PAVE 的 online LR 候选、共同验证流、选择规则及最终运行连起来的记录。**可以验证最终参数，但无法验证这些参数当初如何选择。** 重复运行显示有多个候选，尚无法判定有无依据 test 指标进行选择。
- **VERIFIED_CODE / PARTIALLY_VERIFIED（ZDG6、6htc）**。原始 DSOF 用已成熟完整样本支持 slow replay，用当前可见的不完整监督和相邻预测的时间差分支持 fast stream。已归档 `log/dsof_20260903_000847/` 的 DSOF 使用 `legacy_delayed`，如 ETTh2-24 连到 `result/results286/dsof_ETTh2_pl24_olfull_optadam_tb1/itr_0`。本地 DSOF 的 student 更新发生在完整窗口出队后，优化当前 teacher 残差，H>1 时增加上一条已处理样本的 detached student 位移一致性。Teacher 从已入 buffer 的成熟样本均匀抽取，当前样本在更新后入 buffer。对于完整 N 个 origin、H 步、batch 1 的这条路径，预计 student `max(0,N-H)` 步，teacher `max(0,N-H-1)` 步；无逐步计数文件时这是代码推导。该本地适配与原始 fast stream 的监督时点有实质差异；已用该运行记录的干净 commit `d6a0c14` 复查同一分支。[原始 DSOF §4](https://openreview.net/pdf?id=I0n3EyogMi)
- **CONTRADICTION（限 HEAD 与指定入口的支持范围）**。`utils/run_config.py:19,129–155` 和 `scripts/run_progressive_baseline_control.sh:50–58` 均只接受 FSNet/OneNet/DynaME。仅改 `METHODS` 无法在 HEAD 运行 DSOF/ER + PF。确实找到三份 FSNet/OneNet/DynaME 短流 PF 诊断，尚未定位 Table 2 DSOF/DynaME/ER 的实际命令和结果。无法确认 Table 2 的 DSOF + PF 与主表 DSOF 的差异、fast/slow 更新预算和可见监督数。历史提交与 `../PAVE` 中的对应文件亦已搜索。见 MAN-003。

## C · Capability 与验证

- **VERIFIED_CODE（fvv、ZDG6、KaYu）**。`exp/exp_multi_expert.py:100–120,158–177,268–278,397–413`。FSNet sketch 使用时间池化 encoder 表示，FSNet-Time 使用 channel 池化，均 320 维。每 expert 使用固定 Gaussian buffer（除以 `sqrt(320)`），默认 32 维，默认独立 seed 2025，L2 归一化。存预测时 sketch，完整成熟时用同一历史输入和当前 expert 重算，alignment 是截断 cosine 后 `(1+cos)/2`。诊断和重评使用状态抑制上下文；状态真正改变的条件见 `models/ts2vec/fsnet_.py:100–112`。诊断 forward 不提供所有版本、随机层模式或浮点行为完全一致的运行证明。
- **VERIFIED_CODE**。相同 encoder 输入和参数下，单改预测头可以改变输出而保持 sketch。更早/后续 checkpoint 的持久状态和数值差异要单独核实。分类器 `utils/expert_memory.py:22–60` 先算 loss degradation，再在 alignment 高于阈值时返回 retained，即使 loss 已变差。计算过 loss 与把 loss 用作 decision gate 是不同的行为。见 `tests/test_capability_sketch.py`、`tests/test_directional_recovery.py`。
- **VERIFIED_CODE**。初次参考 loss 为预测时 expert MSE，低 alignment 且无性能降级的候选用已比较过的当前 sketch/loss rebase。周期重评先用旧参考分类，再迁移或 rebase。完整记录在当前优化步之前分类，在该步之后入 buffer，供未来步使用，见 `exp/exp_multi_expert.py:1781–1970,2054–2160,2450–2504`。诊断来自 admission 前，普通 exporter 是按运行的尾部 deque，最大 10,000。论文 10,000 样本的选择脚本仍待提供。
- **VERIFIED_CODE**。Reversal 比较历史责任最高 expert 和当前同一已成熟 record 责任最高 expert，跨专家平均 alignment 是内置 binned analyzer 的口径，历史 owner alignment 可另算。高 alignment 的 loss increase 是配对 MSE 变化，标签与 reversal 不同。见 `utils/credit_diagnostic_analysis.py:63–149` 和统计报告。

## D · Memory、Subspace、Recovery

- **VERIFIED_CODE**。完整成熟后，从历史 sample responsibility 取 top-K，再以责任阈值过滤（常见 1、0.3）。Stable credit 为 `confidence × responsibility × alignment`，Recovery credit 使用 `1-alignment` 并经 recent loss/失败惩罚排序。按 expert 管理，典型容量各 32，sample id 和 sketch cosine 近重复过滤、容量替换均会影响 admission。`utils/expert_memory.py:123–141,197–210,240–287,388–439`。周期基于完成数而非原始日志行数，`exp/exp_multi_expert.py:2370–2427`。
- **VERIFIED_CODE**。Stable 低 alignment 且有害时可降至 Recovery；低 alignment 且无降级可 rebase；Recovery 在性能恢复或 alignment 达标且 loss 低于门槛时尝试升回 Stable；Stable 拒收可导致已恢复样本被丢弃。Replay 失败达到上限会丢弃。周期重评默认不消耗 replay attempts。Stable、Recovery 的短周期迁移计数是事件，一个 record 在多个 expert 或重评里可重复出现。见 `utils/expert_memory.py:503–675,677–778`。
- **VERIFIED_CODE**。子空间使用真实头输入。FSNet 是末时间步 `[B,320]`；FSNet-Time 是头实际读取的 `[B,C,320]`，每 channel 权重为 sample weight/C。`C=Fᵀdiag(w)F/sum(w)` 为非中心化二阶矩。默认 launcher rank 16、最少 4 个 memory samples、每 100 个完整记录刷新。Unweighted 分支给 basis 用均匀样本权重，保护强度仍用真实 Stable credit mass。见 `exp/exp_multi_expert.py:2297–2359`、`utils/subspace_protection.py:184–269`。
- **VERIFIED_CODE**。保护乘子 `gamma=clip(1/(1+LR×lambda×mass/(mass+scale)), bounds)`，按 Expert 实际 optimizer LR 过滤头 weight 的并行梯度，保留垂直分量；bias 和 encoder 梯度没有该投影。没有 Stable evidence 时有效保护关闭。Recovery loss 和监督 loss 同次 backward，共用后续梯度过滤与 clip。AdamW 的最终参数步仍包含动量与 decoupled decay。见 `utils/subspace_protection.py:121–151,278–345`、`exp/exp_multi_expert.py:1446–1502,3216–3293`。
- **MISSING_EVIDENCE（fvv、ZDG6）**。Replay 从 Recovery 排序抽样，pre-step objective 有预测和 sketch loss，post-step 同 expert/record 重测后才更新 memory reference，见 `exp/exp_multi_expert.py:1170–1325`、`utils/recovery_learning.py`。现有逐 origin/累计计数无法提供 Figure 3(b) 的 120 组配对事件、当时未 rebase 的分母、采样和 improvement 判据，见 MAN-006。

## E · 消融、容量和开销

| 消融 | 源码层实际变化 | 解释边界 |
|---|---|---|
| w/o Online Correction | `disable_online_correction`，停止 correction 更新并将 correction 视为零 | partial Router 仍训练，改变实际 gating 和后续状态。 |
| w/o Local Credit | `local_credit_weight=0` | 关闭 partial Router 的局部 credit loss 权重，历史责任、事件和 sample credit 仍计算。 |
| w/o Version Awareness | `disable_version_awareness`，effective alignment=1，方向与 rebase 同时关 | 以有效 alignment 进入 Stable，Recovery 仍可在配置上启用，但正常低 alignment admission/migration 不再触发；credit/ranking 和子空间 evidence 随之改变。 |
| w/o Direction Awareness | `disable_directional_recovery` | Recovery 开启时，满足基本责任条件的低 alignment 候选可进 Recovery，实际还受去重/容量；关闭方向分类及 rebase。 |
| w/o Subspace Protection | `expert_update_strategy=plain`（候选日志） | 去掉 subspace 梯度过滤；核对具体 runs 的其余 flags。 |
| Unweighted Subspace | `disable_credit_weighted_subspace` | basis 样本权重变为均匀，gamma 的 evidence mass 仍加权。 |
| w/o Recovery | `disable_recovery` | 容量置零、无 replay/migration，保留 direction/rebase；有害低 alignment 初次 admission 跳过，已在 Stable 的有害样本在 refresh 中不会迁入 Recovery。 |

依据 `scripts/run_progressive_credit_subspace.sh:49–78,120–137`、`scripts/run_remaining_ablations_seed0.sh`、`exp/exp_multi_expert.py:550–576,642–675,1735–1753,2054–2160,2297–2359,2596–2623`。历史 9 月运行有重复启动。`log/ablations_seed0/wo_version_awareness.out` 与 `wo_recovery.out` 都指向 `20260912_160858_progressive_subspace`，逐设置日志使用相同文件名及重定向写入。结果 root 的编号也由并行进程各自扫描选择，存在相同 leaf 被复用的条件（`main.py:459–499`）。已确认日志路径冲突，且至少五个相关 leaf 的 stdout MSE 与对应数组 MSE 有差异。对应 JSON 与 stdout 对 Recovery 开关也不一致。例 `results381` 的 ETTm1-1 为 0.085187632 对 0.084934918。证据表将冲突标为 `CONTRADICTION`，保留原始出处并排除出单因素候选统计。尚不能从现存文件确定覆盖顺序。w/o Recovery 的候选用 9 月 19 日独立复跑。

**MISSING_EVIDENCE（ZDG6）**。现有 structural ablation、router uniform 初始化和早期 MoE/TSB 对照不足以确认 Appendix F.2 的 Uniform MoE/Matched MoE。未定位其运行入口或明确标记的 seed 0/1/2 原始产物。需要 offline 初始化和 expert/router 参数签名、每 origin 的更新 schedule、训练参数量、replay draw 及有效更新计数。ECL-48 的 Table 11 需要硬件、计时边界、同步与 warmup、峰值显存和 persistent state 测量原始记录。日志的整段 test wall-clock 只能作为粗粒度耗时证据。等参数量、等 update/replay 数和等 wall-clock 是三个独立对照。

## F · 划分与因果反馈

- **VERIFIED_CODE（KaYu、ZDG6）**。`data/data_loader.py:49–113,153–215` 的两套 ETT 类用相同 observation index。训练行 `[0,2880)`；验证 context 起点 `2880-L`，首 target 2880，验证 target 终点不含 3600；测试 context 起点 `3600-L`，target 范围 `[3600,min(14400,N))`，最后一个有效 origin 的首 target 为 `min(14400,N)-H`。正常实验 `label_len=0`，训练 scaler 和 time-feature scaler 都 fit 训练行。ETTh2 的训练/验证分别 120/30 天，ETTm1 15 分钟采样对应 30/7.5 天。本仓库的 conventional 12/4/4 个 30 天月划分是注释，minute 版本乘 4。相邻 `../DynaME/data_provider/data_loader.py:92–96,193–197` 有可用实现。当前 OTSF loader 无 split 开关，历史搜索未发现 OTSF 常规划分结果。WTH/ECL 走 custom 按 20% train、75% test、剩余 validation。旧 artifacts 的流长度另与 HEAD 核查。
- **VERIFIED_CODE**。滚动 stride=1。设 forecast origin 为上下文末尾相对索引，预测时只见最后一条 observation；旧 origin 的第 `h` 个 target 在 `origin+h` 释放，`H` 成熟后才完整学习。PAVE 每 origin 依次 release、逐事件修正、批量 partial Router 更新、完整记录更新、预测和入队；评估真值只用于计算分数。末尾待成熟记录留在队列，评分仍覆盖已有预测。Legacy delayed 预先保存 future target tensor，但完整窗口到期才交更新。PF baseline 的 masked feedback 允许更多已成熟监督，在 batch 化分支合并成至多一个 optimizer step/origin；PAVE 可同时做 partial Router 及完成记录引发的额外步骤。N 个 origin 的 PF 唯一位置数为 `sum_h max(0,N-h)`，而完整到期策略只释放 `H×max(0,N-H)` 个 horizon 位置，均未乘 channel。实际 loss 重用与 optimizer steps 需独立计量。见 `utils/progressive_feedback.py`、`utils/progressive_baseline_feedback.py`、`exp/exp_multi_expert.py:2586–2645`、`exp/exp_stream_baselines.py:178–265`、`exp/exp_dyname.py:153–177`。
- `tests/test_progressive_feedback.py`、`tests/test_progressive_online_order.py`、`tests/test_progressive_baseline_feedback.py` 覆盖释放、顺序及合并步数；三份实际四步诊断支持 H=2 的短流行为。主表级别仍需要各真实运行的 `origin, observation_timestamp, released_h, consumer, optimizer_step` 或可等价核查的轨迹，尤其是 DSOF/ER + PF 和尾部处理。

## G · Recurring

**VERIFIED_CODE / MISSING_EVIDENCE（ZDG6）**。`scripts/run_recurring_drift_maintext.sh:5–61` 默认 dry-run，seed 0/1/2、full/no_direction/no_recovery、H=24、L=60、C=3、各阶段 300，noise 0.05，separation 1.0。分离度 1 时，A/B 的 AR 为 0.72/-0.28，季节振幅 0.32/0.62、周期 24/9、趋势 0.08/-0.18，创新噪声 SD 都为 0.16。A2 复用 A 参数与连续生成的 A 潜在过程；绝对时间、相位和 realized noise 随时间推进。NumPy seed 控制 A/B 创新项、观测噪声，初始 AR 状态为零；Torch 也设 seed，基准的 expert 初值、bias、router 由固定张量构造。默认 launcher 的 A1/B/A2 时间戳边界为 0/300/600/900，共 817 个有效 origin，首目标时间戳为 origin+60，三个 origin 半开区间为 [0,240)/[240,540)/[540,817)。phase 以首目标归类。**Recurring 指标输入是 first-step MSE**，完整 horizon MSE 另存。使用 horizon-wide MSE 会改变统计定义。A1 reference 是最后 pre_window，B early/late 与 A2 early 取首/尾窗口；reacquisition 搜 A2 内最早一段连续达标的 rolling window 起点，允许 0，失败为 null。excess 为 A2 全区间 `sum(max(MSE-A1_ref,0))`。见 `data_provider/synthetic_drift.py:156–179,274–384,426–440`、`utils/drift_metrics.py:80–127,186–287`。原始 timeline 和 Table 9 版本未定位，三个百分比待复算。需确认窗口起点/终点、尾部裁切、未恢复 seed 及比率聚合口径。

## 本地数据身份和入口索引

| CSV | 行数（不含表头） | SHA-256 | 本地划分依据 |
|---|---:|---|---|
| ETTh2 | 17,420 | `a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b` | 验证起点 2016-10-29 00:00，测试起点 2016-11-28 00:00，最后使用标签 2018-02-20 23:00。 |
| ETTm1 | 69,680 | `6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e` | 验证起点 2016-07-31 00:00，测试起点 2016-08-07 12:00，最后使用标签 2016-11-27 23:45。 |
| WTH | 35,064 | `153fead3ea2a83dc5c4127d3994af6b167322f6f5e292c99491f43a2caf14cd4` | Custom 20%/剩余/75%。 |
| ECL | 26,304 | `0a96854fc5c6dd6775147eca20668ba571dde1848fcb2df470a666d1623f6362` | Custom 20%/剩余/75%。 |

- 预训练和结构对照入口包括 `scripts/run_multi_expert_pretrain_only.sh`、`run_multi_expert_moe_tsb_ablation_wth.sh`、`run_structural_ablations.sh`、`scripts/dataset_horizon/` 和 `scripts/hp_search/`。
- 诊断入口包括 `utils/credit_diagnostic_analysis.py`、`utils/evaluation_diagnostics.py`、`utils/online_diagnostics.py`、`utils/iteration_diagnostics.py`，图表脚本包括 `scripts/plot/`、`scripts/analyze_recurring_drift.py`。直接源码测试见 `tests/test_progressive_feedback.py`、`test_progressive_online_order.py`、`test_capability_sketch.py`、`test_directional_recovery.py`、`test_recovery_lifecycle.py`、`test_subspace_protection.py` 和 `test_synthetic_drift.py`。
- 轻量执行结果为 `tests/test_run_config.py` 15 passed。Torch 相关测试在现有 `text4ts` 环境 import `libtorch_cpu.so` 时因 `iJIT_NotifyEvent` 未定义而无法收集，动态模型行为以存档诊断与静态证据核查。数组统计使用同环境的 NumPy 2.2.5。
