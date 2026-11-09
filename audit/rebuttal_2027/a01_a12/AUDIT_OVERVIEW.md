# ICLR 2027 PAVE #43115 · A01–A12 只读审计

日期 2026-11-09（Asia/Shanghai）。工作区起始为 `master` / `367d19d0f65f545734c849dfd6af6e6a3c0c99ec`，Git 显示已有未跟踪 `audit/`，tracked diff 为空。可读范围包括 `result/`、`log/`、`checkpoints/`、`data/`、`scripts/`、本地 Git 历史及同级 `../PAVE`、`../DynaME`、`../OneNet`、`../paper`，并按名称搜查 `../tmp` 和 `/tmp`。日志指向的 `/autodl-tmp/online` 未挂载。公开检索未取得 #43115 论文，表图描述来自附件，作者需提供提交版和制表来源。原始 DSOF 依据其论文核对。[DSOF 原论文](https://openreview.net/pdf?id=I0n3EyogMi)

统计用 `text4ts` Python 3.10.0、NumPy 2.2.5。该环境安装的 PyTorch 为 2.4.1，导入因 `libtorch_cpu.so` 缺少 `iJIT_NotifyEvent` 失败。此前相同 HEAD 的配置测试 15 项通过。本轮执行只读统计和静态核对。新文件在本目录，原始产物、旧审计文件、代码及论文保留。复算入口为 [build_tables.py](build_tables.py)。

## 资产和风险摘要

- [实验清单](EXPERIMENT_INVENTORY.csv) 包含 597 条，462 条关联本地结果、135 条为独立日志线索。此前独立复算的 431 对预测/真值与保存 MSE/MAE 一致，216 个父汇总与逐迭代结果一致。覆盖原始 221 份配置、671 份含父汇总的 metrics、130 个 credit archive、3 份 PF 诊断及 414 个 checkpoint/optimizer 文件。早期流、截断、空指标、重复设置和配置冲突有单独标记。时间只取日志目录或 run setting 名称，时区未记录。原始 shell argv 缺失时 `command=UNKNOWN`，实际 Namespace 保留证据路径。可确定的配置与相符的日志归 `VERIFIED_RUN`，论文采用状态独立留空。
- 多个版本并存。可定位的主候选 baseline/PAVE seed 0 多为 `d6a0c14`。部分后续消融使用 `528aad6`、`d9f486c` 或 `367d19d`；同级发布候选 `../PAVE` 为 `ae06390`。一组 9 月 12 日并行消融共用日志路径，6 项开关冲突，5 项还存在 stdout 与数组 MSE 不同。见 [原先完整技术核查](../AUDIT_OVERVIEW.md)。
- 每节作者三字段留空。各 CSV 也有对应空列。作者选择最终采用版本和数值。

## A01 · Baseline 超参数选择与公平性 · P0

**状态**　`VERIFIED_RUN`（有匹配配置、日志的候选）；`PARTIALLY_VERIFIED`（仅日志的候选）；`MISSING_EVIDENCE`（历史选择）。

**事实**　[参数清单](HYPERPARAMETER_AUDIT.csv) 给出 120 条实际运行。七个 baseline 的 84 个候选全为 seed 0。各方法 offline 初始 LR 为 ECL 0.003、其他 0.001，日志为 batch 32/test batch 1、retrain、`n_inner=1`、epoch 上限 15、patience 3、offline AdamW、weight decay 0.001、`lradj=type1`。共享 scheduler 每 epoch 以 `args.learning_rate × 0.5^(epoch-1)` 设定 LR，PAVE 两个优化器也调用同一 helper。checkpoint early stopping 使用 offline validation MSE。见 `utils/tools.py:4–50`、`exp/exp_multi_expert.py:1007–1127`、`exp/exp_stream_baselines.py:116–146`。

| 方法 | 实际候选 online 基础 LR | 在线优化器、裁剪和有效 replay |
|---|---|---|
| PAVE | Expert ETTh2/ETTm1 1e-4、WTH 5e-5、ECL 1e-5，router 1e-5 | 两个 AdamW，clip 1.0/0.5。Stable/Recovery 容量各 32，Recovery batch 2。完整更新用 dynamic 缩放，partial router 继承 optimizer 当时 LR。 |
| DSOF | teacher 和 student 均 1e-3 | AdamW/Adam，此分支无显式梯度裁剪。Teacher replay 128/16，TD 0.2。 |
| DynaME | 1e-3 | Adam gate，clip 1.0，线上 backbone 冻结，KRR concept 路径独立。 |
| ER | 1e-3 | AdamW，replay 128/16，当前及 replay loss 同一步。 |
| FSNet / OneNet | 各 1e-3 | AdamW，当前成熟样本更新。全局 replay 参数在 Namespace 可见，该分支不采样 replay。 |
| PROCEED | 1e-3 | Adam adapter，冻结 backbone，`observe` 单独更新状态。 |
| PatchTST-DGrad | 1e-3 | AdamW，clip 1.0，线上显式循环 `n_inner`。 |

Adam gate、adapter、student 的构造未传入 0.001 decay。共享 Namespace 打印的 PAVE controller、router LR 或通用 replay 选项只有在对应方法路径被消费时才属于有效配置。见 `exp/exp_stream_baselines.py:148–176,267–274`、`exp/exp_dyname.py:125–177,249–259`、`exp/exp_patchtst_dgrad.py:54–108,139`。

**论文对应**　比较公平性依赖实际选择历史。ETTm1-24 历史 launcher 提供 12 个 offline Expert/Router LR 组合及另 2 个和 diversity 联动的组合。`log/pretrain_*/pretrain_vali_loss.csv` 有 offline validation MSE，例如 20260701 的 ECL-1/24/48 为 0.585669/1.014454/1.126318，未建立候选 LR 到这些值的配对。

**问题**　Insufficient evidence。共同因果 online validation、每方法真实预算、选值依据和 test-set tuning 都待查。**Final value verified; selection history unverified.**

**作者核对**　A01-CHECK-01，确认八种方法的实际采用运行、初始化及选择规则。见 CHECK-001、MAN-001。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可直接给出参数和更新路径。若选择日志无法恢复，公平在线 LR 搜索与 PAVE 敏感性需补。

## A02 · 主实验和异常方差 · P0

**状态**　`VERIFIED_RESULT`（本地 arrays）；`PARTIALLY_VERIFIED`（候选跨 seed）；`MISSING_EVIDENCE`（提交表映射和 baseline 其余种子）。

**事实**　[主结果 CSV](MAIN_RESULTS_RECHECK.csv) 有 192 行，分别覆盖 MSE/MAE 的 96 个设置。PAVE 12 个设置有三种子 console，只有 seed 0 有本地数组；七个 baseline 各 12 个候选仅 seed 0。PAVE ETTh2-24 的 0/1/2 MSE 为 2.463477/2.664164/2.797185，mean 2.641609，SD(ddof0/ddof1) 为 0.137166/0.167993。MAE 为 0.680910/0.694978/0.689364，mean 0.688417，SD 0.005782/0.007082。

| Method/setting | 附件 Table 1 | 本地 seed 0 MSE / MAE | N | P99 / max origin MSE | Result |
|---|---:|---:|---:|---:|---|
| DSOF ETTh2-24 | 3.552 ± 0.443 | 3.239262 / 0.767444 | 10777 | 79.887 / 837.866 | results286 |
| PROCEED ETTh2-48 | 17.438 ± 0.831 | 17.438402 / 0.969616 | 10753 | 228.421 / 5079.774 | results311 |
| OneNet WTH-24 | 1.362 ± 1.261 | 0.632996 / 0.523105 | 26275 | 2.524 / 208.865 | results278 |
| FSNet ETTm1-48 | 1.416 ± 0.067 | 1.340280 / 0.894333 | 10753 | 5.928 / 11.103 | results268 |

四组 seed 0 基础 online LR 均 1e-3、完整流长度与 HEAD 一致。各 checkpoint 和日志见 CSV。seed 1/2 原始数值保留 UNKNOWN。[逐 origin MSE](A02_ANOMALY_ORIGIN_MSE.csv.gz)

**论文对应**　`scripts/analyze_progressive_baseline_control.py:84–97` 使用 ddof=0，提交表 ddof 未证实。96/96 个本地 `mae_results.csv` 单元格吻合 seed 0 到三位，PAVE 候选三 seed 均值仅 4/12 吻合。ETTh2-24 本地 CSV 的 PACE/ER 均舍入为 0.681，原数值为 0.680909721/0.680869442。

**问题**　Confirmed issue（本地 seed 0 CSV 和该组 PAVE 跨 seed 聚合不同）。Insufficient evidence（Table 1/6 实际来源、异常原因和三种子方差）。旧 H=24 ETTh2 的 449 origins 符合旧 stride H 口径，现行是 10777。

**作者核对**　A02-CHECK-01，确认 Tables 1/6 每格 run ID、ddof、舍入、复跑与 checkpoint。见 CHECK-002/003、MAN-002。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可报告已验证历史数据和误差尖峰，不能据此归因训练发散或确定提交表统计。

## A03 · DSOF 原生反馈与 PF 对照 · P0

**状态**　`VERIFIED_CODE` / `VERIFIED_RUN`（本地 legacy）；`CONTRADICTION`（HEAD 入口支持与 Table 2 描述）；`MISSING_EVIDENCE`（实际 Table 2）。

**事实**

| 对照 | 本地 legacy DSOF | Table 2 DSOF + PF |
|---|---|---|
| 证据 | `result/results286/dsof_ETTh2_pl24_olfull_optadam_tb1/itr_0`，commit `d6a0c14`，`legacy_delayed` | 未定位实际命令、实现、配置或 traces。 |
| 反馈 | t 到期时先出队 t−H 的整窗，线上更新后预测 t 并入队 | HEAD launcher/validator 仅允许 FSNet、OneNet、DynaME。 |
| student | 成熟窗残差 MSE；H>1 时与上一 student 预测的重叠位置加 TD MSE，权重 0.2 | UNKNOWN |
| teacher | 从先前成熟窗均匀抽样 replay，之后才存当前成熟窗；容量 128，batch 16 | UNKNOWN |
| optimizer | 10,777 origins，H=24，按代码推算 student 10753 步、teacher 10752 步 | UNKNOWN |

原生 DSOF 快流利用新到的部分反馈和 TD，慢流使用完整反馈 replay。本地 TD 项保留移位一致性，但 student 的触发时间是整窗成熟。当前可验证的三个 PF smoke 是 ETTm1 H=2 seed 0，FSNet/OneNet/DynaME 各 4 origins、5 releases、3 optimizer steps、2 pending。见 `exp/exp_stream_baselines.py:148–176,241–274`、`utils/run_config.py:21,129–150`、`scripts/run_progressive_baseline_maintext.sh:6`。[原始 DSOF](https://openreview.net/pdf?id=I0n3EyogMi)

**论文对应**　附件 Table 2 为 DSOF/DynaME/ER，公开入口支持的集合不同。Table 1 是否选用归档 legacy 分支及 Table 2 对照的快慢流差异仍待确认。

**问题**　Confirmed issue（HEAD 入口不支持 DSOF/ER PF）；Insufficient evidence（提交实验版本及实际步数）。 
   
**作者核对**　A03-CHECK-01，确认正式 DSOF 两臂的源码、命令、steps/replay traces。见 CHECK-004、MAN-003。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可说明本地适配如何使用因果反馈。Table 2 的效果和训练预算需原始证据。

## A04 · Version-aware validation / Loss-only · P0

**状态**　`VERIFIED_CODE` / `PARTIALLY_VERIFIED`（候选配置），loss-only 对照 `MISSING_EVIDENCE`。

**事实**　预测时存各 expert 输出与 sketch；成熟 feedback 累积预测时误差并生成 sample responsibility。当前 expert 以同一 input 和已成熟 target 重算诊断，再按历史责任 top-K 与责任阈值选 owner。已见 full 候选为 top-1、责任阈值 0.3、alignment 0.8。Sketch 从 FSNet 时间池化、FSNet-Time 通道池化的 encoder 表征生成，固定高斯投影 320→32，L2 归一化，相似度 `(1+cosine)/2`。先分类与构造候选，再执行当前专家更新和 memory commit。见 `utils/progressive_feedback.py:192–279`、`exp/exp_multi_expert.py:100–182,268–278,397–413,1781–1964,2009–2210,2450–2504`。

| 分支 | 条件 | Reference / 迁移 |
|---|---|---|
| Retained | alignment ≥ 阈值 | 保留旧 reference；初次候选进 Stable，已有 Stable 保留。 |
| Beneficial/Neutral | alignment 低，current ≤ ref×(1+margin)+eps | 将 sketch 与 loss 同步 rebase，Stable 路径的有效 alignment 重置 1。 |
| Harmful | alignment 低，且 current > ref×(1+margin)+eps | 保留 reference，Recovery 可用时候选进 Recovery，已有 Stable 可迁入；初始有害候选在 Recovery 关闭时跳过。 |

各候选仍受责任、去重、容量和优先级约束。周期 reassessment 与 replay 后走相应迁移。见 `utils/expert_memory.py:22–59,388–441,468–778`。

| Variant | 关闭 | 保留与附带影响 |
|---|---|---|
| w/o Version Awareness | version gating、direction 与 rebase | 有效 alignment 固定 1，常规初始候选走 Stable，改变 credit/排名与证据质量。 |
| w/o Direction Awareness | 基于 loss 的分流及 rebase | 低 alignment 直接走可用的 Recovery 分支，高 alignment 保留 Stable。 |
| w/o Recovery | Recovery admission/replay/迁移 | 保留 direction 与可用的 beneficial rebase。 |

**论文对应**　Eq. 9–11 的附件语义与这些实现可逐项对照，公式原页和实验版本未绑定。旧日志和配置出现并行冲突。

**问题**　Potential issue。高对齐优先 retained，纯 head drift 可被 encoder sketch 忽略。未找到独立 loss-only。最小对照在 classification、admission、refresh、post-replay 引入统一 policy。Stable/Recovery 原排名分别用 `confidence×responsibility×alignment` 和 `confidence×responsibility×(1−alignment)`，高 alignment 的有害样本可能得极低 Recovery credit。为隔离验证条件可先保留原评分，记录 eligibility/实际 replay；另独立做 matched-budget 或评分敏感性，避免混用。

**作者核对**　A04-CHECK-01，确认公式、owner 阈值、rebase、三个变体及 loss-only 预算。见 CHECK-005、MAN-004。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可直接解释 gating 和消融的耦合；loss-only 独立效果需额外对照。

## A05 · 高对齐样本 · P0

**状态**　`VERIFIED_RESULT`（本地重算）；`NOT_AUDITABLE`（Table 8 样本映射）。

**事实**　130 份诊断 NPZ 中 125 份可用、3 缺字段、2 空。ETTh2-24 seed 0 候选 `results253` 有 10,000 条尾段记录，唯一 origin 753–10752。取每记录历史责任 argmax，先配同一记录旧/新 expert MSE，取高对齐 ≥0.8；分母 9,811，责任 ≥0.3 的 top-1 owner 其中 3,119。以 `(new-old)/(old+1e-8)` 严格比较，未按责任再过滤主分母。

| 类别 | 本地 count / 9811 | 附件 Table 8 / 10000 |
|---|---:|---:|
| No | 9063 / 92.376% | 9580 / 95.80% |
| Any | 748 / 7.624% | 420 / 4.20% |
| >5% | 529 / 5.392% | 未提供 |
| >10% | 396 / 4.036% | 180 / 1.80% |
| >20% | 254 / 2.589% | 84 / 0.84% |
| >50% | 101 / 1.029% | 13 / 0.13% |

**论文对应**　本地 exporter 在初次 memory admission 之前记录所有已成熟记录的 per-expert 字段，`deque(maxlen=10000)` 保存尾段。审计按 archive、dataset/H/seed、流位置三等分和 high/low 分层，记录缺覆盖或非有限的限制。论文的抽样范围、随机化、eligible 筛选、去重和参考 loss 版本尚未获得。见 `exp/exp_multi_expert.py:722–728,1781–1964,2802–3026`。[机制统计](MECHANISM_STATISTICS.csv)

**问题**　Insufficient evidence。Table 8 给出的计数由比例换算；本地数字用于说明可查的诊断，无法认定两者用同一记录。

**作者核对**　A05-CHECK-01，确认 Table 8 取样 manifest、责任阈值、是否跨运行、同一 owner loss 和脚本。见 CHECK-006、MAN-004。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可用本地配对直接展示高对齐漏报；论文百分比需对应记录。

## A09 · ETT 划分 · P0

**状态**　`VERIFIED_CODE`（代码、原始 CSV）；`PARTIALLY_VERIFIED`（论文采用和 conventional 结果）。

**事实**　两个 loader 都使用原始行号 `[0,2880)`、`[2880,3600)`、`[3600,14400)`，后两段仅前置 context 取 `seq_len` 历史行。当前候选 `seq_len=60,label_len=0`，val/test slice 起点为 2820/3540，首目标仍为 2880/3600。train-only scaler/time-feature scaler 依各代码路径拟合训练段。每段最后目标窗口必须完整在该段。见 `data/data_loader.py:49–113,153–215`。[原 CSV 哈希](../AUDIT_OVERVIEW.md)

| 数据 / 采样 | Train 首/末 | Validation 首/末 | Test 首/末 | Train/Val/Test |
|---|---|---|---|---|
| ETTh2 / 1h | 2016-07-01 00:00 / 2016-10-28 23:00 | 2016-10-29 00:00 / 2016-11-27 23:00 | 2016-11-28 00:00 / 2018-02-20 23:00 | 120/30/450 天 |
| ETTm1 / 15min | 2016-07-01 00:00 / 2016-07-30 23:45 | 2016-07-31 00:00 / 2016-08-07 11:45 | 2016-08-07 12:00 / 2016-11-27 23:45 | 30/7.5/112.5 天 |

**论文对应**　附件 B.1 的 2880/3600 和当前观察数切分一致。常用 12/4/4 个 30 天月 split 的行边界为 ETTh2 8640/11520/14400，ETTm1 34560/46080/57600。当前 OTSF loader 保留注释，邻库 `../DynaME/data_provider/data_loader.py:92–96,193–197` 有实现，未定位 OTSF 的对应运行。

**问题**　No issue found in audited scope（附件索引与当前数据的对应）；Potential issue（真实训练时长差异与 conventional 稳健性）。

**作者核对**　A09-CHECK-01，确认实际使用的 CSV、版本及边界。见 CHECK-010、MAN-007。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可直接回应数据时长和上下文。conventional 下效果需现有产物或额外实验。

## A06 · Historical-credit reversal · P1

**状态**　`VERIFIED_RESULT`（本地固定候选）；`NOT_AUDITABLE`（Figure 3(a)(c) 的原样本）。

**事实**　用同一已成熟记录的历史责任 argmax 与当前 argmax 比较。分组使用历史 owner 或各 expert 平均 alignment，bins 为 [0,.25)、[.25,.5)、[.5,.75)、[.75,1]。固定汇总范围为四数据集、三个 H、PAVE full 候选 seed 0，每 archive 10,000 条尾段记录。每 archive 的 dataset/seed/分箱、N、count、比例和 Wilson CI 见 [机制统计](MECHANISM_STATISTICS.csv)。

| Horizon | 本地 N | Reversal count | Rate / Wilson 95% | 附件 |
|---|---:|---:|---|---:|
| 1 | 40000 | 0 | 0% [0, 0.0096%] | 21.0% |
| 24 | 40000 | 25886 | 64.715% [64.245%, 65.182%] | 44.7% |
| 48 | 40000 | 26357 | 65.893% [65.426%, 66.356%] | 84.1% |
| All | 120000 | 52243 | 43.536% [43.256%, 43.817%] | 49.9%，N=2000 |

H=1 四个 archive 的 `expert_update_delta` 均为 0。实现先复评该成熟样本，再对 expert 做本轮更新；partial feedback 在此之前只更新 router。在这一实现和样本上，reversal 与预测性能退化是不同事件。见 `exp/exp_multi_expert.py:1781–1964,2450–2504,2577–2650`。

**论文对应**　两组比例有显著口径差异，未建立论文抽样版本/记录身份，不能按同样本差异认定。重叠窗口的 Wilson 是逐记录描述，跨 seed 不确定性需另估。

**问题**　Potential issue。H=1 尤其需查图表是否在 expert 更新后复评，是否另有实现或标签定义。

**作者核对**　A06-CHECK-01，确认 2000 行名单、H=1 的版本切换时机、各组分母和 CI。见 CHECK-007、MAN-004。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可陈述现有机制及结果范围。论文曲线和 49.9% 待重建。

## A07 · Memory / Subspace / Recovery · P1

**状态**　`VERIFIED_CODE` / `PARTIALLY_VERIFIED`（日志事件）；`NOT_AUDITABLE`（Figure 3(b)）。

**事实**　候选先做 owner 筛选，再走 Stable/Recovery。Buffer 按 ID/sketch 相似性去重，满时以评分决定替换，失败上限触发丢弃。周期重评可 rebase、Stable→Recovery 和 Recovery→Stable。Subspace 从实际 head input 估计未中心化二阶矩 `XᵀWX/sum(w)`。FSNet 取末时间步，FSNet-Time 展开 channel、每 channel 权重用 record stable credit/C。候选配置 rank 16、刷新 100、lambda 1e4；`gamma=clip(1/(1+lr×lambda×mass/(mass+8)),min,max)`。只过滤 head weight，bias 和 encoder 不投影，Recovery loss 与当前 loss 合并后经过同一 head 过滤。见 `utils/expert_memory.py:240–287,503–778`、`utils/subspace_protection.py:121–151,184–345`、`exp/exp_multi_expert.py:2297–2427,3216–3293`。

ETTh2-24 seed 0 有 10,753 matured records、1,189 replay 尝试、83 次判定恢复、77 次晋升、6 次恢复后未被 Stable 接收、373 次尝试耗尽、183 次 rebase。事件计数允许同一 record 多次出现。Stable admission 和 reassessment 总数缺一致导出，eligible 只在 NPZ top-1 归档子集可算。见 [机制统计](MECHANISM_STATISTICS.csv)。

**论文对应**　Figure 3(b) 报 120 对、1.47→1.12、97.5%。现有导出是计数和诊断，缺单事件冻结 reference、before/after、采样与阈值。117/120 只是报告百分比隐含计数。

**问题**　Insufficient evidence。不能把 recovery_successes 等同于 loss 改善。

**作者核对**　A07-CHECK-01，确认事件单位、独立样本和 120 对来源。见 CHECK-008、MAN-006。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可说明机制和已记录的迁移，图中改善率需配对原始数据。

## A08 · Capacity matched · P1

**状态**　`VERIFIED_RUN`（full 候选）；`MISSING_EVIDENCE`（Table 10 对照）。

**事实**　Full 候选配置四 experts，代码 mixed 为两 FSNet、两 FSNet-Time，horizon-channel router、progressive feedback、版本感知与 Recovery。Uniform 初始化的代码存在，不能把零初始化 Router 等同于整个实验恒定均匀。旧 WTH H=24 MoE-only/TSB-off 有 MSE 0.431835、MAE 0.436583，未绑定 Table 10。

| Table 10 方法 | Architecture、初始化、反馈及预算 | 已定位结果 |
|---|---|---|
| Full PAVE | 候选 architecture 已知；Table 10 的共享 checkpoint、有效 replay/steps 待查 | 主候选三 seed console，缺 Table 10 映射 |
| Uniform MoE | UNKNOWN | seed 0/1/2 未绑定，dataset/H 需表页 |
| Matched MoE | UNKNOWN | seed 0/1/2 未绑定，dataset/H 需表页 |

见 `exp/exp_multi_expert.py:228–290,2450–2504`、`log/20260721_134717_moe_tsb_ablation_wth/results_summary.csv`、[对照 CSV](ABLATION_AND_CONTROL_RECHECK.csv)。

**论文对应**　Appendix F.2 / Table 10 需要三个方法的实际命令、初始化与计数。附件未列该表各 setting，缺失清单按已知方法和 seed 列举，setting 保留 UNKNOWN。

**问题**　Insufficient evidence。容量、有效更新/replay 预算与 wall-clock 是三套比较。

**作者核对**　A08-CHECK-01，确认 Table 10 的实际设置和三个方法的 run ID。见 CHECK-009、MAN-005。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　现有材料支持 Full 的架构说明；匹配有效性和 multi-seed 效果仍需记录或补做。

## A10 · 计算开销 · P1

**状态**　`VERIFIED_RESULT`（历史整流耗时除法）；`NOT_AUDITABLE`（Table 11 profiling）。

**事实**

| ECL-48 指标 | PAVE（附件） | Matched MoE（附件） | 独立测量证据 |
|---|---:|---:|---|
| ms/origin | 165 | 125 | 缺 profiler；历史 PAVE seed 0 另有 33161.612 s / 19681 = 1684.956 ms/origin |
| Total parameters | 未提供 | 未提供 | 缺参数清单或配对模型实例 |
| Trainable parameters | 未提供 | 未提供 | 缺 optimizer param-group/冻结快照 |
| Peak GPU memory | 未提供 | 未提供 | 缺原始 GPU memory profile |
| Persistent state | 未提供 | 未提供 | 缺状态口径与测量 |

历史值来自 `result/results259/multi_expert_directional_recovery_v1_ECL_pl48_olfull_optadam_tb1/itr_0/metrics.npy`，包含该历史运行完整测试路径的 elapsed。硬件身份、同步、warmup、重复次数、计时范围和统计方法无法与 Table 11 对齐。[对照 CSV](ABLATION_AND_CONTROL_RECHECK.csv)

**论文对应**　165/125 相差 40 ms，比例 1.32；两个报告值未独立复算。历史整流的不同口径无法与表直接判冲突。

**问题**　Insufficient evidence。参数容量、更新预算和实测延迟应分别给出。

**作者核对**　A10-CHECK-01，确认 profiler 原件、设备、warmup、同步和测量区间。见 CHECK-011、MAN-009。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　目前可解释计时口径，compute-matched 主张需实测依据。

## A11 · Ablation 与 recurring · P1

**状态**　`VERIFIED_RESULT`（可用 seed 0 MSE）；`PARTIALLY_VERIFIED`（候选相对退化）；`CONTRADICTION`（历史产物冲突）；`NOT_AUDITABLE`（recurring 表图）。

**事实**　78 个候选 seed 0 配对，七个消融中 Version Awareness 只有 6 个可归属设置。78 对指向同一现存 checkpoint 文件，但 39 对记录 commit 不同。以每个 dataset 的 full MSE 作分母，然后四数据集等权，实际可用数写在表中。mean±SD 表保留缺失 seed 和单点 population SD=0 的统计意义。[对照 CSV](ABLATION_AND_CONTROL_RECHECK.csv)

| 消融 | H=1 | H=24 | H=48 |
|---|---:|---:|---:|
| w/o Online Correction | +0.883% | −4.231% | −1.859% |
| w/o Local Credit | +0.129% | +0.450% | +3.668% |
| w/o Version Awareness | 缺 | +2.651%（3 个） | +16.970%（3 个） |
| w/o Direction Awareness | +0.261% | −0.554% | +5.223% |
| w/o Subspace | +0.037% | −0.068% | +5.185% |
| Unweighted Subspace | +0.090% | +0.037% | +1.434% |
| w/o Recovery | +0.233% | +0.133% | +1.472% |

其余四个消融的候选行为也已对照源码。w/o Online Correction 停止更新和应用 correction，partial Router 仍训练。w/o Local Credit 设局部 credit loss 权重为 0，历史责任与 sample credit 仍计算。w/o Subspace 的候选将更新策略设为 plain。Unweighted Subspace 给估计 basis 的样本统一权重，保护强度的 evidence mass 仍按原 Stable credit 计算。见 `exp/exp_multi_expert.py:507–519,554–600,1482–1509,1619–1625,1731–1750,2297–2359`。

Recurring 默认 launcher 三段各 300，L=60/H=24、三 seeds、full/no_direction/no_recovery。A2 复用 A 的过程；0-based first-target origin 区间为 [0,240)、[240,540)、[540,817)。计算 B-Early/B-Late/A2-Early、持续恢复与 excess 使用 first-step MSE，另存 horizon-wide MSE。参考为 A1 末窗口，末段超额误差累加 `max(MSE-A1_ref,0)`。见 `scripts/run_recurring_drift_maintext.sh:5–61`、`utils/synthetic_drift_benchmark.py:675–729`、`utils/drift_metrics.py:80–127,186–287`。

**论文对应**　Figure 2 / Table 7 仍需正式 seed、版本与采纳清单。Table 9 / Figure 4 的 per-seed 时间序列缺失，附件 32.9%、40.6%、51.8% 保留为 paper 值，复算字段为空缺状态。

**问题**　Confirmed issue（9 月 12 日六条配置冲突，五条数值冲突）。Potential issue（混合 commit 或 first-step 与全 H 定义混用）。Insufficient evidence（正式聚合）。

**作者核对**　A11-CHECK-01，确认七变体采纳、冲突原件、recurring 时间窗和聚合。见 CHECK-012、MAN-002/008。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可解释消融方向及复算边界。三 seed 置信度和 recurring 收益仍需原始数据。

## A12 · 论文与代码一致性 · P2

**状态**　`VERIFIED_CODE`（命名、defaults 和机制）；`NOT_AUDITABLE`（提交稿版面及逐式比对）。

**事实**　同级发布副本和本地 README 使用不同方法名；运行使用 `multi_expert`。Full launcher 显式传入通用 CLI 中默认不同的设置。

**论文对应**

| Paper location | 论文描述或附件问题 | 代码/文档证据 | 差异与严重度 |
|---|---|---|---|
| 标题 / 方法命名 | PAVE，Version-aware | `../PAVE/README.md:1–16` 用 PAVE，本地 `README.md:288` 用 DRSE，`mae_results.csv` 用 PACE，CLI `multi_expert` | 已知历史命名不同，低。Version-aware 与 credit-aware 的正文一致性待原稿。 |
| Expert / Router | 混合专家和 Router | `exp/exp_multi_expert.py:228–290`，候选 mixed 4 experts | 实现可确认，论文架构及各版本映射待查，中。 |
| Full 默认值 | 渐进 feedback、horizon-channel、subspace | `main.py:205,213,251` 的通用默认是 false/channel/rank 0，full launcher 显式传 progressive/horizon_channel/rank 16 | 两入口默认不同，复现时需指定入口，中。 |
| Eq. 9–11 | sketch / alignment / reference / validation | A04 的分支及 rebase，A07 的 memory/Recovery 与 subspace | 阈值 0.8、top-1、责任 0.3 有候选记录，公式逐式待原页，中。 |
| §4.2 / Figure 1 | 重复段落；Stable/Recovery 标签 | 现有资产未定位提交 PDF/TeX/图原件 | 待作者逐页核对，严重度待确认。 |

**问题**　Confirmed issue（CLI 通用默认与 full launcher 参数不同）。Insufficient evidence（稿件文字与图标注）。

**作者核对**　A12-CHECK-01，确认提交 PDF/TeX、Figure 1、§4.2 和版本映射。见 CHECK-013、MAN-010。author_confirmed [ ] · author_correction [ ] · author_note [ ]

**Rebuttal 价值**　可提交准确的复现入口说明。稿件一致性需论文原件。

## 交付入口

[主张证据矩阵](CLAIM_EVIDENCE_MATRIX.md) · [审稿人矩阵](REVIEWER_EVIDENCE_MATRIX.md) · [已找到证据的作者确认](AUTHOR_REVIEW_REQUIRED.md) · [缺失材料清单](MANUAL_VERIFICATION_REQUIRED.md) · [下一轮实验计划](NEXT_EXPERIMENT_PLAN.md)
