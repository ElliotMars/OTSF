# 统计复核

## 原始数据和可复算范围

索引有 **597 条记录**，含 462 个本地结果条目和 135 个无法唯一绑定本地结果的日志条目。**431 份**有效预测/真值的 MSE/MAE 独立复算与对应 metrics 一致。另有 24 份无预测的 NaN placeholder 和 93 份流长度偏离 HEAD 的历史结果。241 个本地条目缺明确 seed。216 份父目录汇总与逐迭代 metrics 一致，另 7 个迭代 root 缺可比父汇总。索引中的状态与论文采纳情况分别判断。

- [实验索引](EXPERIMENT_INVENTORY.csv) 逐次记录 `metrics.npy`、checkpoint、可关联 console、seed、commit 和数据流长度。历史文件夹不写明 seed 时保留 `UNKNOWN`。同一 result root 下的 `itr_* or aggregate` 只将迭代纳入；[汇总副本索引](AGGREGATE_ARTIFACTS.csv) 保留关系。相同方法、数据、H、seed 在不同 root 的多次执行始终独立列为复跑，只有明确归入候选批次的单次结果进入该批次统计。
- `utils/metrics.py` 的 metrics 顺序为 MAE、MSE、RMSE、MAPE、MSPE，线上再附 elapsed。对每个可用的 prediction/target 对，转 float64 后对全部 origin、预测步、channel 取 `mean(abs(yhat-y))` 与 `mean((yhat-y)^2)`，使用一致性容差 `rtol=3e-5, atol=1e-7`，另记录非有限指标和预测/真值。batch=1 的完整流每个 origin 贡献同样多值。旧版本、截断或 batch 语义不明的运行用原样本数列出，HEAD 长度仅作筛查。部分 7 月旧结果的长度吻合更早代码的 `delay_fb` stride H 口径，例如 ETTh2-24 为 449，现行滚动 stride 1 为 10,777。旧 loader 的 stride 证据见 `git show bf049fc:data/data_loader.py`，旧文件实际版本仍由其运行元数据确定。每 origin MSE 的 p99 用 NumPy 默认 linear quantile。保留原精度汇总，报告时才四舍五入至三位。
- 跨 seed 的每次完整流先产出一个 MSE/MAE。均值按 seed 等权。Population SD 为 `sqrt(sum((x-mean)^2)/n)`，sample SD 用 `n-1`。对于 n=3 后者是前者的 `sqrt(3/2)` 倍；n=1 的 population 0 仅表示单点，sample SD 为空。HEAD `main.py:606–611` 的 `np.nanstd` 默认 ddof=0，`scripts/analyze_progressive_baseline_control.py:95–97` 和 `scripts/analyze_recurring_drift.py:124–125` 同样使用 ddof=0。这个脚本事实不能确定提交表采用同一统计方式。PF analyzer 的聚合 key 仅含 method/dataset/H，不去重 seed 或筛选 commit、短流。Recurring analyzer 对有效值算均值和 SD，`num_seeds` 却计全部目录，故遇到 null reacquisition 时须额外报告实际有限值分母。
- [候选批次逐 seed](CAMPAIGN_PER_SEED.csv) 和 [批次统计](CAMPAIGN_STATS.csv) 将 stdout 中的 MSE/MAE 和现存的独立数组复算分开。历史批次选择依据为已有 wrapper/seed 汇总与实际 Namespace，详见 `summarize.py`。论文 Table 1/6/7 的最终 cell-to-run manifest 仍缺。
- 本地 MAE 矩阵只有一组舍入并列。ETTh2-24 的 PACE 和 ER 同为 0.681，原始 seed 0 分别为 0.6809097209、0.6808694424，ER 低约 0.00004028。提交表名次仍应基于其未舍入均值和既定 tie policy。当前 `mae_results.csv` 提供的是四数据集 × 三 horizon × 八方法的三位小数矩阵，没有 seed、误差条或版本。96/96 个 cell 与相应 PAVE/七 baseline 的 seed 0 历史日志四舍五入后三位一致；PAVE 的 3-seed 候选均值仅 4/12 一致。比如 ETTh2-24 候选 seeds 0/1/2 的 MAE 为 0.680910、0.694978、0.689364，均值 0.688417，而该 CSV 为 0.681。**CONTRADICTION 仅限把这份本地 CSV 解释为该组三 seed 聚合**。Table 6 是否直接取自这份 CSV 尚待制表映射。

## Tables 1/6 · 异常方差

| 条目 | 论文报告 MSE ± SD | 本地优先核查批次 | 其余种子 |
|---|---:|---|---|
| DSOF ETTh2-24 | 3.552 ± 0.443 | `log/dsof_20260903_000847/dsof_ETTh2_24_full.out` | 待 MAN-002 |
| PROCEED ETTh2-48 | 17.438 ± 0.831 | `log/proceed_20260903_125908/proceed_ETTh2_48_full.out` | 待 MAN-002 |
| OneNet WTH-24 | 1.362 ± 1.261 | `log/onenet_20260902_133552/onenet_WTH_24_full.out` | 待 MAN-002 |
| FSNet ETTm1-48 | 1.416 ± 0.067 | `log/fsnet_20260902_133552/fsnet_ETTm1_48_full.out` | 待 MAN-002 |

四项实际 seed 0 MSE 依序为 3.239262、17.438402、0.632996、1.340280，MAE 依序为 0.767444、0.969616、0.523105、0.894333。其结果根目录分别为 results286、311、278、268。PROCEED 的 17.438402 舍入后与报告均值相同，单次吻合无法确定另外两个 seed 或 SD。逐 origin MSE 检查如下。四个候选的原始预测与真值差均有限，最坏误差集中在不同 origin，现有结果只能量化尖峰，不能判定训练发散。
 
| 本地 seed 0 | Origins | P99 | 最大 origin MSE | 最大值 origin（0-based） |
|---|---:|---:|---:|---:|
| DSOF ETTh2-24 | 10,777 | 79.887 | 837.866 | 3,318 |
| PROCEED ETTh2-48 | 10,753 | 228.421 | 5,079.774 | 3,332 |
| OneNet WTH-24 | 26,275 | 2.524 | 208.865 | 25,418 |
| FSNet ETTm1-48 | 10,753 | 5.928 | 11.103 | 8,441 |
需要同一 cell 的三个种子、训练及 online 状态后，才能在统计口径、配置混用、方法不稳定、协议差异和证据缺口间区分。**提交版本的四组 SD 为 NOT RECOMPUTABLE。**

## Table 7 / Figure 2 · Ablation

7 个候选消融可做 78 个 seed 0 配对，其中 w/o Version Awareness 只有 6 个可归属设置。78 对均有相同 checkpoint 目录名，39 对的记录 commit 不同。配置 dirty 的运行仍有未跟踪差异的可能。

[seed 0 逐设置相对变化](ABLATION_SEED0_RECHECK.csv) 用同 setting full candidate 原 MSE 作为分母，`(ablation-full)/full`。每个 H 的跨数据集 [等权汇总](ABLATION_HORIZON_RECHECK.csv) 同时报告可用数据集数、相同已知 commit 与同 checkpoint 目录名的数量。对照使用 `20260828_154332` full 候选。代码版本/训练初始化比较有缺口的配对只报告描述性差异。剔除同一路径的 stdout/配置或指标冲突配对。9 月 12 日至少五个设置遇到此问题，详见总览和主索引。Table 7 的最终三种子 SD 和 Figure 2 的最终基准及聚合仍 **NOT RECOMPUTABLE**。

## Appendix E · 高对齐 predictive degradation

在 130 份 credit archive 中，125 份具有非空的可复算字段，3 份缺至少一个分析字段，2 份为空。[CAPABILITY_RECHECK.csv](CAPABILITY_RECHECK.csv) 按 NPZ、dataset/H/seed/variant 和 stream stage 分层。每个 archived record 取预测时 sample responsibility 的 argmax 对应 expert，查同一 record 的预测时与当前版本 expert MSE。阶段用原始 origin 的整数三等分，`floor(3*origin/total_origins)`；`total_origins` 可由现有数组得到时采用实际值。高 alignment 阈值用运行配置，缺失时用审计分析约定 0.8。审计公式为 `(current-prediction)/(prediction+1e-8)`，分别计严格 `>0, >0.1, >0.2, >0.5`。NaN/Inf 行另计，分母是各层有效记录。前后状态在 initial admission 之前，采用同一预测时参考；无需依赖之后的 rebase。此处不会按 confidence/responsibility threshold 再筛 owner，因此低责任可入。若论文使用源码 `capability_degraded`，其 `current > ref×(1+margin)+eps` 与严格相对 `>0` 在极小差值边缘可能不同。

每文件比较 `retained_records` 与 `unique_origins`。同一数值 origin 在不同 run 本来可以重复，故需 run ID 一起识别。普通 archive 的采样来自尾部 deque。ECL 在 H=1 的一个 full-length 候选中共有 19,728 个 origin，archive 仅覆盖 9,727–19,726。ETTh2-24 `result/results253/multi_expert_directional_recovery_v1_ETTh2_pl24_olfull_optadam_tb1/itr_0/credit_diagnostics.npz` 的 10,000 条 origin 为 753–10,752。这里 10,000 是文件实际条数，也是 deque 上限，不能据此确认论文选取哪一组。论文的 95.80/4.20/1.80/0.84/0.13% 在严格 10,000 分母下对应 9580/420/180/84/13 的报告隐含计数，选样映射尚未获得。该 ETTh2-24 archive 按审计口径得到 9,811 个高 alignment owner，其中 748/396/254/101 条超过四个递增阈值（7.624%/4.036%/2.589%/1.029%），全 archive reversal 7,218/10,000。这是一个可追溯历史运行的独立诊断，未找到论文对应的抽样名单。论文统计 **NOT RECOMPUTABLE**，本地 archive 分层可复算。

## Figure 3 · Historical-credit reversal 与 Wilson

[REVERSAL_RECHECK.csv](REVERSAL_RECHECK.csv) 按 run 和 alignment 口径独立统计，`historical_owner` 用历史 owner 的 alignment，`mean_expert` 使用各 expert 平均，后者对应内置 analyzer。Reversal 为两套 responsibility argmax 不同。旧/新预测评价的是同一已成熟记录。使用 overall `[0,1]` 以及 `[0,.25), [.25,.5), [.5,.75), [.75,1]`，给出每个分母、reversal count 和比例。Wilson 95% 以 `z=1.959963984540054, p=k/n`，`(p+z²/(2n) ± z sqrt(p(1-p)/n+z²/(4n²)) / (1+z²/n)`，空 bin 不输出区间。区间是逐记录二项描述，重叠滚动窗口存在串行相关，不能直接作为跨 seed 区间。无论文抽样名单，约 2,000 的 49.9% 和三个 H 的 21.0/44.7/84.1% 及其 Wilson 为 **NOT RECOMPUTABLE**。不能用 loss increase 标签代替 reversal。

## Memory、Recovery 和 recurring

- [MEMORY_EVENT_COUNTS.csv](MEMORY_EVENT_COUNTS.csv) 保存现有逐运行累计事件和完成记录数。Retained、harmful 或迁移事件可能跨 expert/重评重复；独立记录计数只在逐记录文件可得时由明确的 key 统计。Replay attempt、success、promotion、成功后被 Stable 拒收、耗尽丢弃分别列出。例如 ETTh2-24 `results253` 的 10,777 origin 有 10,753 个完整成熟 record，1,189 次 replay 尝试、83 次判定恢复、77 次晋升、6 次恢复后丢弃、373 次尝试耗尽、183 次 rebase。完整 Stable admission 和周期重评次数缺一致导出，不能从 occupancy 或总事件反推出 unique records。
- Figure 3(b) 的 120 对 before/after、normalized loss 中位数 1.47→1.12、97.5% 为 **NOT RECOMPUTABLE**。97.5% 若分母确为 120，隐含 117 次，缺独立的成对数据验证。现有事件累计计数不能恢复配对 loss、冻结的 reference 或重复 replay 样本。
- Table 9 的 32.9%、40.6%、51.8% 为 **NOT RECOMPUTABLE**。需每个 seed 与 variant 的 origin timeline 和窗口。源码把 first-step MSE 喂给 `compute_drift_metrics`，完整 horizon MSE 另存，二者必须区分。A1 参考用末窗口 MSE，A2-Early 取首窗口，reacquisition 使用持续达标 rolling 起点，cumulative excess 是对 A2 的 `max(loss-reference,0)` 求和。未恢复值如何纳入及相对百分比汇总方式待确认。
- Table 2 的 DSOF/DynaME/ER 渐进对照、Table 10 Matched/Uniform、Table 11 profiler、Table 8 和 Figure 4 缺可绑定原始来源。这里保留三份有诊断的 PF smoke（`results417–419`），四个 origin，H=2，每份 5 事件、3 optimizer steps、末尾 2 个未成熟 record，泄漏诊断 false。同一输入用现有 analyzer 复算，见 [PF 短流统计](pf_smoke_recheck/summary.csv) 和 [协议诊断](pf_smoke_recheck/protocol_summary.csv)。这三份的范围仅限短流。

## 重复执行审计

[重复设置索引](REPEATED_SETTINGS.csv) 按显式 seed、方法、dataset/H、已知变体和协议列出各次运行。同 seed 的多个设置可用于追溯复跑和比较环境，不能当作 seed 独立样本。[Checkpoint 清单](CHECKPOINT_ARTIFACTS.csv) 仅按真实路径或日志 setting 的目录名关联，未反序列化模型参数。

在仓库根目录可以用可用 NumPy 环境执行 `python -B audit/rebuttal_2027/recheck.py` 后再执行 `python -B audit/rebuttal_2027/summarize.py`。仅更新日志来源和诊断而复用已有数组复算可给第一条命令加 `--reuse-arrays`。
