# 论文主张与证据矩阵

状态按证据对象分开标注。可信度高表示本地源码或原始产物可查。采纳关系与作者确认见 [手动核查](MANUAL_VERIFICATION_REQUIRED.md)。

| Claim | Paper location | Evidence path | Status | Reviewer | Risk |
|---|---|---|---|---|---|
| 主结果 MSE 采用 3 个独立 seed | Table 1 | `EXPERIMENT_INVENTORY.csv`、`log/baselines_seed0/`、`log/20260914_203827_progressive_subspace/`、`log/20260914_203833_progressive_subspace/` | PARTIALLY_VERIFIED | 6htc / fvv / ZDG6 | 高。存在历史数值，缺 table-to-run 映射和完整的 baseline seed 1/2 证据。 |
| 主结果 MAE 和三位小数排名 | Table 6 | `mae_results.csv`、`STATISTICAL_RECHECK.md`、本地预测与真值 | PARTIALLY_VERIFIED | fvv / 6htc | 高。PACE 命名的矩阵没有 seed/聚合信息。 |
| 参数经公平验证选择 | Tables 1/6, 方法比较 | `scripts/hp_search/`、`log/pretrain_*/`、配置及 Namespace | MISSING_EVIDENCE | 6htc | 高。能确认实际 LR 的运行子集，缺共同 online validation 搜索链。 |
| DSOF 原始 fast stream 能用部分标签 | Table 1 相关 | `exp/exp_stream_baselines.py:153–165,241–249`、`log/dsof_20260903_000847/` | VERIFIED_CODE | ZDG6 / fvv | 高。本地 delayed 适配把 student 也推迟到整窗成熟。Table 1 实际选用运行需确认。 |
| 渐进 feedback 对照覆盖 DSOF/DynaME/ER | Table 2 | `scripts/run_progressive_baseline_maintext.sh:5–25`、`scripts/run_progressive_baseline_control.sh:50–58`、`utils/run_config.py:19,129–155` | CONTRADICTION（当前入口范围）；MISSING_EVIDENCE（论文运行） | ZDG6 / 6htc | 高。当前入口不能运行 DSOF/ER PF。三个实际本地 PF 样例为 FSNet/OneNet/DynaME 四 origin。 |
| 组件消融和相对退化 | Figure 2 / Table 7 | `log/ablations_seed0/`、`log/ablation_rerun_seed0/`、`scripts/run_progressive_credit_subspace.sh:120–137` | PARTIALLY_VERIFIED | 6htc / ZDG6 | 高。有 seed 0 多组候选；需要每个 cell 的原始 seed 和 checkpoint。 |
| Table 8 的具体 claim、组别和分母 | Table 8 | 任务仅列编号，未给表题、数值或表源 | NOT_AUDITABLE | fvv / ZDG6（待作者确认） | 需提交版本对应表页，不能用 Table 9 或 Appendix E 替代。 |
| 10,000 个 high-alignment records 的 95.80% 无降级 | Appendix E | `CAPABILITY_RECHECK.csv`、`exp/exp_multi_expert.py:722–728,1907–1964` | PARTIALLY_VERIFIED | fvv / ZDG6 / KaYu | 高。可复算本地留存数据；样本池、精确选择及分层对应关系待确认。 |
| 约 2,000 个历史信用 reversal，按 H 和 alignment 分箱 | Figure 3 | `REVERSAL_RECHECK.csv`、`utils/credit_diagnostic_analysis.py:89–149` | PARTIALLY_VERIFIED | fvv / ZDG6 | 高。内置 analyzer 用 mean-expert alignment，论文 owner/bin/sample manifest 待查。 |
| 120 次 replay 的配对改善 | Figure 3(b) | `exp/exp_multi_expert.py:1211–1325`、`online_diagnostics_summary.json` | MISSING_EVIDENCE | fvv / ZDG6 | 高。事件总计无 paired loss 样本及分母。 |
| Figure 4 的样本选择、区间和图源 | Figure 4 | 任务仅列编号；本地绘图脚本、NPZ/JSON 无已证实映射 | NOT_AUDITABLE | fvv / ZDG6（待作者确认） | 补图页、plot 输入和生成命令后可映射，暂不推测对应实验。 |
| A1→B→A2 的性能与重获取 | Table 9 / F.1 | `scripts/run_recurring_drift_maintext.sh`、`utils/drift_metrics.py:186–287` | VERIFIED_CODE；MISSING_EVIDENCE（论文 3 seed） | ZDG6 | 缺各变体 timeline、events 和删失口径。 |
| Uniform/Matched MoE 的容量、replay/update 匹配 | Table 10 / F.2 | `scripts/run_structural_ablations.sh`、`log/20260721_134717_moe_tsb_ablation_wth/` | MISSING_EVIDENCE | ZDG6 | 高。结构相关产物无 Table 10 对照签名，缺 seed 1/2 与步数。 |
| ECL-48 参数、延迟、峰显存、持久状态 | Table 11 | 普通 test elapsed 与在线 diagnostics；缺 profiler trace | NOT_AUDITABLE | ZDG6 / fvv | 模型容量、步数、时间须分列；同步与计时边界不可从总耗时反推。 |
| 相同 ETT observation index 和因果 stride | Appendix B.1 / 方法 | `data/data_loader.py:49–113,153–215`、`exp/exp_multi_expert.py:2586–2645` | VERIFIED_CODE | KaYu / ZDG6 | observation-count 相同，ETTh2 和 ETTm1 的真实时长不同。 |
| Encoder alignment 可用作 capability proxy | 方法 / Appendix E | `exp/exp_multi_expert.py:100–120,158–177,268–278,397–413`、`utils/expert_memory.py:22–60` | VERIFIED_CODE（实现）；PARTIALLY_VERIFIED（proxy） | KaYu / fvv / ZDG6 | 高。只变预测头时 sketch 不直接变化，高 alignment 会保留已有能力分类。 |
| Stable/Recovery、子空间和更新顺序 | 方法 / Figure 2 | `utils/expert_memory.py:503–778`、`utils/subspace_protection.py:121–151,184–345`、`exp/exp_multi_expert.py:2450–2504` | VERIFIED_CODE | ZDG6 / fvv | 成功计数区分性能恢复、被 Stable 接收和恢复后丢弃。 |
