# 论文主张与证据矩阵

论文报告值以作者附件为准。当前证据范围与论文实际采用的运行分别判断，作者确认字段见 [作者确认表](AUTHOR_REVIEW_REQUIRED.md)。

| ID | 论文位置与主张 | 证据 | 状态与边界 | Reviewer / 作者确认 |
|---|---|---|---|---|
| A01 | Tables 1/6 · 参数选择公平 | [实际配置与搜索清单](HYPERPARAMETER_AUDIT.csv)，`scripts/hp_search/`，`log/pretrain_*/` | VERIFIED_RUN（有配置及日志的候选）；MISSING_EVIDENCE（online 搜索） | 6htc/ZDG6 · CHECK-001 |
| A02 | Table 1 · 三 seed MSE 和方差 | [主结果 CSV](MAIN_RESULTS_RECHECK.csv)，[逐 origin MSE](A02_ANOMALY_ORIGIN_MSE.csv.gz) | VERIFIED_RESULT（96 个 seed 0）；PARTIALLY_VERIFIED（PAVE 三 seed 日志）；baseline 其余种子待绑定 | fvv/6htc · CHECK-002 |
| A02 | Table 6 · 三 seed MAE | [主结果 CSV](MAIN_RESULTS_RECHECK.csv)，`mae_results.csv` | CONTRADICTION 限本地 CSV 与所查 PAVE 三 seed 聚合；提交表映射未找到 | fvv/6htc · CHECK-003 |
| A03 | Table 1 DSOF 与 Table 2 DSOF/DynaME/ER + PF | `exp/exp_stream_baselines.py:148–176,241–274`，`utils/run_config.py:21,129–150`，[短流证据](ABLATION_AND_CONTROL_RECHECK.csv) | VERIFIED_RUN（legacy DSOF 和 PF smoke）；CONTRADICTION（HEAD 支持集合）；正式 Table 2 MISSING_EVIDENCE | ZDG6 · CHECK-004 |
| A04 | Eq. 9–11 · Version-aware validation / loss-only | `utils/expert_memory.py:22–59,388–441`，`exp/exp_multi_expert.py:1781–2210,2450–2504` | VERIFIED_CODE；独立 loss-only 对照 MISSING_EVIDENCE | ZDG6/KaYu · CHECK-005 |
| A05 | Appendix E / Table 8 · 10,000 条高对齐 | [本地按 archive 和流位置分层](MECHANISM_STATISTICS.csv) | VERIFIED_RESULT（本地）；NOT_AUDITABLE（论文样本身份） | fvv/ZDG6/KaYu · CHECK-006 |
| A06 | Figure 3(a)(c) · 2,000 条 reversal | [分组及 Wilson](MECHANISM_STATISTICS.csv)，`exp/exp_multi_expert.py:2450–2504` | VERIFIED_RESULT（本地）；NOT_AUDITABLE（论文分组）。H=1 本地 0/40000，专家更新间隔为 0 | fvv/ZDG6 · CHECK-007 |
| A07 | Figure 3(b) · 120 对 replay 改善 | `utils/expert_memory.py:503–778`，[事件表](MECHANISM_STATISTICS.csv) | VERIFIED_CODE（机制）；PARTIALLY_VERIFIED（事件日志）；NOT_AUDITABLE（120 对） | fvv/ZDG6 · CHECK-008 |
| A08 | Appendix F.2 / Table 10 · Uniform / Matched / PAVE | `exp/exp_multi_expert.py:228–290`，[三方法种子缺口](ABLATION_AND_CONTROL_RECHECK.csv) | VERIFIED_RUN（Full 候选）；MISSING_EVIDENCE（Table 10 映射、步数、预算、种子） | ZDG6/6htc · CHECK-009 |
| A09 | Appendix B.1 · 观察数划分 | `data/data_loader.py:49–113,153–215`，[实际日期和长度](AUDIT_OVERVIEW.md) | VERIFIED_CODE；conventional 结果 MISSING_EVIDENCE | KaYu · CHECK-010 |
| A10 | Table 11 · ECL-48 计算成本 | [paper 值与历史耗时分开](ABLATION_AND_CONTROL_RECHECK.csv) | VERIFIED_RESULT（历史整流 elapsed/N）；NOT_AUDITABLE（profiler、参数和显存） | ZDG6 · CHECK-011 |
| A11 | Figure 2 / Table 7 · 七项消融 | [逐 seed 与等权汇总](ABLATION_AND_CONTROL_RECHECK.csv)，[冲突运行](EXPERIMENT_INVENTORY.csv) | VERIFIED_RESULT（可用 seed 0 数值）；PARTIALLY_VERIFIED（候选比较）；CONTRADICTION（历史配置/日志冲突） | 全体 · CHECK-012 |
| A11 | Table 9 / Figure 4 · recurring 三种子 | `scripts/run_recurring_drift_maintext.sh:5–61`，`utils/drift_metrics.py:80–127,186–287` | VERIFIED_CODE；原 timeline 和三个百分比 NOT_AUDITABLE | 全体 · CHECK-012 |
| A12 | §4.2 / Figure 1 / Full config / Eq. 9–11 | `README.md:288`、`../PAVE/README.md:1`、`main.py:205,213,251`，[差异表](AUDIT_OVERVIEW.md) | VERIFIED_CODE（历史命名、入口默认）；NOT_AUDITABLE（论文版面及逐式核对） | 全体 · CHECK-013 |
