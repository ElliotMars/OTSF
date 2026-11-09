# Rebuttal suite · 本地代码审计

审计基线为当前工作树及 `367d19d`。实验参数以 96 组本地 seed 0 stdout 快照为基础，C1 已区分原始 LR、实际优化器步进 LR 和测试结果。对应档案、运行边界、版本差异和可用 checkpoint 见 [C1 审计](../c1_lr_tuning/C1_CODE_AUDIT.md) 与 [A01–A12 审计](../../audit/rebuttal_2027/a01_a12/AUDIT_OVERVIEW.md)。

| 实验 | 本地入口和已有能力 | 缺口及改动边界 |
|---|---|---|
| C1 | `main.py`、`utils/run_config.py`、`scripts/c1_lr_sweep.py`、`scripts/c1_lr_worker.py`。八种方法已有单独 optimizer LR 映射、val replay、冻结选择和重置。 | 保留入口和已有计划。统一编排补逐 job 导出、元数据、结果/SD。 |
| C2 | `exp/exp_multi_expert.py` 初始化 alignment、Stable/Recovery、subspace lambda 和 adaptive controller；原训练方法不消费这些在线状态。 | 用同 offline 快照按预注册 OFAT 改动，memory 容量两个字段作为一组。LR 敏感性引用 C1。 |
| C3 | `utils/expert_memory.py:22`、`add_candidate_with_result`、`refresh`、`update_recovery_result`；`exp/exp_multi_expert.py` 的 matured record 分类、更新及 commit。 | 显式 validation policy 同时覆盖初次入库、复评和 replay 后晋升。损失参考初始来自已成熟的 prediction-time outputs，rebase 一并更新 loss/sketch。 |
| C4 | `_predict_and_create_record` 保存 prediction-time expert outputs；`_evaluate_completed_record` 已对同一 context 运行只读当前前向；已有 bounded deque。 | 复用这两个位置，另写新的全流记录，初次成熟记录与复评计数分开；拒绝将新抽样映射成原 Appendix E。 |
| C5 | 当前实验可核四个专家及 horizon-channel router、Recovery replay 数和 optimizer steps。`scripts/run_multi_expert_moe_tsb_ablation_wth.sh` 是旧 MoE/TSB 对照。 | 在源码、shell、Git 历史和同级 `../PAVE` 未找到可绑定 Table 10 的正式 Matched/Uniform 实现或更新 trace。任何新对照须明确标记为 prospective。历史复现待提供代码、命令、预算和 checkpoint。 |
| C6 | `data/data_loader.py` 的 ETT 目标区间是 0/2880/3600/14400；val/test 只前置合法历史 context；scaler 使用 train。邻库 DynaME loader 有 conventional 12/4/4。 | 显式 split mode，按 CSV 时间频率验证固定 30 天月协议；conventional 有独立 checkpoint 身份。 |
| C7 | `exp/exp_dsof.py` 指向共享 `ExpDSOF`；`models/dsof.py` 是 residual student；`scripts/run_dsof.sh` 与归档主候选启用 legacy delay。 | 原始适配快慢流都等完整窗口，和 DSOF 原论文的 incomplete fast stream 不同。新增单独命名的部分反馈适配；正式同构复现待核。 |

已检查八个 experiment/model 入口、C1 配置与 tests、`scripts/run_progressive_credit_subspace.sh`、`scripts/run_progressive_baseline_{control,maintext}.sh`、`scripts/run_multi_expert.sh` 和各方法 launcher。C7 详细对照见 [反馈审计](DSOF_PROTOCOL_AUDIT.md)。
