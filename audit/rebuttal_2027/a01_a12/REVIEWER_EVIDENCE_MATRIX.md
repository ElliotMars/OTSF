# 四位审稿人的证据与回应范围

| Reviewer | 关切 | 可引用的现有证据 | 需补的证据与下一轮条件 | 作者核对 |
|---|---|---|---|---|
| 6htc | LR 公平性、复杂度、超参敏感性 | 七 baseline 84 条日志的线上基础 LR 都为 1e-3。PAVE Expert 1e-4/5e-5/1e-5，Router 1e-5。Offline checkpoint 使用 validation MSE。七消融有 78 对候选 seed 0，39 对记录 commit 不同。 | 取回 online validation、候选预算、选择规则。材料不足时做公平 LR 与 PAVE 敏感性。实际更新、replay 和耗时分开。 | CHECK-001/002/009/011/012 |
| fvv | 异常方差、sketch、recovery | 四个异常条目可核 seed 0 原始预测和尖峰。ETTh2-24 高对齐 748/9811 有降级。H=1 本地 0/40000 reversal，专家更新间隔均为 0。Recovery 事件数可查。 | 取回其余 seed、Figure 3 和 Table 8 抽样、120 对 before/after。材料不足时补 sketch proxy 与稳定性复核。 | CHECK-002/003/006/007/008 |
| ZDG6 | Loss-only、high-alignment failure、capacity-matched、DSOF/PF、成本 | HEAD 分类优先 alignment。归档 DSOF fast/slow 均等整窗。三个其他方法 PF smoke 各 4 origins/3 steps。本地 ECL-48 elapsed 为整流耗时。 | Table 2 两臂命令及 budgets、Table 10 模型和种子、Table 11 profiler。材料不足时补独立 loss-only、always-on 及 matched-budget。 | CHECK-004/005/006/009/011 |
| KaYu | ETT 划分和 encoder alignment 局限 | 两数据集均用 2880/720/10800 个目标观察，真实覆盖分别为 120/30/450 和 30/7.5/112.5 天。高对齐 owner 仍可能有 degradation。 | 确认数据版本和原稿 split。需要 conventional ETT 稳健性及 head/encoder drift 对比。 | CHECK-005/006/010/013 |

见 [完整审计](AUDIT_OVERVIEW.md)、[已找到证据确认](AUTHOR_REVIEW_REQUIRED.md) 和 [待补材料](MANUAL_VERIFICATION_REQUIRED.md)。
