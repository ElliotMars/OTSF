# 下一轮实验计划

先恢复原始结果和提交版映射。以下优先级以现有证据仍不足为前提，本轮未启动实验。

| 优先级 | 实验 | 依据和最小条件 |
|---|---|---|
| MUST RUN | Fair LR tuning | 6htc/ZDG6。共用严格因果 online validation 流和指标，固定候选预算、初始化、seed 与并列规则，存逐候选与最终选择。 |
| MUST RUN | PAVE hyperparameter sensitivity | 6htc。固定其余配置，分别测试 Expert/Router 基础 LR、阈值、memory 和 rank，记录有效 LR、replay/更新步与误差。 |
| MUST RUN | Loss-only validation | ZDG6。分类、admission、refresh、post-replay 采用统一 loss policy。先保留原评分并报告实际 replay 曝光，再分开比较等预算或替代评分。 |
| MUST RUN | Matched MoE multi-seed | ZDG6。确认 Table 10 settings，补 Uniform/Matched/Full 缺 seed，固定初始化和标签可用性，记录有效 update/replay 与时间。 |
| MUST RUN | Baseline stability re-evaluation | fvv/6htc。若缺失 seed 无法找回，冻结四异常设置的版本、LR、checkpoint，保留 per-seed 及 per-origin 指标、更新状态与预先确定的 ddof。 |
| RECOMMENDED | Always-on loss check | ZDG6/fvv。为高对齐样本增加同一 loss 检验，记录 eligibility、Recovery credit 和真实 replay 触达。 |
| RECOMMENDED | Sketch proxy comparison | KaYu/fvv/ZDG6。按 horizon、专家更新数与流位置对比 encoder sketch、head-only 与输出/loss 代理，按 seed 或时间块估计不确定性。 |
| RECOMMENDED | Conventional ETT split | KaYu。按确定的 12/4/4 边界重做 train-only scaler 和 checkpoint，固定预测范围、因果反馈与训练预算。 |
| OPTIONAL | Compute-matched control | ZDG6。先取回 Table 11 profile，再用同硬件、精度、同步、warmup 和测量区间比较。分别记录参数容量、实际步数和墙钟时间。 |

现成入口和复用条件见 [上一轮实施条件](../NEXT_EXPERIMENT_PLAN.md)。
