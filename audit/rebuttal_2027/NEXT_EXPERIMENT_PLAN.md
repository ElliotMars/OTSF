# 后续实验准备度

顺序优先回应 6htc 与 ZDG6。若作者取回足够的原始证据，可将对应重跑降级为复核。只列出需要的实施内容，参数、budget、数据切分和采纳标准由作者确认后固定。

| 优先级 / 实验 | 已有结果与入口 | 最小实施、复用条件 | 必须固定及作者确认 |
|---|---|---|---|
| MUST RUN · Fair online-LR tuning（6htc） | 完整验证搜索证据缺。各 baseline launcher、PAVE launcher 和在线 LR 参数现成。旧 HP 脚本为部分材料。 | 增加隔离的 validation stream 入口和逐候选记录。复用同一方法/setting/seed 兼容 offline checkpoint，每候选从相同 model/optimizer/memory 初态开始。保存候选、验证 MSE、选择日志和最终一次 test。 | 确认验证边界、共同选择指标、候选预算、是否按 dataset/H 选择和 tie rule。原始测试集用于最终报告。 |
| MUST RUN · PAVE sensitivity（6htc） | 有早期 HP、dataset-specific 和 structural 入口，缺当前版本成套结果。 | 用 `scripts/run_progressive_credit_subspace.sh` 和 `run_structural_ablations.sh` 的现有开关组织固定候选表，保留完整 config、动态 LR、memory/evidence、每 seed 指标。可重用架构和 split 一致的 checkpoint。 | 确认主要参数、范围和对照数，逐项敏感性与组合交互分开。记录 controller 后实际 LR，选参与固定协议展示分开。 |
| MUST RUN · Loss-only validation（ZDG6） | 无确认结果或独立入口，已存在 paired current/reference loss 和统一分类器。 | 增加 validation mode，从 admission、periodic refresh、post-replay 三处调用同一决策策略。用现有当前/参考 loss，保持 sketch 计算、真实 alignment ranking/子空间质量权重、router、expert 和其他更新相同。统一用 `current > ref×(1+margin)+eps` 定义 harmful，等值落在预定义保留规则。相同 checkpoint 可用。 | 确认 loss-only 如何区分 retained 与 beneficial/neutral、何时 rebase、是否保留独立 promotion 规则。对等 rank/admission 机会与自然有效 replay 数分别记录。避免直接使用 `disable_version_awareness` 代替。 |
| RECOMMENDED · Always-on high-alignment loss check（ZDG6/fvv） | 无独立结果。HEAD 完整记录已经计算 loss，直接加入 gate 有清晰入口。 | 在保留现有 low-alignment 规则的前提下，让 high-alignment + degraded 按预注册规则进入新分支，联通 admission/refresh/post-replay。复用相同 checkpoint。保存二元 gate 计数、reference 和 replay/保护强度。 | 明确 high-alignment harmful 的 Recovery eligibility、credit `1-alignment` 接近零时的 ranking 处理、rebase 和预算。PAVE 与变体的额外计算/更新次数单独报告。 |
| RECOMMENDED · Sketch vs output drift（KaYu/fvv/ZDG6） | 本地 diagnostics 有 paired expert MSE 和 sketch alignment。逐记录配对 raw expert output 及独立干预证据待取。 | 先用已有记录按 dataset/H/seed/stage 计算损失变化与 alignment 的关系；后续从同一 checkpoint 固定 encoder，仅改变 head，或收集同记录旧/新输出。存预测距离、head/encoder 改动量和配对误差。 | 固定观测样本、尺度、阈值与采样分层，区分相似度、输出差和性能差。对重叠流使用块或 seed 级不确定性。 |
| MUST RUN · Matched MoE 3-seed（ZDG6） | Table 10 报告 single seed，未定位明确产物/入口。结构消融可复用 launcher。 | 先取回 seed 0 原始对照。建立 Uniform、Matched、PAVE 的架构、offline init、trainable set 对照和 per-origin replay/update counters。用同一兼容 offline checkpoint 或每 seed 同一独立预训，补 seed 1/2。 | 确认 router 和 replay policy、按调用或按样本/梯度的 matching 口径。输出实际 draw、有效样本、各 optimizer steps 和 wall-clock。 |
| RECOMMENDED · Conventional ETT split（KaYu） | HEAD 无可选常规 split；代码注释及同级 baseline 有参考入口。未找到成套结果。 | 增加显式 split 配置或独立 loader wrapper，记录时间和 index 边界、scaler fit、测试 origin 数；新的 split 需要新 checkpoint 或经验证适配的预训练来源。 | 固定 ETT 文件、seq/H、seed、feedback stride、scaler 与训练协议。确认 conventional 12/4/4 的具体定义和 ETTm1 ×4。 |
| MUST RUN · Baseline variance re-evaluation（fvv/6htc） | 9 月 seed 0、本地预测与误差轨迹可复核。四项论文 cell 的其余 seed 来源待恢复。 | 先检查原始 seed 文件，缺失时按固定命令补运行。输出 per-origin MAE/MSE、有限性、gradient/update 状态、checkpoint/commit/split、protocol 和两个 ddof。 | 保留自然失败或异常流证据并预先约定排除规则。多次执行同 seed 按复跑报告，控制 checkpoint 混用。 |
| OPTIONAL · Compute-matched baseline（ZDG6） | 无 Table 11 原始 profile 和等时间对照，普通 elapsed 可作为初步参考。 | 先加 profiler/counter 报告，固定 checkpoint 后用同 GPU、精度、batch、同步边界和重复。若要等时间，按验证端预先固定 budget policy。 | 分离模型容量、有效 update/replay、时间三种资源，确认显存定义、warmup、冷启动和 persistent state 口径。 |

DSOF Table 2 优先恢复作者脚本和原始输出。若确需补跑，应先明确以 native incomplete-fast 为主对照，还是以 full-delayed 适配为主对照，并分别记录 fast/slow 的监督和步数。HEAD 渐进 baseline 入口需要补 DSOF 与 ER 的专用路径才能用于这两个方法。
