# 实验计划

| ID | Pilot | Full | 比较与输出 |
|---|---|---|---|
| C1 | ETTh2/H24、WTH/H24、ETTm1/H48，PAVE、DSOF、OneNet、FSNet | 4 数据集 × 3 horizons × 8 方法 | 原始与验证选择 LR，R1 |
| C2 | ETTm1/H48、WTH/H48 | 加 ECL/H48 | alignment、memory capacity、subspace lambda、controller OFAT，R2 |
| C3 | ETTm1/H48、WTH/H48 | 4 × 3 settings | Full、loss-only gate、pure loss-only、两种 high-alignment guard，R3 |
| C4 | C3 同范围 | C3 同范围 | Full PAVE 同次 replay 的首次成熟样本、复评及 proxy，R4/R5 |
| C5 | ECL/H48 | ETTm1/H48、WTH/H48、ECL/H48 | Full、同专家结构 matched/uniform 对照，另可加验证计时对照，R6 |
| C6 | ETTh2/H48、ETTm1/H48，全部方法 | 两数据集的 H24/H48，全部方法 | Conventional ETT 固定 LR，R7 |
| C7 | ETTh2/H24 | ETTh2/H24、ETTm1/H24、WTH/H48、ECL/H48 | Archived adapted DSOF、partial-TD adaptation、PAVE，各做公平 LR 选择，R8 |

正式测试 seeds 为 0、1、2。验证默认 seed 0，可在运行前固定多个 validation seeds。Smoke 默认 seed 0、一个 offline epoch、在线最多执行本组最大 horizon 加四步。C1/C7 只跑倍率 1 的流程，C6 只用 PAVE 检查两种 cadence，其余按首个 pilot setting 检查。Smoke 的 checkpoint 与正式运行独立，汇总排除所有 smoke 指标。

## 选参与统计

- C1、C7 使用相同倍率 0.1、0.3、1、3、10。每个验证任务重新载入同一 seed 的 offline checkpoint。以 validation online MSE 的平均值选参，近似相等时先取最接近 1 的 log-scale，再取较小倍率。测试开始前冻结选择及 checkpoint 指纹。
- C2 以归档 LR 为基准。两个 memory 容量字段共同改变。候选与基准相同的组只运行一次。LR 灵敏度另附 C1 的验证 trial 表。
- C3 首次比较使用同一成熟目标上的旧预测与当前预测。退化定义为 `current > reference * (1 + margin) + eps`。两个 loss-only 门控都将明显退化送入 Recovery，将超过 epsilon 的改善 rebase，其余保留。Pure 使用 `confidence × responsibility` 的 Stable/Recovery evidence，保留 sketch 去重及 replay 正则。两个 guard 以同一 loss 阈值约束高对齐案例，安全版 Recovery 系数为 `max(1 − a, 0.1)`。配置的 replay 上限与更新时钟一致，实际可抽取的 Recovery 数受入库、晋升和容量影响，逐 seed 报告。
- C4 按 dataset、horizon、seed、在线时间四分位分层。第一阶段分母是所有首次成熟 expert-owner，第二阶段通过 top-k 和 responsibility threshold，第三阶段是实际构造的 memory candidates，最后按 Stable/Recovery 实际入库分开。退化率的分母为对应阶段的高对齐数。每条流最多 10,000 个 eligible high-alignment owners 按稳定哈希采样。样本索引以 origin 和 expert ID 连接。历史 Appendix E 样本归属缺失。复评以单独事件文件记录，并另汇总复评次数与迁移计数。
- C4 proxy 比较使用同 context 的历史预测和当前预测。Sketch 使用运行配置中的 32 维投影。参数距离使用每位专家最多 128 个固定位置的 RMS，按 `4 × 实际位置数 × 专家数` 字节估算每条待成熟记录的参数样本，`--parameter-proxy-size 0` 可关闭。可选完整表示余弦使用 `--full-encoder-proxy`，每条记录额外保存一份 FP32 表示。原样报告实际耗时和占线上耗时的比例。Loss degradation 仅作为标签 oracle。使用已注册 sketch 阈值和阈值无关指标，预测对照未据此选参。
- C5 prospective 对照沿用同一架构和 offline checkpoint，移除 version-aware memory、保护与定向恢复，用完全成熟记录的 FIFO replay。基线预算为每个成熟 origin 一次 expert 步、至多两个 FIFO 历史记录。Uniform 在 online 冻结 router 并均匀混合。测量 optimizer calls、实际 replay records 和 expert-record draws、参数量、整流与每 origin 耗时及 GPU allocated/reserved 峰值。
- C5 计时版在固定 validation 流测量 Full 与四个候选，以 `abs(log(candidate latency / Full latency))` 最小选择，按候选顺序打破平局。calibration 和对应 test 在同 host、同物理 GPU 串行执行。即便最接近的候选仍有偏差，也直接报告实测值。
- C6 按 12/4/4 个 30 天月行数验证小时和 15 分钟 cadence。scaler 仅拟合训练段。后段仅使用历史 context overlap，目标严格处于该段。纸面划分和 conventional 的绝对 MSE 对应不同测试流。
- R1、R2、R3、R6、R7、R8 的正式预测均值与样本标准差只使用完整 seed 0、1、2。R4、R5 是逐 seed 诊断，可逐个展示已验证流。失败、缺失依赖、损坏 completion 与历史失败尝试保留在原始表。
