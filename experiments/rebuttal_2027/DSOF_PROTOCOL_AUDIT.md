# C7 DSOF 反馈协议审计

本地主表候选配置来自 [native profiles](../c1_lr_tuning/native_profiles.json)，本地 launcher 在 [run_dsof.sh](../../scripts/run_dsof.sh#L8) 使用 `--delay_fb` 与 `--online_learning full`。当前入口 [exp_dsof.py](../../exp/exp_dsof.py#L1) 绑定 ExpDSOF。上游描述 fast learner 使用 incomplete feedback 做 TD，slow learner 使用 complete feedback 做 replay。参见 [DSOF 论文](https://openreview.net/pdf?id=I0n3EyogMi) 和 [官方实现](https://github.com/yyalau/iclr2025_dsof)。

| 项目 | 归档本地适配 | 新增本地部分反馈适配 |
|---|---|---|
| 推理 | PatchTST teacher 加 residual student，预测后入延迟队列 | 保留同模型与推理结构 |
| Student | 启用。队列成熟后以完整窗口的 residual loss 更新，并使用相邻输出重叠项 | 每个新观测校正上一 origin 首目标，剩余重叠以当前 teacher + student 预测作 detached bootstrap |
| Teacher | 成熟窗口触发，已有 buffer 时从旧成熟记录 replay | 仍按完整窗口成熟触发。窗口由逐步出现的历史观测重建，当前窗口在 replay 后才入库 |
| Online optimizer | Teacher AdamW、student Adam。各 LR 见原始 profile，默认 student 为 1e-3 | 两个 optimizer 同类型，同倍率验证网格分别调选 |
| 更新频率 | Student 每个成熟窗口最多一次，teacher 在旧 buffer 可用时每个成熟窗口最多一次 | Student 在首 origin 后每步一次，teacher 只在成熟且旧 buffer 可用时更新 |
| 因果边界 | 当前窗口只进入延迟队列和评估，满 horizon 后才参与训练 | 训练只依赖历史 context 最后一条新观测、已输出的预测和完全成熟的 replay。当前 y 供评估 |
| 与官方机制的关系 | 快慢流都等待完整窗口，时序与官方 fast incomplete stream 不同 | 局部恢复 fast/slow 的反馈时序，保留本地模型、teacher/student 损失和 replay 细节，协议标为 `dsof_partial_td_adaptation` |
| 与通用 progressive matched control 的差异 | 使用本地 DSOF 双网、完整窗口队列 | 使用双网与 TD bootstrap。通用 progressive control 是另一入口的 masked partial MSE，覆盖 FSNet、OneNet、DynaME |

本地细节见 [模型初始化和 residual forward](../../exp/exp_stream_baselines.py#L48)、[完整窗口更新](../../exp/exp_stream_baselines.py#L156)、[通用 progressive 更新](../../exp/exp_stream_baselines.py#L183)、[延迟队列](../../exp/exp_stream_baselines.py#L242)、[部分 TD 分支](../../exp/exp_stream_baselines.py#L272) 和 [协议校验](../../utils/run_config.py#L121)。R8 保留两个本地 DSOF 变体及 PAVE 的默认与验证选择 LR、完整三 seed 指标和实际更新计数。
