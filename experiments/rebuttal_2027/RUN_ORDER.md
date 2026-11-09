# 执行顺序与资源

先运行 `inspect` 并查看 [代码审计](CODE_AUDIT.md)、[DSOF 审计](DSOF_PROTOCOL_AUDIT.md) 和 C6 manifest 的日期。用已配置 PyTorch 和 GPU 的正式解释器执行脚本。命令默认 dry-run，`EXECUTE=1` 才启用生成的 launcher。现有日志可以先做只读汇总。

1. 第一批执行 C1 公平 LR、C2 OFAT、C3 loss-only。先 pilot，再按同一预注册配置扩至 full。C1 先 prepare、sweep、select，最后 test。C2、C3 各自 prepare 后 test。
2. 第二批汇总 C3 Full 的 C4 diagnostics。C7 审计后可启动 C7 pilot，顺序同 C1。C6 先检查日期、scaler 与独立 checkpoint，再按需启动 pilot。
3. 第三批执行 C5，需计时时使用单 GPU calibration、freeze、同硬件测试。之后扩展 C6 conventional 和 C4 可选 full encoder proxy。Full proxy 需要单独 run-id；优先从较小 setting pilot 估算开销。

| 实验 | Pilot prepare / val / test | Full prepare / val / test | 相对成本 |
|---|---:|---:|---|
| C1 | 36 / 60 / 72 | 288 / 480 / 576 | Full 的八方法网格和三 seed 测试数量最多 |
| C2 | 6 / 0 / 66 | 9 / 0 / 99 | 单 PAVE 多在线控制，复用训练 |
| C3 | 6 / 0 / 30 | 36 / 0 / 180 | 带 gzip 与可选快照的五种 PAVE 回放 |
| C4 | 6 / 0 / 6 | 36 / 0 / 36 | 与同规模 C3 共用，新增 GPU job 为 0 |
| C5（带计时） | 3 / 5 / 12 | 9 / 15 / 36 | 标准 Full 和候选校准须串行测时 |
| C6 | 48 / 0 / 48 | 96 / 0 / 96 | 独立预训练与长 conventional 流 |
| C7 | 6 / 15 / 18 | 24 / 60 / 72 | 两个 DSOF 反馈变体与 PAVE 的 LR 网格 |

Prepare 数是各组请求数，跨组相同 offline key 会合并。Select、freeze、collect、report 为 CPU。C5 去掉 time-match 时每个 setting 少五个验证 job 和三个测试 job。GPU 峰值、墙钟时间、累计 GPU 小时目前为 unknown。先用 smoke 确认依赖和 GPU 余量，再以同卡 pilot 的 `runtime`、`mean_origin_ms`、`peak_gpu_memory_bytes` 与 `peak_gpu_reserved_bytes` 估算 full。不同硬件上的测量分别保留。完整清单同时列出每个 job 的期望 origin 数和前置条件。
