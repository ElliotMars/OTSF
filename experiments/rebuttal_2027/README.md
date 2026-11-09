# PAVE ICLR 2027 rebuttal 实验

本目录提供 C1–C7 的代码入口、可独立提交的任务和 R1–R8 汇总。工作区准备阶段仅执行 CPU 测试和命令核查。正式指标由手动运行后生成。

| 文件 | 内容 |
|---|---|
| [代码审计](CODE_AUDIT.md) | 历史证据与本地实现定位 |
| [实验计划](EXPERIMENT_PLAN.md) | 比较臂、范围、结果分母和计时定义 |
| [配置来源](CONFIG_PROVENANCE.md) | 原始参数、改动项、反馈与缓存键 |
| [DSOF 协议审计](DSOF_PROTOCOL_AUDIT.md) | 历史 baseline、部分反馈适配与原生机制 |
| [执行顺序](RUN_ORDER.md) | 批次、依赖、资源估算 |
| [手动命令](RUN_COMMANDS.md) | 单 GPU、多 GPU、单 job、nohup 与恢复 |
| [GPU_JOB_MANIFEST.csv](GPU_JOB_MANIFEST.csv) | smoke、pilot、full 的逐任务清单 |
| [汇总模板](reports/final/README.md) | 报表规则及正式结果目录 |

在仓库根目录运行。计划、dry-run 和结果汇总仅依赖 Python 标准库。正式 worker 使用项目的 PyTorch 环境。

```bash
python -B scripts/rebuttal_suite.py inspect --scope pilot --time-match
python -B scripts/rebuttal_suite.py run --experiment C3 --scope pilot --phase test --dry-run
python -B scripts/rebuttal_suite.py report --experiment all --scope pilot --run-id rebuttal
```

默认 run-id 为 `rebuttal`。同一 scope 和 run-id 下，C4 读取 C3 的 Full PAVE 任务。修改数据路径、注册参数、源码、Python 环境或候选设置后，为新正式运行选用新的 run-id。
