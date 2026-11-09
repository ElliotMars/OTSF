# 手动命令

在仓库根目录运行。为 `python` 选用能导入项目依赖和 PyTorch 的正式运行环境。生成的脚本用当前 PATH 中的 `python`。如作业调度系统重映射 GPU，请将可见的物理数字 ID 传给 `GPU_IDS`。worker 每次只暴露一张卡，在进程内使用逻辑 `cuda:0`。

```bash
base=/opt/data/private/online/experiments/rebuttal_2027
python -B scripts/rebuttal_suite.py inspect --experiment all --scope pilot --time-match
python -B scripts/rebuttal_suite.py status --experiment all --scope pilot --time-match
```

## 每组检查和 dry-run

```bash
bash "$base/generated_commands/pilot/C1/run.sh"
bash "$base/generated_commands/pilot/C2/run.sh"
bash "$base/generated_commands/pilot/C3/run.sh"
bash "$base/generated_commands/pilot/C4/run.sh"
bash "$base/generated_commands/pilot/C5/run.sh"
bash "$base/generated_commands/pilot/C6/run.sh"
bash "$base/generated_commands/pilot/C7/run.sh"
```

也可用 `python -B scripts/rebuttal_suite.py inspect --experiment C6 --scope pilot` 检查某组。指定 `run --phase` 可以只看一阶段，加 `--dry-run` 可覆盖任何已有的 `--execute`。

## Pilot 和 full

将下例的 `exp` 分别设为 `C1`、`C2`、`C3`、`C4`、`C5`、`C6` 或 `C7`。同一组可按表任选一种执行方式。C4 与同 scope、run-id 的 C3 Full 共享任务，C3 已完成时直接运行汇总。Pilot/full 脚本包含 C5 可选计时对照，多卡 C5 的 prepare 可分发，校准与对应测试串行绑定同一张物理卡。

```bash
exp=C1
# Pilot 单卡
EXECUTE=1 CUDA_VISIBLE_DEVICES=0 bash "$base/generated_commands/pilot/$exp/run.sh"
# Pilot 多卡
EXECUTE=1 GPU_IDS=0,1 MAX_PER_GPU=1 bash "$base/generated_commands/pilot/$exp/run.sh"
# Full 单卡
EXECUTE=1 CUDA_VISIBLE_DEVICES=1 bash "$base/generated_commands/full/$exp/run.sh"
# Full 多卡
EXECUTE=1 GPU_IDS=0,1 MAX_PER_GPU=1 bash "$base/generated_commands/full/$exp/run.sh"
```

C5 单独固定 calibration GPU 时使用 `CALIBRATION_GPU=1 GPU_IDS=0,1`。测时期间应让校准 GPU 专用。若只需要 C5 的三个 capacity arms，用另一个 run-id 导出基础版本。

```bash
python -B scripts/rebuttal_suite.py plan --experiment C5 --scope pilot --run-id capacity-only \
  --export --export-root "$base/generated_commands/capacity-only"
# 上一行仅生成命令。下行是可选的手动执行。
EXECUTE=1 GPU_IDS=0,1 bash "$base/generated_commands/capacity-only/pilot/C5/run.sh"
```

Smoke 用 `generated_commands/smoke/$exp/run.sh`，同样以 `EXECUTE=1` 手动启用。每组的 `planned_manifest.json` 记录 settings、配置、预计 origins 和数据指纹。

## 单 job 与 nohup

从 [GPU_JOB_MANIFEST.csv](GPU_JOB_MANIFEST.csv) 取任一 phase 和 task_id。每个命令也有独立文件。

```bash
job=multi_expert__ETTm1__48__full__s0__full
# 先检查，再执行。C3 test 依赖同 scope 的 prepare。
bash "$base/generated_commands/pilot/C3/jobs/test/$job.sh"
EXECUTE=1 CUDA_VISIBLE_DEVICES=1 bash "$base/generated_commands/pilot/C3/jobs/test/$job.sh"

# 对任一实验设置 exp。输出为新建的 launcher attempt 目录。
exp=C2
attempt=$(EXECUTE=1 GPU_IDS=0,1 bash "$base/generated_commands/pilot/$exp/nohup.sh")
cat "$attempt/pid"
tail -f "$attempt/run.log"
cat "$attempt/exit_code"
```

后台 wrapper 每次先建立独立日志目录，记录 PID、stdout/stderr 和完成后的退出码。停止前可用 `pstree -p "$(cat "$attempt/pid")"` 查出调度器和 worker，按 PID 从 worker 开始发送 `kill -TERM PID`，再检查 GPU 进程已退出。已完成的结果仍保留。再次运行同一脚本会以 `--resume` 校验产物并重试未完成项。人工启动单 job 时也可用 `--job`，或按 `--seed`、`--variant` 筛选已注册任务。

## 汇总

```bash
python -B scripts/rebuttal_suite.py report --experiment all --scope pilot --run-id rebuttal \
  --time-match --report-root "$base/reports/final"
```

将 `--scope pilot` 改为 `--scope full` 汇总完整规模。将 `--experiment all` 改成 C1–C7 的任一项，只刷新该项。可将不同 scope 写入不同 `--report-root`。若只有早先独立 C1 结果，用 `--c1-root /absolute/path/to/c1/run` 补充；已有同 scope/run-id 的统一 C1 manifest 时优先读取统一结果。正式结果目录包含 R1–R8 的 CSV、JSON、Markdown、LaTeX，以及每组的 raw trial、失败尝试和日志索引。
