# 可用于 rebuttal 的证据

## 6htc · 公平 online LR 选择与敏感性

- **关切**　baseline 调参公平，PAVE 超参是否稳健。
- **代码**　方法的实际 optimizer 和控制参数入口不同。PAVE Expert 基准 LR 分数据集，Router 基准共享；dynamic 调整完整更新，局部 Router 有单独的更新路径，并继承 optimizer 当时的 LR。
- **现有实验**　本地近期 baseline seed 0 的实际参数可查。旧 HP 脚本及 offline validation logs 表示存在候选入口和 checkpoint 选择信息。可复算统计见 [统计复核](STATISTICAL_RECHECK.md)。
- **待核与补做**　MAN-001/002。先取回共同在线验证流与选择证据。若无法恢复，独立的公平 online-LR tuning 和 sensitivity 实验最直接，见 [实验计划](NEXT_EXPERIMENT_PLAN.md)。
- **回应范围**　陈述已查日志的具体配置及当时有材料可证的过程。对最终选择的验证来源和公平性，使用作者确认后有凭据的陈述。

## ZDG6 · Loss-only、high-alignment failure、capacity-matched、DSOF

- **关切**　alignment 是否贡献于验证，漏报如何量化，容量/步数能否匹配，以及 DSOF 是否保留原生 partial feedback。
- **代码**　高 alignment 直接 retained；loss 已计算。现有 w/o version 同时关闭 direction 和 rebase，也会改变记忆和证据权重。matched 对照入口未定位。归档 DSOF delayed 代码给 teacher replay 和 student residual TD，但 student 等整窗。公开 PF 只支持 FSNet/OneNet/DynaME。
- **现有实验**　本地逐记录诊断可量化定义明确的高对齐降级。三份真实 PF 短流各 4 origins、5 released events、3 steps，属于另外三种方法的 H=2 smoke。
- **待核与补做**　MAN-003/004/005。需要 loss-only、always-on high-alignment loss check、匹配 3 seed 以及能追溯 Table 2 的 fast/slow 和 replay budget。
- **回应范围**　报告 gating 局限和受限样本口径，明确列出 local DSOF 的 supervision 时点。将 architecture、update/replay 和耗时匹配分别表述。

## fvv · 异常方差与 capability sketch

- **关切**　四个 baseline cell 的异常 MSE/SD 是否由记录、统计、配置、漂移或不稳定造成。
- **代码**　population 与 sample SD 对三种子相差 `sqrt(3/2)`；评估器和 table script 是否共用定义需追踪。sketch 是 encoder proxy，输出头改动可能漏检。
- **现有实验**　可按历史运行复核预测、真值、样本数和 origin-level spikes。四项报告值需要对应三个 seed 的出处，详见 [统计复核](STATISTICAL_RECHECK.md)。
- **待核与补做**　MAN-002/004/006。取回三 seed 的 checkpoint、协议和输出；必要时在固定已确认的配置下做 variance re-evaluation，记录可定位的更新和错误轨迹。
- **回应范围**　区分已算出的历史数据与论文报告行。仅凭大误差或 spike 不能判定发散，配对 replay 的 97.5% 等待原始事件。

## KaYu · ETT 划分与 alignment 的局限

- **关切**　ETTh2/ETTm1 的训练验证时间跨度，及 encoder alignment 对 head drift 的可见性。
- **代码**　相同 2880/720 observations 和从 3600 开始的测试目标，ETTh2 为 120/30 天，ETTm1 为 30/7.5 天。scaler fit 训练段。FSNet/FSNet-Time sketch 来自不同轴的池化，预测头输入独立。
- **现有实验**　本地数组可按样本数核对当前 observation split；不能据此确认 conventional split 的表现。
- **待核与补做**　MAN-004/007。核对数据校验和及版本，补 conventional ETT split 和 capability sketch vs output drift。
- **回应范围**　说明实际 observation-count 协议、真实时间范围、head-only drift 边界及逐记录诊断的覆盖范围。
