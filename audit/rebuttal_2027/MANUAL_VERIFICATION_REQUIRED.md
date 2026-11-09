# 作者手动核查

已搜索仓库及其 Git 历史、本地 `result/`、`checkpoints/`、`log/`、`scripts/`（含被 Git 忽略的子目录）、数据、同级 `../PAVE` 和相关名称匹配的 `../paper`、`../tmp`、`/tmp` 文件。优先核查训练机器、原论文制表目录和作业记录。每项只需提供列出的最小材料。

### MAN-001 · P0 · 6htc / fvv / ZDG6 · online LR 选择

已知 9 月 baseline 日志有实际 LR、epoch 级 validation 和 seed 0。PAVE 有分数据集基准 Expert LR 和 dynamic controller。旧版 HP launcher 与 offline validation 比较表可定位。现有材料缺搜索结果与最终采用运行的链接，且无法确定是否参考 test 指标。请在训练机的 `hp_search/`、作业记录、调参表或笔记中找候选命令、dataset/H、validation 时间范围、在线初始化、各候选 validation MSE、选择与并列规则，以及最终采用配置。若当时没有同一套搜索，请说明每种方法真实采用依据。这可验证调参公平性与是否使用测试集。

### MAN-002 · P0 · 6htc / fvv / ZDG6 · Tables 1/6/7 和 Figure 2 的制表来源

已知有本地 baseline seed 0 结果、历史重跑、PAVE seed 1/2 日志和 PACE MAE 汇总，复算范围见统计报告。缺完整论文版本、每格的 run 路径及采纳历史。请提供提交 PDF/TeX 或对应表页、制表/绘图脚本和一份 cell-to-run 清单，至少含 method/variant、dataset/H、seed、commit、原始产物目录、checkpoint、split、mean/std 及舍入规则。优先核对 9 月 12 日两个消融共用日志目录的原始运行，现存 stdout 与同名 JSON/数组在开关和 MSE 上出现冲突。另取回 DSOF ETTh2-24、PROCEED ETTh2-48、OneNet WTH-24、FSNet ETTm1-48 的 seed 1/2，和全部消融的最终 seeds。搜索 9 月 baseline/ablation wrapper 指向目录、远端 `/autodl-tmp/online/result/results*/`、jobs archive。收到后可复算全表、检查 tie、异常方差和重跑污染。

### MAN-003 · P0 · ZDG6 / 6htc · DSOF native fast stream 与 Table 2

已知归档本地 DSOF 整窗才更新 student，现公开 PF 入口只支持 FSNet/OneNet/DynaME，存在三份 H=2 短流诊断。缺 Table 1 DSOF 的最终运行链接、Table 2 的实际 DSOF/DynaME/ER 两臂及其实现版本。请从其他训练机、未 push 分支或提交时脚本中提供两臂命令、代码 diff 或 commit、配置、per-seed metrics、checkpoint 和一个含每 origin 可见标签、fast/slow 或 ER update/replay count 的轨迹。收到后可核对 native TDL 是否保留、PF 对照是否改变监督与步数，以及 HEAD 与论文是历史分叉还是同版本冲突。

### MAN-004 · P0 · ZDG6 / fvv / KaYu · Appendix E 与 Figure 3 样本

已知本地 `credit_diagnostics.npz` 支持尾窗 owner、alignment、损失和 reversal 复核；常规 exporter 最多 10,000，内置 binner 使用 mean-expert alignment。论文的 10,000 与约 2,000 没有 run ID、采样定义和列表。请从 plot 工作目录、notebook 和导出脚本提供记录 key（dataset/H/seed/run/origin/expert）、抽样 seed、范围/分层和去重规则、阈值、低责任过滤、loss 分母、reversal alignment/bins，以及最终图输入。收到后可验证高对齐降级、各 horizon 样本量和 Wilson 区间，区分完整流覆盖与 tail。

### MAN-005 · P0 · ZDG6 · Loss-only / always-on / Matched MoE

已知 HEAD 没有定位到可归属论文的 loss-only 或 always-on 实验，也未定位 Table 10 的真实入口，uniform 初始化与早期 MoE/TSB 对照可查。请在未公开对照分支、Figure/Table 源目录和 job archive 查 `*loss_only*`、`*always_on*`、`*matched*moe*`、`*uniform*moe*`。提供存在的代码或命令、模型和 optimizer 参数签名、seed 0 原始运行、其他 seeds（如有），以及每 origin 更新、replay 记录数和候选池。另确认 loss-only 的 rebase、ranking、promotion 及 replay budget 的定义。收到后可区分可复用结果和必须补跑的范围。

### MAN-006 · P1 · fvv / ZDG6 · Figure 3(b) Recovery 120 对

已知运行目录的 `online_diagnostics_summary.json` 可查事件计数，源码可跟踪 before objective 和 after migration。缺逐事件 paired loss 及采样 provenance。请在作图目录或专用 profiler/export 查 `*recovery*.{csv,npz,jsonl}`，提供 event ID、run/seed、origin、record/expert ID、update index、before/after MSE、冻结的 reference loss、epsilon、过滤和 improvement 阈值。可核对 1.47→1.12、中位数、117/120 的可能分子和重复 replay。

### MAN-007 · P1 · KaYu / ZDG6 · ETT conventional split 与原始数据身份

已知当前两套 ETT loader 是相同观察数，calendar-like 12/4/4 在注释中。历史运行日志来自多个版本/路径。请确认论文采用的两个 CSV 版本、checksum、时区/频率、实际 train/validation/test 和重叠边界。若另有 conventional 实验，请提供 split 实现、边界打印、独立 checkpoint 和结果。可确认 split 一致性、因果性和是否已有标准划分证据。

### MAN-008 · P1 · ZDG6 / fvv · Table 9 recurring

已知有 3-seed dry-run launcher 和指标定义，未找到本地可关联的 synthetic 原始 timeline。请在 `result/recurring_drift_maintext/`、远端结果归档或绘图目录找每个 seed/variant 的 `synthetic_timeline.npz`、`predictions_and_mse.npz`、`drift_events.json`、`drift_metrics.json`、`memory_diagnostics.json` 与原始命令。还需确认对照、first-step 或 full-horizon MSE、窗口、未恢复 seed 处理和 ratio-of-means 或 mean-of-ratios。可复算三项收益和检查边界。

### MAN-009 · P1 · ZDG6 / fvv · Table 11 profiling

已知普通日志有整段运行 elapsed，未发现同步计时或显存 trace。请在 ECL-48 profile job 目录提供 `*profile*.{json,csv,nsys-rep}`、实际运行命令、GPU/CUDA/PyTorch、精度和 batch、同步、warmup/重复、训练参数计数脚本、persistent state 的边界和峰值定义。可验证逐 origin 延迟、容量、有效 update/replay 匹配和 wall-clock 匹配。

### MAN-010 · P2 · fvv / ZDG6 · Table 8、Figure 4 与 reviewer 原文

任务提供编号，未提供这两项表图主题、样本来源或完整 reviewer 文本。请提供对应稿件页及图表数据脚本，并附 OpenReview 对应评论段落。可以把当前证据矩阵的未归属项绑定到准确的主张和审稿问题。
