# 缺失或待恢复的材料

### MAN-001 — 公平 LR 历史

- **对应审计**　A01
- **Reviewer**　6htc/ZDG6
- **Priority**　P0
- **已找到**　84 baseline 和 36 PAVE 候选参数，旧 launcher 12+2 组合及 offline validation。
- **缺少的具体内容**　在线候选、共同验证流、真实搜索次数和选值。
- **已搜索的路径**　scripts/hp_search、log/pretrain_*、运行日志、Git。
- **缺失原因**　搜索运行和最终选值未绑定。
- **作者需要提供**　逐方法/设置候选、命令、causal validation MSE、预算和选择规则。
- **推荐文件名或路径模式**　*lr*search*.csv、*validation*.jsonl、job history
- **得到文件后可以验证**　公平性及测试集是否参与选参。
- **是否阻碍 rebuttal**　是
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-002 — 主表和消融的原始种子与映射

- **对应审计**　A02/A11
- **Reviewer**　fvv/6htc/ZDG6
- **Priority**　P0
- **已找到**　96 个主候选 seed 0 及 PAVE seed 1/2 console，78 个消融配对。
- **缺少的具体内容**　baseline seed 1/2、消融种子和 Table 1/6/7/Figure 2 映射，9 月 12 日冲突原件。
- **已搜索的路径**　result、log、checkpoints、聚合 CSV、Git。
- **缺失原因**　证据不全，/autodl-tmp/online 未挂载。
- **作者需要提供**　每格 run ID、seed、commit、checkpoint、原始指标、ddof 和制表脚本。
- **推荐文件名或路径模式**　result/results*/**/itr_*/{metrics.npy,run_config.json,preds.npy,trues.npy}
- **得到文件后可以验证**　均值、方差、旧流和覆盖溯源。
- **是否阻碍 rebuttal**　是
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-003 — Table 2 和原生 DSOF

- **对应审计**　A03
- **Reviewer**　ZDG6
- **Priority**　P0
- **已找到**　有 legacy DSOF 和三份其他方法短流。
- **缺少的具体内容**　Table 2 DSOF/DynaME/ER 两臂原始运行、实际快慢流和 ER 步数。
- **已搜索的路径**　scripts、log、result、../PAVE、Git。
- **缺失原因**　HEAD PF 只支持 FSNet/OneNet/DynaME。
- **作者需要提供**　两臂命令、对应源码/commit、每 seed 指标、release 和 optimizer traces。
- **推荐文件名或路径模式**　*progressive_baseline*、*dsof*pf*、protocol_diagnostics.json
- **得到文件后可以验证**　正式 DSOF 反馈与 Table 2 数值。
- **是否阻碍 rebuttal**　是
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-004 — 验证策略与 Table 8/Figure 3 抽样

- **对应审计**　A04/A05/A06
- **Reviewer**　fvv/ZDG6/KaYu
- **Priority**　P0
- **已找到**　125 个可重算 archive、分类代码和 H=1 零更新证据。
- **缺少的具体内容**　提交表图的逐行样本、运行版本、抽样/阈值和 loss-only/always-on 原始对照。
- **已搜索的路径**　result/**/credit_diagnostics.npz、源码、scripts、Git、相邻副本。
- **缺失原因**　样本 manifest 和对照证据缺失。
- **作者需要提供**　record/expert/run/origin 列表、筛选规则、同一 reference、bin、CI、现有对照命令。
- **推荐文件名或路径模式**　*high*alignment*.csv、*reversal*.npz、*loss*only*
- **得到文件后可以验证**　降级、reversal 与验证条件的独立效果。
- **是否阻碍 rebuttal**　是
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-007 — ETT 数据身份与 conventional

- **对应审计**　A09
- **Reviewer**　KaYu
- **Priority**　P0
- **已找到**　已核行号、日期、哈希及邻库 conventional loader。
- **缺少的具体内容**　论文 CSV/split/checkpoint 绑定及常规划分结果。
- **已搜索的路径**　data、result、Git、../DynaME/data_provider。
- **缺失原因**　当前 OTSF 仅查到观察数划分证据。
- **作者需要提供**　采用的 CSV 校验和、split manifest、运行 ID 和结果。
- **推荐文件名或路径模式**　*conventional*ett*、*split*.json
- **得到文件后可以验证**　数据身份、因果边界及稳健性。
- **是否阻碍 rebuttal**　部分
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-005 — Capacity-matched control

- **对应审计**　A08
- **Reviewer**　ZDG6/6htc
- **Priority**　P1
- **已找到**　Full 候选及旧 WTH MoE-only 记录。
- **缺少的具体内容**　Table 10 settings、三方法初始化、更新/replay 预算及 seed 0/1/2。
- **已搜索的路径**　scripts、log、result、Git、../PAVE。
- **缺失原因**　旧 MoE 产物归属未绑定。
- **作者需要提供**　Table 10 表页和三个方法的 run ID、模型配置及有效 step/replay traces。
- **推荐文件名或路径模式**　*matched*moe*、*uniform*moe*、update_trace.jsonl
- **得到文件后可以验证**　容量、预算及逐 seed 指标。
- **是否阻碍 rebuttal**　是
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-006 — Figure 3(b) replay 配对

- **对应审计**　A07
- **Reviewer**　fvv/ZDG6
- **Priority**　P1
- **已找到**　已有迁移、尝试及 rebase 计数。
- **缺少的具体内容**　120 个 replay 的 before/after、冻结的 reference 和纳入规则。
- **已搜索的路径**　result、diagnostic/exporter、绘图代码、Git。
- **缺失原因**　导出以事件计数和局部诊断为主。
- **作者需要提供**　每事件 before/after、reference、record/expert、重复和阈值。
- **推荐文件名或路径模式**　*replay*.jsonl、*replay*.npz
- **得到文件后可以验证**　中位数和改善率。
- **是否阻碍 rebuttal**　部分
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-008 — Recurring 原始时间序列

- **对应审计**　A11
- **Reviewer**　全体
- **Priority**　P1
- **已找到**　三 seed launcher、窗口实现和首步 MSE 定义。
- **缺少的具体内容**　三种子三变体的原始指标及百分比聚合。
- **已搜索的路径**　scripts、result、log、../PAVE。
- **缺失原因**　未定位对应 synthetic 产物。
- **作者需要提供**　timeline、drift metrics、reference 窗口和未恢复处理。
- **推荐文件名或路径模式**　result/recurring_drift_maintext/**/{synthetic_timeline.npz,drift_metrics.json}
- **得到文件后可以验证**　Table 9 / Figure 4 和三个收益。
- **是否阻碍 rebuttal**　部分
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-009 — ECL-48 计时与资源

- **对应审计**　A10
- **Reviewer**　ZDG6
- **Priority**　P1
- **已找到**　普通测试 elapsed 和 origins。
- **缺少的具体内容**　配对 profiler、参数/显存/持久状态测量。
- **已搜索的路径**　result、log、scripts、Git 及相关归档名。
- **缺失原因**　未取得 Table 11 可关联 profile。
- **作者需要提供**　原始时间样本、硬件环境、同步、warmup、测量范围和参数清单。
- **推荐文件名或路径模式**　*profile*.json、*profile*.csv、*.nsys-rep、*timing*.jsonl
- **得到文件后可以验证**　165/125 ms 和资源口径。
- **是否阻碍 rebuttal**　部分
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### MAN-010 — 提交稿与版面

- **对应审计**　A12
- **Reviewer**　全体
- **Priority**　P2
- **已找到**　已核历史命名、CLI 与 Full launcher 默认差异。
- **缺少的具体内容**　提交 PDF/TeX、Figure 1、§4.2、公式页和评论原文。
- **已搜索的路径**　../paper、../PAVE、/tmp、../tmp、Git 和公开检索。
- **缺失原因**　未取得 #43115 提交稿。
- **作者需要提供**　稿件、图原件或有权访问的链接。
- **推荐文件名或路径模式**　*43115*.pdf、main.tex、fig1.*
- **得到文件后可以验证**　逐式、重复段落与图标签核对。
- **是否阻碍 rebuttal**　部分
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

