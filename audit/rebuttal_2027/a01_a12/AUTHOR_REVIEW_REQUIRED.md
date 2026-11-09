# 已找到证据的作者确认

### CHECK-001 — 八种方法最终参数与选择

- **对应审计**　A01
- **Reviewer**　6htc/ZDG6
- **发现的实际数据**　PAVE Expert 1e-4/5e-5/1e-5、Router 1e-5。DSOF、DynaME、ER、FSNet、OneNet、PROCEED、PatchTST-DGrad 各 12 条 seed 0，线上基础 LR 各 1e-3。所有方法 ECL offline 0.003，其余 0.001。
- **论文对应数据**　Tables 1/6 调参公平性
- **证据路径**　[逐方法和设置的配置](HYPERPARAMETER_AUDIT.csv)
- **当前判断**　候选配置可核，历史在线选择未绑定。
- **为什么需要作者核对**　请逐方法确认正式配置、checkpoint、候选预算和选择依据。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-002 — Table 1 及四组异常

- **对应审计**　A02
- **Reviewer**　fvv/6htc/ZDG6
- **发现的实际数据**　DSOF ETTh2-24、PROCEED ETTh2-48、OneNet WTH-24、FSNet ETTm1-48 seed 0 MSE 为 3.239262、17.438402、0.632996、1.340280。PAVE 有三 seed console。
- **论文对应数据**　3.552±0.443、17.438±0.831、1.362±1.261、1.416±0.067
- **证据路径**　[逐 seed 与 checkpoint](MAIN_RESULTS_RECHECK.csv) · [轨迹](A02_ANOMALY_ORIGIN_MSE.csv.gz)
- **当前判断**　四组 seed 0 有原始数组；所报标准差未复核。
- **为什么需要作者核对**　请确认 Tables 1 的 run/seed/commit、复跑取舍、checkpoint 和 ddof。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-003 — Table 6 MAE 与舍入

- **对应审计**　A02
- **Reviewer**　fvv/6htc
- **发现的实际数据**　本地矩阵 96/96 格与候选 seed 0 吻合。PAVE 三 seed 候选仅 4/12 吻合，ETTh2-24 均值 0.688417，CSV 为 0.681。
- **论文对应数据**　Table 6 三种子 MAE
- **证据路径**　本地 `mae_results.csv` · [主结果 CSV](MAIN_RESULTS_RECHECK.csv)
- **当前判断**　本地矩阵与所查跨 seed 聚合有差异。
- **为什么需要作者核对**　请确认 CSV 的用途、Table 6 输入与舍入/并列规则。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-004 — DSOF legacy 与 PF 两臂

- **对应审计**　A03
- **Reviewer**　ZDG6
- **发现的实际数据**　本地 DSOF student 等整窗，TD 0.2，teacher replay 128/16。三份其他方法 smoke 各 4 origins、5 releases、3 steps。
- **论文对应数据**　Table 1 DSOF、Table 2 DSOF/DynaME/ER
- **证据路径**　results286 · `utils/run_config.py:21` · [对照 CSV](ABLATION_AND_CONTROL_RECHECK.csv)
- **当前判断**　HEAD 与 Table 2 的支持范围不同。
- **为什么需要作者核对**　请确认正式两臂源码、命令与真实 steps/replay。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-005 — Version-aware 分支与消融

- **对应审计**　A04
- **Reviewer**　ZDG6/KaYu
- **发现的实际数据**　候选 top-1、责任 0.3、alignment 0.8。高对齐 retained，低对齐无损 rebase，有害进入可用的 Recovery。
- **论文对应数据**　Eq. 9–11 和 Figure 2/Table 7
- **证据路径**　 `utils/expert_memory.py:22–59,388–441` · [实现和消融](AUDIT_OVERVIEW.md)
- **当前判断**　可核代码；提交版和独立 loss-only 待查。
- **为什么需要作者核对**　请确认三个消融 flags、reference 和 loss-only 的 scoring/replay 口径。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-006 — Table 8 高对齐样本

- **对应审计**　A05
- **Reviewer**　ZDG6/fvv/KaYu
- **发现的实际数据**　ETTh2-24 seed 0 的 9811 个高对齐 owner 中，748 any、529 >5%、396 >10%、254 >20%、101 >50%。
- **论文对应数据**　N=10000，95.80% no、4.20% any、1.80% >10%、0.84% >20%、0.13% >50%
- **证据路径**　results253 的 credit NPZ · [分层统计](MECHANISM_STATISTICS.csv)
- **当前判断**　本地重算成立，论文样本身份未绑定。
- **为什么需要作者核对**　请确认是否责任筛选、重复、抽样窗口、reference loss、分母。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-007 — Figure 3(a)(c) reversal

- **对应审计**　A06
- **Reviewer**　fvv/ZDG6
- **发现的实际数据**　四数据集本地 seed 0 的 H=1/24/48 为 0/40000、25886/40000、26357/40000，H=1 的 expert_update_delta=0。
- **论文对应数据**　N=2000，总体 49.9%，分 H 21.0/44.7/84.1%
- **证据路径**　[两种 alignment 定义和 Wilson CI](MECHANISM_STATISTICS.csv)
- **当前判断**　本地与论文摘录比例不同，样本及版本映射待查。
- **为什么需要作者核对**　请确认 H=1 的版本切换时点，附抽样名单、bins 和分母。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-008 — Memory、Subspace 和 replay

- **对应审计**　A07
- **Reviewer**　fvv/ZDG6
- **发现的实际数据**　ETTh2-24 有 1189 次尝试、83 次恢复、77 次晋升、183 次 rebase。Recovery loss 的 head 梯度也经过过滤。
- **论文对应数据**　Figure 3(b) 120 对，1.47→1.12，97.5%
- **证据路径**　[事件表](MECHANISM_STATISTICS.csv) · `utils/subspace_protection.py:121–345`
- **当前判断**　代码和计数可核，配对原件未取得。
- **为什么需要作者核对**　请确认事件定义、重复样本、固定 reference 和 120 对来源。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-009 — Table 10 三个容量对照

- **对应审计**　A08
- **Reviewer**　ZDG6/6htc
- **发现的实际数据**　Full 候选四 experts、horizon-channel。旧 WTH-24 MoE-only 为 0.431835 MSE / 0.436583 MAE。
- **论文对应数据**　Uniform/Matched/Full 及其 settings
- **证据路径**　 `log/20260721_134717_moe_tsb_ablation_wth/results_summary.csv` · [对照 CSV](ABLATION_AND_CONTROL_RECHECK.csv)
- **当前判断**　旧结果未映射 Table 10；matched seeds 待绑定。
- **为什么需要作者核对**　请确认三个方法构造、checkpoint、有效 replay 和正式 settings/seeds。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-010 — ETT 划分和数据身份

- **对应审计**　A09
- **Reviewer**　KaYu
- **发现的实际数据**　两个数据集各 2880/720/10800 点，时长 120/30/450 与 30/7.5/112.5 天。
- **论文对应数据**　Appendix B.1
- **证据路径**　 `data/data_loader.py:49–113,153–215` · [日期](AUDIT_OVERVIEW.md) · [哈希](../AUDIT_OVERVIEW.md)
- **当前判断**　代码和日期一致，论文采用待查。
- **为什么需要作者核对**　请确认 CSV 哈希、边界及 conventional 是否有原始结果。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-011 — Table 11 计算开销

- **对应审计**　A10
- **Reviewer**　ZDG6
- **发现的实际数据**　历史 PAVE ECL-48 seed 0 为 33161.612 秒/19681 origins = 1684.956 ms/origin。
- **论文对应数据**　PAVE 165、Matched MoE 125 ms/origin 及参数、显存、状态
- **证据路径**　results259 的 metrics.npy · [分口径表](ABLATION_AND_CONTROL_RECHECK.csv)
- **当前判断**　历史整流耗时可查，profiler 待补。
- **为什么需要作者核对**　请确认两份耗时的设备、同步、warmup、区间和重复数。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-012 — 消融和 recurring

- **对应审计**　A11
- **Reviewer**　全体
- **发现的实际数据**　78 个 seed 0 配对引用同一现存 checkpoint 文件，39 对记录 commit 不同。9 月 12 日 6 个开关冲突、5 个 MSE 冲突。Recurring 代码用首步 MSE。
- **论文对应数据**　Table 7/Figure 2；Table 9/Figure 4 的 32.9%、40.6%、51.8%
- **证据路径**　[消融和 recurring](ABLATION_AND_CONTROL_RECHECK.csv) · [冲突行](EXPERIMENT_INVENTORY.csv)
- **当前判断**　本地候选可算，提交映射及 recurring 数据待补。
- **为什么需要作者核对**　请确认各变体实际采用的 run、seed、窗口、聚合与恢复失败处理。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

### CHECK-013 — 命名、默认与稿件版面

- **对应审计**　A12
- **Reviewer**　全体
- **发现的实际数据**　本地有 PAVE/PACE/DRSE；通用 CLI 与 Full launcher 的 feedback、router、rank 默认不同。
- **论文对应数据**　名称、Eq. 9–11、§4.2、Figure 1
- **证据路径**　 `../PAVE/README.md:1`、`README.md:288`、`main.py:205,213,251`
- **当前判断**　本地差异已核，稿件逐式与版面待核。
- **为什么需要作者核对**　请提供提交 PDF/TeX 和图，确认名称、公式和版本。
- **author_confirmed**　[ ]
- **author_correction**　[ ]
- **author_note**　[ ]

