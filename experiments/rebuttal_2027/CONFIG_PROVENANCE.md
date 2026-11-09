# 配置来源与复用

C1 的 96 个 method/setting 原始参数来自 [native_profiles.json](../c1_lr_tuning/native_profiles.json) 指向的本地 stdout 档案。离线和在线 LR 映射、配置历史、checkpoint 证据见 [C1 审计](../c1_lr_tuning/C1_CODE_AUDIT.md)。按当前源码重新训练的 checkpoint 带本地源码指纹和归档来源。

[候选注册表](CONFIGS/suite.json) 固定 C2 的 OFAT、C3 policies、C5 预算与 C1/C7 LR 网格。每次正式运行把 resolved config、checkpoint 和 split 协议写入 manifest。C2、C3、C4、C5、C7 的额外参数只在 online path 生效，离线训练使用原始配置。C6 只改变 split 相关输入，沿用各方法原始训练超参和反馈协议。

共享缓存以 method、offline config、seed、数据指纹、代码指纹和软件版本为键，复用前重新检查 checkpoint 文件 SHA-256。独立在线任务保留 stage、method、setting、seed、variant、实际 LR、split、反馈协议、offline key、online config hash、冻结选择与所需诊断产物。缺少任何必须产物的任务不能标记完成。失败尝试保留，重试使用新日志与 attempt 目录。

- C4 使用 C3 的 Full 请求、同一输出根和相同诊断配置。先跑完 C3 可直接汇总 C4。
- C2 引用 C1 LR 结果。旧 C1 运行可通过 `--c1-root` 提供，只导入验证通过的记录。
- C5 的 Full 关闭 C3 recorder。Capacity-only 运行可与 C2 的 Full 复用完全匹配的三 seed 结果。Time-match 的 Full 和各控制绑定 calibration GPU 并独立测量，继续复用同一离线 checkpoint。
- Conventional ETT 的 split mode 进入 offline key。Smoke 的短 offline schedule 也进入独立缓存键。
- 独立旧 C1 和统一入口可分别选择同一原始配置。现有旧 C1 结果通过 collector 读入，统一 checkpoint cache 只收录本入口已验证的 prepare。
- 无法绑定历史 Table 10 Matched/Uniform 的实现与预算。当前 MoE 选择为 prospective。同样，C7 的本地 partial-TD 维持原有本地模型与离线流程，保留单独协议标签。

按原始 CSV 从 0 开始计数，ETTh2 conventional 边界为 0/8640/11520/14400，ETTm1 为 0/34560/46080/57600。两者的 paper online 边界均为 0/2880/3600/14400。真实边界日期、scaler fit 区间、行数、origin 数写入 C6 manifest 与 R7。
