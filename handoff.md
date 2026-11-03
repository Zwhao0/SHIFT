# SHIFT v2 交接

本轮范围：B存活、外部指定待替换，A+B到A+C的计划性交接；优先使用原生torchFT机制，应用层适配候选PG复用、epoch与资源生命周期。

结果、指标及未完成项统一见 `RESULTS_V2.md`、`results/v2/summary.json`。完整MoE状态预拷贝、任意失败自动回退、长期多次交接、DP内部单个PP/TP节点迁移均未完成。当前备用资源仍是整份DP副本。

最终版p1f：通信组提前准备并复用；B在最终共同边界传最新完整状态；旧Manager同步关闭，旧PG在首个AC共同提交后回收，A/C侧第三个AC提交前完成，B侧在待命退出前join核验完成。精度、batch、归一化及Adam epsilon不变。v1实现/证据保留，原数值失败仍明确报告。

三机容器、FLASH源码/模型/运行时与张量快照保留。本机凭据不上传。公开日志仅作必要去敏，离线解包/校验见README。GitHub目标：Zwhao0/SHIFT，v2副本间慢故障恢复。
