# v1 优化

- 推荐GPU热备：完整恢复中位6.737秒，冷进程基线22.680秒；保留8张GPU备用资源。
- 原生torchFT NCCL/IB、64M元素桶/hook、异步quorum；热备在Manager构造前等待ACTIVATE，加载最新全状态后有效参与。
- Lighthouse独立，rid/generation目录隔离，Linux subreaper回收训练进程树。
- 同拓扑ABBA额外步时4.84%。
- 原TF32路径与严格FP32短轨迹均保留失败结果；近零梯度的AdamW敏感性见 `RESULTS.md`。
- 分项计时与GPU/RSS成本：`results/v1/backup_summary.json`。
