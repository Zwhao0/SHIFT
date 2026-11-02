# SHIFT v1 交接

三机优化与实验已完成，无待运行GPU实验。

- 冷备随机轮转：60/60、3600秒、RPO0；恢复480/480、首共同步480/480。
- 推荐GPU热备：三轮正常计时22.680→6.737秒；新拓扑ABBA额外步时4.84%。
- 数值边界：TF32路径三更新、严格FP32两更新均未全部通过原容差；严格模式路由相同，已定位一个近零梯度经AdamW放大的坐标。未放宽容差或修改ε。
- 复现/环境见 `README.md`，完整结果见 `RESULTS.md`、`results/v1/summary.json`。原始日志、实跑源码快照与校验清单在 `results/v1/`。
- 仅训练进程SIGKILL；Lighthouse/控制面存活，合成数据动态batch8→4。未验证主机掉电与同负载TorchTitan。
- server.md仅本机保存；保留现有三机容器、FLASH源码/模型/运行时和服务器张量快照。
- GitHub目标：Zwhao0/SHIFT，版本v1。
