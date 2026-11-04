# SHIFT — v3 基线与 v4 探索

v3是用户接受实际效果后的研究基线，包含fail-stop完整恢复、活慢源提前准备与边界接替。**常态≤1%、纯冷≤15/10s等性能目标未达；flat固定容差未过。**

|项目|实际验证与边界|
|---|---|
|v1 fail-stop|真实随机kill、fresh第三机、三机备用轮换；schema2九状态、完整RNG、恢复后十joint及健康PID检查通过|
|v2 fail-slow|S_k预传、梯度/RNG/buffer tail relay、外层规约开始前边界接替，多轮及限定回退通过；仍等B本步梯度，非自动慢检测或任意故障保证|
|常态|flat→SHIFT +7.779656%；nativeFT同Megatron接口+7.509763%；适配增量+.251040%，区间跨0；非完整TorchTitan|
|纯冷候选|18.118335/18.295410s；旧21.389271s仅n1，候选n2；默认关闭，不能将全部差异归因机制|
|staged慢化|freeze→joint 2.262413/2.161504s；commit gap 7.157995/6.814702s；非完整迁移耗时|

详见[结果与局限](RESULTS_V3.md)、[复现与依赖](docs/v3_reproduce.md)、[优化记录](docs/v3_optimization_record.md)、[小型证据](evidence/v3/summary.json)。后续开发以v3为基线。

三机、每DP副本8×H800，约0.495B Qwen MoE，seq512、合成batch8，TP/PP/EP/eTP均2；Torch 2.14.1+cu130、TorchFT0.2.0、NCCL2.30.7。训练仍依赖已有FLASH/Megatron源码与模型；不包含模型、环境、私有运行历史和凭据，公开包不具备一键从空机器重建全部外部依赖的能力。版本、哈希及入口见复现文档。

```sh
python3 -m compileall -q shift_full shift_min scripts
python3 -m unittest scripts.test_planned_state scripts.test_v3_recovery
```

GPU/真实故障运行前必须配置真实节点与已有环境，使用独立systemd入口，并保留所有计时、慢步、skip与必要校验。`server.example.md`仅为认证模板；不要提交`server.md`。

历史v1/v2/v2.1结果仍在旧分支与提交中；v3工作树只保留本轮可审查公开材料。

## v4：双副本同步训练中的透明节点替换

两套真实 TorchFT 副本各 TP2/PP2/EP2、8 rank、BF16 Qwen MoE，联合 batch16。替补提前准备；在已提交边界恢复并验证完整状态，再切换内部节点。健康副本可计算下一步，在外层规约处等待双方，共同提交；没有单副本独立提交或追赶。

保留健康模型、Adam、DDP、参数存储及 Manager，支持恢复拒绝/取消/超时保留原副本、同任务连续替换、跨节点 TP，以及冻结后发送 ready 前的受控停源恢复。独立无迁移轨迹验证九状态签名、样本和 Manager 进度。实现、验收、范围和复现命令见 [v4 说明](docs/v4_outer_manager.md)。

三机共享宿主不能证明独立故障域；任意通信中途故障仍可全任务停机。计时包含诊断，尚无受控性能加速结论。复用已有环境，审计后删除可再生 raw/日志，保留必要签名、摘要和私有配置。

Six quick functional trials, complete recovery timing and phase samples: [test record](docs/v4_quick_validation.md), [summary](evidence/v4/quick_validation.json). Cross-TP and the second V2 run retain post-recovery timing regressions for analysis.
