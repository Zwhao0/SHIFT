# SHIFT — v4.1 代码重构统一

`training_core.py`统一真实训练、batch计划、Adam/scheduler和完整九状态；v1/v2/v4仅保留恢复/迁移策略。global batch=8（含v1 solo），初始A/B各完整DP副本、C备用，TP节点内，v4迁移后才跨PP。正常训练不逐步hash，开发审计独立。v4入口默认选择已验收的统一事务配置，旧batch16布局需显式`--legacy-layout`。

109项服务器CPU测试通过；三路径16 workers×16步九状态/样本精确对齐，v1/v2还比较独立梯度，v4参考总32步。每种策略至少2次记录seed/触发点/替换对象的随机恢复/迁移通过，恢复后至少5步；新增候选超时收尾回归检查。

|无逐步hash的性能样例|完整请求→控制器观察共同提交|后续4个共同步平均最慢rank步时|
|---|---|---|
|v1 fail-stop（两个不同目标）|19.63 / 20.06 s|0.792 / 0.816 s|
|v2 整副本p2迁移|22.01 / 21.51 s|0.858 / 0.840 s|
|v4 仅PP阶段迁移|21.60 / 33.67 s|0.755 / 4.061 s|

v4第二例处在高负载窗口，同例迁前为3.937 s，post增约3.2%。匹配常态旧/新为0.738/0.740 s；80步v2相对较快同期native约+5.7%；A/C布局控制0.781 s，v2 post约+9.8%/+7.5%。共享负载下相同native代码也出现2.478 s窗口，不支持隔离加速或统计显著性结论。原生solo的额外计算、首次共同步和迁移等待均单独保留。

原v1冷join超时退化、一次v4候选传输超时/提前收尾审计失败均保留为**拒绝的诊断结果**。修复后v1入quorum前本地热身（成本仍计完整时间），v4等所有actor收尾并检查真实owner决定；匹配复测和受控超时回退审计通过。未把retain当作迁移成功。

详见[设计与复现](docs/v4_1_unified.md)、[精简验收/计时/来源SHA](evidence/v4_1/summary.json)、[比较签名](evidence/v4_1/selected_signatures.json)。沿用已有环境、headers和模型；公开仓库不包含私有配置、模型、anchor或raw。清理后必须新建匹配参考，签名不能替代raw verifier。

## 历史v3与旧v4记录（不作为v4.1计时基线）

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

## 旧v4：双副本同步训练中的透明节点替换

两套真实 TorchFT 副本各 TP2/PP2/EP2、8 rank、BF16 Qwen MoE，联合 batch16。替补提前准备；在已提交边界恢复并验证完整状态，再切换内部节点。健康副本可计算下一步，在外层规约处等待双方，共同提交；没有单副本独立提交或追赶。

保留健康模型、Adam、DDP、参数存储及 Manager，支持恢复拒绝/取消/超时保留原副本、同任务连续替换、跨节点 TP，以及冻结后发送 ready 前的受控停源恢复。独立无迁移轨迹验证九状态签名、样本和 Manager 进度。实现、验收、范围和复现命令见 [v4 说明](docs/v4_outer_manager.md)。

三机共享宿主不能证明独立故障域；任意通信中途故障仍可全任务停机。计时包含诊断，尚无受控性能加速结论。复用已有环境，审计后删除可再生 raw/日志，保留必要签名、摘要和私有配置。

Six quick functional trials, complete recovery timing and phase samples: [test record](docs/v4_quick_validation.md), [summary](evidence/v4/quick_validation.json). Cross-TP and the second V2 run retain post-recovery timing regressions for analysis.
