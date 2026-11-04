# V4：双副本同步训练中的透明节点替换

本轮采用“提前准备、提交边界验证和切换、健康副本计算重叠”的方案。
两副本始终共同规约和提交，global batch16、样本角色不变；没有单副本独立提交或事后追赶。

每副本 8 rank，TP2/PP2/EP2，BF16 Qwen MoE（494712832 参数），seq512，batch8。
使用已安装 TorchFT 0.2、Megatron、模型和 headers，不重建环境。

| 角色 | 机器与物理 GPU | 用途 |
| --- | --- | --- |
| A / B | node0 / node1，GPU0–3 | 副本 0，初始拥有者 |
| C | node3，GPU0–3 | 替换 B；后续可 B→C→B |
| D / E | node0 / node1，GPU4–7 | 副本 1 |

默认 TP 在节点内；`--cross-node-tp` 使 A/D 拥有偶数逻辑 rank，B/C/E 拥有奇数 rank，TP 跨节点。
三机布局的副本共享宿主，不代表独立机器故障域。

## 切换与提交

1. 替补提前加载模型、建立状态通道。节点内 TP 可在隔离本地组预热并复用本地组；跨节点 TP 使用仅构造期的 metadata group 分片，实际通信在激活后建立。
2. 替补未就绪时，原 AB 与 DE 继续共同训练。就绪后选择已提交且通信排空的同一步边界，冻结完整快照并封锁旧 epoch。
3. D/E 保留模型、Adam、DDP、参数存储、内部组和 Manager，立刻计算下一步；异步 quorum 在外层规约前等待。记录 store 因果标记，不比较不同机器的单调时钟。
4. C 通过 GPU packed 传输恢复模型、Adam、scheduler、buffers、progress、CPU/CUDA/Python/tracker RNG，九项 SHA256 全部一致才允许替换。原 B 在验证完成前保留，恢复破坏、取消或超时则保留 AB，在新 epoch 下继续共同训练。
5. 替换后 A/D/E 的十二个健康 Manager 和两个 rank0 ManagerServer 保留；仅改变对应外层通信端点的 namespace。恢复完整 checkpoint 不等于重放历史通信。
6. TorchFT `should_commit()` 的结果可能在副本之间不同。因此 Adam 更新前另有全部 16 个逻辑 worker 的确认；TCPStore compare-set 固化 commit/abort 决定，按 epoch/step 隔离。任一拒绝或缺票不能提交。失败后不回退活 Manager 的 native 计数。

源 worker 在冻结后、发送 ready 前发生受控 CPU 停顿时，可从 E 的冻结快照恢复。
只有 E 快照的全部九项签名与原源状态一致才授权；控制器核实旧自建 unit 原为 loaded/active/running，stop 返回 0，之后 inactive/dead/MainPID=0。
若丢源后替补也验证失败，则全任务停机；不能回退到已丢失的源。

## 验收与证据

108 项 CPU 测试通过；最终 GPU 矩阵共 2784 次真实 AVG 审计、1504 组逐步九状态/样本/Manager 比较，均无差异。
最终验收见 [transparent_summary.json](../evidence/v4/transparent_summary.json)，独立参考签名随摘要保存。
该审计重新计算参考比较，检查完整 step/样本覆盖、旧 owner 拒绝、候选准入、健康对象身份、逐步真实 AVG、16-worker 提交确认，以及三节点资源恢复；不直接相信 controller 的成功布尔值。

| 场景 | 运行 |
| --- | --- |
| 独立无迁移参考，40 步 | ref20 |
| 显式取消，保留原副本继续 | cancel21 |
| 故意破坏恢复参数，保留原副本继续 | restore22 |
| 候选恢复停顿，超时后保留原副本继续 | stall23 |
| 同一任务连续 B→C→B | repeat24 |
| 冻结后停源，从健康快照恢复，再次替换 | loss25 |
| 跨节点 TP 独立参考与替换 | crossref26 / cross27 |

所有 oracle 运行与独立无迁移轨迹逐步比较九状态签名、样本和 Manager 进度。
这是签名比较，不是新一次完整张量 dump 比较。最终保留训练源码与两份参考 manifest 的对应 SHA 全部一致；manifest 额外记录的旧未调用探针与控制器/auditor 条目见 `final_source_verification.json`。参考源码 SHA 必须与被测训练源码一致；不允许 live Manager rewind。
早期正常/连续/跨 TP 协议烟测见 [transaction_smoke_summary.json](../evidence/v4/transaction_smoke_summary.json)，不把无 oracle 烟测算作数值参考验收。
`loss19` 和 `deadline09` 的早期证据不足，单列为 preliminary diagnostics，不能作为最终验收。
历史保留 Manager、计算重叠结果分别见 `manager_retention_summary.json`、`peer_overlap_summary.json`；旧 raw 已删除，不能重新审计旧轨迹。

原生 deadline 探针见 [deadline_summary.json](../evidence/v4/deadline_summary.json)：
Manager=90s、quorum=180s 时，PG=2s 加 3s 延迟复现约 2.002s 的 CUDA stream timer abort；仅把 PG 改为 10s，同样延迟通过。
异常退出 guard 后等待超过 2s 的对照也通过，没有遗留 timer abort。
因此已定位可复现的 2s 截止层；历史 v3 最初为何发生足够长的停顿仍未确定，不能声称已重建原事故。

CPU 测试与 GPU 验收互补：前者覆盖 compare-set 决策竞争、缺票/错 epoch、事务状态、metadata group 恢复及审计缺证据负例。
清理报告保留三机每机 8 个公司 GPU PID 不变、无额外 SHIFT GPU 进程或活动 unit、安装库 SHA；未测试公司应用业务健康。
原始日志、trace、中间快照和临时授权在独立审计后清理，保留源码、私有配置、必要初始 anchor、参考签名与摘要。

## 范围与性能解释

- 支持上述冻结边界后的受控进程停顿；源在 checkpoint 已开始发送之后故障，或任意通信中途/优化器更新中故障，仍采用有界全任务停机，不声称通用通信回放或任意时刻自动恢复。
- 独立双节点副本加备用需要五台独立机器；当前三机结果不能证明宿主故障隔离。
- “激活前健康副本已经计算”由因果标记实测确认。本轮正常连续替换的外层总等待约 6.25–6.63s，跨节点 TP 约 11.39–11.63s；这些是含诊断的单次观测。`peer_outer_gradient_ready_seconds` 包含 quorum 之后的梯度等待；不能只把较短 quorum wait 当作全部停机。
- 快照/hash、诊断和独立 AVG 通道影响计时；未完成同条件重复性能基线，不能宣称固定百分比加速、零等待或零额外显存。
- 跨节点 TP 替补没有执行真实跨节点 shadow iteration 预热；验证完成后才建立正式组并计算。

## 复现入口

```sh
python3 -m unittest discover -s scripts -p 'test*v4*.py' -q
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix REF --transactional-switch --outer-audit \
  --reference-only --cut 4 --post-steps 36
# 等待独立参考审计通过，再部署/启动；每次 prepare 凭据只消费一次。
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix NEW --transactional-switch --outer-audit \
  --reference-run REF --cycles 2 --cut 4 --post-steps 10
```

取消加 `--cancel-round 0`；恢复破坏加 `--fail-restore-rank 4`；恢复停顿加
`--stall-restore-rank 4 --restore-deadline 3`；停源加 `--source-stall-round 0`。
一次仅注入一种故障。跨 TP 的参考和被测运行均加 `--cross-node-tp`，故障 rank 为 1/3/5/7。
每次 suffix 必须唯一。参考长度须覆盖实际就绪边界和全部后续提交。

方案参考 [TrainMover](https://www.usenix.org/system/files/osdi26-lao.pdf) 的布局保持替换与
[TorchFT Manager](https://meta-pytorch.org/torchft/manager.html) 的 quorum/提交控制；实际 API 以安装的 0.2 源码为准，不升级运行库。
