# v4.2 手动恢复优化完成

基于已发布 v4.1 e4d4f51eef0ab7343ccc7af750b17a42644bfa25。一个训练会话启动一次，随后提交命令选择策略；六项验收期间不重新启动训练主会话。

## 三层实现

- `shift_full/training_loop.py`：唯一运行中的迭代/投票/提交循环，继续调用原 `training_core` 的数据、前后向与 Adam/scheduler。拒绝提交不推进训练状态，策略只在已提交边界切换。
- `shift_full/session_control.py`：统一 schema、会话/id/epoch、动作、策略、范围和目标。命令串行，重复 id 幂等，过期 epoch/冲突负载/不可用目标拒绝。
- `shift_full/session_engine.py`：统一策略路由器，同一模型、优化器和进度执行 v1 原生 TorchFT 冷恢复、v2 p2 完整副本预复制/梯度与 tail 交接、v4 部分 PP 事务迁移。`session_outer.py` 封装共同 TorchFT 提交和原生恢复适配。

v1 真正终止源副本后冷启动目标，恢复必须在第一次前向前完成；v2 保留实际梯度接力与 epoch fencing；v4 保留健康副本训练、四候选实际恢复确认、owner 切换与提交屏障。采用新 Manager 时使用独立 TCPStore 槽，发布地址取当前目标节点。未来控制端口在系统临时端口范围外预留，Lighthouse 就绪后才继续。

## 使用

```sh
python3 -m scripts.sessionctl start --session demo
python3 -m scripts.sessionctl status --session demo
python3 -m scripts.sessionctl command --session demo --id move1 --policy v2 --target node3
# status 显示 move1 完成后继续；不再 start
python3 -m scripts.sessionctl command --session demo --id partial1 --policy v4 --target node1
python3 -m scripts.sessionctl command --session demo --id recover1 --policy v1 --target node3
python3 -m scripts.sessionctl stop --session demo
```

`start` 内部仅 prepare 一次并获取现有训练锁。默认不做计时窗口内的开发全状态 hash；`--audit` 开启开发审计，停止后所有16逻辑 worker 完成屏障再做完整九状态检查。`command` 可指定 `--expected-epoch` 和将来的 `--at-step`。必须等待当前命令完成后再下发下一条。目标需要空闲 GPU 槽；示例中的可选目标随此前迁移结果变化。

当前已实现的动态范围是 v1/v2 的 DP1 完整8 rank，以及 v4 的 DP0 PP1（rank4–7）；超出范围拒绝。v4 可迁出再迁回。不是任意 DP/PP 放置调度器，也未扩展为任意通信中途物理故障容错。控制器有运行期限与预分配服务槽上限（最多15条策略指令，仍可 stop）；结束后可启动新会话。

## 验收与计时

```sh
python3 -m scripts.test_session_scenario --start --two-each --session acceptance
# 补80步native与每次v2后的80步稳态：加 --steady-steps 80
# 远端既有 Python，CPU-only：
CUDA_VISIBLE_DEVICES='' python -m scripts.verify_session /workspace/SHIFT/results/session_acceptance --two-each
```

固定顺序为 v2→v4→v1→v4迁回→v2迁回→v1再次恢复。审计按真实 actor/PID、逻辑rank、样本、epoch/owner/交接证据及16 worker终态九状态独立验收。完整恢复计时采用控制器单时钟的请求→首共同提交；v1另给实际故障→首共同提交。命令完成时间含至少5个恢复后共同提交，不能与首共同提交计时混用。步时按每步最慢 worker 再取窗口均值，分准备前/首共同步/其后窗口。

结果见 `evidence/v4_2/summary.json`；原版与去边界开发 hash 对照及拒绝案例见 `evidence/v4_recovery_20261105/summary.json`。共享 GPU 快测用于发现功能和明显耗时退化，短窗口不能证明统计吞吐等价。v4.1 与 runtime 对照改变了开发审计策略，不能把全部收益归于通信优化。

保留现有环境/headers/模型与最小 BF16 初始 anchor；单训练锁、实验串行、每 worker 显存上限0.12。训练中禁止 prepare，不操作公司共享容器。导出审计、比较签名和源码 SHA 后删除可再生 raw/log/trace/中间快照。

最终发布代码验收为 `session_s42_release`（188步、各两次策略）；`session_s42_final` 是额外432步联合稳态对照。最终158项服务器CPU测试通过。v2旧入口30.314秒慢例与未改代码复跑22.950秒均保留；`manager_init` 是包含handoff等待/接收的历史混合计时区间。v2切回通用Native Manager的一次提交间隔约1.42–1.45秒，完整80步与11–80稳态及逐步commit gap同时报告，避免稳态窗口掩盖切换开销。共享负载/首次内核准备/通信竞争未能严格隔离。

计时profile区别：统一会话是手动策略选择，v2源未注入人工slow-compute；旧v2兼容入口与v4.1旧v2对照均注入250ms延迟。因此统一会话与旧入口的完整迁移秒数不作为严格配对加速证明。80步对照统计的是恢复后正常训练，另保留包含切换开销的完整80步commit cadence。
