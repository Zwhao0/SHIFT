# torchFT 0.2.0 原生能力核对

范围：只读本地源码和 SHIFT 集成；未跑 GPU、测试或远端命令。

**源码基准**：项目保存的 wheel `tmp/torch214-download/torchft-0.2.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl`；Python 源码解包在 `tmp/torchft-0.2.0/torchft/`。quorum 决策实现位于 wheel 的编译扩展 `_torchft.cpython-312-x86_64-linux-gnu.so`；`_torchft.pyi` 只给签名，因此 `shrink_only` 的 Rust 内部选择策略未从本地 Python 源码验证。没有读取远端。

| 能力 | 0.2.0 源码 / 当前使用 | 对本实验可直接用 | 最小新增 / 分类 |
|---|---|---|---|
| 固定有效副本 + spare | **原生** `manager.py:129-138, 680-690, 411-479`：`FIXED_WITH_SPARES` 将有效数限制为 `min_replica_size`；`is_participating=False` 的 rank 输入被清零，平均按有效数归一。`train.py:394-399` 目前未传 `world_size_mode`，使用默认 `DYNAMIC`、`min_replica_size=1`。 | 可直接做三副本/两有效副本组件实验。spare 仍在训练 PG 中并收发规约；API 不会跳过其前后向，也不自动让它更新 Adam/scheduler。只要各 rank 收到同一平均梯度并照常更新，它可成为状态跟随候选，需实测。 | 加配置暴露 `world_size_mode`、`min_replica_size=2` 和成员顺序；若 C 不做前后向，另加同序零梯度桶/更新调度。**参数配置适配；跳过计算是新增。** 此门控按 rank，不是按用户指定的 B/C 名称；B 必须从最终 AC quorum 移除，再验证晋升。 |
| `start_quorum` / `allow_heal` / `shrink_only` | **原生** `manager.py:560-627, 629-813`。默认异步 quorum；`allow_heal=True` 时执行 checkpoint 恢复，文档要求所有副本取相同值。`shrink_only` 传入 `_quorum`；Python 层未解释其精确 Rust 策略。当前 `train.py:515` 默认调用、异步；`runtime_patch.py:397-407,568` 使用同步 quorum。 | 默认 `allow_heal` 可用作新 C 加入时的 torchFT 恢复；异步模式可与 forward 重叠。固定步长/提前判定 spare 角色可选择同步等待。`shrink_only` 不是已确认的“指定替换 B”接口。 | 加共同的交接阶段/成员协议，显式安排哪一步允许 heal、何时 B 退出及如何取消；微验证 shrink-only 语义后再用。**成员编排新增，策略核对待微验证。** |
| `init_sync` | **原生** `manager.py:182-229, 642-652`，默认 `True`；当前 `train.py:396` 和 `runtime_patch.py:398` 设 `False`。注释定义为 step 0 权重同步；要求各副本已同权。 | 适合各角色初始权重已一致时跳过启动同步。不能代替运行中 C 的最新模型/Adam 状态恢复；恢复仍由 quorum `heal` 和 checkpoint transport 路径处理。 | 后台状态准备需独立版本与最终一致点。**直接用现有 healing 做边界恢复；后台追赶新增。** |
| `should_commit` / `is_participating` | **原生** `manager.py:855-943, 1022-1053`。前者须在 backward/规约后、optimizer.step 前调用；本地 replica ranks 得到一致决定，失败时不提交。后者报告当前 quorum 是否贡献；spare 和 async healing 时可为 false。当前 `train.py:568-580` 直接门控 optimizer，并记录 participants/contributed。 | 可直接用于两贡献者 gate 与审计；spare 可依据参与标记避免算作有效样本贡献。`is_participating` 是状态读取，不会自行禁止 optimizer/scheduler 更新。 | 维持每步一次且所有本地 rank 一致；若跳过 C 前后向，增加数据/零桶/optimizer 一致性逻辑。**提交门控原生，角色策略适配。** |
| 状态锁与 CUDA 顺序 | **原生** `manager.py:242-244, 367-379, 957-965` 有 state-dict RW lock；应用须主动 `disallow_state_dict_read()` / `allow_state_dict_read()` 包住写状态窗口，manager/optimizer 不自动替训练更新加锁（torchFT 的 LocalSGD 示例会显式调用）。`should_commit` 会等 recovery event 并同步当前 accelerator stream（`manager.py:880-899`）。当前 `train.py:274-292` 导出前同步 CUDA 当前流，但没有调用 state-dict lock；`should_commit` 在 optimizer 更新前。 | 可沿用提交边界导出。已有同步不等于后台线程快照、可变 buffer/RNG 和 CUDA stream 都已安全冻结；新后台拷贝必须证明生产流完成且快照不可变。 | 给 optimizer/scheduler/state 写入加读锁互斥，按版本复制快照并记录 CUDA event/stream 依赖；加入 buffer、RNG、数据进度。**应用层新增。** |
| `CheckpointTransport` | **原生接口** `checkpointing/transport.py:14-68`：metadata/send/recv/disallow/shutdown；`PGTransport` 在 `checkpointing/pg_transport.py:168-280` 支持 PG 传输和兼容目标存储时原位接收。当前用的是 **适配层** `shift_full/transport.py:29-53` 的 `PackedGPUTransport`，绑定 Manager 正在重配置的 PG；Manager 仅在 quorum 有恢复目标时调用它（`manager.py:746-804`）。 | 边界状态恢复可直接沿用现有 transport。接口和 Manager 自动路径没有提供 AB 继续更新时对 C 做任意版本预拷贝/增量追赶。 | P2 需要独立可用的 A/C 或 B/C 通道、不可变版本快照和 stale/final-step 校验；可扩展当前 transport，但不应在活动训练 PG 上无协议混传。**传输接口原生，后台版本传输新增。** |
| shutdown / 生命周期 | **原生** `manager.py:402-409` 关闭 transport、ManagerServer、quorum executor；不关闭 `_pg`。当前 `train.py:630-634`、`runtime_patch.py:677-691` 在 Manager 后另行 abort/shutdown PG；自定义 transport 的 shutdown 是 no-op，因为它共享该 PG。 | 可复用现有显式清理顺序。候选 PG 与活动 PG 并存时，双方都需等待在途 work 后分别清理。 | 新增候选组生命周期、失败/取消清理及切换后旧 B 退出确认。**生命周期协议新增。** |
| 候选 AC PG 预建/复用 | torchFT **原生 PG 是可重配置的**：`process_group.py:283-313, 444-480, 794-889`；Manager 在 quorum id 变化时对唯一 `self._pg` 调用阻塞 `configure`（`manager.py:707-725`），NCCL `_create_pg` 建立并 eager-connect。 | 可以研究独立第二 PG，但源码没有把已预建 PG 注入/复用为 Manager 下一 quorum PG 的 API。当前集成也未预建候选 AC PG。 | 由 A/C 按一致顺序创建独立候选 PG 并做真实预热；新增 Manager/规约适配器在边界安全切换句柄，或先验证必须的最小集成点。检验 AB work 排空、stream 顺序、资源干扰和最终 B 排除。**独立建组可借标准 PG；Manager 复用是新增，不能称原生已支持。** |

## 给实验窗口的限制摘要

- 最有价值的原生起点是 `FIXED_WITH_SPARES(min_replica_size=2)`：规约可固定两份有效梯度，同时让第三份物理 replica 收到相同规约结果。它本身不提供 B→C 的名字级切换，也不免除 spare 的通信/优化器成本。
- 当前集成并未使用固定 spare；当前恢复走 Manager 的 quorum + 自定义 GPU transport，PG 随 quorum 重配置。没有独立候选 AC PG，也没有后台版本状态复制。
- `init_sync=False` 只跳过 step 0 的初始同步。`should_commit` 会做 CUDA 同步后再决定提交；state-dict 锁需应用在状态写窗口主动使用。后台快照需要额外的版本、冻结和 stream 依赖协议。
- wheel 保存了完整 Python 源码和签名，但 quorum 的 Rust 策略在编译扩展中。本文没有推断 `shrink_only` 的内部成员选择含义，也未声称候选 PG 可被 Manager 原生复用。

## 本轮新增适配（不归为 torchFT 原生）

以上“当前使用”指 v1，保留原入口不改。新实验入口是 `shift_full/planned_train.py`：配置原生 FIXED_WITH_SPARES、min=2；应用负责 epoch/角色门控、完整 buffer/RNG/数据版本、独立状态通道、共同更新后边界与旧 Manager/PG 关闭。`PreparedNCCL` 是单 epoch、失败即拒绝的 ProcessGroup 子类：把预建 PG 从新 Manager 构造时传入；Manager 的 configure 仍会调用，但只校验相同成员并复用已连接 parent，后续新 quorum 被拒。没有替换正在工作的 Manager 私有句柄，没有改 torchFT/PyTorch/NCCL 源码，也不是通用容错 PG。

额外核实：Manager 构造用固定 MANAGER_ADDR_KEY / REPLICA_ID_KEY (`manager.py:292–333`)；在存活 A 上重建 Manager 时必须使用独立的本地 TCPStore/epoch，不能让其他本地 rank 读取旧地址。新入口通过原生 `store_addr/store_port` 参数隔离。后台 TCPStore wait 使用独立客户端连接，避免阻塞训练侧 check。
