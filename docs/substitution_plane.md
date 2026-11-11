# SHIFT 自动选择备用节点

在现有会话控制器中加入一个 `Substitution Plane` 模块。它接收已分类的事件，读取当前训练布局，选择并预留目标；现有 SHIFT/TorchFT 执行状态恢复或迁移。第一轮手动指定事件，之后由 FLASH 输出同一格式的事件。备用节点不自行抢占任务。

## 首轮范围

| 手动事件 | 执行策略 | v5 支持范围 | 每个目标需要 |
|---|---|---|---|
| `process_exit`：进程已退出 | v1 恢复 | DP0 或 DP1 的完整副本 | `world` 个槽位 |
| `replica_slow`：按副本处理慢故障 | v2 迁移 | DP1 的完整副本 | `world` 个槽位 |
| `rank_slow`：指定副本内慢 rank | v4 迁移 | DP0/DP1 的 PP0/PP1；迁整个 stage | `world / 2` 个槽位 |

事件建议字段：`event_id`、`session`、`expected_epoch`、`fault_type`、`source_replica`、`affected_rank`（v4）、`evidence_ref`。从会话状态解析当前源节点和物理布局。手动提交事件不注入 SIGKILL 或慢计算。v2/v4 可在健康源上测试指定事件后的真实迁移；v1 真恢复必须有实际退出证据，若进程仍在运行，只输出选址结果。跨 epoch 的事件须拒绝，不能把旧故障套到新 owner 上。

## 发现、选择与通知

1. **发现。** 先从本会话允许的节点和自有 Lab Job 发现候选，复用已有 Pod 内的启动方式。用节点名作为目标键、Pod UID 作为实例键；Pod IP、DNS 或 `namespace/pod` 是联络信息。Pod 可从 [Downward API](https://kubernetes.io/docs/concepts/workloads/pods/downward-api/) 获取身份。冷候选可预检环境，有空闲 GPU 份额后再创建定向 Job；先不常驻占用多个热备。
2. **上报。** 记录健康/就绪、最近实测时间、可用 GPU 槽位及每卡可用显存、与训练配置匹配的模型和运行时版本，以及所需 NCCL/RDMA 连通性。预热状态或预计准备时间用于排序。活跃布局、预留和故障隔离状态由控制器维护。共享 GPU 上，Pod `Ready` 或 GPU 数量充足不能替代显存和连通性复核。
3. **选址。** 先剔除故障源、过期、未就绪、配置不匹配或资源不足的候选。再按预计准备耗时、满足需求后的最小 GPU 余量、稳定节点名排序。没有合格目标就报告原因。第二候选只有在第一候选未接管、旧尝试已被确认终止后才可使用。
4. **通知与接管。** 控制器预留目标后，以 `session + epoch + attempt` 标识启动请求，通过 Lab Job 启动或在自有 runner Pod 执行。候选确认 Pod UID、实际设备、逻辑 rank 和准备状态后，交给 SHIFT 路径恢复状态或迁移并加入通信组。现有 `Command.target` 只传节点名；接入时必须让启动器核对实际设备与预留，不能只靠历史控制器的 `range(8)` 选卡。以所有权和首个双副本共同提交确认成功，再释放已清理资源。

单会话先用一个写入者，记录事件、预留、启动、接管和清理结果。重复请求返回同一事务；超时后先查询实际状态，再决定重试，禁止两个候选拥有同一逻辑 rank。控制器重启先核对 Pod UID 与会话 epoch，再恢复工作。多控制器时可用 [Kubernetes Lease](https://kubernetes.io/docs/concepts/architecture/leases/) 选主；Lease 不代表旧训练实例已被隔离。

## 复用与研究参考

- [Kubernetes Scheduler](https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/) 的过滤、打分、绑定可参考。Kubernetes 管 Pod 与资源，SHIFT 决定哪组逻辑 rank 要接管、何时接管。
- [TorchFT Lighthouse](https://meta-pytorch.org/torchft/assumptions_and_recommendations.html) 继续管 replica quorum，Manager 继续管 native healing。候选发现和故障分类属于会话控制层。
- [TrainMover](https://www.usenix.org/conference/osdi26/presentation/lao) 可参考通用备用机和准备/切换分阶段；[FlashRecovery](https://arxiv.org/abs/2509.03047) 可参考检测与恢复协调的职责划分。后者是公开论文，与本地 FLASH 实现分开看。
- 本地 `FLASH1022` 已有 monitor channel 和按训练迭代发布控制信号的 Store；SHIFT 现有适配也可把 FLASH `RedisWriter` 换成空实现。首版只定义事件接口，后续适配 FLASH 输出。若 FLASH 用 Redis 发布告警，可以把 Redis 当事件入口；资源预留和接管状态仍由 SHIFT 持有。

## 执行与验收

1. **本地原型。** `shift_full/spare_selection.py` 把手动事件映射到 v1/v2/v4 并选址。v2/v4 可生成现有命令；v1 只出计划。测试一台和多台候选、容量差异、过期、配置不兼容、无可用目标及当前策略范围。
2. **接 Lab 控制器。** 将三机复测使用的临时适配器固化为可重复的 Lab 启动入口，再接入候选登记、设备复核、事件去重、预留和事务回读。先用 CPU 测试重放重复、目标启动失败、旧 epoch 和超时。v2/v4 用手动事件各完成一次真实迁移。
3. **v1 真实退出。** Lab 验收允许源副本在共同提交边界正常退出。SHIFT 的 Pod 内监管器确认退出码和旧进程组已清空，控制器才释放边界并启动 TorchFT healing。已在边界等待的健康副本不会再等待已退出 rank。非边界退出可以由 SHIFT 观察并记录，训练中途自动恢复仍需验证失败重试和提交一致性。
4. **GPU 验收。** 用 `campus` 在 `lab` 运行有界 Job，只管理本次资源。各路径记录事件→选址→启动→首共同提交；核对目标物理槽位、逻辑 rank、健康侧 PID、owner/epoch、step/sample 和状态签名，再继续至少 16 步。v4 retain 不算成功。只有一台备用时，GPU 验证接管，多候选优先级用 CPU 验证。导出证据后清理本次 Job 和临时数据。

验收分两场。整副本会话先用 v2 把 DP1 从 gpu-02 迁到 gpu-03，再在 DP1 真实退出后用 v1 恢复到重新入池的 gpu-02。v4 会话单独启动，DP0、DP1 各有两个独立的 stage 进程组。四 rank 配置下，一条覆盖三台目标、每台最多占用四个槽位的 v4 轮换如下。

| 迁移 | 源节点 | 目标节点 |
|---|---|---|
| DP0 PP0 | gpu-01 | gpu-03 |
| DP1 PP0 | gpu-02 | gpu-01 |
| DP0 PP1 | gpu-01 | gpu-02 |
| DP1 PP1 | gpu-02 | gpu-03 |

命令用 `--source-replica` 和 `--pp-stage` 指定范围。控制器要求所选 stage 由独立 actor 承载，以便让该 actor 退出时保留同副本的另一半。迁移 PP0 时，由保留的 PP1 rank 协调交接。由于 rank 0 托管 TorchFT Manager 服务，受影响的副本使用新 Store 重建 Manager，并在新的 epoch 重新建立两副本的 DP 通道。每次验收后，只释放已退出并经过重新探测的槽位。

SHIFT 监管器负责识别进程退出，并按 Pod UID、actor 和 rank 记录证据。TorchFT 的副本心跳提供失联信号，不能单独当作进程退出证明。FLASH 继续定位计算慢故障，向 SHIFT 提交副本或 rank 的慢事件。

两节点可以先验证三条迁移与恢复路径。v4 使用 6+4 个互异 GPU 槽位，依次迁 DP1 PP0、DP0 PP0、DP1 PP1、DP0 PP1。整副本会话使用 8+4 个槽位，先把 DP1 迁到有 8 个槽位且正在运行 DP0 的节点，再通过正常退出恢复回另一台。两节点结果和三节点轮换证据分别保存。

当 12 个份额集中在 gpu-01 和 gpu-03 时，`scripts.lab_training_jobs --colocated-spare` 启动三个独立 runner Pod，各申请四份。node0 和 node3 位于 gpu-03，node1 位于 gpu-01。控制器确认 12 个 UUID 互异后才启动，按 runner 区分 Pod 身份和设备所有权。同主机的 runner 共享故障域。这套布局验证三方选址和进程级恢复；三台物理节点的覆盖单独记录。启动验收时为 `scripts.lab_acceptance` 传入同一个布局开关。

整副本验收可加 `--keep-dp-spread`。DP0 放到 gpu-01，DP1 在 gpu-03 的两个 runner 之间交接，使活跃副本一直分居两台主机。源和目标同主机时仍要求独立 Pod 与互异的 GPU UUID。

受控退出的 v1 可用 `--mode restore --normal-v1-exit --prepare-v1` 单独验收温备。控制器先选址并启动候选。候选只建立本地模型与通信组，所有 rank 就绪后源才在共同提交边界退出。候选要等控制器确认原进程组已经清空，才加入 TorchFT 恢复。记录提前占用的显存、确认退出至共同提交和请求至共同提交。此模式验证受控退出前的准备收益。

Lab 的驱动入口是 `scripts.lab_acceptance`，默认自动选址。`--fixed-reference` 使用已验收的自动运行 JSON，在同一批 Job 中按原顺序重放固定节点和 GPU UUID。`--two-node` 使用 gpu-01 和 gpu-03，整副本验收以 `--normal-v1-exit` 启用共同边界正常退出。

`scripts/lab_spare_pool.py` 用有界 Job 跑三个 agent，默认每个申请一个 GPU 份额，也可把分不到 GPU 的机器以 `--cpu-only-node` 登记。控制器核验 Pod UID、报告时间、可用显存和设备分配。验收让每个合格候选执行一次 CUDA 任务，核对占用期间不重复分配，并在退出后等待新报告才回池。`--warm-cuda` 记录 agent 进程内的 CUDA 准备耗时。训练进程有独立的 CUDA 上下文。`spare_agent --prime-model --model-dir ...` 可提前把暂存权重读入节点文件缓存，记录准备字节数与耗时，默认不开启。

整机失联另做：现有会话控制器、Store 或 Lighthouse 与训练源共故障域，需先拆分部署，且断联不能证明旧实例已停止。[Kubernetes 对强制删除 Pod 的说明](https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/#pod-termination-forced) 也区分了 API 删除与进程实际退出。

## 性能验收

| 观测 | 口径 |
|---|---|
| 端到端恢复/迁移 | 控制器收到事件 → 首个双副本共同提交；v1 另列确认实际退出 → 首共同提交 |
| 控制开销 | 收到事件 → 选址完成 → 预留确认 → 目标启动 → 就绪 → 首共同提交，分段保存；控制器用单一单调时钟记观察时间 |
| 训练中断 | 健康侧同一见证 rank 的最大相邻提交间隔，另报首共同步和随后四步 |
| 稳态步时 | 每个共同 step 取所有 rank 最大耗时；报告操作完成后 1–16 的全窗口及 5–16 的稳态窗口，保存逐步值、均值、中位数和 P95 |

配对比较要求源码、配置、Pod UID、GPU 顺序和每次目标一致，两组独立审计通过。自动运行相对固定目标的预设门槛为端到端耗时、v1 退出至首共同提交及稳态均值最多增加 10%，步时 P95 最多增加 15%。初始无事件窗口和迁后窗口分别比较。两组都保留 agent，固定目标组跳过选址。超限保留原始结果。共享负载波动时，逐轮列出同一逻辑操作的自动与固定比值，并用这些比值的几何均值复核同一门槛。手动慢事件验证迁移路径，实际慢故障定位与收益由 FLASH 联测验证。
