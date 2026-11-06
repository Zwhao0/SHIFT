# v4.3 两节点可用验证方案

## 固定模型与运行配置

- 模型：`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`，revision `4b89028ef7376d0e92acbbc4a7f560c2374aeec0`，完整 **494,712,832** 参数，模型 BF16；不裁层、不缩模型。
- 低显存配置：`seq_len=32`、每个数据 rank 两个 microbatch（各1样本）、global batch 8。CPU `OffloadedAdamW` 保存 CPU 参数副本和完整 Adam 状态，训练参数与梯度仍在 GPU；CPU 更新会明显增加步时，不能与 v4.2 的 GPU Adam、seq512 直接比较。
- 缓冲区有限性检查按块处理。CPU 大状态通过 NCCL 搬运，以 GPU 64 MiB 暂存块分段；接收端按叶子设备恢复 CPU/GPU tensor。FrozenTransport 在发送期间检查只读版本；v4 需等候选端确认后才关闭旧通道；首个切换后归约先等双副本真实梯度就绪，避免把健康副本提前完成的计算等待计入短通信期限。
- 低显存 v4 恢复 deadline 为 45 秒；超时 retain 只表示保留旧 owner，不能算成功迁移。

## 两种独立会话

- **v1 原生恢复**：每副本 8 ranks，TP2 × PP2 × EP2；两副本共 16 个训练 GPU。分别测试 `source_replica=0`→node0、`source_replica=1`→node1，每副本各恢复一次。目标是被清空的原机，验证进程故障恢复，不代表整机离线时有第三机可接管。
- **v2/v4 迁移**：每副本 4 ranks，TP1 × PP2 × EP2；两副本共 8 个训练 GPU。v2 从 replica1 迁完整 4 ranks；v4 从 replica0 迁 PP1 的 ranks 2、3（2 ranks）。额外候选分别使峰值达到 12、10 个 GPU。
- 每台物理 GPU 编号仍为 0–7。目标槽位预检拒绝物理重叠；同机健康副本和候选 Manager Store 使用隔离端口。
- 两种 profile 分别新建会话、各启动一次；**不是**同一会话内动态切换 TP。迁移顺序为 v2→v4→v4 迁回→v2 迁回。

## 启动与监控

每组在两个终端配对启动：终端 A 跑 watch，终端 B 跑 rotation。先完成并清理 native 会话，再启动 migration 会话；不得重叠。

```sh
# 第一次：v1 两次真实恢复
# 终端 A
python3 -m scripts.watch_v43_resources --session v43_native_RUNID
# 终端 B
python3 -m scripts.rotation_six --session v43_native_RUNID --start --two-node-native --steady-steps 16 --seq-len 32 --low-memory
```

```sh
# 第一次结束并清理后，第二次：v2 → v4 → v4 迁回 → v2 迁回
# 终端 A
python3 -m scripts.watch_v43_resources --session v43_migration_RUNID
# 终端 B
python3 -m scripts.rotation_six --session v43_migration_RUNID --start --two-node-migration --steady-steps 16 --seq-len 32 --low-memory
```

监视器每 5 秒检查公司进程 PID、未知 GPU 进程和最低空闲显存；低于 512 MiB 或资源集合变化时，只停止该会话自有控制器，由其执行清理。验收要求两节点前后各自的公司 8 PID 集合完全不变。每个会话仍持有原训练锁、每 worker 显存上限 0.12，并固定 `NCCL_MAX_CTAS=4`、`NCCL_BUFFSIZE=1048576`、`OMP_NUM_THREADS=1`。不操作公司容器或进程。资源监视能限制显存风险，不能证明共享机器上的业务吞吐不受影响。

## 验收与限制

- 每条迁移后至少继续 16 步；报告固定 steps 5–16 的 12 步窗口及完整 16 步轨迹。核对完整状态、step/sample、PID、owner/epoch、交接和最终清理证据；不得把 retain、超时或仅收到状态算作 v4 迁移成功。
- 两会话结束后分别导出到 `evidence/v4_3`，完成独立复核后再清理 raw/log/trace 等可再生文件；保留环境、模型、headers 与初始 anchor。不要公开端点凭据或模型 raw。

```sh
python3 -m scripts.export_v42_session session_v43_native_RUNID --evidence-dir evidence/v4_3
python3 -m scripts.export_v42_session session_v43_migration_RUNID --evidence-dir evidence/v4_3
```

- 旧拒绝样本保留为边界证据：GPU Adam 高显存 OOM；早期仅按三机假设设计的方案不构成两机验证；v4 超时后 retain 不算迁移；512 MiB 监视器曾触发止损。样本来源及解释限制随最终 evidence 一并记录。
- 本方案不证明任意故障、peer fallback 或所有旧功能在新拓扑下均已验证；实际 GPU 结果未验收前不宣称通过。

## 实测结果

8卡迁移会话 `v43_mig8_20261106_a6` 在一次启动中完成 v2→v4→v4迁回→v2迁回，最终 step140，8 workers 九状态、样本、owner/epoch、健康PID与完整交接审计通过。每条命令完成后均继续至少16步。下表迁前取路由前6步，迁后固定取命令完成后第5–16步；每步使用全部worker中最大耗时。

|操作|完整请求→首个共同提交 / s|迁前均步时 / s|迁后稳态 / s|相对迁前|
|---|---:|---:|---:|---:|
|v2 → node0|20.27|1.963|2.029|+3.4%|
|v4 → node1|21.06|2.007|2.052|+2.3%|
|v4 → node0|22.10|2.053|2.043|-0.5%|
|v2 → node1|20.02|2.114|1.884|-10.9%|

初始原生 steps5–16 为 **2.002 s/步**，四个迁后稳态窗口相对初始为 −5.9% 至 +2.5%。首个共同步为4.46–9.48秒，健康进程最大提交间隔5.89–10.22秒，首步与提交间隔在证据中单列，未计入稳态窗口。v2 两次完整请求为20.27/20.02秒，v4 为21.06/22.10秒，未见本组持续增长；每类仅两次，不能据此证明统计显著性或与历史配置等速。

本轮200项CPU回归通过。两机 checkpoint 文件逐一SHA与494,712,832个参数元素一致，见 `evidence/v4_3/model.json`。

16卡会话 `v43_native16_20261106_a1` 完成两个副本各一次真实 SIGKILL 后原机重建，最终 step85，16 workers 最终九状态一致，源PID死亡证明及健康PID检查通过。每次恢复后均继续至少16步。

|v1源/目标|请求→首个共同提交 / s|SIGKILL→首个共同提交 / s|迁前均步时 / s|恢复后稳态 / s|
|---|---:|---:|---:|---:|
|replica0 → node0|27.65|22.48|1.399|1.472|
|replica1 → node1|27.21|22.11|1.400|1.472|

v1 初始 steps5–16 为1.501秒，恢复后相对初始约−1.9%；相对各自紧邻的6步窗口则约+5.2%/+5.2%。因此只能说本组未见持续累积退化，不能宣称所有前后窗口完全等速。健康进程最大提交间隔约10.95/10.97秒，solo步、首次共同步与完整请求时间分别保留。

两组均每5秒监控：公司进程集合保持不变，无未知GPU进程；迁移组最低空闲950 MiB，native组1410 MiB，未触发512 MiB保护线。NVML单worker峰值约3316/2792 MiB（含CUDA上下文等）；迁移组PyTorch allocator峰值约1.55 GiB，不能把allocator数值当作完整进程显存。共享GPU上仍无法承诺业务吞吐或实验步时完全隔离。

node3无法登录，本次没有三机备用轮换结论。两节点v1没有常驻的整机备用，使用故障后原机空出的槽位；v2/v4保留的是实验未占用槽位，物理GPU上仍驻留公司服务。

本轮两机共清理549个可再生结果文件，约83.5 MiB；每个文件SHA清单先保存在本地私有runtime。最终各机公司8个GPU PID不变，本实验GPU进程及活动session unit均为0；临时SSH授权各撤销1条并关闭服务器控制复用连接。保留既有环境、模型、headers和初始anchor。详见 `cleanup.json`、`final_resources.json`。

验收补充：四次 `systemctl kill` 调用返回码均为1，记录PID随后全部退出；原stderr未保留。另从四个精确源unit的systemd journal补取 `code=killed, status=9/KILL`，独立确认源进程确由SIGKILL终止，见 `native_termination.json`。独立verifier的物理碰撞检查覆盖最终active布局，运行控制器在每条命令前检查候选槽位；通用verifier尚未逐命令重验全部中间物理布局。公开包保存审计结果、签名与文件hash，raw清理后无法直接重放该次raw审计。
