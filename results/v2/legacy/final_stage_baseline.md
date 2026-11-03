# Planned handoff final-stage baseline (p1, source B)

离线统计 `ph_timing_*_p1_B` 已存日志：5 次成功运行、每次 A/B/C 各 8 rank（40 rank 样本），cut step 27–32。没有运行训练/GPU。A 用 `physical_node0` 的同机 `monotonic`；B 用 `physical_node1`；C 用 `physical_node3`。没有跨主机相减。逐 rank 原始间隔见 [final_stage_baseline.json](final_stage_baseline.json)。

## A 关键路径

表中数值为中位数 / p90。首个 AC commit 选 cut 后最小 step；`begin→commit` 包含首个 AC 训练步。

| A 上阶段 | 秒（median / p90） |
|---|---:|
| last AB commit → freeze | 0.005 / 0.007s |
| freeze → old Manager closed | 0.587 / 0.613s |
| old Manager closed → epoch activated | 0.826 / 0.884s |
| epoch activated → first AC begin | 0.014 / 0.025s |
| first AC begin → first AC commit（含训练步） | 0.892 / 0.932s |
| last AB commit → first AC commit | 2.329 / 2.364s |

## B / C 各自时钟

| 主机阶段 | 秒（median / p90） |
|---|---:|
| B freeze → old Manager closed | 0.536 / 0.542s |
| snapshot_begin → snapshot | 0.013 / 0.016s |
| snapshot → send event（含发送） | 0.180 / 0.202s |
| transport_send 自报时长 | 0.171 / 0.192s |
| transport_recv 自报时长 | 0.741 / 0.771s |
| state load 自报时长 | 0.025 / 0.028s |
| restored → epoch activated | 0.577 / 0.641s |
| epoch activated → first AC begin | 0.023 / 0.062s |

## 结论与最小优化

A 的 freeze→old-close（0.587s）加 old-close→activated（0.826s）共约 1.41s，是主要切换成本；首个 AC 训练步另约 0.89s。A 的 activate→begin 仅 0.014s。B 每 rank 发送约 749 MB GPU 状态；send 自报中位数 0.171s。C recv 自报 0.741s，恢复本身约 0.025s；C 的 restored→activated 还有 0.577s。B/C 只报告各自主机内时长，不把单调时钟拼成跨机流水线。

建议最多两项，均保留 B 存活且不注入延迟、不 kill：

1. **B 发送与旧 Manager 退役重叠。** 在 AB 最终状态冻结后，用已预热的独立 state PG 同步做 snapshot/send，同时优雅关闭旧 Manager/PG；两者完成后再报告 loaded/激活。最多隐藏约 0.17–0.18s，需维持发送快照不可变及关闭失败时不激活。
2. **评估 C 的 idle state-channel 清理是否可与首个 AC 步重叠。** recv/load 已完成后该 channel 不再用于训练；目前 `close_prepared()` 约占 restored→activated 的 0.58s。先保持它与 candidate AC PG 分离，后台关闭并确认成功/记录错误，随后在首个 AC commit 或退出前 join；这项需小规模验证 GPU/communicator 争用，不做无条件延后释放。

## Manager / quorum 预备判断

P1 已在 AB 期间后台准备并 warmup 独立 candidate AC `ProcessGroup`（`PreparedNCCL`）；这是可安全复用的部分。`planned_train.py` 在 gate 激活后才 `make_manager()`、`start_quorum()`。当前 TorchFT `Manager` 构造会在 group rank 0 创建 `ManagerServer` 并向 Lighthouse 注册；`start_quorum()` 会驱动 quorum/PG reconfigure。仅靠训练门控 `EpochGate.authorize()` 无法阻止这些注册副作用，因此不能在活跃 AB 期间安全预构造 Manager 或启动 AC quorum。关键阶段 A 的 activated→first begin 中位仅 14ms，继续预热 Manager/quorum收益很小。
