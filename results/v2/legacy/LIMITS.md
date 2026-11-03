# 实测边界

- 完整负载：现有同构0.495B MoE，TP/PP/EP/eTP=2，8 GPU/副本，seq512，全局batch8，合成随机token；当前训练精度配置为 `fp32=false / TF32=true / matmul=high`。未搭TorchTitan、未迁移副本内部PP/TP。
- 10次全审计是10次独立启动，每次交接内A PID不变；**同一常驻A连续10次轮转=0**。B存活至全部rank的第三个AC提交，随后正常结束实验；未实现其后再次晋升。
- 精确哈希验证包括恢复模型/Adam/scheduler/progress、所有named buffers（含非persistent）、三种RNG，以及前三个AC共同步共享状态。数据角色按真实rid随机样本序列审计。未新增无交接同轨迹的张量数值oracle；不将共享状态一致等同于对任意原算法的数值等价。
- v1严格FP32第二更新仍超原容差的边界保留；本轮未重跑、未放宽容差或修改Adam ε。
- FIXED_WITH_SPARES更新型影子仅在三物理单GPU Linear微验证成立；完整MoE跳过备用前后向、buffer同步和影子晋升未实现。
- PreparedNCCL是新增的单epoch适配：新Manager从构造起拥有候选PG，configure校验后复用，之后新quorum拒绝。不是torchFT原生候选复用API，也不是通用容错PG。
- 真取消验证在C已经准备完成后进行，AB继续8步；准备阶段失败自动保留AB、任意时刻取消、激活后新epoch回退未实现。失败调试运行会停止并清理本实验。
- 8个拒绝负例和1个取消门控单测在应用协议层；实际旧B尝试走同一训练authorize门控被拒，未强行注入Rust过期RPC。
- 版本快照/计算窗口是单GPU Linear组件，stream事件区间重叠不等于已通过profiler证明硬件核并行。无异步网络传输、无完整MoE P2/P3、无增量/重放追赶；预拷贝落后后最终仍完整传输。
- 正常性能与审计分开。50步native/adapter控制各一次，不能替代多轮ABBA常态开销验证；AC后只有3步，常态长期性能证据有限。
- 未注入延迟、未接检测、未重跑kill恢复。失败清理可能终止自有训练进程；这些运行不计为正常live-B交接。成功主路径全部训练进程自然exit0，Lighthouse随后受控关闭。
