# 常态性能有界探索（2026-11-04）

基点：v3-20261104.1 / 8ee421c；开发分支 codex/steady-perf-20261104。原标签不改。
仅一个持有 runtime/v3-training.lock 的服务器 controller；不重建环境，不操作公司容器。

## 审计与初始计划（修改训练代码之前）

- 固定 Qwen MoE revision、BF16、seq512、TP2/PP2/EP2、microbatch/global batch8、seed20261106、staged digest；完整计时边界与九类状态审计保留。
- flat：16rank Megatron，dense DP4、expert DP2；反向就绪时预缩放并 SUM，可与反向重叠。
- SHIFT/nativeFT：各8rank Megatron，dense DP2、expert DP1；相同本地预缩放/SUM，finish_grad_sync 时逐桶追加对应角色的跨副本 SUM，再由 Manager 回调除参与数。之后才执行 TP/embedding 尾规约。
- nativeFT 仅多一层 ManagedProcessGroup 包装，最终调用同一 Manager.allreduce；不是完整 TorchTitan 对照。
- NCCL 正常路径直接使用 bucket.grad_data，无梯度 pack、D2H/H2D 或 dtype 转换。状态快照 buffer clone/RNG、quorum、跨桶 Future/timeout、should_commit 同步及 RPC 是两条路径共有成本；恢复用 PackedGPUTransport 不应误算常态 pack。
- 必须核查实际 topology、bucket 数量/dtype/字节与 trace，静态调用数不代替执行证据。

先跑一次三路径各12步的 step11 CPU/CUDA 诊断（各 PP 选 rank0/4）；采样不算性能。按 GPU 区间并集、CPU 提交与流依赖分析，不加总重叠活动。现有 trace 先复用，不提前改生产算法。

## 有界假设卡（预估待时间线收紧）

1. **跨副本梯度规约暴露于反向尾部**：train.py finish_grad_sync / bucket_finalizer。本地已就绪桶直到 finalize 才提交 outer；若存在 >=20ms 可重叠窗口，尝试保留本地→outer→TP/PP 顺序的早提交。预计0–30ms，需实测窗口；风险为缓冲区写读/流依赖及跨rank collective顺序。窗口不足5ms或约3次无可信净收益则停，不因历史已失败路线重复堆实验。
2. **Future/timeout/同步造成 CPU 提交间隙**：compat.py RecoveryWork / Manager 两桶 dispatch/wait。若时间线显示可归因的串行主线程等待，尝试只消除重复安全等待/控制工作；预计0–15ms，不能取消必要超时、错误传播或 commit 一致性；若关键路径不足5ms或需大改 Manager 则停。
3. **规约层级/大桶放置降低重叠**：Megatron桶与本地 expert DP1。只有 trace 支持才尝试保持数值规约树的桶切分或无效单成员规约旁路；预计0–25ms。切分可能改变 NCCL 算法/数值、增加提交成本；不能用改变规约顺序掩盖数值问题。预计不足5ms、数值失败或3次无改善则停。

正式验证只做同条件交错A/B（初始 ABBA，若有可信收益补 BAAB），各80步，固定11–80窗口全部保留，同时保存1–80完整步分布/提交间隔。rank/step不当独立重复。报告run均值、p50/p95/max、run间差与区间。目标相对 flat 总开销<5%（旧基线需回收20.262ms）；<=1%仍未达。

有稳定净收益候选才做原anchor受影响状态exact比较与相关恢复回归；无有效候选不扩大故障矩阵。结束记录来源SHA、分布、限制，删除可再生trace/raw/快照，保留源码、环境、私有配置、seed/manifest和最小原anchor。

## 本轮结果与止损

原已发布参考：flat 728.928ms、nativeFT接口783.668ms、SHIFT 785.636ms；SHIFT相对flat +7.779656%，相对接口仅多1.968ms且差异不确定。这些是旧成绩，本轮没有有效优化后成绩替代它们。

完成三路径各一次12步诊断、共6个选中rank的CPU/CUDA时间线；formal计划flat/SHIFT/SHIFT/flat各80步，在第2个run失败后停止。没有训练优化候选，也没有修改训练算法、规约顺序或数值容差。

|证据|结果/解释|
|---|---|
|实际规约|SHIFT/nativeFT每rank每步2次outer，dense DP2/expert DP1后接角色间2副本SUM及参与数除法；flat为DP4/2。选中rank0/4的BF16梯度载荷分别249754112/255201792bytes；nativeFT最终调用相同Manager。|
|打包/复制/转换|NCCL梯度直接使用连续bucket视图，无额外梯度pack、D2H/H2D或dtype转换。正常步的buffer克隆/RNG捕获是另外的状态保障，未删除。恢复transport打包不能算常态梯度打包。|
|CPU控制/等待|诊断中未开profiler的7–10步，边界快照+quorum提交约1.5–1.7ms；2–10步dispatch约1.8–2.0ms、wait约1.1–1.5ms、commit约1.1–1.3ms。仅用于量级判断，不作正式性能成绩，也不是把重叠成本相加后的可回收预算。|
|GPU末端|SHIFT选中rank两次outer的时间关联跨度约10.9–14.2ms，nativeFT约12.8–15.0ms；其内部线程kernel缺少精确CPU关联/PG标签，归属仅是静态路径+独立流末端两次allreduce的时间关联。不能将其与本地/TP/PP驻留相加，不能声称全部可消除。|
|PP等待/CPU间隙|profile中PP4的embedding规约驻留约238–262ms，PP0约2.1ms；两个被采样rank的扰动/晚到可产生等待，不能拿这几百ms当常态优化预算。无证据把旧56.7ms全部归因给TorchFT核心。|
|执行口径限制|同模型/批次/seq/精度/拓扑角色/样本ID；step11的部分TP规约元素数flat与共同路径不同。BF16轨迹/动态MoE执行可能参与差异，尚非从相同完整anchor重放同一步的严格因果归因。不能由此断言全部差异是数值导致。|

formal第1次flat（固定11–80，70步全留）：mean **819.898ms**，p50 **665.482ms**，p95 **1274.043ms**，max **6670.543ms**；全部1–80步和79个提交间隔见full_report.json.gz（summary.json为小型索引）。单次run不提供稳定性区间。

第2次SHIFT提交7步后，跨副本NCCL报aborted/ncclSystemError；每rank记录8次skipped，第9次should_commit拒绝触发max_retries=8退出。16rank的全部begin/commit/skipped/reconfigure相对时间、分布和源码/log SHA均保留。无注入，abort触发原因未证实，不能直接认定硬件、网络或TorchFT核心故障。后2次没有启动，正式A/B无效；不拿前7步与flat窗口比较。

诊断为避免采样超时使用20s通信期限，formal保持v3原2s；这是诊断与正式测量的另一处必须披露的差别。未通过放宽formal超时、删慢步或缩边界获取成绩。

### 假设结论

1. 原桶提前outer：静态路径和时间线确认末端提交；但本轮未采新readiness事件，不能据此否定提前规约。现有证据没有可信>=20.3ms窗口，原v3已有双桶早窗不足记录；未重复改hook。
2. 精简Future/timeout/manager：CPU量级只支持预计0–5ms，且这些工作承担超时与错误传播。当前正式路径失败和波动下，收益不足以支持改动；“控制开销独自解释大部分旧差距”不受支持。
3. 小桶/层级：本轮大桶约束及PP后段关闭分桶得到源码确认；只有分桶结合提前outer/确定性提交顺序才可能增加重叠。预计0–15ms仅为探索预算，非已测收益；改变并发/规约树还需全状态数值和恢复验证。缺少可信收益且触及更广调度，未实施。

**决定：本轮止损，无可报告的优化前后提升，<5%及<=1%均未取得新通过证据。** 不继续在当前失败/波动条件下堆实验，也不把未验证假设写成被否定。下轮应先有“原2s期限无故障基线可重复完成”的证据；再考虑用保留完整BF16 anchor对齐同一步、增加轻量readiness归因。若能证明>=20ms的暴露窗口，再单独规划确定性早规约；本轮未自动启动下一轮。

保留的开发改动仅为benchmark lighthouse端点修复、common-probe/short-abba入口与摘要脚本；10项相关CPU测试通过。没有有效训练候选，因此没有扩大故障恢复矩阵。v3标签不改、不推送发布。

复现入口：已有私有端点自动读取；prepare后以唯一suffix运行 `start_v3_benchmark --schedule common-probe --trace-events --timeline-trace --digest-engine staged` 或 `--schedule short-abba --trace-events --digest-engine staged`。确切配置和原始launch源码manifest在私有 `runtime/v3-reproduce/steady_20261104.json`、`steady_source_manifest.json`；摘要与来源SHA在 `evidence/steady_20261104/`。清理后不能从摘要重建trace，须重跑。

收尾：服务器回收 585463951 bytes，本机回收 394545479 bytes（分别记账）。三机SHIFT CPU/GPU进程和运行unit均0，公司GPU PID基线保持，临时授权marker均0。环境、模型、headers与原始BF16 anchor未动。

完整分析输出按字节gzip压缩为 `evidence/steady_20261104/full_report.json.gz`，解压SHA记录在小型summary.json；重跑摘要脚本先得到完整JSON，再压缩即可。
