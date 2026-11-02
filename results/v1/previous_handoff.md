# SHIFT 下一窗口交接（2026-11-02）

先读AGENTS.md。用户已同意按1→2→3→4执行；本窗口只写交接，未启动新实验。保持简洁，每次优化前说明，进展/卡点及时告知；简单重复任务可交GPT-6 Luna/max，仅回摘要。

## v1之后的新实验（仅讨论，暂不实施）

先完成下列fail-stop四项与验收、上传GitHub v1，再继续FAIL_SLOW_PLAN.md：思路一为外部指定存活DP副本的计划性交接（不注入延迟、不接检测、不重跑v1 kill恢复；先验证状态/未来AC组后台准备，以减少恢复与训练停顿为目标）；思路二为同一PP/TP副本内唯一分片迁移，仅记录。新草案不改变当前执行顺序。

## 已完成

- 三机两活一备接管4次成功；全状态审计14/14，8/8 rank模型/非空AdamW/scheduler/进度/CPU+CUDA+Python RNG哈希一致；首个真实共同步强制核验，16组角色-步全部7项哈希一致，已共同提交步RPO0。
- seq512三次无审计计时中位：请求→8rank加载1.498s；检测/仲裁/shrink 3.091s；分段累计4.623s（非连续RTO）；完整故障→共同提交20.405s。健康最大提交间隔1.049s、solo0.703s、额外间隔中位0.352s；启动→ready约15.993s。
- 证据：results/three_node_20261102/{final_summary,timing_summary,cleanup}.json；全审计shift_takeover_verify_20261102_01（seq16），计时shift_takeover_timing_20261102_01..03（seq512）。
- 旧两机ABBA额外步时5.42%，用户接受约5%；新拓扑未测。旧两机10分钟10故障通过，不等于三机轮转长测。自由两步FP32仍超容差，锚定下一步全8角色通过。

## 1. torchFT + TorchTitan对照，困难即跳过

- 限时30分钟，优先现有资源与兼容版本；依赖/模型/拓扑难对齐即提前跳过，记录原因并进入2，不耗时重搭环境或升级现有训练栈。
- 尽量匹配模型、状态规模、序列、batch、精度、并行和故障方式，固定版本/commit。不同TorchTitan负载只能作诊断，不能声称原生开销可比；已有native_probe也仅匹配状态规模。
- 比较正常步时、健康副本推进、请求→加载、完整恢复与健康最大提交间隔；冷/温/热备分别报告。

## 2. 三机随机kill/接管，通过后1小时60次

- 两活一备：记录随机种子，随机初始rid放置；每轮随机kill当前两个训练节点之一，由唯一空闲节点接管，故障机清理后转备用。覆盖三台物理机、两logical rid；接管者随布局轮转，当轮只有一个空闲候选。
- 解耦physical endpoint/rid/generation，generation单调递增，支持节点/目录复用、旧进程和消息隔离；记录候选、选择、绑定、时间、PID。先修subreaper/init回收。
- **先将Lighthouse独立于rid0训练supervisor生命周期**；rid0被kill/迁移时不能停掉或重复启动控制服务。只kill项目训练进程树，保留控制器/Lighthouse；本轮仍是进程fail-stop，不声称主机掉电。
- 当前--takeover-node3仅首次将rid1从node1迁到node3，后续留在node3，尚不支持随机轮转。
- 先10次随机接管全状态短测：每次8rank完整状态及首真实共同步哈希通过、健康PID不变且推进、已提交步RPO0、无活跃/僵尸残留。
- 通过后故障窗口3600秒，每60秒一次，共60次；不含初始化/清理。上一轮恢复后再注入，不kill仅剩健康副本；错过计划或次数不足须如实报告。
- 长测保留每次恢复与首共同步审计，稳态哈希抽样；报告60/60、损失/进度/参与者、恢复分布、健康停顿与资源。正常性能另开关闭审计的重复实验。

## 3. 定位数值正确性边界

- 保留失败证据和原容差；从同一精确状态比较下一步及短轨迹，定位梯度/更新/MoE路由/规约顺序差异。
- 分开证明恢复状态一致与无故障参考计算一致；解释FP32偏差，不能放宽容差或用锚定一步替代轨迹验证，也不强求不同规约逐位相等。
- 当前动态副本有效batch8→4；核验数据进度/固定batch语义，避免把batch变化误判为恢复错误。

## 4. 优化常态开销、完整恢复及恢复开销

- 用户已允许温备/热备，更新此前不优先CPU常驻/缓存的偏好；保留普通冷torchrun基线，不混用计时。
- 温备候选：CPU运行时/配置准备。热备候选：完整GPU执行副本已构建、本地预热，在仲裁加入前待命；故障后加载最新完整状态，加载/校验通过才贡献梯度和提交，不用过期状态。旧fast_launch/CPU常驻不是已验证解，先读历史证据。
- 按完整执行副本和TP/PP/EP/eTP逻辑角色、分片shape/dtype、优化器组匹配备用资源；当前一机一副本只是特例。真实副本可能跨机，需预留整组兼容资源，不能按物理GPU号对应或假定异构可直接接管。
- 冷/温/热使用同负载与故障调度，先全状态审计，再关审计重复计时；分别报告准备、检测/仲裁、传输/加载、首共同提交、健康停顿，以及备用资源/GPU显存/CPU占用和常态新增通信。
- 新拓扑flat/SHIFT做ABBA，固定seq512/batch/精度、剔warmup；GPU实验串行。原生对照跳过时以flat为基线，不宣称原生等价。优先根本优化。
- 停止标准：轮转长测/正确性通过，数值边界解释清楚，额外步时约5%可接受；如有可比原生基线，恢复/健康停顿无明显额外劣化即可收工。不无限追不同规模论文的绝对数字。

## 环境与操作

- 代码名称：node0=10.10.4.2、node1=10.10.4.1、node3=10.10.4.3（用户称三台node1/2/3）。SSH=test；凭据仅查server.md，不输出。
- 三机宿主机/home/campus/zhangweihao/{SHIFT,FLASH1022}，容器/workspace/{SHIFT,FLASH1022}；环境已部署，无需重新搬大镜像。Python /workspace/SHIFT/runtime/torch214/bin/python；torch2.14.1+cu130/NCCL2.30.7/torchFT0.2.0。
- node0容器flash1022-run；node1容器shift-rdma-node1。Docker入口：env DOCKER_HOST=unix:///run/greyhound-docker.sock /opt/greyhound-docker/docker，node1加sudo -n。
- node3容器shift-rdma-node3，sudo -S + /usr/local/bin/nerdctl --namespace shift；controller已实现密码仅stdin。保留其他容器/k8s.io/vLLM。
- 保留原生torchFT NCCL/IB、GPU梯度/状态、64M桶/hook、async quorum、加入前预热；HCA =mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1，GID3已核验。
- 核心shift_full/{control,supervisor,train,transport,analyze,paper_metrics}.py；原始physical_node0/1/3与合并node0/1日志分开，physical_manifest.json存绑定。配置shift_takeover_verify.json / shift_takeover_timing.json。
- 测试：python3 -m unittest shift_full.test_control_routing shift_full.test_event_reader shift_full.test_topology -q（16项）。部署：SHIFT_INCLUDE_NODE3=1 zsh tmp/deploy-code.sh。session唯一，caffeinate防休眠，运行中勿覆盖代码，只清项目PID。
- 上轮124个项目PID已清理；node1因PID1不回收留32个SIGKILL僵尸，确认无训练后重启专用容器回收；需在第2项根治，不重启其他容器。
- 模型约0.5B MoE，每副本8×H800，TP2/PP2/EP2/eTP2，合成token；论文H200/7B+/真实语料、同负载原生性能尚未复现。
- 旧归档tmp/shift-takeover-20261102.tar.gz保留；最新计划以此文为准，旧交接备份results/three_node_20261102/handoff_before_four_stage_plan.md。论文文本tmp/pdfs/thesis.txt。
