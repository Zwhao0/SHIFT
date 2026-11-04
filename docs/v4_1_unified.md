# v4.1 代码重构统一

统一 `training_core.py` 的模型分片/DDP、batch计划、前后向事务、梯度有限性、Adam/scheduler/提交游标和完整状态接口。策略在反向完成后、Adam之前投票；拒绝不会推进优化器或样本。`train.py`、`planned_train.py`、`v4_train.py` 保留各自启动、诊断和成员/通信策略。

固定 global batch=8、seq512、BF16、TP/PP/EP/eTP=2。初始物理A=node0和B=node1各有一个8-rank完整DP副本；C=node3备用。v4的旧演员名A/B分别是DP0的两个PP阶段，D/E是DP1的两个PP阶段，不应把演员名与物理机器混淆。TP始终在节点内，v4迁移后才由node0/node3分担DP0的PP。

- v1：真实SIGKILL目标机器的8个SHIFT worker；健康副本继续batch8单副本计算；原生TorchFT Manager从健康完整状态恢复至C。
- v2：p2整副本预复制，慢源仍完成必要梯度；在外层规约之前传递梯度、buffers和完整RNG尾部，再交接给C。
- v4：只替换DP0的PP1四个逻辑rank，健康完整DP1及DP0的保留阶段继续存活；验证候选后在已提交边界切owner，重建受影响通信。

DP配对分片的RNG是复制状态，初始化按逻辑rank而非物理GPU；样本随机性独立使用`100000+sample_id`生成器。这样原生TorchFT可恢复完整复制状态。旧v2独有的replica种子起点已移除。模型/Adam/scheduler、持久和非持久buffer、提交进度、CPU/CUDA/Python/Megatron tracker RNG仍分别严格校验，未减少九状态字段。

正常运行不逐步计算hash。开发审计额外记录完整九状态、独立梯度/轨迹和owner/成员/提交证据。审计耗时不作为性能基线；性能迁移仍包含必要的边界状态快照/验证，相关成本必须计入完整时间。

入口：`scripts.prepare_v3_controller`每轮一次，随后`start_quick_recovery v1/v2`或`start_v4 --unified --transactional-switch`。实验共用`runtime/v3-training.lock`且严格串行；沿用已安装环境和模型，显存比例上限0.12。

`run_v41_suite`接受私有manifest并串行执行基线、参考与随机场景；任一失败即停。计时保留正常、准备中、首次共同提交、恢复后窗口、完整控制器时间与等待。v4另有`--post-layout --reference-only --performance-no-audit`，专门测迁移后跨节点PP布局的无迁移控制。该控制不属于初始训练布局。

验收结果和来源SHA以随代码发布的精简证据为准。旧batch16 v4日志、其他公司负载下的数据以及逐步hash审计时间均不能替代本轮匹配基线。共享宿主上的短测试只能给出功能和诊断计时，不能证明任意通信故障容忍、独立故障域或统计显著加速。


本轮107项CPU测试通过。v1/v2连续16步、16逻辑worker的九状态和独立梯度一致；v4 32步独立参考通过，公共前16步九状态/样本与v1完全一致。详见`evidence/v4_1/summary.json`，其中保留所有接受/拒绝案例、来源SHA、固定窗口全样本、完整时间及等待，未将旧batch16审计耗时作为基线。

v1只在kill后启动的新目标上先做本地前后向热身，再加入原生FT quorum；这不预备备用进程，热身时间包含在恢复总时间内。性能模式关闭恢复过程中的开发hash，在恰好五个共同提交后单独严格比较16个角色的最终九状态；完整恢复审计另有两次运行，未削弱其源/目标恢复签名检查。

v4候选超过验证期限可保留原owner；控制器必须等每个未被明确停止的actor全部四个device出现done/retired/standby_done，才结束实验。发布验收核对实际decision，retain不算成功迁移。一次早期失败记录与相同seed/触发点/10秒期限的修复后复测、受控超时回退结果均保留。

计时控制没有重建或调换环境；共享公司负载非平稳，正常native的80步窗口分别出现2.478和0.722秒。v2的80步为0.763秒，相对更快的native约+5.7%；A/C整副本布局控制0.781秒，v2两例post为0.858/0.840秒。v1无计时hash的完整恢复19.627/20.061秒，v2迁移22.008/21.515秒，v4迁移21.600/33.666秒；后者高负载例的同例pre/post约3.937/4.061秒。首次共同步、必要边界验证和同步等待另列，不声称恢复完整延迟的隔离因果加速。

新建参考示例：

```sh
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix ref --unified --transactional-switch --outer-audit --reference-only --cut 16 --post-steps 16
# 等待参考和资源检查完成；每轮重新prepare，禁止训练中prepare。
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix move --unified --transactional-switch --outer-audit --reference-run ref --candidate-request-step 9 --cut 12 --post-steps 5 --slow-ms 250
```

V1/V2通过`start_quick_recovery`运行，V2显式`--variant p2`；计时用`--performance-no-audit`，开发校验单独运行。所有GPU实验使用单训练锁、串行cgroup和0.12显存上限。参考必须覆盖实际cut和后续步数；不允许源码/profile不匹配或比较失败后继续空等。
