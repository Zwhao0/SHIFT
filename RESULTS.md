# v1 实验结果

## 故障恢复

- 短测：10/10通过。长测：60/60，3600.003秒，schedule_late=0。
- 恢复全状态：480/480 rank-hash；首个共同提交：480/480 rank-hash；RPO0：60/60。Lighthouse存活60/60，健康PID不变60/60。稳态审计293步、2344角色-步。
- 长测审计开启、seq16：完整故障→共同步p50/p95 27.893/29.921秒；这是带审计结果。seq512关闭审计的三轮计时见下表，口径分开。
- 全状态模式审计：cold 1/1, cpu 1/1, gpu 1/1均通过。

## 备份计时（seq512，关闭审计）

| 模式 | 准备p50 | 故障→共同步p50 | 请求→就绪p50 | 恢复请求→加载p50 | 故障期健康max间隔p50 | 准备期健康max间隔p50 |
|---|---:|---:|---:|---:|---:|---:|
| cold | 0.000 | 22.680 | 16.726 | 1.519 | 3.581 | n/a |
| cpu | 3.250 | 19.427 | 13.876 | 1.287 | 3.659 | 0.943 |
| gpu | 16.799 | 6.737 | 0.465 | 1.587 | 3.511 | 0.949 |

最快实测中位故障→共同步为 `gpu`；GPU热备需要额外预留整组8-GPU副本。保留cold作为基线。GPU/RSS/CUDA分项见 `results/v1/summary.json` 的backup_timing。预备期跨副本通信为结构性零（Manager在ACTIVATE前未构造）；这不是带宽测量结论。

ABBA flat→SHIFT→SHIFT→flat；步时变化 4.84%，吞吐变化 -4.62%。

## 数值轨迹

以相同step1精确状态起点，比较commit step2/3/4；batch与路由计划覆盖pre-update step1/2/3。16个角色的7项起始哈希、batch、输入和participants/global_batch检查通过。原比较容差保留；数值比较通过=`False`，超容差角色-步对=16。逐step/类别最大绝对误差、relative-L2见 `results/v1/summary.json`。

首次有序路由差异发生于pre-update step 2；完整签名与误差见汇总JSON。路由边界margin见汇总JSON。观测不证明根因；FP32 reduction/addition顺序差异只是可能解释。

## 范围

故障是训练进程树SIGKILL，Lighthouse/控制面存活；动态batch由8变4，固定batch8只用于数值轨迹。每节点一个8-GPU副本。未运行完整同负载TorchTitan对照，未验证主机掉电。精选公开证据不含凭据、模型.pt或runtime上传。


## 严格FP32与路由诊断

旧轨迹为FP32参数、TF32-high路径；严格复核禁用TF32并设highest（flat/shift各16条ready记录均核验）。起点、batch、participants均匹配；严格两更新共736个角色/层/微批ordered与sorted路由签名相同。step2八个rank通过，step3八个rank仍按相同容差失败。旧路径在pre-update step1/2/3的路由差异数为0/32/152；TF32放大了旧路径差异，但不是唯一解释。

单副本复放对flat/shift各368个step2有序路由和16个样本计划均复现；跨条件按角色/rank、层、微批序号比较有2个top-k成员变化、30个仅选择顺序变化（不是token计数）。

AdamW标量例：rank1 `decoder.layers.4.mlp.experts.15.down_proj.weight[16002]`，step2两侧p相同，g2/m2/v2为0；step3的极小梯度为2.41137e-08与-3.52024e-08，符号相反。经AdamW归一化后参数更新为-3.71709e-05与4.28362e-05；保存矩量代入标量公式后，预测p3与观测差分别2.3e-10、2.5e-11。ε=1e-08。这是一个坐标的数值敏感性证据，不证明全局reduction根因或整段轨迹等价；未改容差或ε。

冷备为冷进程启动，文件页缓存未清空。实验不含真实语料dataloader状态验证。原始日志与源码快照有SHA256清单；大型张量快照保留服务器、不进入GitHub。
