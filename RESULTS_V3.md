# v3 功能、实测与未达项

发布日期：2026-11-04。用户接受当前实际效果，授权公开v3并作为后续开发基线；此授权不改变原性能/数值验收结果。所有总体性能目标仍为false。

## 功能范围

- v1：真实随机fail-stop，fresh第三机冷启动，三机备用轮换；schema2的model、optimizer、scheduler、progress、Python/NumPy/Torch CPU/CUDA/tracker RNG及buffers完整恢复（九hash分组），非空Adam、源版本只读、首次joint、后续十joint、健康PID保持和继续前进、退役进程树为空均有证据。
- v2：存活慢源提前准备，S_k状态预传，B本步梯度及RNG/buffer尾部relay，外层规约开始前接替；多轮、取消/拒绝及限定异常回退通过。仍等待B本步梯度，不是自动慢检测或任意故障保证。PreparedNCCL只支持既有受控单epoch能力；任意未知stale GPU/硬件故障未穷尽。
- nativeFT/SHIFT同anchor有十步160 rank-step exact证据；C2十二次迁移、六基线共5424 rank-step功能覆盖。状态一致不等于与flat数值轨迹等价。flat固定容差0/80通过，strictseq16亦失败；所测dense切片定位只能解释限定切片，不能宣称全模型无害。

## 性能（保持原负载与边界）

|口径|实测|限制|
|---|---|---|
|flat→SHIFT step|+7.779656%，四对run t95 [4.532637,11.054131]%|≤1%未过|
|flat→nativeFT接口 step|+7.509763%，[5.243376,9.783425]%|本地TorchFT0.2同Megatron接口，非完整Titan/已证下限|
|nativeFT→SHIFT step|+.251040%，[-2.913002,3.449611]%|不确定，不能代替总开销|
|完整commit interval flat→SHIFT|+7.782814%|仍未过|
|P11纯冷 Popen前→首次16rank joint|18.118335 / 18.295410s|≤15/10s未过；候选默认off|
|旧同研究路径纯冷|21.389271s（n1）|对候选n2，不能全归因CPU加载重叠|
|最近staged受控慢化freeze→joint|2.262413 / 2.161504s|非总迁移时间，2s仍未过|
|对应完整commit gap|7.157995 / 6.814702s|含切换间隔；不能与旧正常audit-off额外gap混比|

常态三模式各4×80步、预定11–80统计窗口，15360 raw rank-step核验；全部原始首步、慢步、skip仍保留，rank/step不当独立重复。step均值flat728.928ms、nativeFT783.668ms、SHIFT785.636ms。审计、staged/reference digest与慢化条件必须随成绩一起说明；旧v2.1正常/audit-off成绩不可直接替代本轮受控慢化。

跨主机墙钟未独立校时，同rank monotonic仅作分段；state_loaded/局部joint不是全球恢复终点。P11故障→joint首样本20.702180s，与Popen口径不同。正式注入12、历史排障5；限制已取消，原始失败记录保留。

## 默认与候选

staged digest需要显式选择；CPU load/import重叠、lazy导入、ownership、shadow/recompute与细分桶/trace均为实验候选，优化开关默认off，未证明收益的路线不改成生产通过。仅将部署默认私有地址替换为RFC文档地址、移除私有部署文档打包项；训练计算与验收语义未作发布性修改。公开源码SHA与实验冻结源码SHA分别记录，不能混称逐字相同。

公开小证据是从本机不可变原始材料选取的摘要与SHA索引，不包含大张量/raw归档。可核对摘要与公开源码身份，但没有私有原始归档就不能独立重算全部GPU成绩；必要原件及唯一SHA归档仍保留。见[evidence/v3](evidence/v3/summary.json)和[复现入口](docs/v3_reproduce.md)。
