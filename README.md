# SHIFT v2 — 副本间慢故障恢复

两份有效DP副本A+B继续训练，待命C提前初始化并与A预建、预热未来外层NCCL通信组；在共同提交边界继承仍存活B的最新状态，切为A+C。最终版复用通信组，并把旧组的物理回收放到首个AC共同提交之后；A/C侧在第三个AC提交前完成回收。

这是外部指定待替换副本的计划性交接原型；没有注入真实慢故障。状态仍在最终边界从B完整同步，短暂等待由实测报告，不承诺零停机。实现、最终对照、未完成项与复现见 [RESULTS_V2.md](RESULTS_V2.md)，机器可读结果见 [summary.json](results/v2/summary.json)。

负载约0.495B MoE、合成token；两活一备，每副本8×H800，TP/PP/EP/eTP均为2。依赖现有 `/workspace/FLASH1022/{flash_v1,Greyhound-master/Megatron-LM,models/qwen_moe}` 与 `/workspace/SHIFT/runtime/torch214/bin/python`；版本清单见 `runtime/torch214-manifest/`。凭据仅存Git排除的 `server.md`，模板为 `server.example.md`。

```bash
python3 -m shift_full.deploy --include-node3
python3 scripts/run_planned_handoff.py --name UNIQUE_V2 --variant p1f --source B --audit --post-steps 6
python3 scripts/unpack_v2_evidence.py --output tmp/v2-evidence
python3 scripts/analyze_planned_handoff.py tmp/v2-evidence/v2_final_timing_10_p1f
```

v1的进程SIGKILL恢复实验、60次/3600秒长测及公开证据保留，见 [RESULTS.md](RESULTS.md) 与 [results/v1](results/v1)。严格FP32第二次更新仍未通过原容差，未放宽容差或修改Adam epsilon。v2状态交接通过不代表该数值问题已解决。
