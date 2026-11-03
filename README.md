# SHIFT v2.1 — v2继续优化

新增完整MoE状态预传、本步梯度接替、B/C原进程连续轮换，以及限定异常的新epoch恢复。保持两份有效贡献，精度、batch和归一化不变。

P1f/P2各5次新对照的交接间隔中位数为1.195/1.188秒，差异区间跨0，未证明P2稳定提速。**常规单次替换推荐P1f；P2提供预传/本步接替、多轮与本轮受控恢复能力。** 详细结果、边界与复现见 [RESULTS_V2_1.md](RESULTS_V2_1.md)、[summary.json](results/v2_1/summary.json)。

```bash
python3 -m shift_full.deploy --include-node3
python3 scripts/verify_overlap_source.py tmp/deployed-v21.json
python3 scripts/run_planned_handoff.py --name UNIQUE_REPLACE --variant p1f --source B --post-steps 50
python3 scripts/run_planned_handoff.py --name UNIQUE_RELAY --variant p2 --source B --audit --post-steps 6
python3 scripts/run_overlap_cycles.py --name UNIQUE_CYCLES --cycles 3 --audit --post-steps 6
python3 scripts/unpack_v2_evidence.py --evidence results/v2_1 --output tmp/v21-evidence
```

三机，每DP副本8×H800，约0.495B MoE、合成batch8，TP/PP/EP/eTP均为2。依赖已有FLASH/Megatron、模型与`runtime/torch214`；环境沿用v2。凭据仅存Git排除的`server.md`，模板为`server.example.md`。

v2的计划性交接结果保留：[RESULTS_V2.md](RESULTS_V2.md)、[results/v2](results/v2)。v1的进程故障恢复结果保留：[RESULTS.md](RESULTS.md)、[results/v1](results/v1)。v1严格FP32数值失败未改变；本轮状态/梯度审计通过不代表已解决数值轨迹等价问题。
