# SHIFT v1

三机进程故障恢复实验已完成。长测60次/3600秒通过，状态恢复、首个共同提交与RPO0均经离线原始日志核验。版本 v1：kill 进程恢复。

- 数值三步对照：`false`；按原容差报告，未放宽。
- ABBA步时变化：4.84%；备份建议：gpu（GPU热备需预留8张GPU）。
- 详细证据：`RESULTS.md`、`results/v1/summary.json`。

```bash
python3 -m shift_full.deploy --include-node3
python3 -m shift_full.rotation --session UNIQUE_LONG --config configs/shift_rotation_verify.json --seconds 4800 --faults 60 --interval 60 --duration 3600 --seed 20261102
python3 -m shift_full.rotation --session UNIQUE_GPU_HOT_AUDIT --config configs/shift_rotation_audit512.json --seconds 900 --faults 1 --backup-mode gpu --seed 20261102
python3 scripts/unpack_v1_evidence.py --evidence results/v1 --output tmp/v1-evidence
python3 -m shift_full.verify_rotation tmp/v1-evidence/v1_rotation_long_20261102_01 --expected-faults 60 --minimum-duration 3600
```

严格FP32复核在首个更新后通过、第二个更新仍失败，路由签名一致；原容差保留。单坐标AdamW分析见 `RESULTS.md` 与 `results/v1/adam_coordinate.json`。

负载约0.5B MoE、合成token；两活一备，每副本8×H800，TP/PP/EP/eTP均为2。依赖现有 `/workspace/FLASH1022/{flash_v1,Greyhound-master/Megatron-LM,models/qwen_moe}` 与 `/workspace/SHIFT/runtime/torch214/bin/python`；版本清单见 `runtime/torch214-manifest/`。本机凭据仅存Git排除的 `server.md`，模板为 `server.example.md`。
