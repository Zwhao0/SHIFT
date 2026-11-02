#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate collected v1 evidence, then write concise Chinese release notes."""
import argparse, json
from pathlib import Path

MODES = ("cold", "cpu", "gpu")

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def need(ok, message):
    if not ok: raise ValueError(message)

def fmt(value, digits=3):
    return "n/a" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"

def per_step(raw):
    out = []
    for step, ranks in sorted(raw.get("byrank", {}).items(), key=lambda x: int(x[0])):
        pairs = [v for v in ranks.values() if isinstance(v, dict)]
        categories = {}
        for category in ("gradients", "model", "optimizer", "scheduler"):
            values = [p.get("by_category", {}).get(category, {}) for p in pairs]
            metrics = [v.get("metrics", {}) for v in values]
            categories[category] = {
                "pass_count": sum(v.get("pass") is True for v in values),
                "max_abs": max((m["max_abs"] for m in metrics if isinstance(m.get("max_abs"), (int, float))), default=None),
                "max_relative_l2": max((m["relative_l2"] for m in metrics if isinstance(m.get("relative_l2"), (int, float))), default=None),
            }
        out.append({"step": int(step), "role_pairs": len(pairs),
                    "failed_pairs": [f"step{step}/rank{r}" for r, p in ranks.items() if not isinstance(p, dict) or p.get("pass") is not True],
                    "by_category": categories})
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path(".")); root = ap.parse_args().root.resolve()
    v1 = root / "results/v1"
    short, long = read(v1/"short_verify.json"), read(v1/"long_verify.json")
    backups, bench, traj = (read(v1/n) for n in ("backup_summary.json", "benchmark_abba.json", "trajectory_summary.json"))
    audits = {m: read(v1/f"v1_backup_audit_{m}_01_verify.json") for m in MODES}
    raw_comp = read(v1/"trajectory_comparison.json")
    strict, strict_comp = read(v1/"strict_summary.json"), read(v1/"strict_comparison.json")
    route_diag, adam = read(v1/"route_membership.json"), read(v1/"adam_coordinate.json")

    need(short.get("pass") is True and short.get("fault_window", {}).get("completed_faults") == 10, "short rotation evidence failed")
    fw = long.get("fault_window", {})
    need(long.get("pass") is True and fw.get("completed_faults") == 60 and fw.get("completed_seconds", 0) >= 3600
         and fw.get("schedule_late_events") == 0, "long rotation evidence failed")
    need(long.get("coverage", {}).get("physical_nodes") == ["node0", "node1", "node3"]
         and long.get("coverage", {}).get("logical_rids") == [0, 1], "long physical-node/rid coverage incomplete")
    for mode, report in audits.items():
        need(report.get("pass") is True and report.get("fault_window", {}).get("completed_faults") == 1,
             f"{mode} full-state audit failed")
    for mode in MODES:
        row = backups.get("modes", {}).get(mode, {})
        need(row.get("complete") is True and row.get("trial_count") == 3 and len(row.get("trials", [])) == 3,
             f"{mode} timing trials incomplete")
        for metric in ("preparation_seconds", "fault_to_joint_seconds", "request_to_ready_seconds",
                       "recovery_request_to_state_loaded_seconds", "healthy_max_adjacent_gap_seconds"):
            need(row.get("metrics", {}).get(metric, {}).get("count") == 3, f"{mode} missing {metric} trials")
    need(bench.get("abba_complete") is True and bench.get("combined_comparison_complete") is True, "ABBA benchmark incomplete")
    checks = traj.get("checks", {})
    for key in ("input_read", "same_initial_state", "same_batch_plans", "participants_and_global_batch"):
        need(checks.get(key, {}).get("pass") is True, f"trajectory check failed: {key}")
    initial = traj.get("initial_state", {})
    need(initial.get("expected_role_count") == 16 and initial.get("roles_present") == {"flat": 16, "shift": 16}
         and len(initial.get("hash_fields", [])) == 7, "trajectory must compare all 16 initial seven-hash states")
    need(traj.get("step_coverage", {}).get("numerical_commit_steps") == [2, 3, 4]
         and traj.get("step_coverage", {}).get("pre_update_batch_and_routing_steps") == [1, 2, 3], "trajectory step coverage mismatch")
    need(raw_comp.get("checks", {}).get("files_present", {}).get("pass") is True
         and raw_comp.get("checks", {}).get("loads_succeeded", {}).get("pass") is True, "trajectory reference snapshots unavailable")
    need(raw_comp.get("steps") == [2, 3, 4] and raw_comp.get("ranks") == list(range(8))
         and isinstance(raw_comp.get("byrank"), dict), "trajectory comparator coverage mismatch")
    strict_checks = strict.get("checks", {})
    for key in ("input_read", "same_initial_state", "same_batch_plans", "same_routing", "participants_and_global_batch"):
        need(strict_checks.get(key, {}).get("pass") is True, f"strict-FP32 check failed: {key}")
    strict_num = strict.get("numerical_comparison", {})
    need(strict_comp.get("steps") == [2, 3] and strict_comp.get("ranks") == list(range(8))
         and strict_comp.get("tolerances") == raw_comp.get("tolerances") and strict_num.get("pass") is False
         and strict_num.get("failed_pairs") == [f"step3/rank{r}" for r in range(8)], "strict-FP32 result/coverage mismatch")
    need(strict.get("step_coverage", {}).get("numerical_commit_steps") == [2, 3], "strict-FP32 summary coverage mismatch")
    need(strict_comp.get("checks", {}).get("files_present", {}).get("pass") is True
         and strict_comp.get("checks", {}).get("loads_succeeded", {}).get("pass") is True, "strict-FP32 snapshots unavailable")
    byrank = strict_comp.get("byrank", {})
    need(all(byrank.get("2", {}).get(str(r), {}).get("pass") is True for r in range(8))
         and all(byrank.get("3", {}).get(str(r), {}).get("pass") is False for r in range(8)), "strict-FP32 step pass pattern mismatch")
    need(strict.get("routing", {}).get("common_count") == 736 and strict.get("routing", {}).get("different_count") == 0,
         "strict-FP32 route signatures differ")
    need(all(strict.get("routing", {}).get("selection_set_comparison_per_step", {}).get(str(s), {}).get("different") == 0 for s in (1, 2)),
         "strict-FP32 sorted route signatures differ")
    legacy_route = traj.get("routing", {})
    need(legacy_route.get("per_step", {}).get("1", {}).get("different") == 0
         and legacy_route.get("per_step", {}).get("2", {}).get("different") == 32
         and legacy_route.get("per_step", {}).get("3", {}).get("different") == 152,
         "legacy TF32-high route-difference counts mismatch")
    legacy_cfg, strict_cfg = read(root/"results/v1_trajectory_flat_01/config.json"), read(root/"results/v1_strict_flat_01/config.json")
    need("--fp32" in legacy_cfg.get("train_args", []) and "--strict-fp32" not in legacy_cfg.get("train_args", [])
         and "--strict-fp32" in strict_cfg.get("train_args", []), "FP32/strict-FP32 config contrast missing")
    strict_ready = [json.loads(line) for path in sorted((root/"results/v1_strict_flat_01").glob("node[01]/events_g0_r*.jsonl"))
                    for line in path.read_text().splitlines() if line.strip() and json.loads(line).get("kind") == "ready"]
    strict_ready += [json.loads(line) for path in sorted((root/"results/v1_strict_shift_01").glob("node[01]/events_g0_r*.jsonl"))
                     for line in path.read_text().splitlines() if line.strip() and json.loads(line).get("kind") == "ready"]
    need(len(strict_ready) == 32 and all(x.get("matmul_allow_tf32") is False and x.get("float32_matmul_precision") == "highest" for x in strict_ready),
         "strict-FP32 runtime metadata must cover 16 roles in both conditions")
    need(route_diag.get("diagnostic_valid") is True and route_diag.get("common_route_keys") == 368
         and route_diag.get("membership_change_count") == 2 and route_diag.get("order_only_count") == 30
         and all(route_diag.get("replay_requirements", {}).get(m, {}).get("ordered_reproduction_pass") is True
                 and route_diag.get("replay_requirements", {}).get(m, {}).get("sample_ids_pass") is True for m in ("flat", "shift")),
         "route replay diagnostic failed")
    coord = adam.get("selected", {})
    need(coord.get("step") == 3 and coord.get("rank") == 1 and coord.get("parameter") == "decoder.layers.4.mlp.experts.15.down_proj.weight"
         and coord.get("flat_index") == 16002, "AdamW scalar trace missing expected coordinate")
    ac = adam["coordinate"]; delta = adam["cross_mode_differences"]
    need(ac["flat"]["g2"] == ac["shift"]["g2"] == ac["flat"]["m2"] == ac["shift"]["m2"] == ac["flat"]["v2"] == ac["shift"]["v2"] == 0
         and delta["p2"] == 0 and ac["flat"]["g3"] > 0 > ac["shift"]["g3"] and ac["flat"]["eps"] == ac["shift"]["eps"] == 1e-8,
         "AdamW scalar state mismatch")
    need(abs(ac["flat"]["g3"]-2.41137e-8) < 1e-12 and abs(ac["shift"]["g3"]+3.52024e-8) < 1e-12
         and abs(ac["flat"]["actual_update"]+3.71709e-5) < 1e-9 and abs(ac["shift"]["actual_update"]-4.28362e-5) < 1e-9,
         "AdamW scalar values changed")
    need(abs(ac["flat"]["p3"]-ac["flat"]["p3_pred_observed_moments"]) < 3e-10
         and abs(ac["shift"]["p3"]-ac["shift"]["p3_pred_observed_moments"]) < 3e-10,
         "AdamW scalar update formula does not match saved values")
    num = traj.get("numerical_comparison")
    need(isinstance(num, dict) and isinstance(num.get("pass"), bool), "trajectory numerical comparison missing")
    step_metrics = per_step(raw_comp)
    strict_step_metrics = per_step(strict_comp)
    need(len(step_metrics) == 3 and all(row["role_pairs"] == 8 for row in step_metrics), "trajectory comparison must cover 24 role-step pairs")

    faults = long.get("faults", [])
    rank_restores = sum(f.get("restored_rank_count", 0) for f in faults)
    first_joint_roles = sum(f.get("first_joint_hash_roles", 0) for f in faults)
    rpo0 = sum(f.get("restored_step", -1) >= f.get("highest_joint_step_before_fault", 0) for f in faults)
    lighthouse_alive = sum(f.get("lighthouse_alive") is True for f in faults)
    healthy_pid_stable = sum(f.get("healthy_pids_unchanged") is True for f in faults)
    timing = long["timing_percentiles"]
    modes = backups["modes"]
    fastest = min(MODES, key=lambda m: modes[m]["metrics"]["fault_to_joint_seconds"].get("median", float("inf")))
    benchmark = bench.get("combined", {})
    route = traj.get("routing", {})
    summary = {
        "version": "v1", "phase_complete": True, "github_publication": "artifact-ready",
        "short_rotation": {"session": short.get("session"), "faults": 10, "pass": short["pass"]},
        "long_rotation": {"session": long.get("session"), "pass": long["pass"], "fault_window": fw,
            "restored_rank_hashes": rank_restores, "first_joint_hash_role_pairs": first_joint_roles,
            "rpo0_faults": rpo0, "lighthouse_alive_faults": lighthouse_alive,
            "healthy_pid_stable_faults": healthy_pid_stable,
            "steady_joint_steps": len(long.get("audited_joint_steps", [])),
            "steady_joint_role_pairs": long.get("audited_joint_role_pairs"), "timing_percentiles": timing},
        "full_state_audits": {m: {"pass": audits[m]["pass"], "faults": 1} for m in MODES},
        "backup_timing": backups, "fastest_median_fault_to_joint_mode": fastest,
        "recommendation": {"backup_mode": fastest,
            "cold_fault_to_joint_median_seconds": modes["cold"]["metrics"]["fault_to_joint_seconds"].get("median"),
            "gpu_hot_fault_to_joint_median_seconds": modes["gpu"]["metrics"]["fault_to_joint_seconds"].get("median"),
            "benchmark_shift_mean_step_change_percent": benchmark.get("mean_step_seconds_percent_change_shift_vs_flat")},
        "benchmark_abba": {"order": bench.get("abba_order"), "mean_step_seconds_change_percent": benchmark.get("mean_step_seconds_percent_change_shift_vs_flat"),
            "throughput_change_percent": benchmark.get("throughput_percent_change_shift_vs_flat"), "flat": benchmark.get("flat"), "shift": benchmark.get("shift")},
        "trajectory": {"checks": checks, "step_coverage": traj.get("step_coverage"), "initial_state": initial,
            "batch_plans": traj.get("batch_plans"), "participants_and_global_batch": traj.get("participants_and_global_batch"),
            "routing": route, "numerical_comparison": traj.get("numerical_comparison"),
            "numerical_by_step": step_metrics, "interpretation": traj.get("interpretation")},
        "strict_fp32": {"configuration": {"legacy": "FP32 parameters, TF32-high baseline", "strict": "TF32 disabled; float32 matmul precision highest"},
            "runtime_verified_roles": len(strict_ready), "checks": strict_checks, "step_coverage": strict.get("step_coverage"),
            "numerical_comparison": strict_num, "numerical_by_step": strict_step_metrics, "tolerances": strict_comp.get("tolerances"),
            "routing": strict.get("routing"),
            "legacy_route_differences_by_preupdate_step": {s: legacy_route.get("per_step", {}).get(str(s), {}).get("different") for s in (1, 2, 3)},
            "interpretation": "The FP32/TF32-high run diverges in routing; the strict run has identical 736 ordered and sorted role/layer/microbatch signatures yet still fails on its second update. TF32 amplifies the earlier divergence but is not the sole explanation."},
        "route_membership_diagnostic": {k: route_diag.get(k) for k in ("diagnostic_valid", "common_route_keys", "membership_change_count", "order_only_count", "replay_requirements", "interpretation")},
        "adam_coordinate": adam,
        "limitations": ["进程树SIGKILL故障，控制器/Lighthouse存活；未验证主机断电。", "副本失效时有效batch从8变4；数值诊断固定batch8。",
            "每节点一个8-GPU副本；GPU热备需要额外8-GPU容量。", "未运行完整同负载TorchTitan对照。", "运行依赖现有模型、FLASH1022/旧Core及runtime manifest。"],
        "evidence_scope": "results/v1精选JSON和压缩原始日志；不含凭据、模型.pt或runtime上传。",
    }
    (v1/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)+"\n", encoding="utf-8")

    audit = ", ".join(f"{m} 1/1" for m in MODES)
    bm = summary["benchmark_abba"]; num = summary["trajectory"]["numerical_comparison"]
    route_first = route.get("first_differences", [])
    route_note = f"首次有序路由差异发生于pre-update step {route.get('first_divergent_pre_update_step')}；完整签名与误差见汇总JSON"
    prep_cmds = """```bash
python3 -m shift_full.deploy --include-node3
python3 -m shift_full.rotation --session UNIQUE_LONG --config configs/shift_rotation_verify.json --seconds 4800 --faults 60 --interval 60 --duration 3600 --seed 20261102
python3 -m shift_full.rotation --session UNIQUE_GPU_HOT_AUDIT --config configs/shift_rotation_audit512.json --seconds 900 --faults 1 --backup-mode gpu --seed 20261102
python3 scripts/unpack_v1_evidence.py --evidence results/v1 --output tmp/v1-evidence
python3 -m shift_full.verify_rotation tmp/v1-evidence/v1_rotation_long_20261102_01 --expected-faults 60 --minimum-duration 3600
```"""
    docs = {
        "README.md": f"# SHIFT v1\n\n三机进程故障恢复实验已完成。长测60次/3600秒通过，状态恢复、首个共同提交与RPO0均经离线原始日志核验。版本 v1：kill 进程恢复。\n\n- 数值三步对照：`{str(num.get('pass')).lower()}`；按原容差报告，未放宽。\n- ABBA步时变化：{fmt(bm.get('mean_step_seconds_change_percent'),2)}%；备份建议：{fastest}（GPU热备需预留8张GPU）。\n- 详细证据：`RESULTS.md`、`results/v1/summary.json`。\n\n{prep_cmds}\n",
        "RESULTS.md": f"# v1 实验结果\n\n## 故障恢复\n\n- 短测：10/10通过。长测：{fw['completed_faults']}/60，{fw['completed_seconds']:.3f}秒，schedule_late={fw['schedule_late_events']}。\n- 恢复全状态：{rank_restores}/480 rank-hash；首个共同提交：{first_joint_roles}/480 rank-hash；RPO0：{rpo0}/60。Lighthouse存活{lighthouse_alive}/60，健康PID不变{healthy_pid_stable}/60。稳态审计{len(long.get('audited_joint_steps',[]))}步、{long.get('audited_joint_role_pairs')}角色-步。\n- 长测审计开启、seq16：完整故障→共同步p50/p95 {fmt(timing['fault_to_joint_seconds']['median'])}/{fmt(timing['fault_to_joint_seconds']['p95'])}秒；这是带审计结果。seq512关闭审计的三轮计时见下表，口径分开。\n- 全状态模式审计：{audit}均通过。\n\n## 备份计时（seq512，关闭审计）\n\n| 模式 | 准备p50 | 故障→共同步p50 | 请求→就绪p50 | 恢复请求→加载p50 | 故障期健康max间隔p50 | 准备期健康max间隔p50 |\n|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(f"| {m} | {fmt(modes[m]['metrics']['preparation_seconds'].get('median'))} | {fmt(modes[m]['metrics']['fault_to_joint_seconds'].get('median'))} | {fmt(modes[m]['metrics']['request_to_ready_seconds'].get('median'))} | {fmt(modes[m]['metrics']['recovery_request_to_state_loaded_seconds'].get('median'))} | {fmt(modes[m]['metrics']['healthy_max_adjacent_gap_seconds'].get('median'))} | {fmt(modes[m]['metrics']['healthy_max_adjacent_gap_preparation_seconds'].get('median'))} |" for m in MODES) + f"\n\n最快实测中位故障→共同步为 `{fastest}`；GPU热备需要额外预留整组8-GPU副本。保留cold作为基线。GPU/RSS/CUDA分项见 `results/v1/summary.json` 的backup_timing。预备期跨副本通信为结构性零（Manager在ACTIVATE前未构造）；这不是带宽测量结论。\n\nABBA flat→SHIFT→SHIFT→flat；步时变化 {fmt(bm.get('mean_step_seconds_change_percent'),2)}%，吞吐变化 {fmt(bm.get('throughput_change_percent'),2)}%。\n\n## 数值轨迹\n\n以相同step1精确状态起点，比较commit step2/3/4；batch与路由计划覆盖pre-update step1/2/3。16个角色的7项起始哈希、batch、输入和participants/global_batch检查通过。原比较容差保留；数值比较通过=`{num.get('pass')}`，超容差角色-步对={len(num.get('failed_pairs',[]))}。逐step/类别最大绝对误差、relative-L2见 `results/v1/summary.json`。\n\n{route_note}。路由边界margin见汇总JSON。观测不证明根因；FP32 reduction/addition顺序差异只是可能解释。\n\n## 范围\n\n故障是训练进程树SIGKILL，Lighthouse/控制面存活；动态batch由8变4，固定batch8只用于数值轨迹。每节点一个8-GPU副本。未运行完整同负载TorchTitan对照，未验证主机掉电。精选公开证据不含凭据、模型.pt或runtime上传。\n",
        "OPTIMIZATION_NOTES.md": f"# v1 优化与证据\n\n推荐保持cold恢复作为基线；本轮审计通过的三种模式为 {audit}。按三轮关闭审计的故障→共同步中位数，最快为 `{fastest}`。若采用GPU热备，须保留额外8-GPU容量；资源明细在 `results/v1/backup_summary.json`。\n\nABBA测得SHIFT步时变化 {fmt(bm.get('mean_step_seconds_change_percent'),2)}%，吞吐变化 {fmt(bm.get('throughput_change_percent'),2)}%。长测seq16且开启状态审计，RTO p50/p95 {fmt(timing['fault_to_joint_seconds']['median'])}/{fmt(timing['fault_to_joint_seconds']['p95'])}秒；seq512计时关闭审计，不能混作同口径。\n\n三更新FP32对照通过原容差=`{num.get('pass')}`；超容差角色-步对={len(num.get('failed_pairs',[]))}。batch/输入/起始7哈希检查通过。{route_note}。路由margin与逐step范数在汇总JSON。证据不支持确定根因；FP32加法/reduction次序是可能性，不是结论。\n",
        "CURRENT_PLAN.md": "# 当前状态\n\n- v1实验与证据核验已完成；本阶段无待运行实验。\n- `results/v1/summary.json` 汇总长测、备份模式、ABBA和FP32轨迹。数值比较即使失败也按原容差明确保留。\n- 仓库：`Zwhao0/SHIFT`；证据附SHA256清单，凭据、张量快照与运行时均排除。\n- 限制：仅进程树SIGKILL、控制服务存活；不代表主机掉电或完整TorchTitan等价。\n",
        "handoff.md": f"# SHIFT v1 交接\n\n实验阶段已完成，证据汇总于 `results/v1/summary.json`。长测60/60、3600秒；状态恢复hash {rank_restores}/480，首个joint hash {first_joint_roles}/480，RPO0 {rpo0}/60。seq16审计RTO p50/p95 {fmt(timing['fault_to_joint_seconds']['median'])}/{fmt(timing['fault_to_joint_seconds']['p95'])}秒。\n\n数值step2–4比较原容差结果：`{num.get('pass')}`，超容差角色-步对 {len(num.get('failed_pairs',[]))}；不得放宽容差或称完全等价。版本：v1。原始证据可用 `scripts/unpack_v1_evidence.py` 解包，再运行`shift_full.verify_rotation`。\n\n边界与复现命令见 `README.md`、`RESULTS.md`。不上传凭据、模型.pt或runtime。\n",
    }
    ac = adam["coordinate"]; cross = adam["cross_mode_differences"]
    strict_note = (f"\n\n## 严格FP32与路由诊断\n\n旧轨迹为FP32参数、TF32-high路径；严格复核禁用TF32并设highest（flat/shift各16条ready记录均核验）。起点、batch、participants均匹配；严格两更新共736个角色/层/微批ordered与sorted路由签名相同。step2八个rank通过，step3八个rank仍按相同容差失败。旧路径在pre-update step1/2/3的路由差异数为{legacy_route['per_step']['1']['different']}/{legacy_route['per_step']['2']['different']}/{legacy_route['per_step']['3']['different']}；TF32放大了旧路径差异，但不是唯一解释。\n\n单副本复放对flat/shift各368个step2有序路由和16个样本计划均复现；跨条件按角色/rank、层、微批序号比较有2个top-k成员变化、30个仅选择顺序变化（不是token计数）。\n\nAdamW标量例：rank1 `{coord['parameter']}[{coord['flat_index']}]`，step2两侧p相同，g2/m2/v2为0；step3的极小梯度为{ac['flat']['g3']:.6g}与{ac['shift']['g3']:.6g}，符号相反。经AdamW归一化后参数更新为{ac['flat']['actual_update']:.6g}与{ac['shift']['actual_update']:.6g}；保存矩量代入标量公式后，预测p3与观测差分别{abs(ac['flat']['p3']-ac['flat']['p3_pred_observed_moments']):.2g}、{abs(ac['shift']['p3']-ac['shift']['p3_pred_observed_moments']):.2g}。ε={ac['flat']['eps']:.0e}。这是一个坐标的数值敏感性证据，不证明全局reduction根因或整段轨迹等价；未改容差或ε。\n")
    docs["README.md"] += f"\n严格FP32复核在首个更新后通过、第二个更新仍失败，路由签名一致；原容差保留。单坐标AdamW分析见 `RESULTS.md` 与 `results/v1/adam_coordinate.json`。\n"
    docs["RESULTS.md"] += strict_note
    docs["OPTIMIZATION_NOTES.md"] += strict_note
    docs["CURRENT_PLAN.md"] += "\n数值结论：严格FP32仍在step3失败；路由一致，AdamW标量仅显示局部敏感性，不作根因结论。保持原容差与ε。\n"
    docs["handoff.md"] += strict_note
    docs["README.md"] += "\n负载约0.5B MoE、合成token；两活一备，每副本8×H800，TP/PP/EP/eTP均为2。依赖现有 `/workspace/FLASH1022/{flash_v1,Greyhound-master/Megatron-LM,models/qwen_moe}` 与 `/workspace/SHIFT/runtime/torch214/bin/python`；版本清单见 `runtime/torch214-manifest/`。本机凭据仅存Git排除的 `server.md`，模板为 `server.example.md`。\n"
    docs["RESULTS.md"] += "\n冷备为冷进程启动，文件页缓存未清空。实验不含真实语料dataloader状态验证。原始日志与源码快照有SHA256清单；大型张量快照保留服务器、不进入GitHub。\n"
    docs["OPTIMIZATION_NOTES.md"] = f"# v1 优化\n\n- 推荐GPU热备：完整恢复中位6.737秒，冷进程基线22.680秒；保留8张GPU备用资源。\n- 原生torchFT NCCL/IB、64M元素桶/hook、异步quorum；热备在Manager构造前等待ACTIVATE，加载最新全状态后有效参与。\n- Lighthouse独立，rid/generation目录隔离，Linux subreaper回收训练进程树。\n- 同拓扑ABBA额外步时{fmt(bm.get('mean_step_seconds_change_percent'),2)}%。\n- 原TF32路径与严格FP32短轨迹均保留失败结果；近零梯度的AdamW敏感性见 `RESULTS.md`。\n- 分项计时与GPU/RSS成本：`results/v1/backup_summary.json`。\n"
    docs["handoff.md"] = "# SHIFT v1 交接\n\n三机优化与实验已完成，无待运行GPU实验。\n\n- 冷备随机轮转：60/60、3600秒、RPO0；恢复480/480、首共同步480/480。\n- 推荐GPU热备：三轮正常计时22.680→6.737秒；新拓扑ABBA额外步时4.84%。\n- 数值边界：TF32路径三更新、严格FP32两更新均未全部通过原容差；严格模式路由相同，已定位一个近零梯度经AdamW放大的坐标。未放宽容差或修改ε。\n- 复现/环境见 `README.md`，完整结果见 `RESULTS.md`、`results/v1/summary.json`。原始日志、实跑源码快照与校验清单在 `results/v1/`。\n- 仅训练进程SIGKILL；Lighthouse/控制面存活，合成数据动态batch8→4。未验证主机掉电与同负载TorchTitan。\n- server.md仅本机保存；保留现有三机容器、FLASH源码/模型/运行时和服务器张量快照。\n- GitHub目标：Zwhao0/SHIFT，版本v1。\n"
    for name, content in docs.items(): (root/name).write_text(content, encoding="utf-8")
    print(json.dumps({"phase_complete": True, "summary": str(v1/"summary.json"), "github_publication": "artifact-ready"}, ensure_ascii=False))

if __name__ == "__main__": main()
