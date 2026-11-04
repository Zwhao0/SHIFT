# Three quick functional tests and timing (2026-11-04)

Six independent functional audits passed: two V4 intra-replica replacements, two V1 real-kill whole-DP cold recoveries, and two V2 P1f B-to-C whole-DP migrations. Seed manifest: `quick_random_manifest.json` (1127942288). Full derived samples: `evidence/v4/*_timing.json`; summary: `quick_validation.json`.

V4: DP2, each replica TP2/PP2/EP2 with 8 ranks, 16 training GPUs plus a 4-GPU spare role. Both replicas share physical hosts. Local layout places PP stages across node0/node1; cross layout also spans TP across hosts. V1/V2 put each 8-rank DP replica on one node and reserve the third node. No independent host failure-domain claim.

All durations below are seconds. Step time is the maximum worker duration at each step, averaged over the window. The pre-request window excludes fixed startup steps1-2; all samples remain saved. Post-recovery averages cover only the next4 or5 steps and do not establish steady state.

|Trial|Pre-request step|First recovered step|Following steps|Full observed time|
|---|---:|---:|---:|---:|
|v4_q4_local|2.496|9.674|2.469|28.516|
|v4_q4_cross|2.912|23.122|11.556|34.522|
|v1_quick_20261104_q1_0_r1|0.818|6.119|0.875|25.088|
|v1_quick_20261104_q1_1|0.877|5.763|0.850|24.111|
|v2_q2_q2_0|5.247|12.584|5.222|55.525|
|v2_q2_q2_1|4.549|8.725|10.669|52.282|

Full time uses node0 controller monotonic: request (kill marker for V1) to observing all16 ranks at their first recovered common commit. This includes polling, SSH and log collection latency. Cross-host clocks are never subtracted. Local worker phase durations may overlap and their extrema cannot be added as controller phases. Hashing/diagnostic costs are included; only two trials per type, with different batch/topology/hash/standby conditions. No speedup or significance claim.

V4 requests/cuts: local6/11, cross4/7. V1 placements: node3 to node1, then node1 to node0; healthy replica can commit solo during cold recovery. V2 requests/cuts:9/18 and10/20, both node1(B) to node3(C), fixed random batch8 and hash every step.

V4 cross-TP and the second V2 run show higher post-recovery step times. Cause remains unlocalized; slow samples are retained for optimization. V4 local A0 first-common-step proxy is24.406s versus28.516s for all-rank observation; the4.110s difference includes collection/polling and is not pure switching compute.

Controller stage intervals and worker import/group/model/snapshot/transfer/load/hash durations are preserved in the summary and timing artifacts. V1 first joint has all-nine-state matching hashes and five validated post-recovery commits. V2 validates all-nine-state B-to-C restoration, shared model/optimizer/scheduler across A/C, epoch/membership, contributions and global sample coverage.

Preliminary failures are explicit in quick_failures.json: missing private GPU baseline before training, then a legacy documentation Lighthouse endpoint during initialization (no kill). The first successful V1 GPU run had a post-processing Path-expression error; it passed a fresh independent offline audit on the same collected evidence after that expression was fixed. Its original status remains saved.

V2 q2_0 and q2_1 have minor startup-wrapper differences: the latter explicitly sets the CUDA device and microbatches=2 (the earlier default was also2). Training/P1f sources are unchanged. Per-run deployed SHA manifests and q2_0_runner_delta.patch preserve provenance. Do not attribute trial timing differences to the migration algorithm.

Reuse existing private endpoints/environment. Run scripts.prepare_v3_controller before each trial, then scripts.start_v4 or scripts.start_quick_recovery. Never prepare while the training lock is held. Workers are memory-capped at0.12 and owned by exact bounded systemd units; company containers are untouched. After audit/export, delete regenerable raw/log/archive/trace/intermediate data and temporary authorization; retain source, installed environment/headers/model, private configs/seeds/manifests, initial BF16 anchors and comparison signatures.

## V1 timing follow-up (same day)

The retained v1 profile explicitly selects `--cold-launcher --digest-engine staged --overlap-cpu-model-load` through `scripts.start_quick_recovery`. It retains fresh cold workers, the original native forward/backward and full TP/PP gradient finalization, complete nine-state audits, the2s communication deadline and0.12 memory cap.

Same-seed normal-load baseline→candidate:25.116→21.264s controller-observed recovery (exploratory15.34%, one matched pair). A second placement passed at20.141s; its following15 steps averaged0.822s versus0.810s pre-fault, with last10 averaging0.810s. All initial-role nine-state hashes match the baseline. The bounded healthy commit gap remains about9.4–9.5s.

Fresh v2 and PP-only v4 regressions passed independent audits:31.685s and27.016s, respectively. Their post windows were2.005s vs1.962s pre, and2.446s vs2.578s pre. These are functional regressions under their declared workloads, not cross-strategy speedup evidence. V1/v2/v4 currently differ in entrypoint, input data, global batch, physical placement and auditing; the user's intended comparison requires a common training engine/workload with only recovery decisions varying.

Additional v1 attempts were rejected: incomplete gradient paths failed guards/exact model-Adam hashes; a complete native gradient-only round passed correctness but took25.069s; overlapping cold launch with retired-generation cleanup passed but changed21.264s to21.239s, within observation uncertainty. All were reverted. Selected source identity was restored exactly; the user canceled redundant reruns. High-load baseline failures and the39.034s/30s-deadline diagnostic are retained without performance credit.

See `evidence/v1_optimization_20261104/` for all timings, audits, initial signatures, source identities, failed attempts and cleanup. Original public V4 remains commit7c8fe1e; follow-up changes remain local. Regenerable raw was removed after export; current reference verification requires fresh raw on a future run.
