# V4 GPU state handoff (2026-11-04)

`v4_train.py` uses `PackedGPUTransport` on four independent B→C NCCL pairs.
Model, Adam and buffer snapshots stay on GPU. Small CPU RNG/state tensors and
tree metadata use the bounded control payload. Full-state hashing still stages
tensors through CPU for audit; this is not an audit-free GPU-only execution path.
Snapshots are independent clones; Adam restore clones again to prevent future
updates from modifying the snapshot. A retains its model/optimizer/DDP objects,
parameter objects and parameter storage. The baseline rebuilds all training groups.

Audited runs are in `evidence/v4/summary.json`, including exact source hashes:

| Run | Reference replay | GPU bytes / control bytes | Result |
| --- | --- | --- | --- |
| gpu01 | 10 steps | 6,124,856,320 / 3,299,914 | 80 rank-step comparisons of nine states and final full tensors: zero differences |
| gpu02 | disabled | 3,062,428,160 / 1,631,542 | complete restore, fencing, sample cursor and 10 further steps passed |

gpu01 sends both the initial snapshot and diagnostic final reference; gpu02
sends only the initial snapshot. CPU tensor payloads total 62,848 / 31,424 bytes.
Both runs preserve four healthy worker PIDs and retire four source workers.

gpu02 requests replacement at step 1, injects a one-second source slow step,
and selects step 4 when standby is ready. Boundary→activation is 8.819 s;
boundary→first commit is 11.144 s. B0 pack/send is 0.144 s. These are single-run
diagnostics with full hashing, not a controlled performance or speedup claim.
Per-worker phase durations are retained in the compact summary; overlapping
durations across workers must not be added together.

Reproduce using the existing environment:

```sh
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix NEW --gpu-prepare --gpu-warmup
# Prepare the controller again before another run (credentials are consumed).
# Add --no-oracle --cut 1 --slow-ms 1000 for gpu02's diagnostic mode.
```

Still unverified: outer TorchFT Manager/DP isolation, the old v3 outer deadline
failure, cross-host TP migration, repeated replacements, cancellation/rollback,
and hung-rank recovery. Extra standby/snapshot memory is required. Next work:
reduce exposed group setup/audit costs, then reconnect the outer Manager.

The optional `--reuse-local-groups --preconnect-state` path preserves host-local
groups across default-world teardown. It pins SHA256 of the installed c10d,
Megatron parallel-state and FLASH adapter sources; a mismatch fails closed.
Only the existing four-devices-per-stage layout is supported. No installed
runtime file is edited. A keeps 13 groups per rank; C reuses 12 and closes four
standby-only groups. Default and cross-host groups are rebuilt. B→C channels
are prepared on a separate CUDA stream before the update boundary.

`groups01` passes another 80 rank-step comparisons of nine states and final
full-tensor comparison, with zero differences. Registry lifecycle tests cover
preservation through WORLD shutdown, remapping standby ranks, unused-group
closure, pending-coalescing rejection, and restoring the constructor on exceptions.
These options remain disabled by default; outer TorchFT integration is pending.

Adding `--defer-source-local-close` preserves B's 12 drained local groups until
A reports activation. B remains fenced before sending state and cannot train;
all source-local groups are then closed before B reports retirement. This option
requires `--reuse-local-groups`. `groups03` passes another 80 rank-step oracle
checks and final full-tensor comparison. All 160 new oracle cases pass.

Matched diagnostic runs use cut 8, ten further steps, seq512, the same seed and
a one-second injected slow step, without reference replay:

| Path | Boundary→activation | Boundary→first commit |
| --- | --- | --- |
| Baseline, groupsbase01 | 9.044 s | 11.406 s |
| Local reuse + preconnect, groups02 | 9.249 s | 11.018 s |
| Also defer B local close, groups04 | 7.798 s | 9.693 s |
| Repeat, groups05 | 7.825 s | 9.642 s |

All four runs have identical aggregate nine-state signatures across all eight
logical ranks, both at the boundary and at the end. The first optimization alone
does not demonstrate an end-to-end improvement. Deferring source-local teardown
removes a remaining receive-side wait in these samples. One baseline and two
final-path samples, with uncontrolled shared-host load and full audit hashing,
do not establish stable performance. Raw evidence was summarized and deleted;
see `evidence/v4/group_setup_comparison.json` and `groups_cleanup.json`.

```sh
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix NEW --gpu-prepare --gpu-warmup \
  --reuse-local-groups --preconnect-state --defer-source-local-close
# Add --no-oracle --cut 8 --slow-ms 1000 for the matched diagnostic mode.
```
