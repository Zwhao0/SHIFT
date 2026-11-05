# v4 运行验证与开发审计

基于 v4.1 的统一 batch8 事务路径，训练、数据、Adam和数值规约不变。

- 默认开发模式（`--boundary-validation full`）：完整九状态hash、独立梯度/轨迹审计；可用 `--digest-engine staged`，SHA语义不变。
- `--performance-no-audit`：默认 `--boundary-validation runtime`，关闭计时窗口内的开发hash/逐步参考。候选在完整结构/版本检查、严格恢复、CUDA完成后报告readiness；epoch、owner、四rank确认、全局提交屏障、超时回退和actor排空仍有效。
- 实验结束后，所有16逻辑worker先完成共同窗口，才开始完整九状态终态审计。`--final-reference-run SUFFIX`额外与新建独立参考的同rank/step比对，不在训练窗口读取参考状态或逐步hash。
- `--performance-no-audit --boundary-validation full`保留边界hash，用于成本对照；不能称为零计时hash路径。

运行模式对schema、逻辑rank、epoch/cut/owner、完整模型/Adam/buffer/RNG结构、Manager游标和实际恢复完成进行检查，不提供任意静默字节篡改的在线hash检测。静默篡改检查属于开发审计。缺字段、错误结构/版本、恢复异常不会取得接管资格；通信中途故障仍为既有fail-stop范围。

每轮使用现有prepare入口，单训练锁、GPU串行和0.12显存上限不变。不得在训练期间prepare。参考要求源码及训练profile匹配；终态参考只允许开发审计开关不同，模型/批次/样本/seed/布局及来源SHA检查仍保留。

```sh
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix ref --reference-only --outer-audit --cut 16 --post-steps 16 --digest-engine staged
# 完成并独立核验参考；下一轮prepare后启动
python3 -m scripts.prepare_v3_controller
python3 -m scripts.start_v4 --suffix move --candidate-request-step 10 --cut 13 --post-steps 5 --slow-ms 250 --performance-no-audit --final-reference-run ref
```

参考必须覆盖实际cut和后续步数。实验入口与报告工具不会将retain计为迁移成功；正式耗时包含冷候选准备，不通过提前热备用移动成本。分项计时只在同一worker/宿主内相减，跨rank重叠阶段不加总。

本轮结果、原版对照、所有接受/拒绝案例和限制保存在 `evidence/v4_recovery_20261105/summary.json`；复现计划在私有 `runtime/v3-reproduce/v42_recovery_manifest.json`。`scripts.run_v42_suite`逐例串行运行并独立审计；任一失败停止后续队列。

新的单会话主循环、统一命令和动态策略入口见 `docs/v4_2_unified_session.md`。上述 `start_v4` 保留为兼容和独立审计入口；正式单会话入口为 `scripts.sessionctl`。
