# v3 复现入口与外部依赖

此版本保留研究源码与必要分析/验证入口，GPU结果来自已记录的既有环境；发布验证只做CPU/编译/引用检查，不重跑GPU矩阵。不能从少量公开摘要重算未公开的大张量证据。

## 环境

Linux x86_64、Python3.12.3；每DP副本8×H800，三机轮换；Torch2.14.1+cu130、TorchFT0.2.0、NCCL2.30.7、Transformers4.53.2。锁文件位于`runtime/torch214-manifest/`，pip freeze里的Torch本地wheel URL在公开版改为版本约束；原wheel SHA保存在依赖JSON。不要在现有服务器删除环境再重建。

训练需要`/workspace/FLASH1022/flash_v1/{megatron_adapter,pipeline_adapter,tensor_parallel,train}.py`及其`Greyhound-master/Megatron-LM`；这是外部保留依赖，不随本发布重新许可或打包。没有可核验的完整Git HEAD，已记录实际文件SHA，见`evidence/v3/dependencies.json`。需要具有这些依赖的现有环境才能完整运行；仅安装pip包不足。模型`bayliner1980/Qwen2.5-0.5B-MoE-A0.3B`固定revision与文件SHA也在该JSON；模型不随Git发布。TorchFT/TorchTitan参考版本、上游归属见同文件；本项目兼容补丁位于`shift_full/compat.py`，不修改安装版源码。

## 配置与启动

公开默认IP是RFC5737文档地址。先在本机和三台服务器配置`SHIFT_NODE0_IP/SHIFT_NODE1_IP/SHIFT_NODE3_IP`、相应`SHIFT_NODE*_HOST_ROOT`（主机侧SHIFT目录）；真实训练配置的lighthouse/master地址与`--host-ip/--lighthouse/--flash`必须一致。现有实验路径约定`/workspace/SHIFT`与`/workspace/FLASH1022`；需要相应挂载/符号链接。不要把示例IP直接用于实验。认证使用SSH或本地`server.md`模板，禁止上传凭据。

`prepare_v3_controller`恢复有限来源专用控制身份，同步源码SHA与一次性凭据，不建立新环境；其后每次实验由独立systemd托管，单owner，不能并发训练。服务器控制器还需要同样的真实端点环境（systemd环境须显式配置）；默认端点不推断现场地址。以下均从仓库根目录，suffix必须唯一：

```sh
python3 -m scripts.prepare_v3_controller --smoke
python3 -m scripts.start_v3_benchmark --suffix UNIQUE_PROBE --schedule probe --digest-engine staged
python3 -m scripts.start_v3_benchmark --suffix UNIQUE_TRIAD --schedule triad --digest-engine staged
python3 -m scripts.start_v3_cold --suffix UNIQUE_COLD --digest-engine staged --seed 20261106 --overlap-cpu-model-load
python3 -m scripts.start_v3_planned --suffix UNIQUE_SLOW --scenario slow-compute --slow-ms 100 --digest-engine staged --baseline PRIVATE_BASELINE.json
```

每次独立controller之前重新prepare一次，等待前次完全退出；真实kill仅在cold模块的受控SHIFT树内执行。`--slow-ms 100`仅展示参数入口，不代表已发表样本的固定强度；应从保留的原配置复现相应样本。planned基线必须先在现场记录公司GPU PID；`scripts.inspect_v3_resources --baseline ... --output ...`用于检查，不能以PID保持替代业务健康。

生产/研究候选默认off；staged摘要必须显式选择。完整schema2检查入口`shift_full.verify_v3_recovery`；原raw重算使用`reanalyze_v3_benchmark.py`、`summarize_v3_cold_path.py`、`summarize_v3_cold_steps.py`、`summarize_v3_planned_path.py`等。各脚本`--help`给出输入路径，不附带现场私有raw。

## CPU与源码身份

```sh
python3 -m compileall -q shift_full shift_min scripts
python3 -m unittest scripts.test_planned_state scripts.test_v3_recovery
# 在已有torch214环境另跑不初始化CUDA的完整CPU套件（清理证据中记录结果）
```

`evidence/v3/source_manifest.json`区分公开source SHA与实验冻结source provenance文件SHA。发布只调整部署默认地址和公开打包清单，没有重新证明所有GPU行为逐字相同。公开包不含私有AGENTS/handoff、调度IDs、server.md、socket、模型、wheel、虚拟环境或大日志；旧版本仍可在旧Git提交查阅。
