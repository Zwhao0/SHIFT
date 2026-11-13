"""Emit two bounded Lab Jobs for the frozen FLASH v4 16-rank layouts."""
import argparse
import json
import re


RUN = r'''
set -euo pipefail
cd "$FLASH_LAB_ROOT"
export PYTHONDONTWRITEBYTECODE=1 FLASH_MEGATRON_CORE_ONLY=1 FLASH_V4_KERNEL_PROFILE=1
export HOME=/tmp/flash-home XDG_CACHE_HOME=/tmp/flash-home/cache
mkdir -p "$XDG_CACHE_HOME"
export PYTHONPATH="$PWD:$PWD/pylib"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export FLASH_V4_ALLOCATOR_BUDGET=available
export FLASH_V4_MONITOR_CONTROL=r92_store_v1 FLASH_V4_MONITOR_DEGRADATION=0.1
export FLASH_V4_SAMPLE_STRIDE=64 NCCL_SOCKET_IFNAME=bond0 GLOO_SOCKET_IFNAME=bond0
export NCCL_NET=IB NCCL_IB_DISABLE=0 NCCL_IB_HCA='=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1'
export NCCL_CUMEM_ENABLE=0 NCCL_CUMEM_HOST_ENABLE=0 NCCL_NVLS_ENABLE=0
export NCCL_MAX_NCHANNELS=4 NCCL_BUFFSIZE=1048576 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
site="$PWD/runtime/torch214/lib/python3.12/site-packages/nvidia"
export LD_LIBRARY_PATH="$site/cu13/lib:$site/nccl/lib"
export FLASH_V4_CUBLAS_LIBRARY="$site/cu13/lib/libcublas.so.13"
export FLASH_V4_CUBLASLT_LIBRARY="$site/cu13/lib/libcublasLt.so.13"
export FLASH_V4_NCCL_LIBRARY="$site/nccl/lib/libnccl.so.2"
export FLASH_V4_DELAY_CUDART_LIBRARY="$site/cu13/lib/libcudart.so.13"
digest=$(printf "%s" "$1" | sha256sum)
export FLASH_V4_MONITOR_RUN_ID="v4_${digest:0:32}"
rank=$2; layout=$3; mode=$4; port=$5; steps=$6; length=$7; slots=$8; rank_base=$9; split=${10}
if [[ "$split" == yes ]]; then
  export NCCL_SHM_DISABLE=1 NCCL_P2P_DISABLE=1
fi
out="$PWD/runs/$1/node$rank"
mkdir -p "$out"
trap 'chmod -R g+rX "$out"' EXIT
export FLASH_V4_OUTPUT_DIR="$out/observations"
export FLASH_V4_ONLINE_LOG_DIR="$out/control"
export FLASH_V4_ACTUATION_DIR_PRIVATE="$out/actuation"
mkdir -p "$FLASH_V4_OUTPUT_DIR" "$FLASH_V4_ACTUATION_DIR_PRIVATE"
chmod 700 "$FLASH_V4_ACTUATION_DIR_PRIVATE"
if [[ "$mode" == delay ]]; then
  mode=online
  export FLASH_V4_STREAM_DELAY_PRIVATE='{"12":6}'
  export FLASH_V4_ACTOR_WINDOW_PRIVATE='{"first":36,"last":67}'
  export LD_PRELOAD="$PWD/libobserver.so:$PWD/libdelay.so"
elif [[ "$mode" == gemm-delay ]]; then
  mode=online
  export FLASH_LAB_GEMM_DELAY_RANK=12
  export LD_PRELOAD="$PWD/libobserver.so:$PWD/libgemm-delay.so"
elif [[ "$mode" != off ]]; then
  export LD_PRELOAD="$PWD/libobserver.so"
fi
export FLASH_V4_OBSERVE_MODE="$mode"
runtime/torch214/bin/python -c 'import sys, torch; assert torch.cuda.device_count()==int(sys.argv[1])' "$slots"
nvidia-smi --query-gpu=uuid,memory.free,utilization.gpu,clocks.sm --format=csv > "$out/devices-before.csv"
train=(--layout "$layout" --steps "$steps" --warmup 3 --seq-len "$length"
       --model "$PWD/models/qwen_moe" --output "$out/training")
if [[ "$split" == yes || -v FLASH_LAB_GEMM_DELAY_RANK ]]; then
  time timeout --kill-after=10s 180s runtime/torch214/bin/python v4_rank_entry.py --rank-base "$rank_base" --slots "$slots" --master-port "$port" --master-addr "$FLASH_LAB_MASTER_ADDR" "${train[@]}"
else
  time timeout --kill-after=10s 180s runtime/torch214/bin/python -m torch.distributed.run --nnodes 2 --nproc-per-node 8 --node-rank "$rank" --master-addr "$FLASH_LAB_MASTER_ADDR" --master-port "$port" --max-restarts 0 -m flash_v4.train_entry "${train[@]}"
fi
'''


def jobs(name, layout, mode, port, steps, seq_len, peer, split_peer, stage, master_addr):
    if not re.fullmatch(r'flash-[a-z0-9-]{1,38}', name):
        raise ValueError('use a unique flash-* Job name')
    if not 1024 <= port <= 32767 or not 8 <= steps <= 120 or not 16 <= seq_len <= 1024:
        raise ValueError('invalid bounded Lab case')
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,54}', stage):
        raise ValueError('expected a session-owned /data/lab staging directory')
    items = []
    placements = [('gpu-01', 0, 8), (peer, 8, 8)]
    if split_peer:
        placements = [('gpu-01', 0, 8), (peer, 8, 4), (peer, 12, 4)]
    for rank, (node, rank_base, slots) in enumerate(placements):
        mounts = [dict(name='lab-data', mountPath=f'/data/lab/{stage}', subPath=stage),
                  dict(name='shm', mountPath='/dev/shm')]
        labels = {'experiment': name, 'flash-node': str(rank)}
        container = dict(name='flash',
                         image='registry.reinfra.cn/reinfra/infrastructure/images/vllm-openai:v0.32.0',
                         command=['bash', '-ec'], args=[RUN, 'case', name, str(rank), layout,
                                                        mode, str(port), str(steps), str(seq_len), str(slots),
                                                        str(rank_base), "yes" if split_peer else "no"],
                         env=[{'name': 'FLASH_LAB_ROOT', 'value': f'/data/lab/{stage}'},
                              {'name': 'FLASH_LAB_MASTER_ADDR', 'value': master_addr}],
                         resources={'requests': {'cpu': '8', 'memory': '12Gi'},
                                    'limits': {'memory': '40Gi', 'nvidia.com/gpu': slots,
                                               'rdma/rdma_shared_device_a': 1}},
                         securityContext={'capabilities': {'add': ['IPC_LOCK', 'SYS_RESOURCE']}},
                         volumeMounts=mounts)
        pod = dict(restartPolicy='Never', serviceAccountName='lab', automountServiceAccountToken=False,
                   hostNetwork=True, dnsPolicy='ClusterFirstWithHostNet',
                   nodeSelector={'kubernetes.io/hostname': node},
                   securityContext={'runAsNonRoot': True, 'runAsUser': 1001, 'runAsGroup': 1001},
                   containers=[container], volumes=[
                       {'name': 'lab-data', 'hostPath': {'path': '/data/lab', 'type': 'Directory'}},
                       {'name': 'shm', 'emptyDir': {'medium': 'Memory', 'sizeLimit': '2Gi'}}])
        items.append(dict(apiVersion='batch/v1', kind='Job',
                          metadata=dict(name=f'{name}-{rank}', namespace='lab', labels=labels),
                          spec=dict(activeDeadlineSeconds=230, backoffLimit=0,
                                    ttlSecondsAfterFinished=900,
                                    template=dict(metadata=dict(labels=labels), spec=pod))))
    return dict(apiVersion='v1', kind='List', items=items)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('layout', choices=('legacy', 'node_local_pp'))
    parser.add_argument('mode', choices=('off', 'online', 'profile', 'delay', 'gemm-delay'))
    parser.add_argument('port', type=int)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--master-addr', required=True)
    parser.add_argument('--split-peer', action='store_true', help='place two four-GPU runners on the peer host')
    parser.add_argument('--peer', choices=('gpu-02', 'gpu-03'), default='gpu-03')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--seq-len', type=int, default=256)
    print(json.dumps(jobs(**vars(parser.parse_args()))))
