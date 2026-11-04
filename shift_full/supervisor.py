"""Run and recover one replica's training launcher for a bounded session."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess as sp
import sys
import time


# Use the same process-tree ownership and teardown code as shift_min.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shift_min.supervisor import descendants, terminate  # noqa: E402


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config must be a JSON object")
    mode = config.get("mode")
    if mode not in {"shift", "flat"}:
        raise ValueError("config.mode must be 'shift' or 'flat'")
    train_args = config.get("train_args", [])
    if not isinstance(train_args, list) or not all(isinstance(arg, str) for arg in train_args):
        raise ValueError("config.train_args must be a list of CLI strings")
    target_steps = config.get("target_steps")
    if not isinstance(target_steps, int) or isinstance(target_steps, bool) or target_steps < 1:
        raise ValueError("config.target_steps must be a positive integer")
    lighthouse_port = config.get("lighthouse_port", 29771)
    master_port = config.get("master_port", 29860)
    for name, port in (("lighthouse_port", lighthouse_port), ("master_port", master_port)):
        if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
            raise ValueError(f"config.{name} must be a valid TCP port")
    lighthouse_ip = config.get("lighthouse_ip", "192.0.2.10")
    if not isinstance(lighthouse_ip, str) or not lighthouse_ip:
        raise ValueError("config.lighthouse_ip must be a non-empty string")
    external_cpu_launcher_socket = config.get("external_cpu_launcher_socket")
    if (external_cpu_launcher_socket is not None
            and (not isinstance(external_cpu_launcher_socket, str)
                 or not external_cpu_launcher_socket.strip())):
        raise ValueError("config.external_cpu_launcher_socket must be a non-empty path string")
    cpu_launcher_generation_offset = config.get("cpu_launcher_generation_offset", 0)
    if (not isinstance(cpu_launcher_generation_offset, int)
            or isinstance(cpu_launcher_generation_offset, bool)
            or cpu_launcher_generation_offset < 0):
        raise ValueError("config.cpu_launcher_generation_offset must be a non-negative integer")
    persistent_cpu_launcher = config.get("persistent_cpu_launcher", False)
    cold_launcher = config.get("cold_launcher", False)
    if cold_launcher and (mode != 'shift' or config.get('fast_launch') or persistent_cpu_launcher
                          or external_cpu_launcher_socket or config.get('single_gpu_visibility')):
        raise ValueError('cold_launcher requires SHIFT fresh workers without preload or visibility remapping')
    if external_cpu_launcher_socket is not None and not persistent_cpu_launcher:
        raise ValueError("external_cpu_launcher_socket requires persistent_cpu_launcher")
    return {
        "mode": mode,
        "entry_module": config.get("entry_module", "shift_full.train"),
        "nccl_library": config.get("nccl_library"),
        "python": config.get("python", sys.executable),
        "train_args": train_args,
        "target_steps": target_steps,
        "lighthouse_port": lighthouse_port,
        "master_port": master_port,
        "master_addr": config.get("master_addr", "192.0.2.10"),
        "lighthouse_ip": lighthouse_ip,
        "env": config.get("env", {}),
        "fast_launch": config.get("fast_launch", False),
        "cold_launcher": cold_launcher,
        "persistent_cpu_launcher": persistent_cpu_launcher,
        "external_cpu_launcher_socket": external_cpu_launcher_socket,
        "cpu_launcher_generation_offset": cpu_launcher_generation_offset,
        "heartbeat_timeout_ms": config.get("heartbeat_timeout_ms", 1000),
        "join_timeout_ms": config.get("join_timeout_ms", 1000),
        "quorum_tick_ms": config.get("quorum_tick_ms", 100),
        "recycle_after_worker_fault": config.get("recycle_after_worker_fault", False),
        "single_gpu_visibility": config.get("single_gpu_visibility", False),
        "external_lighthouse": config.get("external_lighthouse", False),
        "min_replicas": config.get("min_replicas", 1),
    }


def session_name(value):
    if not value or not all(char.isalnum() or char in "_-" for char in value):
        raise argparse.ArgumentTypeError("session may contain only letters, digits, '_' and '-'")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True, type=session_name)
    parser.add_argument("--replica", required=True, type=int, choices=(0, 1))
    parser.add_argument("--physical-node", default=None)
    parser.add_argument("--initial-generation", type=int, default=0)
    parser.add_argument("--result-subdir", default=None)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--seconds", required=True, type=int)
    parser.add_argument("--config", required=True, help="JSON training configuration")
    args = parser.parse_args()
    if args.seconds < 1:
        parser.error("--seconds must be positive")
    if not args.host_ip:
        parser.error("--host-ip must not be empty")
    if args.initial_generation < 0:
        parser.error("--initial-generation must be nonnegative")
    args.physical_node = args.physical_node or f"node{args.replica}"

    config = load_config(args.config)
    from shift_full.process_tree import enable_subreaper, reap_orphans
    enable_subreaper()
    root = Path(args.root).resolve()
    if config["external_cpu_launcher_socket"]:
        socket_path = Path(config["external_cpu_launcher_socket"]).expanduser()
        if not socket_path.is_absolute():
            socket_path = root / socket_path
        config["external_cpu_launcher_socket"] = str(socket_path.resolve())
    out = root / "results" / args.session
    if args.result_subdir:
        if not all(c.isalnum() or c in '_-' for c in args.result_subdir):
            parser.error('invalid result subdirectory')
        out = out / args.result_subdir
    # Each replica runs on its own node-local result tree; refuse accidental reuse.
    out.mkdir(parents=True, exist_ok=False)
    supervisor_log = (out / "supervisor.jsonl").open("a", buffering=1, encoding="utf-8")
    controller_log = (out / "controller.jsonl").open("a", buffering=1, encoding="utf-8")
    log_handles = []

    generation = args.initial_generation
    training = None
    lighthouse = None
    cpu_launcher = None
    completed = False
    faulted_generations = set()
    started = time.monotonic()
    deadline = started + args.seconds

    def row(kind, **fields):
        record = dict(kind=kind, time=time.time(), rid=args.replica, rank=-1,
                      generation=generation, physical_node=args.physical_node,
                      host_ip=args.host_ip, pid=os.getpid())
        record.update(fields)
        return record

    def event(kind, **fields):
        supervisor_log.write(json.dumps(row(kind, **fields), sort_keys=True) + "\n")

    def controller_event(kind, **fields):
        controller_log.write(json.dumps(row(kind, **fields), sort_keys=True) + "\n")

    env = os.environ.copy()
    env['SHIFT_RESULT_DIR'] = str(out)
    flash = env.get("FLASH", "/workspace/FLASH1022")
    sources = list((root / "shift_full").glob("*.py")) + list((Path(flash) / "flash_v1").glob("*.py"))
    provenance = {"config": config, "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    env.update(
        PYTHONPATH=(f"{root}:{flash}" if config["python"] != sys.executable else f"{root}/vendor:{root}:{flash}"),
        SHIFT_ISOLATED_RUNTIME=str(int(config["python"] != sys.executable)),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS="1",
        MALLOC_ARENA_MAX="2",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
        NCCL_MAX_CTAS="4",
        NCCL_LAUNCH_ORDER_IMPLICIT="0",
        NCCL_RAS_ADDR="127.0.0.1:29859",
        NCCL_BUFFSIZE="1048576",
        NCCL_CUMEM_HOST_ENABLE="0",
        NCCL_CUMEM_ENABLE="0",
        NCCL_NVLS_ENABLE="0",
        NCCL_DEBUG="WARN",
        NCCL_NET="Socket",
        NCCL_IB_DISABLE="1",
        NCCL_SOCKET_IFNAME="bond0",
        NCCL_SOCKET_RETRY_CNT="4",
        NCCL_SOCKET_RETRY_SLEEP_MSEC="50",
        GLOO_SOCKET_IFNAME="bond0",
        TORCH_NCCL_ASYNC_ERROR_HANDLING="0",
        TORCH_NCCL_USE_COMM_NONBLOCKING="0",
        TORCHFT_USE_OTEL="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False",
        TORCH_NCCL_NONBLOCKING_TIMEOUT="10",
        NO_PROXY="*",
        no_proxy="*",
        RUST_LOG="warn",
        TOKIO_WORKER_THREADS="2",
    )
    env.pop("LD_PRELOAD", None)
    overrides = config["env"]
    if not isinstance(overrides, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in overrides.items()):
        raise ValueError("config.env must map strings to strings")
    env.update(overrides)
    if config['fast_launch']:
        env['SHIFT_ENTRY_MODULE'] = config['entry_module']
    if config['single_gpu_visibility']:
        if not config['fast_launch']: raise ValueError('single GPU visibility requires fast_launch')
        env['SHIFT_SINGLE_GPU_VISIBLE']='1'
    if config["nccl_library"]:
        library = Path(config["nccl_library"])
        if not library.is_file():
            raise FileNotFoundError(library)
        env["LD_PRELOAD"] = str(library)

    def start_lighthouse():
        nonlocal lighthouse
        stream = (out / "lighthouse.log").open("w", encoding="utf-8")
        log_handles.append(stream)
        cmd = [
            config["python"], str(root / "vendor/bin/torchft_lighthouse"),
            "--bind", f"[::]:{config['lighthouse_port']}",
            "--min_replicas", str(config['min_replicas']),
            "--heartbeat_timeout_ms", str(config['heartbeat_timeout_ms']),
            "--join_timeout_ms", str(config['join_timeout_ms']),
            "--quorum_tick_ms", str(config['quorum_tick_ms']),
        ]
        lighthouse = sp.Popen(
            cmd, env=env, cwd=root, stdout=stream, stderr=sp.STDOUT,
            start_new_session=True,
        )
        event("lighthouse_started", lighthouse_pid=lighthouse.pid, command=cmd)
        time.sleep(0.5)
        if lighthouse.poll() is not None:
            raise RuntimeError("Lighthouse failed to start")

    def launch():
        port = config["master_port"] + generation
        if port > 65535:
            raise ValueError("master_port + generation exceeds 65535")
        stream = (out / f"torchrun_g{generation}.log").open("w", encoding="utf-8")
        log_handles.append(stream)
        launcher = ('shift_full.cold_launch' if config['cold_launcher'] else
                    'shift_full.fast_launch' if config['fast_launch'] else 'torch.distributed.run')
        cmd = [config["python"], "-m", launcher]
        entry = (["--start-method=fork", "--run-path", str(root / 'shift_full/worker_entry.py')]
                 if config['fast_launch'] else ["--module", config['entry_module']])
        if config["mode"] == "shift":
            cmd += [
                "--nnodes=1", "--nproc_per_node=8", "--max_restarts=0",
                f"--master_addr={args.host_ip}", f"--master_port={port}",
                *entry,
                "--session", args.session,
                "--replica", str(args.replica),
                "--generation", str(generation),
                "--host-ip", args.host_ip,
                "--root", str(root),
                "--target-steps", str(config["target_steps"]),
                "--lighthouse", f"http://{config['lighthouse_ip']}:{config['lighthouse_port']}",
            ]
        else:
            cmd += [
                "--nnodes=2", "--nproc_per_node=8", f"--node_rank={args.replica}",
                "--max_restarts=0", f"--master_addr={config['master_addr']}", f"--master_port={port}",
                *entry,
                "--session", args.session,
                "--replica", str(args.replica),
                "--generation", str(generation),
                "--host-ip", args.host_ip,
                "--root", str(root),
                "--target-steps", str(config["target_steps"]),
            ]
        cmd += config["train_args"]
        process_started_time = time.time()
        if config['persistent_cpu_launcher']:
            from shift_full.launch_server import launch as cached_launch
            external_socket = config["external_cpu_launcher_socket"]
            launcher_socket = (Path(external_socket).expanduser()
                               if external_socket else out/'cpu_launcher.sock')
            launcher_generation = (config["cpu_launcher_generation_offset"] + generation
                                   if external_socket else generation)
            proc = cached_launch(launcher_socket, cmd[3:], out/f'torchrun_g{generation}.log',
                                 launcher_generation, env=env)
            process_started_time = getattr(proc, 'started_time', None) or process_started_time
        else:
            process_started_time = time.time()
            proc = sp.Popen(
                cmd, env=env, cwd=root, stdout=stream, stderr=sp.STDOUT,
                start_new_session=True,
            )
        event("launched", generation=generation, launcher_pid=proc.pid,
              launcher_generation=(launcher_generation if config['persistent_cpu_launcher'] else generation),
              external_cpu_launcher=bool(config["external_cpu_launcher_socket"]),
              process_started_time=process_started_time, master_port=port, command=cmd)
        return proc

    def local_rank_ready_pid(rank, current_generation):
        path = out / f"events_g{current_generation}_r{rank}.jsonl"
        if not path.is_file():
            return None
        pid = None
        try:
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if (record.get("kind") == "ready"
                            and record.get("generation") == current_generation
                            and record.get("rank") == rank
                            and record.get("rid") == args.replica):
                        candidate = record.get("pid")
                        if isinstance(candidate, int) and candidate > 1:
                            pid = candidate
        except OSError:
            return None
        return pid

    def all_local_ranks_stopped(current_generation):
        for rank in range(8):
            path = out / f"events_g{current_generation}_r{rank}.jsonl"
            stopped = False
            if not path.is_file():
                return False
            try:
                with path.open(encoding="utf-8") as stream:
                    for line in stream:
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if (record.get("kind") == "stopped"
                                and record.get("generation") == current_generation
                                and record.get("rank") == rank
                                and record.get("rid") == args.replica):
                            stopped = True
            except OSError:
                return False
            if not stopped:
                return False
        return True

    def inject_launcher_kill():
        if training is None or training.poll() is not None:
            return False
        now = time.time()
        victims = terminate(training, signal.SIGKILL)
        training.wait(timeout=15)
        reaped = reap_orphans(exclude_pids=[p.pid for p in (lighthouse, cpu_launcher) if p is not None])
        fields = dict(generation=generation, signal="SIGKILL", fault_kind="launcher_kill",
                      pids=victims, count=len(victims), time=now, reaped=reaped)
        event("injected", **fields)
        controller_event("injected", **fields)
        faulted_generations.add(generation)
        return True

    def inject_worker_kill(rank):
        if training is None or training.poll() is not None:
            return False
        if not 0 <= rank < 8:
            raise ValueError(f"WORKER_KILL rank must be in [0, 7], got {rank}")
        pid = local_rank_ready_pid(rank, generation)
        if pid is None:
            raise RuntimeError(f"no ready pid for generation={generation} localrank={rank}")
        owned = descendants(training.pid)
        if pid not in owned:
            raise RuntimeError(f"ready pid {pid} is not a descendant of launcher {training.pid}")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            raise RuntimeError(f"ready worker pid {pid} exited before SIGKILL")
        now = time.time()
        fields = dict(generation=generation, rank=rank, signal="SIGKILL",
                      fault_kind="worker_kill", localrank=rank,
                      pids=[pid], count=1, time=now)
        event("injected", **fields)
        controller_event("injected", **fields)
        faulted_generations.add(generation)
        if config['recycle_after_worker_fault']:
            victims=terminate(training,signal.SIGKILL)
            training.wait(timeout=15)
            event('failed_replica_recycled',pids=victims,count=len(victims),cause='confirmed_worker_failure')
        return True

    def stop_and_wait(proc):
        if proc is None:
            return None
        if proc.poll() is None:
            terminate(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=4)
            except sp.TimeoutExpired:
                terminate(proc, signal.SIGKILL)
                proc.wait(timeout=10)
        return proc.poll()

    try:
        if generation > 0:
            fields = dict(generation=generation, previous_generation=generation - 1,
                          endpoint_takeover=True)
            controller_event("restart_requested", **fields)
            event("restart_requested", **fields)
        if config["mode"] == "shift" and args.replica == 0 and not config['external_lighthouse']:
            start_lighthouse()
        if config['persistent_cpu_launcher']:
            if not config['fast_launch']:
                raise ValueError('persistent CPU launcher requires fast_launch')
            external_socket = config["external_cpu_launcher_socket"]
            if external_socket:
                socket_path = Path(external_socket).expanduser()
                socket_deadline = min(deadline, time.monotonic() + 45)
                while not socket_path.is_socket():
                    if time.monotonic() >= socket_deadline:
                        raise RuntimeError(f'external CPU launcher socket is unavailable: {socket_path}')
                    time.sleep(.05)
                event('cpu_launcher_external_ready', socket=str(socket_path),
                      generation_offset=config["cpu_launcher_generation_offset"])
            else:
                stream=(out/'cpu_launcher.log').open('w',encoding='utf-8');log_handles.append(stream)
                preload_started=time.monotonic()
                cpu_launcher=sp.Popen([config['python'],'-m','shift_full.launch_server','--socket',str(out/'cpu_launcher.sock')],
                                      env=env,cwd=root,stdout=stream,stderr=sp.STDOUT,start_new_session=True)
                while not (out/'cpu_launcher.sock').exists():
                    if cpu_launcher.poll() is not None or time.monotonic()-preload_started>45:
                        raise RuntimeError('CPU preload service failed to start')
                    time.sleep(.05)
                rss=[line for line in Path(f'/proc/{cpu_launcher.pid}/status').read_text().splitlines() if line.startswith('VmRSS:')]
                event('cpu_launcher_ready',server_pid=cpu_launcher.pid,preload_seconds=time.monotonic()-preload_started,cpu_rss=rss)
        training = launch()
        event("supervisor_ready", mode=config["mode"], target_steps=config["target_steps"])

        while True:
            if (out / "STOP").exists():
                event("stopping", requested=True)
                break
            if time.monotonic() >= deadline:
                event("timeout", seconds=args.seconds)
                raise TimeoutError(f"session exceeded --seconds={args.seconds}")

            if training is not None and training.poll() is None and generation not in faulted_generations:
                kill_marker = out / f"KILL_g{generation}"
                worker_marker = out / f"WORKER_KILL_g{generation}"
                if kill_marker.is_file():
                    inject_launcher_kill()
                elif worker_marker.is_file():
                    raw_rank = worker_marker.read_text(encoding="utf-8").strip()
                    try:
                        worker_rank = int(raw_rank)
                    except ValueError as exc:
                        raise ValueError(f"{worker_marker.name} must contain a localrank integer") from exc
                    inject_worker_kill(worker_rank)

            if training is not None and training.poll() is not None:
                returncode = training.returncode
                if all_local_ranks_stopped(generation):
                    completed = True
                    event("completed", generation=generation, returncode=returncode)
                    # Keep Lighthouse alive until STOP so the experiment controller
                    # can collect all node-local evidence before teardown.
                    while not (out / "STOP").exists():
                        if time.monotonic() >= deadline:
                            event("timeout", seconds=args.seconds, completed=True)
                            raise TimeoutError(f"completed trainer waited past --seconds={args.seconds}")
                        if lighthouse is not None and lighthouse.poll() is not None:
                            raise RuntimeError("Lighthouse exited before STOP")
                        time.sleep(0.1)
                    event("stopping", requested=True, completed=True)
                    break
                if generation in faulted_generations:
                    next_generation = generation + 1
                    restart_marker = out / f"RESTART_g{next_generation}"
                    if restart_marker.is_file():
                        generation = next_generation
                        controller_event("restart_requested", generation=generation,
                                         previous_generation=generation - 1)
                        event("restart_requested", generation=generation,
                              previous_generation=generation - 1)
                        training = launch()
                    else:
                        if lighthouse is not None and lighthouse.poll() is not None:
                            raise RuntimeError("Lighthouse exited while awaiting restart")
                        time.sleep(0.1)
                        continue
                elif (out / f"EXPECT_PEER_FAILURE_g{generation}").is_file():
                    faulted_generations.add(generation)
                    event("peer_fault_exit", generation=generation, returncode=returncode)
                    continue
                else:
                    event("unexpected_exit", generation=generation, returncode=returncode)
                    raise RuntimeError(f"trainer exited unexpectedly with code {returncode}")

            if lighthouse is not None and lighthouse.poll() is not None:
                raise RuntimeError("Lighthouse exited unexpectedly")
            if cpu_launcher is not None and cpu_launcher.poll() is not None:
                raise RuntimeError('CPU preload service exited unexpectedly')
            time.sleep(0.1)
    except BaseException as error:
        event("supervisor_error", error=repr(error))
        raise
    finally:
        (out / "STOP").touch(exist_ok=True)
        if training is not None and training.poll() is None:
            try:
                training.wait(timeout=30)
            except sp.TimeoutExpired:
                pass
        trainer_returncode = stop_and_wait(training)
        cpu_launcher_returncode = stop_and_wait(cpu_launcher)
        lighthouse_returncode = stop_and_wait(lighthouse)
        reaped = reap_orphans()
        event("cleanup", completed=completed, trainer_returncode=trainer_returncode,
              lighthouse_returncode=lighthouse_returncode,
              cpu_launcher_returncode=cpu_launcher_returncode,
              reaped=reaped,
              own_children=descendants(os.getpid()))
        for stream in log_handles:
            stream.close()
        controller_log.close()
        supervisor_log.close()


if __name__ == "__main__":
    main()
