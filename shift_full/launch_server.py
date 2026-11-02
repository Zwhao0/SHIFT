"""Persistent CPU preload service that forks fresh torchrun agents."""

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


_MAX_REQUEST = 1024 * 1024


class LaunchProcess:
    """Small Popen-like handle backed by the server's atomic exit record."""

    def __init__(self, pid, status_path, generation):
        self.pid = int(pid)
        self.status_path = Path(status_path)
        self.generation = int(generation)
        self.returncode = None

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        try:
            record = json.loads(self.status_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        if int(record.get("pid", -1)) != self.pid:
            raise RuntimeError(f"status PID mismatch in {self.status_path}")
        self.returncode = int(record["returncode"])
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            result = self.poll()
            if result is not None:
                return result
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(f"generation {self.generation}", timeout)
                time.sleep(min(0.05, remaining))
            else:
                time.sleep(0.05)

    def send_signal(self, sig):
        if self.poll() is None:
            os.kill(self.pid, sig)

    def terminate(self):
        self.send_signal(signal.SIGTERM)

    def kill(self):
        self.send_signal(signal.SIGKILL)


def launch(socket_path, args, log, generation):
    """Request a launch and return a small process-compatible handle."""
    request = {"args": list(args), "log": os.fspath(log), "generation": int(generation)}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(15)
        client.connect(os.fspath(socket_path))
        client.sendall(json.dumps(request, separators=(",", ":")).encode() + b"\n")
        response = bytearray()
        while b"\n" not in response:
            chunk = client.recv(4096)
            if not chunk:
                raise RuntimeError("launch server closed without a response")
            response.extend(chunk)
            if len(response) > _MAX_REQUEST:
                raise RuntimeError("launch response exceeded size limit")
    packet = json.loads(bytes(response).split(b"\n", 1)[0])
    if "error" in packet:
        raise RuntimeError(f"launch server: {packet['error']}")
    info = packet.get("launch")
    if not isinstance(info, dict):
        raise RuntimeError(f"invalid launch response: {packet!r}")
    return LaunchProcess(info["pid"], info["status_path"], request["generation"])


def _reply(conn, packet):
    conn.sendall(json.dumps(packet, separators=(",", ":")).encode() + b"\n")


def _read_request(conn):
    data = bytearray()
    while b"\n" not in data:
        chunk = conn.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > _MAX_REQUEST:
            raise ValueError("request exceeded size limit")
    if not data:
        raise ValueError("empty request")
    return json.loads(bytes(data).split(b"\n", 1)[0])


def _write_exit(status_path, pid, generation, returncode):
    record = {"pid": pid, "generation": generation, "returncode": returncode}
    temp_path = status_path.with_name(status_path.name + f".{pid}.tmp")
    temp_path.write_text(json.dumps(record, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temp_path, status_path)


def _child_agent(conn, listener, args, log_path):
    try:
        conn.close()
        listener.close()
        fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        if fd > 2:
            os.close(fd)
        sys.argv = ["torchrun", *args]
        from torch.distributed.run import main as torchrun_main
        torchrun_main()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    except BaseException:
        traceback.print_exc()
        try:
            sys.stderr.flush()
        finally:
            os._exit(1)


def _handle_launch(conn, listener, socket_path, active, used_generations, fast_launch):
    try:
        request = _read_request(conn)
        if not isinstance(request, dict):
            raise ValueError("request must be a JSON object")
        args = request.get("args")
        log = request.get("log")
        generation = request.get("generation")
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            raise ValueError("args must be a list of strings")
        if not isinstance(log, str) or not log:
            raise ValueError("log must be a non-empty path")
        if not isinstance(generation, int) or isinstance(generation, bool) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if active:
            raise RuntimeError("an agent is already active")
        if generation in used_generations:
            raise RuntimeError(f"generation {generation} has already been used")

        status_path = socket_path.parent / f"exit_g{generation}.json"
        if status_path.exists():
            raise RuntimeError(f"generation {generation} already has an exit status")
        log_path = Path(log).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab"):
            pass

        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError("server CUDA initialized before agent fork")
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            _child_agent(conn, listener, args, str(log_path))
        active[pid] = generation
        used_generations.add(generation)
        _reply(conn, {"launch": {"pid": pid, "status_path": str(status_path)}})
    except Exception as exc:
        try:
            _reply(conn, {"error": f"{type(exc).__name__}: {exc}"})
        except OSError:
            pass


def serve(socket_path):
    socket_path = Path(socket_path).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    from shift_full import fast_launch
    fast_launch.preload()
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError("preload initialized CUDA; refusing to start launch server")

    used_generations = set()
    for status in socket_path.parent.glob("exit_g*.json"):
        try:
            used_generations.add(int(json.loads(status.read_text()).get("generation")))
        except (OSError, ValueError, TypeError):
            continue

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(os.fspath(socket_path))
    os.chmod(socket_path, 0o600)
    listener.listen(16)
    listener.settimeout(0.1)
    active = {}
    try:
        while True:
            for pid, generation in list(active.items()):
                reaped, status = os.waitpid(pid, os.WNOHANG)
                if reaped:
                    _write_exit(socket_path.parent / f"exit_g{generation}.json",
                                pid, generation, os.waitstatus_to_exitcode(status))
                    del active[pid]
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(10)
                _handle_launch(conn, listener, socket_path, active, used_generations, fast_launch)
    finally:
        listener.close()
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True, help="Unix socket path")
    args = parser.parse_args(argv)
    serve(args.socket)


if __name__ == "__main__":
    main()
