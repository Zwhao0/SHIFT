"""Run an unmodified FLASH pipeline under the fixed runtime adapter.

The adapter is source-version-bound and intentionally changes data and update
semantics. See runtime_patch.py and the per-rank patch manifest before using it.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from shift_full.runtime_patch import RuntimePatch, StopTraining, sha256_file  # noqa: E402

if os.environ.get("SHIFT_BABY_COMPAT") == "1":
    from shift_full.compat import install_baby_event_fix, install_managed_callback_fix

    install_baby_event_fix()
    install_managed_callback_fix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    parser.add_argument("--replica", type=int, required=True, choices=(0, 1))
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--host-ip", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--lighthouse", required=True)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--script", help="original FLASH train_pipeline.py entrypoint")
    parser.add_argument("--backend", choices=("blocking", "baby", "hybrid"), default="hybrid")
    args = parser.parse_args()
    if args.generation < 0 or args.target_steps < 1 or args.max_seconds < 1:
        parser.error("generation must be nonnegative; target steps and max seconds must be positive")
    if not args.host_ip or not args.lighthouse:
        parser.error("host-ip and lighthouse must be non-empty")
    if not args.session or not all(char.isalnum() or char in "_-" for char in args.session):
        parser.error("session may contain only letters, digits, '_' and '-'")

    env_keys = (
        "FLASH_PARALLEL_LAYOUT", "TORCHFT_USE_OTEL", "PYTHONPATH",
        "PYTHONUNBUFFERED", "TORCH_NCCL_USE_COMM_NONBLOCKING", "SHIFT_BABY_COMPAT",
    )
    env_before = {key: os.environ.get(key) for key in env_keys}
    path_before = list(sys.path)
    argv_before = sys.argv
    runtime = None
    graceful = False
    try:
        root = Path(args.root).resolve()
        flash = Path(os.environ.get("FLASH", "/workspace/FLASH1022")).resolve()
        script = Path(args.script).resolve() if args.script else flash / "flash_v1" / "train_pipeline.py"
        os.environ["FLASH_PARALLEL_LAYOUT"] = "legacy"
        os.environ["TORCHFT_USE_OTEL"] = "false"
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "0"
        module_paths = [str(root), str(flash)]
        if os.environ.get("SHIFT_ISOLATED_RUNTIME") != "1":
            module_paths.insert(0, str(root / "vendor"))
        os.environ["PYTHONPATH"] = ":".join(module_paths)
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
        sys.path[:0] = module_paths
        if args.backend == "baby":
            os.environ["SHIFT_BABY_COMPAT"] = "1"
            from shift_full.compat import install_baby_event_fix, install_managed_callback_fix

            install_baby_event_fix()
            install_managed_callback_fix()
        else:
            os.environ.pop("SHIFT_BABY_COMPAT", None)

        session_dir = root / "results" / args.session
        session_dir.mkdir(parents=True, exist_ok=True)
        output = session_dir / f"native_g{args.generation}"
        output.mkdir(parents=True, exist_ok=True)
        runtime = RuntimePatch(
            script=script,
            root=root,
            session=args.session,
            replica=args.replica,
            generation=args.generation,
            host_ip=args.host_ip,
            lighthouse=args.lighthouse,
            target_steps=args.target_steps,
            max_seconds=args.max_seconds,
            backend=args.backend,
        )
        try:
            runtime.install()
            sys.argv = [
                str(script),
                "--tensor-parallel-size", "2",
            "--model", str(flash / "models" / "qwen_moe"),
                "--steps", "1000000000",
                "--warmup", "0",
                "--seq-len", "16",
                "--batch-size", "1",
                "--microbatches", "2",
                "--output", str(output),
                "--compute-delay-ms", "0",
            ]
            runpy.run_path(str(script), run_name="__main__")
            runtime.verify_guard()
            runtime.mark_stopped("source_returned")
            graceful = True
        except StopTraining as stop:
            runtime.verify_guard()
            runtime.mark_stopped(str(stop))
            graceful = not str(stop).startswith("timeout")
        except BaseException as error:
            runtime.mark_error(error)
            raise
        finally:
            try:
                after = sha256_file(script)
                runtime.write_patch_manifest(after)
            finally:
                runtime.close(graceful=graceful)
    finally:
        sys.argv = argv_before
        sys.path[:] = path_before
        for key, value in env_before.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == "__main__":
    main()
