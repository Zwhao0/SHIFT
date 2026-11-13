# FLASH1022 companion files

These files apply to the FLASH v4 R97 public bundle in the FLASH1022 checkout at `work/final_r97/public_bundle`. The SHIFT driver also uses that checkout's FLASH v1 training and Megatron adapters. See [integration and evidence](../../docs/flash_shift.md).

Apply `observer.patch` with `patch -p1` from the root of a FLASH checkout containing the R97 observer. Copy `deploy/lab` into that same checkout. The unpatched `flash_v4/native/observer.cpp` SHA-256 is `1347eefa157672c1581bd5848217b7cd94e4fccbf7c9bb2844df91a61ec282f9`. The SHIFT Lab run used the v1 adapter with SHA-256 `c1da9ff1db8628290bdfdf372a43c5043d3f93ee04c7a6be3b4d94a5c4e3d76e` from the archived v3 native preflight inputs.

Build the observer against the Lab runtime's CUDA and NCCL headers and libraries. The SHIFT runner expects `runtime/flash-libs/libobserver.so` and, for simulated faults, `runtime/flash-libs/libgemm-delay.so`. Build the Lab observer with `-DFLASH_LAB_GEMM_DELAY` and link the delay library against the compiled `flash_v4/native/stream_delay.cu` kernel. The strong SHIFT trial used `-DFLASH_LAB_WAIT_NS=20000000ULL -DFLASH_LAB_MAX_LAUNCHES=64`. The standalone FLASH trial used `2500000ULL` and `72`, respectively.

```bash
# Set SHIFT_ROOT, FLASH_ROOT and CUDA_TOOLKIT to the staged source and toolkit directories.
site="$SHIFT_ROOT/runtime/torch214/lib/python3.12/site-packages/nvidia"
libs="$SHIFT_ROOT/runtime/flash-libs"
mkdir -p "$libs"
g++ -DFLASH_LAB_GEMM_DELAY -std=c++17 -O2 -fPIC -shared -pthread \
  -I"$CUDA_TOOLKIT/include" -I"$site/nccl/include" "$FLASH_ROOT/flash_v4/native/observer.cpp" \
  -L"$site/cu13/lib" -l:libcudart.so.13 -ldl -o "$libs/libobserver.so"
"$CUDA_TOOLKIT/bin/nvcc" -std=c++17 -O2 -Xcompiler=-fPIC -arch=sm_90 \
  -c "$FLASH_ROOT/flash_v4/native/stream_delay.cu" -o "$libs/delay.o"
g++ -DFLASH_LAB_WAIT_NS=20000000ULL -DFLASH_LAB_MAX_LAUNCHES=64 -std=c++17 -O2 -fPIC -shared \
  -I"$CUDA_TOOLKIT/include" "$FLASH_ROOT/deploy/lab/v4_gemm_delay.cpp" "$libs/delay.o" \
  -L"$site/cu13/lib" -l:libcudart.so.13 -o "$libs/libgemm-delay.so"
```

The standalone job generator takes the session-owned Lab directory via `--stage` and the rendezvous address via `--master-addr`. Use `--split-peer` for the 8+4+4 runner layout. Stage the launcher as `v4_rank_entry.py` next to the FLASH Python packages and build the CUDA libraries there. The model and runtime are staged separately.
