// Bounded Lab-only simulation. Delay the target worker's existing GEMM stream.
#include <atomic>
#include <cuda_runtime_api.h>

extern "C" cudaError_t flash_v4_delay_launch_wait(cudaStream_t, unsigned long long);

#ifndef FLASH_LAB_WAIT_NS
#define FLASH_LAB_WAIT_NS 750000ULL
#endif
#ifndef FLASH_LAB_MAX_LAUNCHES
#define FLASH_LAB_MAX_LAUNCHES 256
#endif

namespace {
std::atomic<bool> enabled{false};
std::atomic<int> launches{0};
constexpr int max_launches = FLASH_LAB_MAX_LAUNCHES;
constexpr unsigned long long wait_ns = FLASH_LAB_WAIT_NS;
static_assert(wait_ns > 0 && max_launches > 0 && wait_ns * max_launches <= 1300000000);
}

extern "C" int flash_lab_gemm_delay(int active) {
    enabled.store(active != 0);
    return launches.exchange(0);
}

extern "C" cudaError_t flash_lab_before_gemm(cudaStream_t stream) {
    int reserved = launches.load();
    while (enabled.load() && reserved < max_launches) {
        if (launches.compare_exchange_weak(reserved, reserved + 1))
            return flash_v4_delay_launch_wait(stream, wait_ns);
    }
    return cudaSuccess;
}
