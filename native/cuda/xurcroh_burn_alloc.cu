#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdint>
#include <sys/types.h>

static std::atomic<uint64_t> g_scrubbed_bytes{0};
static std::atomic<uint64_t> g_scrub_calls{0};

__global__ void xurcroh_shared_scrub_kernel(uint32_t* sink) {
    extern __shared__ volatile uint32_t shared[];
    volatile uint32_t local[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) local[i] = 0;
    for (int index = threadIdx.x; index < (48 * 1024) / sizeof(uint32_t); index += blockDim.x) {
        shared[index] = local[index & 31];
    }
    __syncthreads();
    __threadfence_system();
    if (sink != nullptr && threadIdx.x == 0 && shared[0] == 0xFFFFFFFFU) {
        sink[blockIdx.x] = shared[0];
    }
}

__global__ void xurcroh_leftover_probe_kernel(uint32_t* output, int words) {
    extern __shared__ volatile uint32_t shared[];
    for (int index = threadIdx.x; index < words; index += blockDim.x) {
        output[blockIdx.x * words + index] = shared[index];
    }
}

__global__ void xurcroh_uniform_touch_kernel(const uint8_t* pages, size_t count, size_t stride, volatile uint32_t* sink) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const volatile uint8_t* page = pages + index * stride;
    uint32_t value = 0;
    for (int offset = 0; offset < 128; ++offset) value ^= page[offset];
    if (sink != nullptr && value == 0xFFFFFFFFU) sink[index] = value;
}

extern "C" {
void* xurcroh_burn_malloc(ssize_t size, int device, cudaStream_t stream) {
    if (size <= 0) return nullptr;
    if (cudaSetDevice(device) != cudaSuccess) return nullptr;
    void* pointer = nullptr;
    if (cudaMalloc(&pointer, static_cast<size_t>(size)) != cudaSuccess || pointer == nullptr) {
        return nullptr;
    }
    if (cudaMemsetAsync(pointer, 0, static_cast<size_t>(size), stream) != cudaSuccess) {
        cudaFree(pointer);
        return nullptr;
    }
    return pointer;
}

void xurcroh_burn_free(void* pointer, ssize_t size, int device, cudaStream_t stream) {
    if (pointer == nullptr) return;
    cudaSetDevice(device);
    if (size > 0 && cudaMemsetAsync(pointer, 0, static_cast<size_t>(size), stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess) {
        g_scrubbed_bytes.fetch_add(static_cast<uint64_t>(size), std::memory_order_relaxed);
        g_scrub_calls.fetch_add(1, std::memory_order_relaxed);
    }
    cudaFree(pointer);
}

uint64_t xurcroh_get_scrubbed_bytes() {
    return g_scrubbed_bytes.load(std::memory_order_relaxed);
}

uint64_t xurcroh_get_scrub_calls() {
    return g_scrub_calls.load(std::memory_order_relaxed);
}

int xurcroh_launch_shared_scrub(int num_sms, cudaStream_t stream) {
    if (num_sms <= 0) return static_cast<int>(cudaErrorInvalidValue);
    xurcroh_shared_scrub_kernel<<<num_sms, 256, 48 * 1024, stream>>>(nullptr);
    return static_cast<int>(cudaGetLastError());
}

int xurcroh_run_leftover_probe(uint32_t* output, int num_sms, cudaStream_t stream) {
    if (output == nullptr || num_sms <= 0) return static_cast<int>(cudaErrorInvalidValue);
    xurcroh_leftover_probe_kernel<<<num_sms, 256, 48 * 1024, stream>>>(output, (48 * 1024) / sizeof(uint32_t));
    return static_cast<int>(cudaGetLastError());
}

int xurcroh_uniform_page_touch(const void* base, size_t count, size_t stride, cudaStream_t stream) {
    if (base == nullptr || count == 0 || stride < 128) return static_cast<int>(cudaErrorInvalidValue);
    xurcroh_uniform_touch_kernel<<<count, 1, 0, stream>>>(
        static_cast<const uint8_t*>(base), count, stride, nullptr);
    return static_cast<int>(cudaGetLastError());
}
}
