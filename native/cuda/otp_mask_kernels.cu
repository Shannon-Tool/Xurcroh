#include <cuda_runtime_api.h>
#include <cstdint>

__device__ inline uint32_t rotate_left(uint32_t value, unsigned shift) {
    return (value << shift) | (value >> (32 - shift));
}

__device__ inline void quarter_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d) {
    a += b; d ^= a; d = rotate_left(d, 16);
    c += d; b ^= c; b = rotate_left(b, 12);
    a += b; d ^= a; d = rotate_left(d, 8);
    c += d; b ^= c; b = rotate_left(b, 7);
}

__device__ inline uint32_t load32(const uint8_t* input) {
    return static_cast<uint32_t>(input[0]) |
           (static_cast<uint32_t>(input[1]) << 8) |
           (static_cast<uint32_t>(input[2]) << 16) |
           (static_cast<uint32_t>(input[3]) << 24);
}

__device__ void chacha20_block(const uint32_t* key, uint32_t counter, const uint8_t* nonce, uint32_t* output) {
    uint32_t state[16] = {
        0x61707865U, 0x3320646eU, 0x79622d32U, 0x6b206574U,
        key[0], key[1], key[2], key[3], key[4], key[5], key[6], key[7],
        counter, load32(nonce), load32(nonce + 4), load32(nonce + 8)
    };
    uint32_t working[16];
    for (int i = 0; i < 16; ++i) working[i] = state[i];
    for (int round = 0; round < 10; ++round) {
        quarter_round(working[0], working[4], working[8], working[12]);
        quarter_round(working[1], working[5], working[9], working[13]);
        quarter_round(working[2], working[6], working[10], working[14]);
        quarter_round(working[3], working[7], working[11], working[15]);
        quarter_round(working[0], working[5], working[10], working[15]);
        quarter_round(working[1], working[6], working[11], working[12]);
        quarter_round(working[2], working[7], working[8], working[13]);
        quarter_round(working[3], working[4], working[9], working[14]);
    }
    for (int i = 0; i < 16; ++i) output[i] = working[i] + state[i];
}

__global__ void mask_kernel(uint16_t* data, size_t count, const uint32_t* key, const uint8_t* nonce, bool unmask) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    uint64_t block = index / 32;
    uint32_t stream[16];
    chacha20_block(key, static_cast<uint32_t>(block), nonce, stream);
    uint32_t word_index = index % 32;
    uint32_t word = (word_index % 2 == 0) ? (stream[word_index / 2] & 0xFFFFU) : (stream[word_index / 2] >> 16);
    if (unmask) {
        data[index] = static_cast<uint16_t>(data[index] - static_cast<uint16_t>(word));
    } else {
        data[index] = static_cast<uint16_t>(data[index] + static_cast<uint16_t>(word));
    }
}

extern "C" int xurcroh_mask_words(uint16_t* data, size_t count, const uint32_t* key, const uint8_t* nonce, bool unmask, cudaStream_t stream) {
    if (data == nullptr || key == nullptr || nonce == nullptr) return static_cast<int>(cudaErrorInvalidValue);
    size_t threads = 256;
    size_t blocks = (count + threads - 1) / threads;
    mask_kernel<<<blocks, threads, 0, stream>>>(data, count, key, nonce, unmask);
    return static_cast<int>(cudaGetLastError());
}
