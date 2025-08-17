
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>  // for memset on host paths only
#include "../mailbox_peripheral.h"

// NOTE: This is a skeleton: byte-wise copies, no vectorization, minimal checks.
// Align record_stride (>=128B) in practice for coalesced access.

__device__ bool pbm_try_push(PBM_Header* h, uint8_t* payload_base,
                             const void* src, uint32_t len) {
    // Fixed-size records only; clamp to record_stride
    const uint32_t stride = h->record_stride;
    if (len > stride) return false;

    // Atomic "loads" of indices (monotonic counters)
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);

    // Capacity check (number of items currently enqueued = w - r)
    if ((w - r) >= h->capacity_records) {
        return false; // full
    }

    // Compute slot and destination pointer
    unsigned long long slot = w % h->capacity_records;
    uint8_t* dst = payload_base + (slot * stride);

    // Byte copy (optimize later with vectorized loads/stores if needed)
    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t i = 0; i < len; ++i) dst[i] = s[i];
    // If len < stride you may want to zero the tail for debugging predictability
    // for (uint32_t i = len; i < stride; ++i) dst[i] = 0;

    // Ensure payload visible before publishing the write index
    __threadfence();

    // Publish: increment write index (monotonic)
    atomicAdd((unsigned long long*)&h->write_idx, 1ULL);
    return true;
}

__device__ int pbm_try_pop_bulk(PBM_Header* h, uint8_t* payload_base,
                                void* dst, uint32_t dst_stride,
                                uint32_t max_records, uint32_t* out_bytes) {
    const uint32_t stride = h->record_stride;
    // Atomic loads
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);

    unsigned long long available = (w - r);
    if (available == 0ULL) {
        if (out_bytes) *out_bytes = 0;
        return 0; // empty
    }
    unsigned long long to_pop = available < max_records ? available : max_records;

    uint8_t* out = reinterpret_cast<uint8_t*>(dst);
    for (unsigned long long i = 0ULL; i < to_pop; ++i) {
        unsigned long long slot = (r + i) % h->capacity_records;
        uint8_t* src_ptr = payload_base + (slot * stride);
        uint8_t* dst_ptr = out + (i * dst_stride);
        // Copy one fixed-size record
        for (uint32_t b = 0; b < stride; ++b) {
            dst_ptr[b] = src_ptr[b];
        }
    }

    // Make sure copies are visible before updating read index
    __threadfence();

    // Publish: advance read index by to_pop
    atomicAdd((unsigned long long*)&h->read_idx, (unsigned long long)to_pop);

    if (out_bytes) *out_bytes = (uint32_t)(to_pop * stride);
    return (int)to_pop;
}

// Minimal host helper: allocate header & payload in device memory and initialize.
extern "C" bool pbm_host_init(PBM_Header** d_header_out,
                              uint8_t** d_payload_out,
                              uint32_t record_stride,
                              uint64_t capacity_records) {
    if (!d_header_out || !d_payload_out) return false;
    if (record_stride == 0 || capacity_records == 0) return false;

    PBM_Header h{};
    h.magic = 0x50424D31; // 'PBM1'
    h.version = 1;
    h.record_stride = record_stride;
    h.capacity_records = capacity_records;
    h.write_idx = 0ULL;
    h.read_idx = 0ULL;

    // Allocate device header & payload
    PBM_Header* d_hdr = nullptr;
    uint8_t* d_payload = nullptr;
    size_t payload_bytes = (size_t)record_stride * (size_t)capacity_records;

    cudaError_t err;
    err = cudaMalloc((void**)&d_hdr, sizeof(PBM_Header));
    if (err != cudaSuccess) return false;
    err = cudaMalloc((void**)&d_payload, payload_bytes);
    if (err != cudaSuccess) { cudaFree(d_hdr); return false; }

    // Initialize device header
    err = cudaMemcpy(d_hdr, &h, sizeof(PBM_Header), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_hdr); cudaFree(d_payload); return false; }

    *d_header_out = d_hdr;
    *d_payload_out = d_payload;
    return true;
}
