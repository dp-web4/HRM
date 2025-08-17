// Combined CUDA implementation to avoid linking issues
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "../mailbox_peripheral.h"
#include "../mailbox_focus.h"

// ========== PERIPHERAL MAILBOX IMPLEMENTATION ==========

__device__ bool pbm_try_push(PBM_Header* h, uint8_t* payload_base,
                             const void* src, uint32_t len) {
    const uint32_t stride = h->record_stride;
    if (len > stride) return false;

    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);

    if ((w - r) >= h->capacity_records) {
        return false; // full
    }

    unsigned long long slot = w % h->capacity_records;
    uint8_t* dst = payload_base + (slot * stride);

    const uint8_t* s = reinterpret_cast<const uint8_t*>(src);
    for (uint32_t i = 0; i < len; ++i) dst[i] = s[i];

    __threadfence();
    atomicAdd((unsigned long long*)&h->write_idx, 1ULL);
    return true;
}

__device__ int pbm_try_pop_bulk(PBM_Header* h, uint8_t* payload_base,
                                void* dst, uint32_t dst_stride,
                                uint32_t max_records, uint32_t* out_bytes) {
    const uint32_t stride = h->record_stride;
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);

    unsigned long long available = (w > r) ? (w - r) : 0;
    unsigned long long to_pop = (available < max_records) ? available : max_records;
    if (to_pop == 0) {
        if (out_bytes) *out_bytes = 0;
        return 0;
    }

    uint8_t* d = reinterpret_cast<uint8_t*>(dst);
    for (unsigned long long i = 0; i < to_pop; ++i) {
        unsigned long long slot = (r + i) % h->capacity_records;
        uint8_t* src = payload_base + (slot * stride);
        uint8_t* dst_rec = d + (i * dst_stride);
        for (uint32_t j = 0; j < stride; ++j) dst_rec[j] = src[j];
    }

    __threadfence();
    atomicAdd((unsigned long long*)&h->read_idx, (unsigned long long)to_pop);

    if (out_bytes) *out_bytes = (uint32_t)(to_pop * stride);
    return (int)to_pop;
}

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

    PBM_Header* d_hdr = nullptr;
    uint8_t* d_payload = nullptr;
    size_t payload_bytes = (size_t)record_stride * (size_t)capacity_records;

    cudaError_t err;
    err = cudaMalloc((void**)&d_hdr, sizeof(PBM_Header));
    if (err != cudaSuccess) return false;
    err = cudaMalloc((void**)&d_payload, payload_bytes);
    if (err != cudaSuccess) { cudaFree(d_hdr); return false; }

    err = cudaMemcpy(d_hdr, &h, sizeof(PBM_Header), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_hdr); cudaFree(d_payload); return false; }

    *d_header_out = d_hdr;
    *d_payload_out = d_payload;
    return true;
}

// ========== FOCUS TENSOR MAILBOX IMPLEMENTATION ==========

__device__ bool ftm_try_push(FTM_Header* h, FTM_Record* ring, const FTM_Record* rec) {
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);
    if ((w - r) >= h->capacity_records) return false;

    unsigned long long slot = w % h->capacity_records;
    ring[slot] = *rec;

    __threadfence();
    atomicAdd((unsigned long long*)&h->write_idx, 1ULL);
    return true;
}

__device__ bool ftm_try_pop(FTM_Header* h, FTM_Record* ring, FTM_Record* out) {
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);
    if (r == w) return false;

    unsigned long long slot = r % h->capacity_records;
    *out = ring[slot];

    __threadfence();
    atomicAdd((unsigned long long*)&h->read_idx, 1ULL);
    return true;
}

extern "C" bool ftm_host_init(FTM_Header** d_header_out, FTM_Record** d_ring_out, uint64_t capacity_records) {
    if (!d_header_out || !d_ring_out) return false;
    if (capacity_records == 0) return false;

    FTM_Header h{};
    h.magic = 0x46544D31; // 'FTM1'
    h.version = 1;
    h.capacity_records = capacity_records;
    h.write_idx = 0ULL;
    h.read_idx = 0ULL;

    FTM_Header* d_hdr = nullptr;
    FTM_Record* d_ring = nullptr;

    cudaError_t err;
    err = cudaMalloc((void**)&d_hdr, sizeof(FTM_Header));
    if (err != cudaSuccess) return false;
    err = cudaMalloc((void**)&d_ring, sizeof(FTM_Record) * capacity_records);
    if (err != cudaSuccess) { cudaFree(d_hdr); return false; }

    err = cudaMemcpy(d_hdr, &h, sizeof(FTM_Header), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_hdr); cudaFree(d_ring); return false; }

    *d_header_out = d_hdr;
    *d_ring_out = d_ring;
    return true;
}

// ========== KERNEL LAUNCHERS ==========

__global__ void k_pbm_push(PBM_Header* h, uint8_t* payload,
                           const uint8_t* src, int len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        pbm_try_push(h, payload, src, (uint32_t)len);
    }
}

__global__ void k_pbm_pop(PBM_Header* h, uint8_t* payload,
                          uint8_t* dst, int max_records, int record_stride, int* d_count) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t out_bytes = 0;
        int got = pbm_try_pop_bulk(h, payload, dst, record_stride, max_records, &out_bytes);
        *d_count = got;
    }
}

void pbm_push_kernel_launch(PBM_Header* hdr, uint8_t* payload, const uint8_t* src, int len, cudaStream_t stream) {
    k_pbm_push<<<1,1,0,stream>>>(hdr, payload, src, len);
}

void pbm_pop_kernel_launch(PBM_Header* hdr, uint8_t* payload, uint8_t* dst, int max_records, int record_stride, int* d_count, cudaStream_t stream) {
    k_pbm_pop<<<1,1,0,stream>>>(hdr, payload, dst, max_records, record_stride, d_count);
}

__global__ void k_ftm_push(FTM_Header* h, FTM_Record* ring, FTM_Record rec) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ftm_try_push(h, ring, &rec);
    }
}

__global__ void k_ftm_pop(FTM_Header* h, FTM_Record* ring, FTM_Record* out, int* d_success) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        bool success = ftm_try_pop(h, ring, out);
        *d_success = success ? 1 : 0;
    }
}

void ftm_push_kernel_launch(FTM_Header* h, FTM_Record* ring, const FTM_Record& rec, cudaStream_t stream) {
    k_ftm_push<<<1,1,0,stream>>>(h, ring, rec);
}

void ftm_pop_kernel_launch(FTM_Header* h, FTM_Record* ring, FTM_Record* out, int* d_success, cudaStream_t stream) {
    k_ftm_pop<<<1,1,0,stream>>>(h, ring, out, d_success);
}