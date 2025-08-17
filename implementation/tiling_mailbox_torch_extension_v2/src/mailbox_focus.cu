
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../mailbox_focus.h"

// Skeleton: ring buffer of FTM_Record entries.
// NOTE: Pointer lifetime is NOT managed here; producer/consumer must coordinate ownership.

__device__ bool ftm_try_push(FTM_Header* h, FTM_Record* ring, const FTM_Record* rec) {
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);
    if ((w - r) >= h->capacity_records) return false; // full

    unsigned long long slot = w % h->capacity_records;
    // Copy the record (shallow copy; dev_ptr is passed by value)
    ring[slot] = *rec;

    __threadfence(); // ensure record visible
    atomicAdd((unsigned long long*)&h->write_idx, 1ULL);
    return true;
}

__device__ bool ftm_try_pop(FTM_Header* h, FTM_Record* ring, FTM_Record* out) {
    unsigned long long w = atomicAdd((unsigned long long*)&h->write_idx, 0ULL);
    unsigned long long r = atomicAdd((unsigned long long*)&h->read_idx,  0ULL);
    if (r == w) return false; // empty

    unsigned long long slot = r % h->capacity_records;
    *out = ring[slot]; // shallow copy

    __threadfence(); // ensure read happens before advancing
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
