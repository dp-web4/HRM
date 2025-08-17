
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "../mailbox_peripheral.h"
#include "../mailbox_focus.h"

// --- Peripheral kernels ---
__global__ void k_pbm_push(PBM_Header* h, uint8_t* payload,
                           const uint8_t* src, int len) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        pbm_try_push(h, payload, src, (uint32_t)len);
    }
}

__global__ void k_pbm_pop(PBM_Header* h, uint8_t* payload,
                          uint8_t* dst, int max_records, int record_stride) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t out_bytes = 0;
        pbm_try_pop_bulk(h, payload, dst, record_stride, max_records, &out_bytes);
    }
}

void pbm_push_kernel_launch(PBM_Header* hdr, uint8_t* payload, const uint8_t* src, int len, cudaStream_t stream) {
    k_pbm_push<<<1,1,0,stream>>>(hdr, payload, src, len);
}

void pbm_pop_kernel_launch(PBM_Header* hdr, uint8_t* payload, uint8_t* dst, int max_records, int record_stride, cudaStream_t stream) {
    k_pbm_pop<<<1,1,0,stream>>>(hdr, payload, dst, max_records, record_stride);
}

// --- Focus kernels ---
__global__ void k_ftm_push(FTM_Header* h, FTM_Record* ring, FTM_Record rec) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ftm_try_push(h, ring, &rec);
    }
}

__global__ void k_ftm_pop(FTM_Header* h, FTM_Record* ring, FTM_Record* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ftm_try_pop(h, ring, out);
    }
}

void ftm_push_kernel_launch(FTM_Header* h, FTM_Record* ring, const FTM_Record& rec, cudaStream_t stream) {
    k_ftm_push<<<1,1,0,stream>>>(h, ring, rec);
}

void ftm_pop_kernel_launch(FTM_Header* h, FTM_Record* ring, FTM_Record* out, cudaStream_t stream) {
    k_ftm_pop<<<1,1,0,stream>>>(h, ring, out);
}
