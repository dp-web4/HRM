
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "../tiling_mailbox_cuda_pack/mailbox_focus.h"

__global__ void push_ftm(FTM_Header* h, FTM_Record* ring, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        FTM_Record rec{};
        rec.dev_ptr = (void*)0xDEADBEEF; // dummy
        rec.shape[0] = 1; rec.shape[1] = 2; rec.shape[2] = 3; rec.shape[3] = 4;
        rec.stride[0]= 4; rec.stride[1]= 2; rec.stride[2]= 1; rec.stride[3]= 1;
        rec.ndim = 4; rec.dtype = 1; rec.tag = tid; rec.ttl = 2;
        ftm_try_push(h, ring, &rec);
    }
}

__global__ void pop_ftm(FTM_Header* h, FTM_Record* ring, FTM_Record* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ftm_try_pop(h, ring, out);
    }
}

int main() {
    FTM_Header* d_hdr = nullptr;
    FTM_Record* d_ring = nullptr;
    if (!ftm_host_init(&d_hdr, &d_ring, /*capacity=*/256)) {
        printf("INIT FAIL\\n");
        return 1;
    }

    int N = 64;
    push_ftm<<<(N+127)/128, 128>>>(d_hdr, d_ring, N);
    cudaDeviceSynchronize();

    FTM_Record* d_out = nullptr;
    cudaMalloc((void**)&d_out, sizeof(FTM_Record));
    cudaMemset(d_out, 0, sizeof(FTM_Record));
    pop_ftm<<<1,1>>>(d_hdr, d_ring, d_out);
    cudaDeviceSynchronize();

    FTM_Record hrec{};
    cudaMemcpy(&hrec, d_out, sizeof(FTM_Record), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    if (hrec.dev_ptr == nullptr) {
        printf("POP dev_ptr null — FAIL\\n");
        return 2;
    }
    printf("FTM PASS — tag=%d ndim=%d\\n", hrec.tag, hrec.ndim);
    return 0;
}
