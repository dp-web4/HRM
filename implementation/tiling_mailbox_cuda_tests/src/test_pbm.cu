
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "../tiling_mailbox_cuda_pack/mailbox_peripheral.h"

__global__ void push_kernel(PBM_Header* h, uint8_t* payload, int n) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        uint8_t rec[64];
        for (int i=0;i<64;i++) rec[i] = (uint8_t)(tid + i);
        pbm_try_push(h, payload, rec, 64);
    }
}

__global__ void pop_kernel(PBM_Header* h, uint8_t* payload, uint8_t* out, int max_pop) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t out_bytes = 0;
        int got = pbm_try_pop_bulk(h, payload, out, 64, max_pop, &out_bytes);
        // store count in out[0]
        if (out_bytes > 0) out[0] = (uint8_t)got;
    }
}

int main() {
    PBM_Header* d_hdr = nullptr;
    uint8_t* d_payload = nullptr;
    if (!pbm_host_init(&d_hdr, &d_payload, /*record_stride=*/64, /*capacity=*/1024)) {
        printf("INIT FAIL\\n");
        return 1;
    }

    // push N records
    int N = 256;
    push_kernel<<<(N+127)/128, 128>>>(d_hdr, d_payload, N);
    cudaDeviceSynchronize();

    uint8_t* d_out = nullptr;
    cudaMalloc((void**)&d_out, 64 * 1024);
    cudaMemset(d_out, 0, 64 * 1024);
    pop_kernel<<<1,1>>>(d_hdr, d_payload, d_out, 512);
    cudaDeviceSynchronize();

    uint8_t h0 = 0;
    cudaMemcpy(&h0, d_out, 1, cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    if (h0 == 0) {
        printf("POP got 0 — FAIL\\n");
        return 2;
    }
    printf("PBM PASS — popped %d records\\n", (int)h0);
    return 0;
}
