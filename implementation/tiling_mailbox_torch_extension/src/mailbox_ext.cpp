
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <vector>
#include "../../tiling_mailbox_cuda_pack/mailbox_peripheral.h"
#include "../../tiling_mailbox_cuda_pack/mailbox_focus.h"

// Bind minimal functions

torch::Tensor pbm_init(int64_t record_stride, int64_t capacity) {
    PBM_Header* d_hdr = nullptr;
    uint8_t* d_payload = nullptr;
    bool ok = pbm_host_init(&d_hdr, &d_payload, (uint32_t)record_stride, (uint64_t)capacity);
    TORCH_CHECK(ok, "pbm_host_init failed");
    // Return pointers as 64-bit ints (unsafe but practical for testing)
    auto t = torch::empty({2}, torch::dtype(torch::kInt64).device(torch::kCPU));
    t[0] = (int64_t)d_hdr;
    t[1] = (int64_t)d_payload;
    return t;
}

torch::Tensor ftm_init(int64_t capacity) {
    FTM_Header* d_hdr = nullptr;
    FTM_Record* d_ring = nullptr;
    bool ok = ftm_host_init(&d_hdr, &d_ring, (uint64_t)capacity);
    TORCH_CHECK(ok, "ftm_host_init failed");
    auto t = torch::empty({2}, torch::dtype(torch::kInt64).device(torch::kCPU));
    t[0] = (int64_t)d_hdr;
    t[1] = (int64_t)d_ring;
    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbm_init", &pbm_init, "Peripheral mailbox init");
    m.def("ftm_init", &ftm_init, "Focus mailbox init");
}
