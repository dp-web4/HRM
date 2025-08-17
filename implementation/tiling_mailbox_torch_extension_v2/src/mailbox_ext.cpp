
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <pybind11/stl.h>
#include <vector>
#include <array>
#include <map>
#include <stdexcept>
#include "../mailbox_peripheral.h"
#include "../mailbox_focus.h"

// CUDA kernels (defined in mailbox_cuda_all.cu)
void pbm_push_kernel_launch(PBM_Header* hdr, uint8_t* payload, const uint8_t* src, int len, cudaStream_t stream);
void pbm_pop_kernel_launch(PBM_Header* hdr, uint8_t* payload, uint8_t* dst, int max_records, int record_stride, int* d_count, cudaStream_t stream);

// --- Init ---
torch::Tensor pbm_init(int64_t record_stride, int64_t capacity) {
    PBM_Header* d_hdr = nullptr;
    uint8_t* d_payload = nullptr;
    bool ok = pbm_host_init(&d_hdr, &d_payload, (uint32_t)record_stride, (uint64_t)capacity);
    TORCH_CHECK(ok, "pbm_host_init failed");
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

// --- Peripheral push/pop ---
// Expects 'src' to be a CUDA uint8 tensor with length <= record_stride
bool pbm_push_bytes_cuda(int64_t hdr_ptr, int64_t payload_ptr, torch::Tensor src) {
    TORCH_CHECK(src.is_cuda(), "src must be a CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kUInt8, "src must be uint8");
    auto* hdr = reinterpret_cast<PBM_Header*>(hdr_ptr);
    auto* payload = reinterpret_cast<uint8_t*>(payload_ptr);
    cudaStream_t stream = 0; // Use default stream
    pbm_push_kernel_launch(hdr, payload,
                           src.data_ptr<uint8_t>(),
                           (int)src.numel(),
                           stream);
    return true;
}

// Returns a CUDA uint8 tensor containing N*record_stride bytes (N <= max_records)
torch::Tensor pbm_pop_bulk_cuda(int64_t hdr_ptr, int64_t payload_ptr, int max_records, int record_stride) {
    auto* hdr = reinterpret_cast<PBM_Header*>(hdr_ptr);
    auto* payload = reinterpret_cast<uint8_t*>(payload_ptr);
    // Allocate worst-case output on CUDA
    auto out = torch::empty({max_records * record_stride}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    auto d_count = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    cudaStream_t stream = 0; // Use default stream
    pbm_pop_kernel_launch(hdr, payload, out.data_ptr<uint8_t>(),
                          max_records, record_stride,
                          d_count.data_ptr<int>(), stream);
    
    // Synchronize to get count
    auto h_count = d_count.cpu().item<int>();  // This syncs the stream
    
    // Return trimmed tensor with actual data
    if (h_count > 0) {
        return out.narrow(0, 0, h_count * record_stride);
    } else {
        return torch::empty({0}, torch::dtype(torch::kUInt8).device(torch::kCUDA));
    }
}

// --- Focus push/pop ---
bool ftm_push_ptr(int64_t hdr_ptr, int64_t ring_ptr,
                  int64_t dev_ptr,
                  std::vector<int64_t> shape,
                  std::vector<int64_t> stride,
                  int64_t ndim, int64_t dtype, int64_t tag, int64_t ttl) {
    auto* hdr = reinterpret_cast<FTM_Header*>(hdr_ptr);
    auto* ring = reinterpret_cast<FTM_Record*>(ring_ptr);

    FTM_Record rec{};
    rec.dev_ptr = (void*)dev_ptr;
    rec.ndim = (int32_t)ndim;
    rec.dtype = (int32_t)dtype;
    rec.tag = (int32_t)tag;
    rec.ttl = (int32_t)ttl;
    for (int i=0;i<4;i++) {
        rec.shape[i]  = (i < (int)shape.size())  ? shape[i]  : 1;
        rec.stride[i] = (i < (int)stride.size()) ? stride[i] : 1;
    }

    // Launch a tiny kernel to push (defined in mailbox_kernels.cu)
    extern void ftm_push_kernel_launch(FTM_Header*, FTM_Record*, const FTM_Record&, cudaStream_t);
    cudaStream_t stream = 0; // Use default stream
    ftm_push_kernel_launch(hdr, ring, rec, stream);
    return true;
}

std::map<std::string, torch::Tensor> ftm_pop(int64_t hdr_ptr, int64_t ring_ptr) {
    auto* hdr = reinterpret_cast<FTM_Header*>(hdr_ptr);
    auto* ring = reinterpret_cast<FTM_Record*>(ring_ptr);
    extern void ftm_pop_kernel_launch(FTM_Header*, FTM_Record*, FTM_Record*, int*, cudaStream_t);
    
    // Allocate device memory for output record and success flag
    FTM_Record* d_rec = nullptr;
    cudaMalloc(&d_rec, sizeof(FTM_Record));
    auto d_success = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCUDA));
    
    cudaStream_t stream = 0; // Use default stream
    ftm_pop_kernel_launch(hdr, ring, d_rec, d_success.data_ptr<int>(), stream);
    
    // Synchronize to get success flag
    auto h_success = d_success.cpu().item<int>();  // This syncs the stream
    
    std::map<std::string, torch::Tensor> result;
    
    if (h_success > 0) {
        // Copy record from device to host
        FTM_Record host_rec{};
        cudaMemcpy(&host_rec, d_rec, sizeof(FTM_Record), cudaMemcpyDeviceToHost);
        
        auto devptr = torch::empty({1}, torch::dtype(torch::kInt64).device(torch::kCPU));
        devptr[0] = (int64_t)host_rec.dev_ptr;
        result["dev_ptr"] = devptr;

        auto shape = torch::empty({4}, torch::dtype(torch::kInt64).device(torch::kCPU));
        auto stride= torch::empty({4}, torch::dtype(torch::kInt64).device(torch::kCPU));
        for (int i=0;i<4;i++) { shape[i] = host_rec.shape[i]; stride[i] = host_rec.stride[i]; }
        result["shape"] = shape;
        result["stride"] = stride;

        auto meta = torch::empty({3}, torch::dtype(torch::kInt32).device(torch::kCPU));
        meta[0] = host_rec.ndim; meta[1] = host_rec.dtype; meta[2] = host_rec.tag;
        result["meta"] = meta;

        auto ttl = torch::empty({1}, torch::dtype(torch::kInt32).device(torch::kCPU));
        ttl[0] = host_rec.ttl;
        result["ttl"] = ttl;
    } else {
        // Return empty result to indicate no record available
        result["dev_ptr"] = torch::zeros({1}, torch::dtype(torch::kInt64).device(torch::kCPU));
        result["shape"] = torch::zeros({4}, torch::dtype(torch::kInt64).device(torch::kCPU));
        result["stride"] = torch::zeros({4}, torch::dtype(torch::kInt64).device(torch::kCPU));
        result["meta"] = torch::zeros({3}, torch::dtype(torch::kInt32).device(torch::kCPU));
        result["ttl"] = torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCPU));
    }
    
    cudaFree(d_rec);
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pbm_init", &pbm_init, "Peripheral mailbox init");
    m.def("ftm_init", &ftm_init, "Focus mailbox init");
    m.def("pbm_push_bytes_cuda", &pbm_push_bytes_cuda, "Push one record (CUDA uint8 tensor)");
    m.def("pbm_pop_bulk_cuda", &pbm_pop_bulk_cuda, "Pop up to max_records; returns CUDA uint8 tensor");
    m.def("ftm_push_ptr", &ftm_push_ptr, "Push a focus record by device pointer + metadata");
    m.def("ftm_pop", &ftm_pop, "Pop one focus record; returns dict");
}
