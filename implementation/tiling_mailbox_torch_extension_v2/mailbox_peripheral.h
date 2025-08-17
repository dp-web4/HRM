
// mailbox_peripheral.h
// PeripheralBroadcastMailbox: many→many fixed-size records in GPU memory.
// Minimal C-style header to keep language-agnostic bindings easy.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Must be 128B aligned for coalesced access; keep small.
typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved0;
  uint32_t record_stride;       // e.g., 256 bytes
  uint64_t capacity_records;    // ring size
  alignas(8) volatile unsigned long long write_idx; // atomic monotonic
  alignas(8) volatile unsigned long long read_idx;  // atomic monotonic
} PBM_Header;

// Buffer layout in device global memory:
// [ PBM_Header | padding to 128B | payload bytes ... ]
// Payload is capacity_records * record_stride

// Device-side API (to be implemented in CUDA .cu)
__device__ bool pbm_try_push(PBM_Header* h, uint8_t* payload_base,
                             const void* src, uint32_t len);

__device__ int  pbm_try_pop_bulk(PBM_Header* h, uint8_t* payload_base,
                                 void* dst, uint32_t dst_stride,
                                 uint32_t max_records, uint32_t* out_bytes);

// Host helpers (allocate, init header); implement in .cc/.cu
bool pbm_host_init(PBM_Header** d_header_out,
                   uint8_t** d_payload_out,
                   uint32_t record_stride,
                   uint64_t capacity_records);

#ifdef __cplusplus
}
#endif
