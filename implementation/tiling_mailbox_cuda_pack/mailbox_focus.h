
// mailbox_focus.h
// FocusTensorMailbox: few→few pointer-based zero-copy handoff.

#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  FTM_DTYPE_F16 = 1,
  FTM_DTYPE_F32 = 2,
  FTM_DTYPE_BF16 = 3,
  FTM_DTYPE_I32 = 10,
} FTM_DType;

typedef struct {
  void*    dev_ptr;       // device pointer to tensor storage
  int64_t  shape[4];      // up to 4D; unused dims = 1
  int64_t  stride[4];
  int32_t  ndim;          // 1..4
  int32_t  dtype;         // FTM_DTYPE_*
  int32_t  tag;           // semantic/source id
  int32_t  ttl;           // drop if <= 0
} FTM_Record;

typedef struct {
  uint32_t magic;
  uint16_t version;
  uint16_t reserved0;
  uint64_t capacity_records;
  alignas(8) volatile unsigned long long write_idx;
  alignas(8) volatile unsigned long long read_idx;
} FTM_Header;

// Device-side API
__device__ bool ftm_try_push(FTM_Header* h, FTM_Record* ring, const FTM_Record* rec);
__device__ bool ftm_try_pop(FTM_Header* h, FTM_Record* ring, FTM_Record* out);

// Host helpers
bool ftm_host_init(FTM_Header** d_header_out, FTM_Record** d_ring_out, uint64_t capacity_records);

#ifdef __cplusplus
}
#endif
