# Tiling Mailbox Compilation Issues

## Current Status
- ✅ PyTorch 2.3.0+cu121 installed and working
- ✅ CUDA 12.0 compiler (nvcc) available
- ✅ RTX 2060 SUPER (compute capability 7.5) detected
- ❌ Mailbox extensions not compiling due to missing implementations

## Issue Categories

### 1. Header File Path Issues

**Problem**: Headers not found during compilation
```
mailbox_focus.cu:5:10: fatal error: ../mailbox_focus.h: No such file or directory
```

**Affected Files**:
- `tiling_mailbox_torch_extension_v2/src/mailbox_focus.cu`
- `tiling_mailbox_torch_extension_v2/src/mailbox_peripheral.cu`
- `tiling_mailbox_torch_extension_v2/src/mailbox_ext.cpp`

**Current Structure**:
```
implementation/
├── tiling_mailbox_pack/           # Headers are here
│   ├── mailbox_focus.h
│   └── mailbox_peripheral.h
├── tiling_mailbox_cuda_pack/      # Copied headers here
│   ├── mailbox_focus.h
│   └── mailbox_peripheral.h
└── tiling_mailbox_torch_extension_v2/
    └── src/                        # Code expects headers in parent dir
        ├── mailbox_ext.cpp         # Uses ../../tiling_mailbox_cuda_pack/
        ├── mailbox_focus.cu        # Uses ../mailbox_focus.h
        └── mailbox_peripheral.cu   # Uses ../mailbox_peripheral.h
```

### 2. Missing Function Implementations

**Problem**: Undefined external CUDA functions
```
ptxas fatal   : Unresolved extern function 'pbm_try_push'
ptxas fatal   : Unresolved extern function 'ftm_try_push'
```

**Missing Functions**:
- `pbm_try_push` - Should be in mailbox_peripheral.cu
- `pbm_try_pop` - Should be in mailbox_peripheral.cu  
- `ftm_try_push` - Should be in mailbox_focus.cu
- `ftm_try_pop` - Should be in mailbox_focus.cu
- `pbm_host_init` - Host-side initialization
- `ftm_host_init` - Host-side initialization

**Functions Referenced But Not Defined**:
- In `mailbox_ext.cpp`:
  - `pbm_host_init(&d_hdr, &d_payload, record_stride, capacity)`
  - `ftm_host_init(&d_hdr, &d_ring, capacity)`
  - `pbm_push_kernel_launch(...)` 
  - `pbm_pop_kernel_launch(...)`
  - `ftm_push_kernel_launch(...)`
  - `ftm_pop_kernel_launch(...)`

### 3. CUDA Stream API Issues

**Problem**: PyTorch 2.3 stream API incompatibility
```
error: 'getCurrentCUDAStream' is not a member of 'at::cuda'
```

**Current Fix**: Using default stream (0) instead of PyTorch stream
```cpp
cudaStream_t stream = 0; // Use default stream
```

**Better Solution**: Need to properly get PyTorch's current stream

### 4. Kernel Launch Functions

**Problem**: Kernel launch functions declared but not implemented
```cpp
// Declared in mailbox_ext.cpp but not defined anywhere:
void pbm_push_kernel_launch(PBM_Header*, uint8_t*, const uint8_t*, int, cudaStream_t);
void pbm_pop_kernel_launch(PBM_Header*, uint8_t*, uint8_t*, int, int, cudaStream_t);
void ftm_push_kernel_launch(FTM_Header*, FTM_Record*, const FTM_Record&, cudaStream_t);
void ftm_pop_kernel_launch(FTM_Header*, FTM_Record*, FTM_Record*, cudaStream_t);
```

These should be in `mailbox_kernels.cu` but that file has issues.

## Resolution Plan

### Step 1: Fix Header Paths ✅
- [x] Document current structure
- [ ] Choose consistent header location
- [ ] Update all #include paths
- [ ] Verify headers are complete

### Step 2: Implement Missing Device Functions
- [ ] Review header definitions for function signatures
- [ ] Implement `pbm_try_push` in mailbox_peripheral.cu
- [ ] Implement `pbm_try_pop` in mailbox_peripheral.cu
- [ ] Implement `ftm_try_push` in mailbox_focus.cu
- [ ] Implement `ftm_try_pop` in mailbox_focus.cu

### Step 3: Implement Host Initialization Functions
- [ ] Implement `pbm_host_init` for peripheral mailbox
- [ ] Implement `ftm_host_init` for focus mailbox
- [ ] Add proper CUDA memory allocation

### Step 4: Implement Kernel Launch Functions
- [ ] Fix mailbox_kernels.cu compilation
- [ ] Implement `pbm_push_kernel_launch`
- [ ] Implement `pbm_pop_kernel_launch`
- [ ] Implement `ftm_push_kernel_launch`
- [ ] Implement `ftm_pop_kernel_launch`

### Step 5: Fix PyTorch Stream Integration
- [ ] Research correct PyTorch 2.3 stream API
- [ ] Update stream handling code
- [ ] Test stream synchronization

### Step 6: Build and Test
- [ ] Clean build of extensions
- [ ] Run unit tests
- [ ] Run performance benchmarks
- [ ] Verify zero-copy operation

## File Dependencies Graph

```
mailbox_ext.cpp (main interface)
    ├── mailbox_peripheral.h
    │   └── Defines: PBM_Header, PBM_Record structures
    ├── mailbox_focus.h  
    │   └── Defines: FTM_Header, FTM_Record structures
    ├── mailbox_peripheral.cu
    │   └── Should implement: pbm_try_push, pbm_try_pop, pbm_host_init
    ├── mailbox_focus.cu
    │   └── Should implement: ftm_try_push, ftm_try_pop, ftm_host_init
    └── mailbox_kernels.cu
        └── Should implement: all kernel launch functions
```

## Next Actions

1. **Immediate**: Fix header paths to get past initial compilation errors
2. **Then**: Implement missing device functions based on header signatures
3. **Finally**: Complete kernel launch implementations and test

## Testing Strategy

Once compiled:
1. Test initialization functions (memory allocation)
2. Test single push/pop operations
3. Test bulk operations
4. Test concurrent access patterns
5. Profile memory bandwidth and latency