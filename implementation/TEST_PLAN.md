# Tiling Mailbox System Test Plan

*Created: August 18, 2025*  
*Author: Claude (reviewing GPT's implementation)*

## Overview

GPT has created a comprehensive two-tier mailbox system implementing the Sight-Insight tiling principle. The system consists of:
- **Peripheral Broadcast Mailboxes** (PBM): Many small, fast messages for global awareness
- **Focus Tensor Mailboxes** (FTM): Few large, zero-copy tensor handoffs for deep analysis

## Code Structure

### 1. Core Implementation (`tiling_mailbox_pack/`)
- `mailbox_peripheral.h`: Peripheral mailbox header definitions
- `mailbox_focus.h`: Focus mailbox header definitions  
- `IMPLEMENTATION_NOTES_TILING.md`: Architecture overview
- `trust_scheduler_example.py`: CPU-arbitrated trust-weighted scheduling

### 2. CUDA Implementation (`tiling_mailbox_cuda_pack/`)
- `mailbox_peripheral.cu`: GPU kernels for peripheral operations
- `mailbox_focus.cu`: GPU kernels for focus operations
- `CMakeLists.txt`: Build configuration
- `BUILD.md`: Build instructions

### 3. CUDA Tests (`tiling_mailbox_cuda_tests/`)
- `test_pbm.cu`: Standalone peripheral mailbox test
- `test_ftm.cu`: Standalone focus mailbox test
- `README.md`: Test execution instructions
- **Purpose**: PyTorch-independent tests for Jetson validation

### 4. PyTorch Extension v1 (`tiling_mailbox_torch_extension/`)
- `setup.py`: Python extension build script
- `mailbox_ext.cpp`: PyTorch C++ bindings
- `test_ext.py`: Basic extension test

### 5. PyTorch Extension v2 (`tiling_mailbox_torch_extension_v2/`)
- Enhanced push/pop API
- `test_push_pop.py`: Push/pop functionality test
- `test_profile.py`: Performance profiling test

## Test Categories

### Level 1: Unit Tests (Component Validation)

#### 1.1 Peripheral Mailbox Tests
- [ ] **PBM Init**: Verify mailbox initialization with correct sizes
- [ ] **PBM Push**: Single record push succeeds
- [ ] **PBM Pop**: Single record pop retrieves correct data
- [ ] **PBM Bulk Pop**: Multiple records retrieved in order
- [ ] **PBM Overflow**: Graceful handling when full
- [ ] **PBM Alignment**: 128B alignment for coalescing

#### 1.2 Focus Mailbox Tests
- [ ] **FTM Init**: Ring buffer initialization
- [ ] **FTM Push Pointer**: Tensor metadata stored correctly
- [ ] **FTM Pop Pointer**: Tensor metadata retrieved correctly
- [ ] **FTM Zero-Copy**: Verify no host transfers
- [ ] **FTM TTL**: Time-to-live expiration handling
- [ ] **FTM Ownership**: Pointer lifetime management

### Level 2: Integration Tests (Module Interaction)

#### 2.1 Trust Scheduler Tests
- [ ] **Top-K Selection**: Correct tiles promoted to focus
- [ ] **Stream Priority**: Focus stream higher priority than peripheral
- [ ] **Event Synchronization**: Producer→consumer coordination
- [ ] **Backpressure**: Full focus mailbox handling
- [ ] **Fairness**: Per-producer slot limits

#### 2.2 PyTorch Extension Tests
- [ ] **Extension Build**: Successful compilation for target arch
- [ ] **Python API**: All functions accessible from Python
- [ ] **Tensor Round-trip**: Push/pop preserves tensor properties
- [ ] **CUDA Streams**: Correct stream association
- [ ] **Memory Management**: No leaks after repeated ops

### Level 3: Performance Tests (Metrics Validation)

#### 3.1 Bandwidth Tests
- [ ] **Peripheral < 20%**: Memory bandwidth usage
- [ ] **Focus < 80%**: Memory bandwidth usage
- [ ] **Combined < 100%**: Total bandwidth within limits

#### 3.2 Latency Tests
- [ ] **Peripheral Push**: < 0.1ms per record
- [ ] **Peripheral Pop**: < 0.1ms per record
- [ ] **Focus Push**: < 0.5ms per tensor
- [ ] **Focus Pop**: < 0.5ms per tensor
- [ ] **End-to-End**: < 10ms for full cycle

#### 3.3 Throughput Tests
- [ ] **Peripheral Rate**: > 10,000 msgs/sec
- [ ] **Focus Rate**: > 100 tensors/sec
- [ ] **Drop Rate**: < 1% under load
- [ ] **Coherence Pin**: Focus tiles maintain attention

### Level 4: System Tests (End-to-End Validation)

#### 4.1 Vision Pipeline Integration
- [ ] **Camera→Peripheral**: Motion detection flows
- [ ] **Trust→Focus**: High-trust tiles promoted
- [ ] **Focus→Analysis**: Deep processing on focus tiles
- [ ] **Insight Generation**: Measurable insight increase

#### 4.2 FlashAttention Integration
- [ ] **SDPA Compatibility**: Works with scaled_dot_product_attention
- [ ] **Flash Mode**: enable_flash=True succeeds
- [ ] **Memory Efficient**: enable_mem_efficient=True works
- [ ] **Tiling Alignment**: Tile sizes match SRAM capacity

#### 4.3 Multi-GPU Tests (Future)
- [ ] **Cross-GPU Mailbox**: P2P communication
- [ ] **NVLink Usage**: When available
- [ ] **Distributed Trust**: Consensus across GPUs

## Test Execution Plan

### Phase 1: Local Development (Windows/WSL2)
```bash
# 1. Build CUDA tests
cd tiling_mailbox_cuda_tests
mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=86 ..  # RTX 2060 SUPER
cmake --build . -j4

# 2. Run standalone tests
./test_pbm
./test_ftm

# 3. Build PyTorch extension
cd ../../tiling_mailbox_torch_extension_v2
export TORCH_CUDA_ARCH_LIST="8.6"
python setup.py build_ext --inplace

# 4. Run PyTorch tests
python test_push_pop.py
python test_profile.py
```

### Phase 2: Jetson Validation (Orin Nano)
```bash
# Adjust for Jetson architecture
export TORCH_CUDA_ARCH_LIST="8.7"  # Orin
# Or for older Jetson: "7.2"  # Xavier

# Run same test sequence with Jetson-specific metrics
```

### Phase 3: Legion Performance (RTX 4090)
```bash
# High-performance validation
export TORCH_CUDA_ARCH_LIST="8.9"  # Ada Lovelace

# Add stress tests with larger tile counts
```

## Success Criteria

### Functional Success
- ✅ All unit tests pass
- ✅ Zero host copies verified in profiler
- ✅ Trust-based promotion works correctly
- ✅ No memory leaks after 1000 iterations

### Performance Success
- ✅ Meets bandwidth targets (< 20% peripheral, < 80% focus)
- ✅ Meets latency targets (< 1ms peripheral, < 10ms focus)
- ✅ Stable throughput under load
- ✅ < 1% drop rate

### Integration Success
- ✅ Works with real camera feeds
- ✅ FlashAttention integration successful
- ✅ Measurable insight improvement
- ✅ Graceful degradation under stress

## Known Issues & Mitigations

### Issue 1: Hidden Host Transfers
**Detection**: Use nvprof or Nsight Systems  
**Mitigation**: Pin memory, use cudaHostAlloc

### Issue 2: Misaligned Access
**Detection**: CUDA profiler warnings  
**Mitigation**: Pad records to 128B boundaries

### Issue 3: Pointer Lifetime
**Detection**: Segfaults or corruption  
**Mitigation**: Explicit ownership rules, reference counting

### Issue 4: Stream Synchronization
**Detection**: Race conditions, wrong results  
**Mitigation**: Proper event recording and waiting

## Testing Tools

### Profiling
- `nvprof` (deprecated but works)
- `nsys` (Nsight Systems)
- `ncu` (Nsight Compute)
- PyTorch Profiler with CUDA events

### Monitoring
- `nvidia-smi dmon` - Real-time GPU metrics
- `tegrastats` - Jetson-specific monitoring
- Custom telemetry in mailbox implementation

### Validation
- CUDA memcheck for memory errors
- compute-sanitizer for race conditions
- Python memory_profiler for leaks

## Documentation Requirements

For each test:
1. **Purpose**: What does this test validate?
2. **Setup**: Prerequisites and configuration
3. **Execution**: Exact commands to run
4. **Expected Results**: What success looks like
5. **Common Failures**: Known issues and fixes

## Next Steps

1. [ ] Set up CI/CD for automated testing
2. [ ] Create benchmark baselines for each platform
3. [ ] Implement telemetry dashboard
4. [ ] Add fault injection tests
5. [ ] Create performance regression tests

---

*"Testing isn't about proving it works - it's about discovering how it fails."*