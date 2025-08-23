# HRM/SAGE Implementation Plan - Jetson Orin Nano

*Date: August 22, 2025*  
*Machine: Jetson Orin Nano (8GB, CUDA 8.7)*  
*Baseline: 1097 GFLOPS @ FP32, 633 img/s convolution*

## Phase Structure

Each phase builds on the previous, with clear success criteria and deliverables.

---

## Phase 1: Environment & Baseline ✅ COMPLETE

### Achieved:
- PyTorch 2.5.0 with CUDA working
- Baseline performance: 788 GFLOPS (1024x1024 matmul)
- Vision estimate: 1.6ms per frame with early stop
- Language estimate: 0.2ms per token with stabilization

### Key Finding:
- FP16 currently slower than FP32 (need to investigate tensor cores)
- Excellent convolution performance (633 img/s single batch)

---

## Phase 2: Vision IRP Implementation (Current)

### Step 2.1: Lightweight VAE Setup
```python
# models/vision/lightweight_vae.py
- Input: 224x224x3 images
- Latent: 7x7x256 (12.5KB per image)
- Architecture: 4 conv layers encode, 4 deconv decode
- Target: <5ms encode/decode
```

### Step 2.2: Refinement Network
```python
# sage/irp/plugins/vision_impl.py
- Small U-Net in latent space (7x7x256)
- 3 refinement blocks with skip connections
- Energy function: reconstruction + regularization
- Early stop when delta_E < 0.01
```

### Step 2.3: Integration & Testing
```python
# demos/vision_real.py
- Load CIFAR-10 or custom dataset
- Run IRP with real VAE
- Measure: compute savings, quality preservation
- Target: 2x speedup, <1% quality loss
```

### Success Criteria:
- [ ] VAE encode/decode < 5ms
- [ ] Refinement converges in < 20 steps
- [ ] 2x compute savings vs full refinement
- [ ] Quality metrics within 1% of baseline

### Deliverables:
- `models/vision/lightweight_vae.py`
- `sage/irp/plugins/vision_impl.py`
- `demos/vision_real.py`
- `results/vision_irp_metrics.json`

---

## Phase 3: Language IRP Implementation

### Step 3.1: Small Language Model
```python
# models/language/tiny_bert.py
- 6 layers, 256 hidden, 8 heads
- Vocabulary: 10K tokens
- Total params: ~10M (fits in 40MB)
- Target: <2ms per token
```

### Step 3.2: Span Masking Strategy
```python
# sage/irp/plugins/language_impl.py
- Progressive unmasking (80% → 20%)
- Meaning latent from [CLS] token
- Stabilization detection via cosine similarity
- Early stop when meaning_delta < 0.05
```

### Step 3.3: Integration & Testing
```python
# demos/language_real.py
- Test on simple QA dataset
- Measure stabilization speed
- Track meaning preservation
- Target: <10 iterations to stabilize
```

### Success Criteria:
- [ ] Model inference < 2ms per token
- [ ] Meaning stabilizes in < 10 steps
- [ ] 3x speedup on average sequences
- [ ] Downstream task accuracy maintained

### Deliverables:
- `models/language/tiny_bert.py`
- `sage/irp/plugins/language_impl.py`
- `demos/language_real.py`
- `results/language_irp_metrics.json`

---

## Phase 4: HRM Orchestrator

### Step 4.1: Core Orchestration
```python
# sage/orchestrator/hrm_orchestrator.py
class HRMOrchestrator:
    - Plugin registry and lifecycle
    - Trust weight management
    - ATP budget allocation
    - Concurrent execution control
```

### Step 4.2: Budget Allocation
```python
# sage/orchestrator/budget.py
- Initial ATP: 1000 units per cycle
- Trust-weighted distribution
- Dynamic reallocation on early stops
- Value tracking per plugin
```

### Step 4.3: Async Execution
```python
# sage/orchestrator/async_manager.py
- asyncio-based plugin execution
- Progress monitoring
- Timeout handling
- Result aggregation
```

### Success Criteria:
- [ ] 3+ plugins running concurrently
- [ ] ATP properly distributed by trust
- [ ] Early stops trigger reallocation
- [ ] No deadlocks or race conditions

### Deliverables:
- `sage/orchestrator/hrm_orchestrator.py`
- `sage/orchestrator/budget.py`
- `sage/orchestrator/async_manager.py`
- `tests/test_orchestration.py`

---

## Phase 5: SNARC Memory Integration

### Step 5.1: Connect SNARC to IRP
```python
# memory_integration/irp_memory_bridge.py
- SNARC evaluates refinement trajectories
- High-salience patterns stored
- Circular buffer for recent context
- Verbatim storage for important states
```

### Step 5.2: Memory-Guided Refinement
```python
# sage/irp/plugins/memory_aware.py
- Use past refinements to guide current
- Skip low-value iterations
- Cache successful patterns
- Adaptive halting based on memory
```

### Step 5.3: Sleep Consolidation
```python
# demos/sleep_cycle.py
- Collect day's refinements
- Run consolidation during "sleep"
- Extract reusable patterns
- Update trust weights based on value
```

### Success Criteria:
- [ ] SNARC correctly filters refinements
- [ ] Memory improves convergence speed
- [ ] Sleep creates reusable patterns
- [ ] 20% faster refinement after sleep

### Deliverables:
- `memory_integration/irp_memory_bridge.py`
- `sage/irp/plugins/memory_aware.py`
- `demos/sleep_cycle.py`
- `results/memory_impact.json`

---

## Phase 6: Performance Optimization

### Step 6.1: Profiling
```bash
# Profile each component
nsys profile python demos/vision_real.py
nvprof python demos/language_real.py
py-spy record -o profile.svg -- python demos/orchestrator_demo.py
```

### Step 6.2: Optimization Targets
- Enable tensor cores for FP16
- Optimize memory transfers
- Reduce kernel launch overhead
- Implement gradient checkpointing

### Step 6.3: Batching Strategy
- Dynamic batching for similar tasks
- Memory-aware batch sizing
- Latency vs throughput tradeoff

### Success Criteria:
- [ ] FP16 faster than FP32
- [ ] Memory usage < 4GB peak
- [ ] Latency < 10ms for simple tasks
- [ ] Throughput > 100 tasks/second

### Deliverables:
- `optimization/profile_results/`
- `optimization/fp16_fixes.py`
- `optimization/batching_strategy.py`
- `benchmarks/optimized_results.json`

---

## Phase 7: Sleep Consolidation Demo

### Step 7.1: Experience Collection
```python
# demos/experience_collector.py
- Run various tasks throughout "day"
- Collect refinement trajectories
- Track success/failure patterns
- Build experience buffer
```

### Step 7.2: Consolidation Process
```python
# demos/sleep_consolidation_full.py
- Compress experiences via SNARC
- Extract common patterns
- Build pattern library
- Update model weights (H-level)
```

### Step 7.3: Morning Evaluation
```python
# demos/morning_eval.py
- Test on new tasks
- Compare with/without sleep
- Measure improvement
- Calculate value created
```

### Success Criteria:
- [ ] 50% compression of experiences
- [ ] Pattern library < 100MB
- [ ] 30% faster on morning tasks
- [ ] Clear value attribution

### Deliverables:
- `demos/full_day_cycle.py`
- `results/sleep_value_analysis.json`
- `models/consolidated_patterns.pt`
- Video demo of improvement

---

## Phase 8: Full Integration & Demo

### Step 8.1: End-to-End Pipeline
- Camera input → Vision IRP
- Microphone → Language IRP  
- Control commands → Control IRP
- All managed by orchestrator
- Memory tracking everything

### Step 8.2: Real-World Task
- Object detection + description
- Voice command processing
- Action planning and execution
- Continuous learning

### Step 8.3: Performance Dashboard
- Real-time telemetry display
- Energy consumption tracking
- Trust weight visualization
- Value creation metrics

### Success Criteria:
- [ ] Full pipeline < 30ms latency
- [ ] Stable for 1 hour runtime
- [ ] Clear value demonstration
- [ ] Ready for GR00T integration

### Deliverables:
- `demos/full_system_demo.py`
- `dashboard/realtime_monitor.py`
- Video of system in action
- Performance report

---

## Timeline

- **Today (Day 1)**: Phase 1 ✅ + Start Phase 2
- **Day 2-3**: Complete Phase 2 (Vision IRP)
- **Day 4-5**: Phase 3 (Language IRP)
- **Day 6**: Phase 4 (Orchestrator)
- **Day 7**: Phase 5 (SNARC Integration)
- **Day 8**: Phase 6 (Optimization)
- **Day 9**: Phase 7 (Sleep Demo)
- **Day 10**: Phase 8 (Full Integration)

## Next Immediate Step

Let's implement the lightweight VAE for vision (Phase 2.1):

```bash
# We'll create:
1. models/vision/lightweight_vae.py
2. tests/test_vae.py
3. Run initial tests
```

Ready to proceed with Phase 2?