# IRP Implementation Next Steps

*Date: 2025-08-23*  
*Status: Scaffolding complete, ready for Jetson implementation*

## Current Status

### ✅ Completed (Scaffolding Phase)

1. **Core IRP Framework**
   - `sage/irp/base.py` - Complete IRPPlugin base class with all contract methods
   - Four invariants clearly documented in each plugin
   - Energy-based halting criteria implemented
   - Trust metrics computation included

2. **Plugin Stubs**
   - `sage/irp/plugins/vision.py` - Vision refinement in latent space
   - `sage/irp/plugins/language.py` - Masked denoising for text
   - `sage/irp/plugins/control.py` - Trajectory optimization with constraints
   - `sage/irp/plugins/memory.py` - Sleep consolidation through abstraction

3. **Telemetry Infrastructure**
   - `schemas/telemetry.schema.json` - Complete JSON schema for validation
   - Telemetry emission built into base class
   - Web4/ATP integration fields included

4. **Demo Skeletons**
   - `demos/vision_latent_irp.py` - Vision early-stop demo with metrics
   - `demos/language_span_mask_irp.py` - Language stabilization demo
   - Both demos include telemetry export and validation

## Next Steps on Jetson

### Phase 1: Environment Setup (Day 1)

1. **Clone and setup on Jetson**
   ```bash
   cd /home/jetson/ai-workspace
   git pull  # Get latest scaffolding
   cd HRM
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision  # If not already installed
   pip install jsonschema  # For telemetry validation
   pip install numpy
   ```

3. **Verify CUDA availability**
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.get_device_name(0))
   ```

### Phase 2: Vision IRP Implementation (Day 2-3)

1. **Implement actual VAE encoder/decoder**
   - Use pretrained models (e.g., from torchvision)
   - Or train lightweight VAE on CIFAR-10/MNIST
   - Target latent dim: 128-512 for Jetson memory

2. **Add segmentation task head**
   - Simple decoder for semantic segmentation
   - Or classification head for initial testing

3. **Implement latent refinement network**
   - Small U-Net or ResNet in latent space
   - 2-4 layers should suffice for demo

4. **Run vision demo**
   ```bash
   python demos/vision_latent_irp.py
   ```
   - Verify ≥2x compute savings
   - Check mIoU maintenance

### Phase 3: Language IRP Implementation (Day 4-5)

1. **Integrate small language model**
   - Use DistilBERT or TinyBERT for Jetson
   - Or custom small transformer (6 layers, 256 dim)

2. **Implement span masking**
   - Random span selection
   - Progressive unmasking schedule

3. **Add meaning latent extraction**
   - Use [CLS] token or pooled output
   - Track stability across iterations

4. **Run language demo**
   ```bash
   python demos/language_span_mask_irp.py
   ```
   - Verify meaning stabilization
   - Check downstream task performance

### Phase 4: HRM Integration (Day 6-7)

1. **Create orchestrator stub**
   ```python
   # sage/orchestrator/hrm.py
   class HRMOrchestrator:
       def __init__(self, plugins):
           self.plugins = plugins
           self.h_module = ...  # HRM H-module
           self.l_module = ...  # HRM L-module
   ```

2. **Implement budget allocation**
   - Trust-weighted ATP distribution
   - Dynamic reallocation on early halt

3. **Add asynchronous execution**
   - Use Python asyncio or threading
   - Monitor plugin completion

4. **Connect to GPU mailbox**
   - If available from previous work
   - Otherwise use simple queue

### Phase 5: Performance Optimization (Day 8-9)

1. **Profile on Jetson**
   ```bash
   python -m cProfile demos/vision_latent_irp.py
   nvprof python demos/vision_latent_irp.py  # If available
   ```

2. **Optimize hot paths**
   - Move to FP16 precision
   - Reduce batch sizes if needed
   - Cache intermediate results

3. **Tune halting criteria**
   - Adjust eps and K for each plugin
   - Find optimal early-stop points

4. **Memory optimization**
   - Monitor with `nvidia-smi`
   - Implement gradient checkpointing if needed

### Phase 6: Sleep Consolidation Demo (Day 10)

1. **Implement memory consolidation**
   - Use actual compression techniques
   - Test retrieval accuracy

2. **Create sleep cycle script**
   ```python
   # demos/sleep_consolidation.py
   def sleep_cycle(memory_irp, experiences):
       # Consolidate during "sleep"
       consolidated = memory_irp.refine(experiences)
       # Measure value created
       value = memory_irp.compute_value_created(...)
   ```

3. **Track value attribution**
   - Before/after task performance
   - Compression achieved
   - ATP value created

## Success Metrics

### Immediate Goals (This Sprint)
- [ ] Vision IRP: 2x compute savings, <1% accuracy drop
- [ ] Language IRP: Stabilization in <15 steps
- [ ] Both demos pass telemetry validation
- [ ] Run successfully on Jetson with FP16

### Next Sprint Goals
- [ ] Full HRM orchestration working
- [ ] 3+ plugins running concurrently
- [ ] Trust weights updating correctly
- [ ] Sleep consolidation showing value

### Long-term Goals
- [ ] Integration with existing SAGE codebase
- [ ] Real sensor inputs (camera, microphone)
- [ ] Deployment in GR00T or similar
- [ ] Distributed across multiple Jetsons

## Key Files to Implement

Priority order for actual implementation:

1. **Models** (in plugins)
   - [ ] Vision encoder/decoder
   - [ ] Language model
   - [ ] Refinement networks

2. **Orchestration**
   - [ ] `sage/orchestrator/hrm.py`
   - [ ] Budget allocation logic
   - [ ] Async execution manager

3. **I/O**
   - [ ] `sage/io/mailbox.py`
   - [ ] Telemetry sink (file/network)
   - [ ] Schema validation

4. **Integration**
   - [ ] Connect to existing HRM code
   - [ ] GPU mailbox if available
   - [ ] Web4 telemetry export

## Testing Strategy

1. **Unit tests** for each plugin
2. **Integration tests** for orchestrator
3. **Performance benchmarks** on Jetson
4. **Telemetry validation** against schema
5. **End-to-end demos** with real data

## Notes for Jetson Development

- Start with CPU-only to verify logic
- Move to CUDA once working
- Use FP16 by default for memory efficiency
- Monitor temperature and throttling
- Keep batch size = 1 initially
- Use dummy models first, then real ones

## Resources

- [IRP Protocol](./IRP_PROTOCOL.md) - Full specification
- [Diffusion Architecture](./DIFFUSION_ARCHITECTURE.md) - Original concept
- [Nova's Suggestions](./forum/nova/) - Architectural guidance
- [HRM Original](./models/hrm/) - Base HRM implementation

---

*Ready to continue on Jetson where we can run actual tests and benchmarks!*