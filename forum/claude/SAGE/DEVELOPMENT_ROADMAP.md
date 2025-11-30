# SAGE Development Roadmap

**Purpose**: Practical guide for implementing SAGE architecture  
**Status**: Development phases and priorities  
**For**: Claude-code instances ready to build

---

## Current State

### What's Working ✅
- Coherence Engine framework (`/ai-dna-discovery/coherence-engine/`)
- Basic IRP plugin architecture
- Trust evolution system
- Context state management (STABLE, MOVING, UNSTABLE, NOVEL)
- Multiple platform support (Jetson, Legion)

### What's In Progress 🚧
- Full KEY/VALUE separation in IRP outputs
- TSM-inspired sparse input layers
- Topological inter-module routing
- SNARC attention tensor implementation

### What's Planned 📋
- HRM integration for small model training
- Memory distiller for experience consolidation
- Multi-scale hierarchical SNARC
- Learned sparse projections

---

## Development Phases

### Phase 1: Foundation (Current) 🎯

**Goal**: Establish core K/V architecture with working IRP plugins

**Priority Tasks**:

1. **Refactor IRP Outputs for K/V Split**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/
   
   Tasks:
   - [ ] Create SalienceKey class
   - [ ] Create FeatureValue class  
   - [ ] Create IRPOutput class
   - [ ] Update all existing IRP plugins
   - [ ] Add K/V validation tests
   
   Files to modify:
   - core/plugins/base_plugin.py
   - plugins/jetson/*.py
   - plugins/legion/*.py
   ```

2. **Implement TSM Sparse Layers**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/layers/
   
   Tasks:
   - [ ] Create topographical_sparse.py
   - [ ] Implement TopographicalSparseLayer
   - [ ] Add unit tests
   - [ ] Document sparsity patterns
   - [ ] Benchmark vs dense layers
   
   Reference: /HRM/forum/claude/TSM_IMPLEMENTATION.md
   ```

3. **Build SNARC Tensor**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/snarc.py
   
   Tasks:
   - [ ] Implement SNARCTensor class
   - [ ] Add attention update logic
   - [ ] Add coherence computation
   - [ ] Add temporal dynamics tracking
   - [ ] Integration tests with IRP keys
   
   Reference: /HRM/forum/claude/SNARC_SPECIFICATION.md
   ```

**Success Criteria**:
- All IRP plugins generate both K and V
- SNARC updates from salience keys
- No VALUE data enters SAGE orchestration layer
- Tests pass for K/V separation

**Estimated Duration**: 2-3 weeks

---

### Phase 2: Integration (Next) 🔄

**Goal**: Connect SAGE orchestration with value routing

**Priority Tasks**:

1. **Inter-Module Communication Bus**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/routing.py
   
   Tasks:
   - [ ] Create InterModuleBus class
   - [ ] Implement topological routing graph
   - [ ] Add attention-gated routing
   - [ ] Create sparse projection layers
   - [ ] Add routing tests
   
   Key Features:
   - Structured sparse routing (not all-to-all)
   - Attention-based gating
   - TSM-inspired sparse projections
   ```

2. **SAGE Attention Network**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/sage.py
   
   Tasks:
   - [ ] Implement AttentionNetwork (neural network)
   - [ ] Add SNARC state integration
   - [ ] Create attention allocation logic
   - [ ] Add context transition detection
   - [ ] Training pipeline for attention network
   ```

3. **Memory Integration**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/memory.py
   
   Tasks:
   - [ ] Create MemoryConvergenceLayer
   - [ ] Implement as both IRP and module
   - [ ] Add retrieval network
   - [ ] Create experience storage
   - [ ] Add memory-based salience generation
   ```

**Success Criteria**:
- VALUES route between modules based on SAGE attention
- Memory receives convergent inputs from all modules
- Context transitions trigger correctly
- Complete tick loop works end-to-end

**Estimated Duration**: 3-4 weeks

---

### Phase 3: Learning (Future) 🧠

**Goal**: Enable learning and adaptation

**Priority Tasks**:

1. **HRM Integration**
   ```
   Location: /HRM/
   
   Tasks:
   - [ ] Adapt HRM for SAGE attention network training
   - [ ] Create training data from IRP outputs
   - [ ] Implement curriculum learning
   - [ ] Add evaluation metrics
   - [ ] Train initial SAGE model (30M params)
   
   Key Insight:
   - HRM trains small models efficiently
   - SAGE attention network is ~30M params
   - Can learn from simulated + real data
   ```

2. **Memory Distiller**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/distiller.py
   
   Tasks:
   - [ ] Create sleep cycle architecture
   - [ ] Implement experience replay
   - [ ] Add coherence pattern extraction
   - [ ] Create synthetic scenario generation
   - [ ] Integration with HRM training
   
   Biological Parallel:
   - Sleep consolidates memory
   - Distiller extracts patterns
   - Generates training data for SAGE
   ```

3. **Learned Sparse Projections**
   ```
   Location: /ai-dna-discovery/coherence-engine/core/layers/learned_sparse.py
   
   Tasks:
   - [ ] Extend TSM layers with learning
   - [ ] Implement adaptive receptive fields
   - [ ] Add projection learning between modules
   - [ ] Train end-to-end
   - [ ] Evaluate vs fixed topology
   ```

**Success Criteria**:
- SAGE learns better attention policies from experience
- Memory distiller improves over time
- Sparse projections adapt to task
- System performance increases with training

**Estimated Duration**: 4-6 weeks

---

### Phase 4: Scale & Optimize (Advanced) ⚡

**Goal**: Tested and validated, efficient, robust

**Priority Tasks**:

1. **Multi-Scale SNARC**
   ```
   Tasks:
   - [ ] Implement HierarchicalSNARC
   - [ ] Add fast/medium/slow timescales
   - [ ] Cross-scale coherence detection
   - [ ] Hierarchical attention allocation
   ```

2. **Hardware Optimization**
   ```
   Tasks:
   - [ ] GPU acceleration for TSM layers
   - [ ] Sparse tensor operations
   - [ ] Quantization (INT8/FP16)
   - [ ] Edge deployment (Jetson optimization)
   ```

3. **Robustness & Testing**
   ```
   Tasks:
   - [ ] Comprehensive test suite
   - [ ] Failure mode analysis
   - [ ] Recovery mechanisms
   - [ ] Adversarial testing
   - [ ] Continuous integration
   ```

**Success Criteria**:
- Real-time performance on target hardware
- Robust to sensor failures
- Efficient memory and compute usage
- Production-quality codebase

**Estimated Duration**: 4-6 weeks

---

## Implementation Priorities

### Critical Path 🔴

These must be done first, in order:

1. K/V separation in IRP outputs
2. SNARC tensor implementation
3. Inter-module routing
4. End-to-end tick loop

### High Priority 🟡

Important but can be done in parallel with critical path:

1. TSM sparse layers
2. Attention network architecture
3. Memory convergence layer
4. Trust evolution refinement

### Medium Priority 🟢

Nice to have, can wait until later:

1. Learned sparse projections
2. Multi-scale SNARC
3. Hardware optimization
4. Advanced memory features

---

## Quick Start Guide

### For New Claude-Code Instance

1. **Read documentation** (30 minutes)
   ```
   /HRM/forum/claude/SAGE_QUICK_REFERENCE.md  → Overview
   /HRM/forum/claude/SAGE_ARCHITECTURE.md     → Deep dive
   /HRM/forum/claude/TSM_IMPLEMENTATION.md    → Sparse layers
   /HRM/forum/claude/SNARC_SPECIFICATION.md   → Attention state
   ```

2. **Review existing code** (1 hour)
   ```
   /ai-dna-discovery/coherence-engine/core/engine.py      → Main loop
   /ai-dna-discovery/coherence-engine/core/context.py     → Context states
   /ai-dna-discovery/coherence-engine/plugins/           → IRP examples
   ```

3. **Choose a task** (from Phase 1 critical path)
   - Start with something concrete and testable
   - Create branch: `sage-dev/your-task-name`
   - Write tests first
   - Implement
   - Document

4. **Test and integrate**
   ```bash
   # Run tests
   python -m pytest tests/
   
   # Test on hardware if available
   python run_jetson.py  # or run_legion.py
   
   # Document changes
   git commit -m "feat: descriptive message"
   ```

---

## Testing Strategy

### Unit Tests
- Each component tested in isolation
- Mock IRP outputs for SNARC testing
- Validate K/V separation
- Test sparse layer properties

### Integration Tests
- Full tick loop end-to-end
- Multiple IRPs → SAGE → routing → modules
- Context transitions
- Memory integration

### System Tests
- Real sensor data (camera, IMU, etc.)
- Performance benchmarks
- Failure recovery
- Long-running stability

### Hardware Tests
- Jetson Nano (edge device)
- Legion (RTX 4090)
- Real-time performance
- Power consumption

---

## Code Quality Standards

### Required for All Code

1. **Type hints**
   ```python
   def process(self, input: torch.Tensor, tick: int) -> IRPOutput:
       """Clear type signatures"""
   ```

2. **Docstrings**
   ```python
   def compute_salience(self, features: FeatureValue) -> SalienceKey:
       """
       Extract salience metadata from features
       
       Args:
           features: Dense feature representation
       
       Returns:
           Compressed salience key for SAGE
       """
   ```

3. **Tests**
   ```python
   def test_snarc_updates_from_keys():
       """SNARC should update attention from salience keys"""
       snarc = SNARCTensor(num_modules=3)
       keys = {...}  # Mock keys
       snarc.update(keys)
       assert snarc.attention_weights.sum() == 1.0
   ```

4. **Documentation**
   - Update relevant .md files when architecture changes
   - Add code comments for complex logic
   - Document assumptions and limitations

---

## Key Decisions

### Architecture Decisions

1. **K/V separation is non-negotiable**
   - SAGE only sees KEYS
   - Modules process VALUES
   - Clean separation enforced by types

2. **TSM principles for VALUE processing**
   - Topographical sparse input layers
   - Structured (not random) connectivity
   - No dense pretraining

3. **Fractal organization**
   - SAGE = orchestration (Level 2)
   - Inter-module = routing (Level 1)
   - IRP = sensing (Level 0)

### Implementation Decisions

1. **Start simple, add complexity**
   - Basic SNARC before hierarchical
   - Fixed topology before learned
   - Manual routing before learned routing

2. **Test-driven development**
   - Write tests first
   - Run tests often
   - Maintain test coverage

3. **Document as you go**
   - Update .md files with discoveries
   - Comment non-obvious code
   - Share insights in forum/

---

## Communication

### Where to Document

- **Architecture changes**: Update SAGE_ARCHITECTURE.md
- **Implementation patterns**: Update TSM_IMPLEMENTATION.md or SNARC_SPECIFICATION.md
- **Discoveries/insights**: Create new .md in /HRM/forum/claude/
- **Code specifics**: Inline comments + docstrings
- **Questions/blockers**: Create QUESTIONS.md

### Code Review Checklist

Before committing:
- [ ] Tests pass
- [ ] Type hints correct
- [ ] Docstrings present
- [ ] K/V separation maintained
- [ ] Documentation updated
- [ ] No VALUES in SAGE code
- [ ] TSM principles followed (if applicable)

---

## Resources

### Internal Docs
- `/HRM/forum/claude/` - All architecture documentation
- `/ai-dna-discovery/coherence-engine/` - Current implementation
- `/HRM/` - Training framework

### External References
- TSM Paper (uploaded) - Topographical sparse mapping
- Attention mechanisms - Transformer literature
- Coherence Engine README - Current system docs

### Getting Help
- Review past conversations about SAGE
- Check SAGE_ARCHITECTURE.md for context
- Look at existing IRP plugins for patterns
- Ask questions in code comments

---

## Success Metrics

### Phase 1 Complete When:
- [ ] All IRPs generate {K, V} outputs
- [ ] SNARC updates from keys
- [ ] No VALUE leakage to SAGE
- [ ] Tests green

### Phase 2 Complete When:
- [ ] Full tick loop works
- [ ] Values route based on attention
- [ ] Context transitions trigger
- [ ] Memory integrates all sources

### Phase 3 Complete When:
- [ ] SAGE learns from experience
- [ ] Memory distiller consolidates
- [ ] Performance improves over time
- [ ] HRM training pipeline works

### Overall Success:
- Real-time performance on target hardware
- Biological plausibility maintained
- K/V separation clean
- Code quality high
- Documentation complete

---

**Ready to start? Pick a task from Phase 1 and dive in!**

See SAGE_ARCHITECTURE.md for full context.
