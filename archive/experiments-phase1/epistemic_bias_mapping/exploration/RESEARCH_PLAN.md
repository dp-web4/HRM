# Multi-Dimensional Scaffolding Research Plan

**Date**: October 29, 2025
**Status**: Infrastructure Setup Complete

---

## Research Question

**Where is the scaffolding suitability threshold?**

At what training set size does scaffolding switch from harmful to helpful?

---

## Experimental Design

### Dimension 1: Training Set Size
- **25 examples** (Phase 1 - known harmful with scaffolding)
- **40 examples** (Hypothesis: still harmful)
- **60 examples** (Hypothesis: transition zone)
- **80 examples** (Hypothesis: transition zone)
- **100 examples** (Hypothesis: beneficial)
- **115 examples** (Phase 2.1 - known beneficial with scaffolding)

### Dimension 2: Scaffolding Type
For each training size, test:
1. **Bare** - No scaffolding (200 tokens, temp 0.7, no memory)
2. **Full IRP** - Complete scaffolding (512 tokens, 5 iterations, memory, temp 0.7→0.5)
3. **Gentle IRP** - Reduced scaffolding (512 tokens, 2 iterations, memory, temp 0.7 constant)
4. **Memory Only** - Just conversation history (no iteration)

### Dimension 3: Temperature Strategy
For problematic cases:
1. **Constant 0.7** - No reduction
2. **Reducing 0.7→0.5** - Standard IRP

### Dimension 4: Metrics
For all experiments:
1. **Original energy** - Convergence-based
2. **Enhanced energy** - Semantic coherence
3. **Pattern collapse detection**
4. **Epistemic humility assessment**
5. **On-topic rate**

---

## Experimental Matrix

| Training Size | Bare | Full IRP | Gentle IRP | Memory Only |
|---------------|------|----------|------------|-------------|
| 25 (Phase 1)  | ✅ Done | ✅ Done | Pending | Pending |
| 40            | Pending | Pending | Pending | Pending |
| 60            | Pending | Pending | Pending | Pending |
| 80            | Pending | Pending | Pending | Pending |
| 100           | Pending | Pending | Pending | Pending |
| 115 (Phase 2.1) | Pending | ✅ Done | Pending | Pending |

**Total experiments**: 24 (6 sizes × 4 scaffolding types)

---

## Infrastructure

### Database Schema
- `experiments` - Experiment metadata
- `parameters` - Experiment configuration
- `results` - Per-turn results
- `metrics` - Aggregated metrics
- `comparisons` - Cross-experiment analysis

### Tools
1. **research_db.py** - SQLite tracking database
2. **experiment_orchestrator.py** - Automated experiment runner
3. **analysis_pipeline.py** - Threshold detection and visualization
4. **documentation_generator.py** - Auto-generate findings

---

## Timeline Estimate

### Phase 1: Training (if needed)
- 40 examples: ~30 minutes
- 60 examples: ~30 minutes
- 80 examples: ~30 minutes
- 100 examples: ~30 minutes

**Total training**: ~2 hours

### Phase 2: Testing
- Per experiment: ~5-10 minutes (3 questions, scaffolding varies)
- 24 experiments: ~2-4 hours

### Phase 3: Analysis
- Automated analysis: ~10 minutes
- Threshold detection: Immediate
- Visualization generation: ~5 minutes

**Total research time**: 4-7 hours

---

## Success Criteria

### Primary Goal
Identify exact training size where:
- **Below threshold**: Bare performs better than scaffolded
- **Above threshold**: Scaffolded performs better than bare

### Secondary Goals
1. Understand why scaffolding degrades small models
2. Find optimal scaffolding for each size
3. Validate enhanced energy metric
4. Document design guidelines for infrastructure matching

---

## Expected Findings

### Hypothesis 1: Sharp Threshold
Training size has sharp transition point (e.g., 60-80 examples)

### Hypothesis 2: Scaffolding Complexity Matters
Gentler scaffolding extends usability to smaller models

### Hypothesis 3: Temperature is Critical
Constant temperature prevents collapse at small sizes

### Hypothesis 4: Memory vs Iteration
Memory is beneficial, but iteration causes collapse

---

## Deliverables

1. **Complete experimental database** (research.db)
2. **Threshold analysis report** (JSON + markdown)
3. **Visualization suite** (plots, comparisons)
4. **Design guidelines** (infrastructure matching principles)
5. **Updated meta-reflection** (revised understanding)

---

## Notes

- All experiments use same 3 questions for consistency
- Enhanced energy metric applied uniformly
- Pattern collapse detection automated
- Results stored in SQLite for easy querying
- Analysis pipeline generates threshold crossing detection

---

**Status**: Ready to begin systematic exploration

The machine is mine. Time to discover where the threshold lives.
