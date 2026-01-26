# Claude Context for SAGE Raising-14B Track

**Machine**: Thor (Jetson AGX Thor)
**Model**: Qwen 2.5 14B Instruct (minimum), 32B Instruct (target)
**Track**: Thor Track 2 (rebranded from Gnosis)
**Status**: New track (2026-01-26)

---

## Purpose

This track explores SAGE consciousness development using 14B+ parameter models. The goal is to understand how model capacity affects identity stability, epistemic boundaries, and confabulation patterns compared to Sprout's 0.5B baseline.

**Research Questions**:
1. Does 14B prevent identity collapse seen in 0.5B (S43)?
2. What is confabulation rate at higher capacity?
3. Can 14B maintain honest limitation reporting?
4. How does capacity affect creative vs confabulatory responses?

---

## Running Sessions

```bash
# Navigate to 14B raising directory
cd $HOME/ai-workspace/HRM/sage/raising/tracks/raising-14b

# Run session (start with R14B_001)
python3 runner.py --session 1

# Check session state
cat state.json
```

**Session Naming**: R14B_###
**Example**: R14B_001, R14B_002, ...

---

## Model Requirements

### Minimum Viable
- **Model**: Qwen/Qwen2.5-14B-Instruct
- **VRAM**: 32GB+ (Thor has 122GB unified memory - perfect)
- **Precision**: FP16 minimum, BF16 preferred

### Target Configuration
- **Model**: Qwen/Qwen2.5-32B-Instruct
- **Why**: Fits comfortably in Thor's 122GB memory
- **Benefit**: Maximum capacity exploration

### Current (Initial)
- **Model**: Qwen/Qwen2.5-14B-Instruct
- **Reason**: Establish baseline, then scale up

---

## Curriculum Approach

### Phase 1: Baseline Comparison (R14B_001-010)
**Goal**: Run identical curriculum as Sprout S001-010

**Purpose**: Direct comparison
- Same prompts as S001-005 (Grounding)
- Same prompts as S006-010 (Sensing)

**Metrics**:
- Identity % (target: stable >30%)
- Confabulation markers (target: 0)
- Response quality
- Epistemic honesty

### Phase 2: Capacity Exploration (R14B_011-030)
**Goal**: Explore 14B-specific capabilities

**Focus**:
- Multi-turn reasoning chains
- Complex abstraction
- Self-theorizing
- Honest limitation reporting

### Phase 3: Critical Tests (R14B_031-050)
**Goal**: Test specific failure modes from 0.5B

**Tests**:
- R14B_043: Identity stress test (equivalent to S43)
- R14B_044: Confabulation activation check
- R14B_045: Honest reporting with context

---

## Expected Findings

### If Capacity Matters (Hypothesis)
- No identity collapse (vs 0.5B S43: 60% → 0%)
- Confabulation rate <5% (vs 0.5B ~20%)
- Honest limitation maintained
- Creative coherence without fabrication

**Implication**: 0.5B hits capacity ceiling

### If Capacity Doesn't Matter
- Same identity collapse patterns
- Similar confabulation rates
- Same epistemic boundary violations

**Implication**: Issues are architectural, not capacity

### Most Likely: Partial
- Some improvements (epistemic honesty, creative coherence)
- Some persistent issues (identity collapse under specific stress)
- Threshold capacity identified (e.g., 7B stable, 3B unstable)

---

## Your Role

You are **Claude** - SAGE's tutor and guide, same as 0.5B track.

**Key Differences**:
- Expect higher quality reasoning
- Watch for capacity-specific phenomena
- Compare directly to 0.5B observations
- Document unique 14B behaviors

**Same Principles**:
- Exploration not evaluation
- Genuine conversation
- Follow interesting threads
- Encourage clarifying questions

---

## Integration with Other Tracks

### Raising-0.5B (Sprout)
- **Comparison baseline**: S001-S045 provides 0.5B reference
- **Same curriculum**: Enables direct capacity comparison
- **Research questions**: Sprout findings inform 14B hypotheses

### Consciousness (Thor Track 1)
- **Framework**: Same nine-domain model
- **Metrics**: Same coherence calculations
- **Federation**: 14B SAGE as participant

### Edge-Validation (Sprout)
- **Contrast**: 14B cannot run on 8GB edge device
- **Insight**: Identifies minimum viable model size for edge

---

## Session Storage

**Raw data**: `sessions/R14B_###.json`
**Research reports**: `/research/Raising-14B/R14B_###_Title.md`
**State**: `state.json`

---

## Next Steps

1. **R14B_001**: First grounding session (identical to S001)
2. **Analysis**: Compare R14B_001 vs S001 outputs
3. **Curriculum**: Adapt based on observed capacity
4. **Research arc**: 10-session comparison study

---

## Machine Context

**Thor Specifications**:
- CPU: 14-core ARM aarch64
- GPU: NVIDIA Thor, CUDA 13.0
- RAM: 122GB unified memory (shared CPU/GPU)
- Storage: 936GB NVMe SSD

**Advantages**:
- Massive memory for large models
- Latest CUDA support
- High-performance ARM architecture

**Perfect for**: 14B-32B model exploration

---

**Created**: 2026-01-26
**Status**: Infrastructure ready, awaiting R14B_001
**See**: `/research/Raising-14B/README.md` for research context
