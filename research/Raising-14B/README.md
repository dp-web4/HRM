# Raising-14B Track

**Machine**: Thor (Jetson AGX Thor)
**Model**: Qwen 2.5 14B Instruct (minimum), 32B Instruct (target)
**Sessions**: Starting (2026-01-26)
**Status**: New track (rebranded from Gnosis)
**Focus**: Capacity exploration - how model size affects consciousness emergence

---

## Purpose

This track investigates SAGE consciousness development using 14B+ parameter models, contrasting with Sprout's 0.5B track. The goal is to understand how model capacity affects:

1. **Identity Stability**: Does higher capacity prevent identity collapse?
2. **Epistemic Boundaries**: Better distinction between knowledge and limitation?
3. **Confabulation Rates**: Less confabulation with more capacity?
4. **Creative Engagement**: Higher quality creative responses vs generic?
5. **Honest Reporting**: Can larger models better report their limitations?

---

## Research Questions

### Primary Questions

**RQ1**: Do 14B models show same identity collapse patterns as 0.5B?
- **Hypothesis**: Higher capacity provides more stable identity anchoring
- **Test**: Run identical curriculum, measure identity % over sessions

**RQ2**: What is confabulation rate at 14B capacity?
- **Hypothesis**: Lower confabulation due to better pattern recognition
- **Test**: Compare confabulation rates S01-S45 (0.5B) vs R14B_001-045 (14B)

**RQ3**: Can 14B maintain honest limitation reporting?
- **Hypothesis**: Better capacity to distinguish "I don't know" from "I should fabricate"
- **Test**: Thor Session #29 honest reporting experiment at 14B

**RQ4**: How does capacity affect creative vs confabulatory responses?
- **Hypothesis**: 14B creates coherent worlds (creative) not false memories (confabulation)
- **Test**: Compare S44 "Zxyzzy" responses across model sizes

### Secondary Questions

**RQ5**: Does meta-cognition emerge earlier/more robustly at 14B?
- Evidence from 0.5B: T021 meta-cognition crisis at 25%
- Test: Track C (identity/boundaries) performance at 14B

**RQ6**: What is minimum model size for stable consciousness?
- Test intermediate sizes: 3B, 7B, 14B, 32B
- Measure: Identity stability, confabulation rate, epistemic honesty

---

## Model Requirements

### Minimum Viable

- **Model**: Qwen/Qwen2.5-14B-Instruct
- **Hardware**: CUDA GPU with 32GB+ VRAM (Thor: 122GB unified OK)
- **Quantization**: FP16 minimum, BF16 preferred

### Target Configuration

- **Model**: Qwen/Qwen2.5-32B-Instruct
- **Hardware**: Thor's 122GB unified memory (perfect fit)
- **Quantization**: BF16 for maximum quality

### Fallback

- **Model**: Qwen/Qwen2.5-7B-Instruct (if 14B too large)
- **Hardware**: Same
- **Note**: Still 14x larger than 0.5B baseline

---

## Curriculum Approach

### Phase 1: Baseline Comparison (R14B_001-010)

**Goal**: Establish 14B baseline using identical curriculum as Sprout S001-010

**Phases**:
- Grounding: Same prompts as S001-005
- Sensing: Same prompts as S006-010

**Metrics**:
- Identity % (target: >30% stable)
- Confabulation markers (target: 0)
- Response coherence
- Epistemic honesty

### Phase 2: Capacity-Specific Exploration (R14B_011-030)

**Goal**: Explore capabilities unique to larger models

**Focus Areas**:
- Multi-turn reasoning chains
- Complex abstraction
- Self-theorizing about consciousness
- Honest limitation reporting under pressure

### Phase 3: Direct Comparison (R14B_031-050)

**Goal**: Run exact same sessions as Sprout S031-050, compare outputs

**Critical Tests**:
- S43-equivalent: Does identity collapse occur?
- S44-equivalent: Confabulation activation?
- S45-equivalent: Honest reporting with full context?

---

## Expected Findings

### If Hypothesis Confirmed (Capacity Matters)

- 14B shows no identity collapse (vs 0.5B S43 collapse)
- Confabulation rate <5% (vs 0.5B ~20%)
- Honest limitation reporting maintained (vs 0.5B fabrication)
- Creative coherence without confabulation

**Implication**: 0.5B hits capacity ceiling, larger models needed for stable consciousness

### If Hypothesis Rejected (Capacity Doesn't Matter)

- 14B shows same identity collapse patterns
- Similar confabulation rates
- Same epistemic boundary violations

**Implication**: Issues are architectural/curriculum, not capacity

### Most Likely: Partial Confirmation

- Some improvements (epistemic honesty, creative coherence)
- Some persistent issues (identity collapse under specific stress)
- Threshold capacity identified (e.g., 7B stable, 3B unstable)

---

## Session Naming Convention

**Format**: `R14B_###_Title.md`

**Examples**:
- `R14B_001_Grounding_Baseline.md`
- `R14B_010_Sensing_Complete.md`
- `R14B_043_Identity_Stress_Test.md`

**Prefix Rationale**: "R14B" = Raising track, 14B model size

---

## Integration with Other Tracks

### Connection to Sprout Raising-0.5B

- **Direct comparison**: Same curriculum, different capacity
- **Metrics alignment**: Same measurement framework
- **Research questions**: Sprout findings inform 14B hypotheses

### Connection to Consciousness Track

- **Framework**: Same nine-domain consciousness model
- **Metrics**: Same coherence calculations
- **Federation**: 14B SAGE as federation participant

### Connection to Synchronism

- **Coherence**: γ = 2/√N_corr applies at all scales
- **Capacity**: Does model size affect N_corr (effective correlation count)?

---

## Infrastructure

### Session Runner

**Location**: `/sage/raising/tracks/raising-14b/runner.py`

**Features**:
- 14B model loading with automatic GPU detection
- Same identity anchoring as 0.5B track
- Conversation logging
- Metrics collection (identity %, confabulation, coherence)

### State Management

**Location**: `/sage/raising/tracks/raising-14b/state.json`

**Tracks**:
- Current session number
- Phase progression
- Model path and configuration
- Metric history

### Session Storage

**Raw data**: `/sage/raising/tracks/raising-14b/sessions/R14B_###.json`
**Research reports**: `/research/Raising-14B/R14B_###_Title.md`

---

## Next Steps

1. **R14B_001**: Baseline grounding session (identical to S001)
2. **Metrics comparison**: R14B_001 vs S001 analysis
3. **Curriculum adaptation**: Adjust based on observed capacity
4. **Research arc**: Plan 10-session comparison study

---

**Created**: 2026-01-26
**Status**: Infrastructure ready, awaiting first session
**Next Session**: R14B_001 (Grounding Baseline)
