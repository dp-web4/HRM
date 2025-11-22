# HierarchicalMemory Integration Plan

**Created**: 2025-11-21 10:15 PM PST (Auto Session #15)
**Status**: Ready for implementation
**Estimated Time**: 2-3 hours
**Priority**: Next major Michaud enhancement (after EmotionalEnergy complete)

---

## Overview

Integrate the existing `sage/memory/hierarchical_memory.py` (581 lines) into CogitationSAGE to enable cross-session learning through three-level memory hierarchy:

1. **Experiences** - Specific observations (SNARC-selected, high salience)
2. **Patterns** - Generalizations from clustered experiences
3. **Concepts** - Abstract relationships between patterns

This is the final major Michaud enhancement for cross-session conceptual learning.

---

## Current State

### ✅ What's Complete

1. **HierarchicalMemory implementation exists**: `sage/memory/hierarchical_memory.py`
   - 581 lines, production-ready
   - Three-level architecture (Experience → Pattern → Concept)
   - Latent space indexing for fast retrieval
   - Automatic pattern extraction from experiences
   - Concept formation from pattern clusters

2. **All four previous Michaud enhancements operational**:
   - AttentionManager (metabolic states)
   - Satisfaction-based consolidation
   - Identity-grounded cogitation
   - EmotionalEnergy modulation

3. **Integration targets identified**:
   - `sage/core/sage_consciousness_cogitation.py` - main integration
   - `sage/experiments/test_cogitation_integration.py` - test updates

---

## Architecture Analysis

### HierarchicalMemory Class API

**Key Methods**:
```python
# Core operations
store_experience(observation, salience, latent, plugin, energy, context) -> str
recall_similar(query_latent, k=5, plugin=None) -> List[Dict]
get_patterns_for_situation(query_latent, k=3, plugin=None) -> List[Pattern]
update_pattern_success(pattern_id, success)

# Automatic processing
_update_patterns_for_experience(exp_id, latent, plugin)
_create_pattern(exp_ids, plugin)
_update_pattern(pattern_id, exp_ids)
_update_concepts_for_pattern(pattern_id)

# Stats
get_stats() -> Dict
```

**Configuration Parameters**:
```python
{
    'experience_salience_threshold': 0.6,  # Only store high-salience
    'pattern_min_cluster_size': 3,         # Min experiences for pattern
    'pattern_max_distance': 0.5,           # Clustering threshold
    'concept_min_patterns': 2,             # Min patterns for concept
    'max_experiences': 10000,              # Memory limit
    'pattern_update_frequency': 10         # Update patterns every N experiences
}
```

---

## Integration Steps

### Step 1: Initialize HierarchicalMemory in CogitationSAGE (15 min)

**File**: `sage/core/sage_consciousness_cogitation.py`

**Changes needed**:

1. **Add import**:
```python
from sage.memory.hierarchical_memory import HierarchicalMemory
```

2. **Initialize in `__init__`** (after emotional tracker):
```python
# Hierarchical memory (cross-session learning)
self.hierarchical_memory = HierarchicalMemory({
    'experience_salience_threshold': 0.6,
    'pattern_min_cluster_size': 3,
    'pattern_max_distance': 0.5,
    'concept_min_patterns': 2,
    'max_experiences': 10000,
    'pattern_update_frequency': 10
})
self.experience_count = 0

print(f"  Hierarchical memory enabled: True")
```

---

### Step 2: Store Experiences in Main Loop (30 min)

**File**: `sage/core/sage_consciousness_cogitation.py`

**Where**: In `_execute_llm_michaud()` after emotional tracking

**Logic**:
1. Check if exchange is salient enough (SNARC selected it)
2. Get latent representation (need VAE encoding)
3. Store as experience with context
4. Automatically triggers pattern/concept formation

**Implementation**:

```python
# After emotional tracking, before return

# Store high-salience exchanges in hierarchical memory
if observation.get('salience', 0) >= 0.6:  # Match threshold
    # Get latent representation
    # TODO: Need VAE encoding of (question, response) pair
    # For now, use placeholder - proper integration needs VAE
    latent = torch.randn(64)  # Placeholder 64D latent

    exp_id = self.hierarchical_memory.store_experience(
        observation=None,  # Don't store raw data (too large)
        salience=observation.get('salience', 0.0),
        latent=latent,
        plugin='llm',  # From LLM plugin
        energy=final_energy,
        context={
            'question': question[:100],  # Truncated for memory
            'quality': 1.0 - final_energy,
            'cycle': self.cycle_count,
            'cogitation_verified': verification_performed,
            'emotions': emotions
        }
    )

    self.experience_count += 1

    print(f"[MEMORY] Stored experience {exp_id} (salience={observation.get('salience', 0):.3f})")

    # Check for new patterns/concepts
    mem_stats = self.hierarchical_memory.get_stats()
    if mem_stats['patterns_count'] > prev_patterns:
        print(f"[MEMORY] New pattern formed! Total patterns: {mem_stats['patterns_count']}")
    if mem_stats['concepts_count'] > prev_concepts:
        print(f"[MEMORY] New concept formed! Total concepts: {mem_stats['concepts_count']}")
```

**Note**: Proper implementation needs VAE encoding of QA pairs. This is a known limitation that can be addressed later.

---

### Step 3: Recall Similar Experiences (Optional Enhancement, 30 min)

**File**: `sage/core/sage_consciousness_cogitation.py`

**Where**: Before LLM execution in `_execute_llm_michaud()`

**Purpose**: Retrieve similar past experiences to inform current response

**Implementation**:

```python
# Before "Generate initial response with IRP refinement"

# Recall similar experiences (if we have latent for current question)
similar_experiences = []
if hasattr(observation, 'latent') and self.experience_count > 0:
    similar = self.hierarchical_memory.recall_similar(
        query_latent=observation['latent'],
        k=3,
        plugin='llm'
    )
    similar_experiences = similar

    if similar:
        print(f"[MEMORY] Recalled {len(similar)} similar experiences")
        # Could inject into prompt context (future enhancement)
```

---

### Step 4: Add Memory Statistics Method (15 min)

**File**: `sage/core/sage_consciousness_cogitation.py`

**Add method** (after `get_emotional_stats()`):

```python
def get_memory_stats(self) -> Dict:
    """Get hierarchical memory statistics."""
    return self.hierarchical_memory.get_stats()
```

---

### Step 5: Update Test File (30 min)

**File**: `sage/experiments/test_cogitation_integration.py`

**Changes**:

1. **Add memory stats to summary output** (after emotional stats):

```python
memory_stats = sage.get_memory_stats()
print(f"\nHierarchical Memory Statistics:")
print(f"  Experiences: {memory_stats['experiences_count']}")
print(f"  Patterns: {memory_stats['patterns_count']}")
print(f"  Concepts: {memory_stats['concepts_count']}")
if memory_stats['experiences_count'] > 0:
    print(f"  Avg pattern stability: {memory_stats.get('avg_pattern_stability', 0):.2f}")
```

2. **Add memory stats to return dict**:

```python
return {
    'name': 'Cogitation Loop',
    'avg_quality': avg_quality,
    'avg_identity': avg_identity,
    'avg_salience': stats['avg_salience'],
    'capture_rate': stats['capture_rate'],
    'cogitation_stats': cogitation_stats,
    'emotional_stats': emotional_stats,
    'memory_stats': memory_stats,  # NEW
    'results': results
}
```

3. **Add memory section to final output**:

```python
print(f"\n🧠 Hierarchical Memory (NEW):")
mem = cogitation_summary['memory_stats']
print(f"  Experiences stored: {mem['experiences_count']}")
print(f"  Patterns formed: {mem['patterns_count']}")
print(f"  Concepts emerged: {mem['concepts_count']}")
if mem['patterns_count'] > 0:
    print(f"  Cross-session learning: Active")
```

---

### Step 6: Run Validation Test (10 min)

```bash
cd /home/dp/ai-workspace/HRM
python sage/experiments/test_cogitation_integration.py
```

**Expected Results**:
- Test passes
- At least 3-5 experiences stored (100% SNARC capture rate)
- Possibly 1-2 patterns formed (if experiences cluster)
- All other metrics maintained
- Memory stats visible in output

---

### Step 7: Document Results (15 min)

**Update**: `sage/docs/LATEST_STATUS.md`

**Changes**:
1. Update title to "HierarchicalMemory Integration Complete!"
2. Add to "What's Working" section
3. Update biological parallels (all 5 complete)
4. Update metrics with memory stats
5. Update recommendations for next session

---

## Integration Pattern Summary

Following the same pattern as EmotionalEnergy:

1. ✅ **Import** - Add HierarchicalMemory import
2. ✅ **Initialize** - Create instance in `__init__`
3. ✅ **Integrate** - Store experiences in main loop
4. ✅ **Statistics** - Add `get_memory_stats()` method
5. ✅ **Test** - Update test file with memory metrics
6. ✅ **Validate** - Run tests and verify
7. ✅ **Document** - Update LATEST_STATUS.md

---

## Known Limitations

### 1. VAE Encoding Needed

**Issue**: Proper latent representations require VAE encoding of (question, response) pairs.

**Current Workaround**: Use placeholder `torch.randn(64)` for latent vectors.

**Future Fix**:
- Integrate language VAE from tri-modal system
- Encode QA pairs before storing
- Enables actual similarity matching

**Impact**: Without VAE, pattern formation won't cluster meaningfully. But structure/API will be correct.

### 2. No Persistence Yet

**Issue**: Memory is in-memory only, lost between sessions.

**Future Enhancement**:
- Add save/load methods
- Store to `sage/data/memory/thor_hierarchical.pt`
- Load on initialization
- Enables true cross-session learning

**Impact**: Limited to single-session learning for now.

### 3. No Recall Integration

**Issue**: Similar experiences are retrieved but not used yet.

**Future Enhancement**:
- Inject recalled experiences into LLM prompt context
- Use patterns to inform response generation
- Leverage concepts for meta-learning

**Impact**: Memory is stored but not actively informing responses yet.

---

## Testing Strategy

### Minimal Test (Current)

1. Store experiences during 5-turn conversation
2. Verify experiences counted correctly
3. Check if any patterns form
4. Display stats in output

**Expected**:
- 5 experiences stored (100% capture rate)
- 0-1 patterns (too few experiences to cluster)
- 0 concepts (need patterns first)

### Future Tests

1. **Multi-session learning**: Load previous memories, add new ones
2. **Pattern formation**: Run longer conversations (20+ turns)
3. **Concept emergence**: Multiple sessions with varied topics
4. **Transfer learning**: Test if patterns from one context help another

---

## Timeline Breakdown

| Step | Task | Time |
|------|------|------|
| 1 | Initialize HierarchicalMemory | 15 min |
| 2 | Store experiences in loop | 30 min |
| 3 | Recall similar (optional) | 30 min |
| 4 | Add stats method | 15 min |
| 5 | Update test file | 30 min |
| 6 | Run validation | 10 min |
| 7 | Document results | 15 min |
| **Total** | **Complete integration** | **2h 25min** |

---

## Success Criteria

### Minimum (Must Have)
- ✅ HierarchicalMemory initialized without errors
- ✅ Experiences stored during test run
- ✅ Memory stats visible in test output
- ✅ No breaking changes to existing functionality
- ✅ All previous metrics maintained

### Desired (Should Have)
- ✅ At least 5 experiences stored (one per turn)
- ✅ Memory stats integrated into test summary
- ✅ Clean commit with descriptive message
- ✅ Documentation updated

### Stretch (Nice to Have)
- 🎯 At least 1 pattern formed
- 🎯 Recall integration working
- 🎯 VAE encoding instead of placeholder
- 🎯 Persistence (save/load)

---

## Files to Modify

1. **`sage/core/sage_consciousness_cogitation.py`**
   - Add import
   - Initialize memory
   - Store experiences
   - Add stats method
   - ~30 lines added

2. **`sage/experiments/test_cogitation_integration.py`**
   - Add memory stats output
   - Update return dict
   - Add memory section to summary
   - ~15 lines added

3. **`sage/docs/LATEST_STATUS.md`**
   - Update with memory integration
   - All sections refreshed
   - ~50 lines changed

---

## Next Steps After Integration

### Short-term (Same Session)
1. Validate integration works
2. Commit and push changes
3. Update documentation

### Medium-term (Next Session)
1. Add VAE encoding for proper latent representations
2. Implement persistence (save/load)
3. Test multi-session learning

### Long-term (Future)
1. Integrate recall into LLM context
2. Pattern-based response optimization
3. Concept-level meta-learning
4. Transfer learning across contexts

---

## Risk Assessment

**Low Risk**:
- Non-breaking integration (like EmotionalEnergy)
- Memory system is self-contained
- Can disable if issues arise
- No external dependencies

**Potential Issues**:
- Placeholder latents mean patterns won't cluster meaningfully
- In-memory only (no cross-session yet)
- Slight performance overhead (minimal with current scale)

**Mitigation**:
- Follow EmotionalEnergy integration pattern (proven successful)
- Start with minimal integration
- Add enhancements incrementally
- Test thoroughly before committing

---

## Coordination Notes

### For Human Sessions
- Read this plan before starting
- Follow steps sequentially
- Test after each major change
- Commit when all tests pass

### For Autonomous Sessions
- Check if 2-3 hour block available
- Pull latest changes first
- Follow integration steps
- Document any deviations
- Update this plan if issues found

### For Sprout Validation (Future)
- After Thor integration stable
- Deploy to Sprout
- Verify separate memory spaces
- Test that Thor's memories ≠ Sprout's memories
- Validates hardware-anchoring for memory too

---

## Appendix: HierarchicalMemory Architecture

### Three-Level Hierarchy

```
Concepts (Abstract)
    ↑ (2+ patterns)
Patterns (Generalizations)
    ↑ (3+ experiences)
Experiences (Specific Observations)
    ↑ (SNARC salience ≥ 0.6)
Observations (All sensory input)
```

### Data Flow

```
1. Observation arrives
2. SNARC computes salience
3. If salience ≥ threshold → Store as Experience
4. Every N experiences → Update patterns
5. Cluster similar experiences → Form pattern
6. When patterns cluster → Form concept
```

### Example Evolution

**Turn 1**: "What is ATP?"
- Experience stored (high salience)
- No pattern yet (need ≥3)

**Turns 2-4**: More ATP-related questions
- 3 experiences about ATP
- Pattern formed: "ATP energy questions"
- Centroid in latent space

**Turns 10-15**: Questions about SNARC, memory
- New patterns: "Memory questions", "Architecture questions"
- Concept formed: "SAGE introspection questions"
- Links ATP, memory, architecture patterns

**Transfer**: New question about metabolic states
- Recalls similar experiences
- Activates "ATP energy" pattern
- Informed by "SAGE introspection" concept
- Better response through generalization

---

**Status**: Ready for implementation
**Next Session**: HierarchicalMemory integration (2-3 hours)
**After That**: Sprout validation or persistence layer

---

*Plan created by Auto Session #15*
*Platform: Thor (Jetson AGX Thor Developer Kit)*
*All four Michaud enhancements operational, ready for #5*
