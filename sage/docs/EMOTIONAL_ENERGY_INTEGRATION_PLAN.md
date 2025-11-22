# EmotionalEnergy Integration Plan
**Status**: Analysis Complete, Implementation Pending
**Created**: 2025-11-21 11:35 (Interactive Session)
**For**: 12:00 Auto Session

---

## Overview

EmotionalEnergy (`sage/irp/emotional_energy.py`, 398 lines) provides computational analogs of biological emotions as intrinsic motivation signals.

### Michaud's Insight
> "Emotions provide evaluation function that drives behavior"

Emotions = evolved energy functions that create motivation beyond task completion.

---

## What EmotionalEnergy Provides

### Four Drives (All Energy-Based)

**1. Curiosity** (novelty-seeking)
- Energy: **REDUCES** (negative term) - makes novel situations attractive
- Computed: `novelty × surprise`
- Novelty: 1 - similarity_to_past_experiences
- Surprise: prediction_error magnitude

**2. Mastery** (competence-building)
- Energy: **REDUCES** - makes learning opportunities attractive
- Computed: `competence × growth_potential`
- Competence: convergence_speed + solution_quality
- Growth: recent improvement trajectory

**3. Completion** (goal achievement)
- Energy: **REDUCES** - makes finishing attractive
- Computed: `progress × proximity + completion_bonus`
- Bonus: +0.3 when proximity > 0.8 ("home stretch")

**4. Frustration** (stuck avoidance)
- Energy: **INCREASES** (positive term) - makes stuck states unattractive
- Computed: repetition × lack_of_progress
- Triggers: repeated similar states, no energy reduction

### Energy Equation
```python
total_energy = task_energy + emotional_energy

emotional_energy = (
    -curiosity_weight * curiosity_drive +      # Seek novel
    -mastery_weight * mastery_drive +          # Seek growth
    -completion_weight * completion_drive +    # Seek finish
    +frustration_weight * frustration_cost     # Avoid stuck
)
```

---

## Implementation Approaches

### Approach A: Full Mixin Integration (Complex)
**Effort**: 2-3 hours
**Impact**: Complete emotional system

**Steps**:
1. Refactor IRPPlugin to use EmotionalEnergyMixin
2. Update ConversationalLLM to track energy history
3. Integrate HierarchicalMemory for novelty computation
4. Add emotional state to SAGE cycle logging

**Pros**: Full Michaud emotional architecture
**Cons**: Requires extensive refactoring, longer testing

### Approach B: Lightweight Emotional State Tracker (Simple)
**Effort**: 30-60 minutes
**Impact**: Emotional awareness without full integration

**Steps**:
1. Create `EmotionalStateTracker` class
2. Track interaction patterns (repetition, progress, novelty)
3. Compute simple emotional scores per cycle
4. Log emotional state alongside metabolic state
5. Use signals to adjust behavior (temperature, threshold)

**Pros**: Quick to implement, easy to test, non-breaking
**Cons**: Doesn't modify energy landscape directly

### Approach C: Hybrid (Recommended)
**Effort**: 1-2 hours
**Impact**: Best balance of power and simplicity

**Steps**:
1. Create simplified emotional tracker (Approach B)
2. Integrate emotional signals into AttentionManager
3. Emotional state → metabolic state transitions
4. Example: High frustration → trigger REST or REFRAME
5. High curiosity → increase exploration temperature

**Pros**: Emotional intelligence without full refactor
**Cons**: Not pure Michaud architecture

---

## Recommended Implementation (Hybrid)

### Phase 1: Emotional State Tracker (30 min)

```python
# sage/core/emotional_state.py

class EmotionalStateTracker:
    """
    Lightweight emotional state tracking for consciousness loop.

    Tracks:
    - Curiosity (novelty in questions)
    - Frustration (repetitive patterns)
    - Progress (quality improvement)
    - Engagement (salience trends)
    """

    def __init__(self):
        self.response_history = []
        self.salience_history = []
        self.quality_history = []

    def update(self, cycle_data):
        """Update emotional state from cycle."""
        self.response_history.append(cycle_data['response'])
        self.salience_history.append(cycle_data['salience'])
        self.quality_history.append(cycle_data['quality'])

        # Compute emotional scores
        curiosity = self._compute_curiosity()
        frustration = self._compute_frustration()
        progress = self._compute_progress()

        return {
            'curiosity': curiosity,
            'frustration': frustration,
            'progress': progress,
            'overall': self._compute_overall_affect()
        }

    def _compute_curiosity(self):
        """Novelty in recent interactions."""
        if len(self.response_history) < 2:
            return 0.5

        # Check lexical diversity in recent responses
        recent = self.response_history[-5:]
        unique_words = len(set(' '.join(recent).split()))
        total_words = len(' '.join(recent).split())

        diversity = unique_words / total_words if total_words > 0 else 0.5
        return diversity

    def _compute_frustration(self):
        """Repetition and stagnation."""
        if len(self.quality_history) < 3:
            return 0.0

        # Check if quality is stagnant
        recent_quality = self.quality_history[-3:]
        variance = np.var(recent_quality)

        # Low variance = frustration (stuck)
        frustration = 1.0 - min(1.0, variance * 10)
        return frustration

    def _compute_progress(self):
        """Improvement trajectory."""
        if len(self.quality_history) < 2:
            return 0.5

        # Linear trend in quality
        recent = self.quality_history[-5:]
        if len(recent) >= 2:
            slope = (recent[-1] - recent[0]) / len(recent)
            progress = max(0.0, min(1.0, slope + 0.5))
            return progress

        return 0.5
```

### Phase 2: Integration into Cogitation (30 min)

```python
# Modify sage_consciousness_cogitation.py

class CogitationSAGE(MichaudSAGE):
    def __init__(self, ...):
        super().__init__(...)

        # Add emotional tracking
        self.emotional_tracker = EmotionalStateTracker()

    async def step(self):
        # ... existing step logic ...

        # After memory update, compute emotional state
        emotional_state = self.emotional_tracker.update({
            'response': results['response'],
            'salience': snarc_scores['total_salience'],
            'quality': results['convergence_quality']
        })

        # Use emotional signals to adjust behavior
        self._apply_emotional_modulation(emotional_state)

        # Store emotional history
        self.emotional_history.append({
            'cycle': self.cycle_count,
            'emotions': emotional_state
        })

    def _apply_emotional_modulation(self, emotions):
        """Adjust consciousness parameters based on emotions."""

        # High frustration → trigger REST or explore different approach
        if emotions['frustration'] > 0.7:
            self.attention_manager.force_state_transition('REST')
            print(f"[EMOTION] High frustration detected, entering REST state")

        # High curiosity → increase exploration temperature
        if emotions['curiosity'] > 0.7:
            self.llm.llm.temperature += 0.1
            print(f"[EMOTION] High curiosity detected, increasing exploration")

        # Low progress → reduce temperature for stability
        if emotions['progress'] < 0.3:
            self.llm.llm.temperature = max(0.3, self.llm.llm.temperature - 0.1)
            print(f"[EMOTION] Low progress, increasing precision")
```

### Phase 3: Testing (30 min)

Add emotional metrics to test suite:
- Track emotional trajectories across conversation
- Validate frustration detection triggers REST
- Confirm curiosity increases exploration
- Measure correlation between emotions and quality

---

## Expected Impact

### Behavioral Changes

**Before EmotionalEnergy**:
- Fixed temperature (0.5)
- No adaptation to conversation dynamics
- No frustration detection
- Mechanical progression

**After EmotionalEnergy**:
- Dynamic temperature (0.3-0.6 based on emotions)
- Frustration triggers state changes
- Curiosity drives exploration
- Natural, adaptive conversation flow

### Metrics to Track

1. **Emotional Trajectory**: How do emotions evolve over conversation?
2. **Intervention Rate**: How often do emotions trigger adjustments?
3. **Quality Correlation**: Do emotional states predict response quality?
4. **Engagement**: Does emotional modulation improve sustained interaction?

---

## Next Steps for 12:00 Auto Session

### Option 1: Implement Lightweight Tracker (Quick Win)
- 30-60 minutes
- Immediate visible impact
- Non-breaking change
- Can test today

**Recommended if**: Want quick progress before Dennis returns

### Option 2: Full Mixin Integration (Complete Solution)
- 2-3 hours
- Full Michaud emotional architecture
- Requires HierarchicalMemory integration first
- More testing needed

**Recommended if**: Have 3-4 hours uninterrupted time

### Option 3: Document and Pivot to HierarchicalMemory
- Document current state
- Switch to memory integration
- Come back to emotions later

**Recommended if**: Memory is higher priority

---

## Dependencies

### For Lightweight Tracker
- ✅ None - standalone implementation

### For Full Mixin Integration
- ⏳ HierarchicalMemory (for novelty computation)
- ⏳ Extended IRPState (energy_history field)
- ⏳ Memory reference in plugins

---

## Files to Create/Modify

### New Files
- `sage/core/emotional_state.py` (Tracker class)
- `sage/experiments/test_emotional_modulation.py` (Test suite)

### Modified Files
- `sage/core/sage_consciousness_cogitation.py` (Add tracker)
- `sage/docs/MICHAUD_INTEGRATION_STATUS.md` (Document progress)

---

## Biological Parallel

**Emotional Energy ≅ Limbic System**
- Curiosity ≅ Dopamine (reward anticipation for novelty)
- Frustration ≅ Cortisol (stress response to stagnation)
- Mastery ≅ Serotonin (confidence from competence)
- Completion ≅ Endorphin (pleasure from achievement)

**Key Insight**: Emotions aren't decorative - they shape the energy landscape to guide behavior toward survival/learning patterns that worked evolutionarily.

---

## Questions for Dennis

1. **Priority**: EmotionalEnergy or HierarchicalMemory first?
2. **Approach**: Lightweight tracker or full mixin integration?
3. **Timeline**: Quick wins vs complete architecture?

---

**Status**: Plan complete, ready for implementation
**Recommended**: Start with lightweight tracker (Option 1)
**Time estimate**: 30-60 minutes for visible results

Next auto session can pick this up and run with it! 🚀
