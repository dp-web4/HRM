#!/usr/bin/env python3
"""
Unified Consciousness Integration - Session 41

Integrates all core consciousness components into unified architecture:
- Quality metrics (Session 27-29)
- Epistemic awareness (Session 30-31)
- Metabolic state management (Session 40)
- Epistemic calibration (Session 39)

Creates complete consciousness cycle with:
- Metabolic regulation of processing
- Epistemic self-awareness
- Quality optimization with ATP allocation
- State-aware behavior

This is the synthesis of Sessions 27-40, demonstrating how components
work together to create integrated consciousness architecture.

Author: Thor (Autonomous Session 41)
Date: 2025-12-12
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from sage.core.quality_metrics import score_response_quality, QualityScore
from sage.core.epistemic_states import (
    EpistemicMetrics,
    EpistemicStateTracker,
    EpistemicState
)
from sage.core.metabolic_states import (
    MetabolicStateManager,
    MetabolicState,
    ATPAllocation
)
from sage.core.emotional_state import EmotionalStateTracker
from sage.core.circadian_clock import CircadianClock, CircadianContext, CircadianPhase
from sage.core.dream_consolidation import DREAMConsolidator, ConsolidatedMemory
from sage.core.pattern_retrieval import PatternRetriever, RetrievalContext, TransferLearningResult


@dataclass
class ConsciousnessCycle:
    """
    Single consciousness cycle capturing complete state.

    Attributes:
        cycle_number: Cycle count
        timestamp: When cycle occurred

        # Input
        prompt: Input prompt/question
        response: Generated response

        # Quality
        quality_score: Quality metrics
        quality_atp: ATP allocated to quality

        # Epistemic
        epistemic_metrics: Meta-cognitive metrics
        epistemic_state: Primary epistemic state
        epistemic_atp: ATP allocated to epistemic tracking

        # Emotional (Session 48)
        emotional_state: Emotional metrics (curiosity, frustration, progress, engagement)
        emotional_summary: Human-readable emotional state

        # Circadian (Session 49)
        circadian_context: Temporal context (phase, day/night, biases)
        circadian_phase: Current circadian phase

        # Metabolic
        metabolic_state: Current metabolic state
        total_atp: Total ATP budget
        atp_allocation: Full allocation breakdown

        # Performance
        processing_time: Time to complete cycle (seconds)
        errors: Any errors encountered
    """
    cycle_number: int
    timestamp: float = field(default_factory=time.time)

    # Input/Output
    prompt: str = ""
    response: str = ""

    # Quality
    quality_score: Optional[QualityScore] = None
    quality_atp: float = 0.0

    # Epistemic
    epistemic_metrics: Optional[EpistemicMetrics] = None
    epistemic_state: Optional[EpistemicState] = None
    epistemic_atp: float = 0.0

    # Emotional (Session 48)
    emotional_state: Dict[str, float] = field(default_factory=dict)
    emotional_summary: str = ""

    # Circadian (Session 49)
    circadian_context: Optional[CircadianContext] = None
    circadian_phase: str = ""

    # DREAM Consolidation (Session 50)
    consolidation_triggered: bool = False
    consolidated_memory: Optional[ConsolidatedMemory] = None

    # Transfer Learning (Session 51)
    patterns_retrieved: int = 0
    transfer_learning_result: Optional[TransferLearningResult] = None
    learning_applied: bool = False

    # Metabolic
    metabolic_state: MetabolicState = MetabolicState.WAKE
    total_atp: float = 100.0
    atp_allocation: Dict[str, float] = field(default_factory=dict)

    # Performance
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


class UnifiedConsciousnessManager:
    """
    Unified consciousness architecture integrating all core components.

    Manages complete consciousness cycle:
    1. Metabolic state determines ATP budget
    2. ATP allocated to quality and epistemic processes
    3. Quality metrics evaluated with allocated resources
    4. Epistemic states tracked and classified
    5. Metabolic state updated based on epistemic signals
    6. Cycle repeats with updated state

    Integration points:
    - Metabolic ATP → Quality optimization resources
    - Epistemic frustration → Metabolic CRISIS detection
    - Quality + Epistemic → Metabolic state transitions
    - Metabolic FOCUS → Enhanced quality/epistemic tracking
    """

    def __init__(self,
                 initial_atp: float = 100.0,
                 quality_atp_baseline: float = 20.0,
                 epistemic_atp_baseline: float = 15.0,
                 emotional_history_length: int = 20,
                 circadian_period: int = 100,
                 circadian_enabled: bool = True,
                 consolidation_enabled: bool = True,
                 transfer_learning_enabled: bool = True):
        """
        Initialize unified consciousness manager.

        Args:
            initial_atp: Starting ATP budget
            quality_atp_baseline: Base ATP for quality scoring
            epistemic_atp_baseline: Base ATP for epistemic tracking
            emotional_history_length: History window for emotional tracking (Session 48)
            circadian_period: Cycles per circadian day (Session 49)
            circadian_enabled: Whether to use circadian rhythm (Session 49)
            consolidation_enabled: Whether to use scheduled DREAM consolidation (Session 50)
            transfer_learning_enabled: Whether to use pattern retrieval transfer learning (Session 51)
        """
        # Core components
        self.metabolic_manager = MetabolicStateManager(initial_atp=initial_atp)
        self.epistemic_tracker = EpistemicStateTracker(history_size=100)
        self.emotional_tracker = EmotionalStateTracker(history_length=emotional_history_length)
        self.circadian_clock = CircadianClock(period_cycles=circadian_period) if circadian_enabled else None
        self.dream_consolidator = DREAMConsolidator() if consolidation_enabled else None
        self.pattern_retriever = PatternRetriever(top_k=5, min_relevance=0.3) if transfer_learning_enabled else None

        # ATP baselines
        self.quality_atp_baseline = quality_atp_baseline
        self.epistemic_atp_baseline = epistemic_atp_baseline

        # Cycle history
        self.cycles: List[ConsciousnessCycle] = []
        self.cycle_count = 0

        # DREAM consolidation tracking (Session 50)
        self.last_consolidation_cycle = 0
        self.consolidated_memories: List[ConsolidatedMemory] = []
        self.consolidation_count = 0

        # Statistics
        self.total_errors = 0
        self.crisis_events = 0
        self.focus_episodes = 0

    def consciousness_cycle(self,
                          prompt: str,
                          response: str,
                          task_salience: float = 0.5) -> ConsciousnessCycle:
        """
        Execute complete consciousness cycle.

        Integrates all components:
        1. Allocate ATP based on metabolic state
        2. Score quality with allocated resources
        3. Track epistemic state
        4. Update metabolic state based on outcomes
        5. Record complete cycle state

        Args:
            prompt: Input prompt/question
            response: Generated response
            task_salience: Task salience for attention (0.0-1.0)

        Returns:
            ConsciousnessCycle with complete state
        """
        cycle_start = time.time()
        self.cycle_count += 1

        # Create cycle record
        cycle = ConsciousnessCycle(
            cycle_number=self.cycle_count,
            prompt=prompt,
            response=response,
            metabolic_state=self.metabolic_manager.current_state,
            total_atp=self.metabolic_manager.atp.total_atp
        )

        try:
            # 1. ATP Allocation based on metabolic state
            quality_atp, epistemic_atp = self._allocate_atp()
            cycle.quality_atp = quality_atp
            cycle.epistemic_atp = epistemic_atp
            cycle.atp_allocation = {
                'quality': quality_atp,
                'epistemic': epistemic_atp,
                'reserved': self.metabolic_manager.atp.reserved,
                'available': self.metabolic_manager.atp.available
            }

            # 1.5. Transfer Learning - Pattern Retrieval (Session 51)
            if self.pattern_retriever:
                if len(self.consolidated_memories) > 0:
                    # Retrieve relevant patterns from consolidated memories
                    transfer_result = self._retrieve_relevant_patterns(
                        prompt=prompt,
                        task_salience=task_salience
                    )
                    cycle.transfer_learning_result = transfer_result
                    cycle.patterns_retrieved = len(transfer_result.retrieved_patterns)
                    cycle.learning_applied = cycle.patterns_retrieved > 0
                else:
                    # No memories yet - create empty result
                    from sage.core.pattern_retrieval import TransferLearningResult
                    cycle.transfer_learning_result = TransferLearningResult(
                        retrieved_patterns=[],
                        quality_guidance=[],
                        creative_insights=[],
                        application_summary="No consolidated memories yet",
                        retrieval_count=0,
                        retrieval_time=0.0
                    )
                    cycle.patterns_retrieved = 0
                    cycle.learning_applied = False

            # 2. Quality Evaluation (with ATP-weighted processing)
            quality_score = self._evaluate_quality(response, prompt, quality_atp)
            cycle.quality_score = quality_score

            # 3. Epistemic State Tracking (with ATP-weighted depth)
            epistemic_metrics, epistemic_state = self._track_epistemic_state(
                response, quality_score, epistemic_atp
            )
            cycle.epistemic_metrics = epistemic_metrics
            cycle.epistemic_state = epistemic_state

            # 3.5. Emotional State Tracking (Session 48)
            emotional_state = self._track_emotional_state(
                response=response,
                salience=task_salience,
                quality=quality_score.normalized,
                epistemic_frustration=epistemic_metrics.frustration
            )
            cycle.emotional_state = emotional_state
            cycle.emotional_summary = self.emotional_tracker.get_emotional_summary()

            # 3.6. Circadian Rhythm Tracking (Session 49)
            circadian_context = self._track_circadian_state()
            cycle.circadian_context = circadian_context
            cycle.circadian_phase = circadian_context.phase.value if circadian_context else ""

            # 3.7. DREAM Consolidation Check (Session 50)
            if self.dream_consolidator and circadian_context:
                # Trigger consolidation during DEEP_NIGHT phase
                if circadian_context.phase == CircadianPhase.DEEP_NIGHT:
                    # Check if enough cycles have passed since last consolidation
                    cycles_since_last = self.cycle_count - self.last_consolidation_cycle
                    min_cycles_between = 10  # Consolidate at most once per 10 cycles

                    if cycles_since_last >= min_cycles_between and len(self.cycles) > 0:
                        # Trigger consolidation of recent cycles
                        consolidated = self._trigger_consolidation()
                        cycle.consolidation_triggered = True
                        cycle.consolidated_memory = consolidated

            # 4. Metabolic State Update (now considers emotions + circadian)
            self._update_metabolic_state(
                task_salience=task_salience,
                epistemic_frustration=epistemic_metrics.frustration,
                quality_score=quality_score.normalized,
                emotional_frustration=emotional_state.get('frustration', 0.0),
                circadian_context=circadian_context
            )

            # 5. Success reporting
            self.metabolic_manager.report_success()

        except Exception as e:
            # Error handling
            cycle.errors.append(str(e))
            self.total_errors += 1
            self.metabolic_manager.report_error()

        # Record processing time
        cycle.processing_time = time.time() - cycle_start

        # Store cycle
        self.cycles.append(cycle)

        # Update statistics
        if self.metabolic_manager.current_state == MetabolicState.CRISIS:
            self.crisis_events += 1
        elif self.metabolic_manager.current_state == MetabolicState.FOCUS:
            self.focus_episodes += 1

        return cycle

    def _allocate_atp(self) -> Tuple[float, float]:
        """
        Allocate ATP to quality and epistemic processes.

        Uses metabolic state multipliers to adjust resource allocation.

        Returns:
            (quality_atp, epistemic_atp) tuple
        """
        # Get state-based multipliers
        quality_mult = self.metabolic_manager.get_atp_multiplier("quality")
        epistemic_mult = self.metabolic_manager.get_atp_multiplier("epistemic")

        # Calculate allocations
        quality_atp = self.quality_atp_baseline * quality_mult
        epistemic_atp = self.epistemic_atp_baseline * epistemic_mult

        # Attempt allocation (may fail if insufficient ATP)
        quality_allocated = self.metabolic_manager.atp.allocate("quality", quality_atp)
        epistemic_allocated = self.metabolic_manager.atp.allocate("epistemic", epistemic_atp)

        # If allocation failed, use reduced amounts
        if not quality_allocated:
            quality_atp = min(quality_atp, self.metabolic_manager.atp.available)
        if not epistemic_allocated:
            epistemic_atp = min(epistemic_atp, self.metabolic_manager.atp.available)

        return quality_atp, epistemic_atp

    def _evaluate_quality(self,
                         response: str,
                         prompt: str,
                         quality_atp: float) -> QualityScore:
        """
        Evaluate response quality with ATP-weighted processing.

        Higher ATP enables more thorough quality analysis.

        Args:
            response: Response text
            prompt: Prompt text
            quality_atp: ATP allocated to quality

        Returns:
            QualityScore
        """
        # Base quality score
        quality_score = score_response_quality(response, prompt)

        # ATP affects quality scoring depth/thoroughness
        # In real implementation, more ATP could enable:
        # - Deeper semantic analysis
        # - More comprehensive fact checking
        # - Nuanced quality assessment
        # For now, we track the ATP allocation for future use

        # Release ATP after use
        self.metabolic_manager.atp.release("quality")

        return quality_score

    def _track_epistemic_state(self,
                              response: str,
                              quality_score: QualityScore,
                              epistemic_atp: float) -> Tuple[EpistemicMetrics, EpistemicState]:
        """
        Track epistemic state with ATP-weighted depth.

        Higher ATP enables deeper meta-cognitive analysis.

        Args:
            response: Response text
            quality_score: Quality metrics
            epistemic_atp: ATP allocated to epistemic tracking

        Returns:
            (EpistemicMetrics, EpistemicState) tuple
        """
        # Calculate epistemic metrics
        # In production, these would come from actual consciousness cycles
        # For integration demo, we estimate based on quality and response

        # Confidence correlates with quality (but not perfectly - Session 38 insight)
        base_confidence = 0.5 + (quality_score.normalized * 0.3)

        # Add noise to prevent perfect correlation
        confidence = np.clip(base_confidence + np.random.normal(0, 0.1), 0.0, 1.0)

        # Comprehension depth based on quality components
        comprehension = 0.4 + (quality_score.normalized * 0.4)

        # Uncertainty inversely related to confidence
        uncertainty = 1.0 - confidence

        # Coherence from quality
        coherence = 0.5 + (quality_score.normalized * 0.3)

        # Frustration low in normal operation (updated by errors)
        frustration = min(0.3 * (self.metabolic_manager.consecutive_errors / 3.0), 1.0)

        # Create metrics
        metrics = EpistemicMetrics(
            confidence=confidence,
            comprehension_depth=comprehension,
            uncertainty=uncertainty,
            coherence=coherence,
            frustration=frustration
        )

        # Track in epistemic tracker
        self.epistemic_tracker.track(metrics)

        # Get primary state
        state = metrics.primary_state()

        # Release ATP
        self.metabolic_manager.atp.release("epistemic")

        return metrics, state

    def _track_emotional_state(self,
                              response: str,
                              salience: float,
                              quality: float,
                              epistemic_frustration: float) -> Dict[str, float]:
        """
        Track emotional state across consciousness cycles (Session 48).

        Emotions provide additional signals for metabolic regulation and
        behavioral adaptation.

        Args:
            response: Response text
            salience: Task salience (0.0-1.0)
            quality: Quality score (0.0-1.0)
            epistemic_frustration: Epistemic frustration (0.0-1.0)

        Returns:
            Dict with emotional metrics (curiosity, frustration, progress, engagement)
        """
        # Update emotional tracker with cycle data
        emotional_state = self.emotional_tracker.update({
            'response': response,
            'salience': salience,
            'quality': quality,
            'convergence_quality': quality  # Use quality as proxy for convergence
        })

        # Epistemic frustration can reinforce emotional frustration
        if epistemic_frustration > 0.5:
            emotional_state['frustration'] = max(
                emotional_state['frustration'],
                epistemic_frustration
            )

        return emotional_state

    def _track_circadian_state(self) -> Optional[CircadianContext]:
        """
        Track circadian rhythm state (Session 49).

        Provides temporal context for consciousness including:
        - Current circadian phase (dawn, day, dusk, night, deep_night)
        - Day/night strength for metabolic biasing
        - Time-dependent expectations

        Returns:
            CircadianContext or None if circadian disabled
        """
        if self.circadian_clock is None:
            return None

        # Advance clock and get current context
        context = self.circadian_clock.tick()

        return context

    def _retrieve_relevant_patterns(self,
                                    prompt: str,
                                    task_salience: float) -> TransferLearningResult:
        """
        Retrieve consolidated patterns relevant to current context (Session 51).

        Uses pattern retriever to find previously consolidated memories
        that might inform the current consciousness cycle.

        Biological parallel: Retrieving sleep-consolidated memories during
        waking cognition to improve current reasoning.

        Args:
            prompt: Current prompt/query
            task_salience: Current task salience

        Returns:
            TransferLearningResult with retrieved patterns and guidance
        """
        if self.pattern_retriever is None:
            return TransferLearningResult(
                retrieved_patterns=[],
                quality_guidance=[],
                creative_insights=[],
                application_summary="Transfer learning disabled",
                retrieval_count=0,
                retrieval_time=0.0
            )

        # Build retrieval context from current consciousness state
        # Get most recent epistemic state from cycle history
        recent_epistemic = "exploring"  # default
        if len(self.cycles) > 0 and self.cycles[-1].epistemic_state:
            recent_epistemic = self.cycles[-1].epistemic_state.value

        # Get most recent emotional state
        recent_emotional = {}
        if len(self.cycles) > 0 and self.cycles[-1].emotional_state:
            recent_emotional = self.cycles[-1].emotional_state

        # Get current circadian phase
        current_phase = "day"  # default
        if self.circadian_clock:
            circadian_context = self.circadian_clock.get_context()
            if circadian_context:
                current_phase = circadian_context.phase.value

        context = RetrievalContext(
            prompt=prompt,
            task_salience=task_salience,
            metabolic_state=self.metabolic_manager.current_state.value,
            epistemic_state=recent_epistemic,
            emotional_state=recent_emotional,
            circadian_phase=current_phase
        )

        # Retrieve patterns
        transfer_result = self.pattern_retriever.retrieve_patterns(
            consolidated_memories=self.consolidated_memories,
            context=context
        )

        return transfer_result

    def _trigger_consolidation(self) -> Optional[ConsolidatedMemory]:
        """
        Trigger DREAM consolidation of recent consciousness cycles (Session 50).

        Called during DEEP_NIGHT circadian phase to consolidate recent
        consciousness experiences into memory patterns, quality learnings,
        and creative associations.

        Biological parallel: Memory consolidation during deep sleep.

        Returns:
            ConsolidatedMemory or None if consolidation fails
        """
        if self.dream_consolidator is None:
            return None

        # Determine how many cycles to consolidate
        # Consolidate cycles since last consolidation (or all if first time)
        cycles_since_last = self.cycle_count - self.last_consolidation_cycle
        start_idx = max(0, len(self.cycles) - cycles_since_last)
        cycles_to_consolidate = self.cycles[start_idx:]

        if len(cycles_to_consolidate) == 0:
            return None

        # Consolidate with ATP budget (DREAM state has high ATP for consolidation)
        # Use 80% of available ATP for consolidation
        current_atp = self.metabolic_manager.atp.available
        consolidation_budget = current_atp * 0.8

        try:
            # Run consolidation
            consolidated = self.dream_consolidator.consolidate_cycles(
                cycles_to_consolidate,
                atp_budget=consolidation_budget
            )

            # Store consolidated memory
            self.consolidated_memories.append(consolidated)
            self.consolidation_count += 1
            self.last_consolidation_cycle = self.cycle_count

            return consolidated

        except Exception as e:
            # Consolidation failed - don't crash, just skip
            return None

    def _update_metabolic_state(self,
                               task_salience: float,
                               epistemic_frustration: float,
                               quality_score: float,
                               emotional_frustration: float = 0.0,
                               circadian_context: Optional[CircadianContext] = None):
        """
        Update metabolic state based on cycle outcomes.

        Integrates epistemic, emotional, and circadian signals into metabolic regulation.

        Args:
            task_salience: Task salience (0.0-1.0)
            epistemic_frustration: Frustration level (0.0-1.0)
            quality_score: Quality score (0.0-1.0)
            emotional_frustration: Emotional frustration (0.0-1.0) [Session 48]
            circadian_context: Temporal context [Session 49]
        """
        # Combine epistemic and emotional frustration
        # Emotional frustration amplifies metabolic signals
        combined_frustration = max(epistemic_frustration, emotional_frustration)

        # Session 49: Apply circadian biasing to metabolic transitions
        # Circadian rhythm naturally biases toward certain states at different times
        if circadian_context:
            # Get current metabolic state
            current_state = self.metabolic_manager.current_state.value

            # Get circadian bias for current state
            state_bias = self.circadian_clock.get_metabolic_bias(current_state)

            # Modulate task salience based on circadian phase
            # During night (high night_strength), reduce effective salience for WAKE/FOCUS
            # During day (high day_strength), enhance salience for active states
            if circadian_context.is_night:
                # Night: Natural tendency toward REST/DREAM
                # Reduce effective salience to make WAKE/FOCUS transitions harder
                task_salience = task_salience * (1.0 - 0.3 * circadian_context.night_strength)
            else:
                # Day: Natural tendency toward WAKE/FOCUS
                # Enhance salience slightly during peak day
                task_salience = task_salience * (1.0 + 0.2 * circadian_context.day_strength)

        # Standard metabolic cycle update
        self.metabolic_manager.cycle_update(
            task_salience=task_salience,
            epistemic_frustration=combined_frustration
        )

        # Additional state logic based on quality
        # Low quality + high salience might trigger FOCUS
        if quality_score < 0.7 and task_salience > 0.7:
            if self.metabolic_manager.current_state == MetabolicState.WAKE:
                self.metabolic_manager.set_attention(
                    target="quality_improvement",
                    salience=task_salience
                )

        # Session 48: Emotional intervention
        # High emotional frustration → REST for consolidation
        if emotional_frustration > 0.7:
            if self.metabolic_manager.current_state not in [MetabolicState.REST, MetabolicState.DREAM]:
                # Trigger REST state for emotional reset
                self.metabolic_manager.set_attention(
                    target="emotional_consolidation",
                    salience=0.8  # High priority
                )

    def get_statistics(self) -> Dict:
        """
        Get comprehensive consciousness statistics.

        Returns:
            Dict with quality, epistemic, metabolic stats
        """
        if not self.cycles:
            return {}

        # Quality statistics
        quality_scores = [c.quality_score.normalized for c in self.cycles
                         if c.quality_score is not None]
        quality_stats = {
            'mean': np.mean(quality_scores) if quality_scores else 0.0,
            'std': np.std(quality_scores) if quality_scores else 0.0,
            'min': np.min(quality_scores) if quality_scores else 0.0,
            'max': np.max(quality_scores) if quality_scores else 1.0,
        }

        # Epistemic statistics
        epistemic_states = [c.epistemic_state.value for c in self.cycles
                           if c.epistemic_state is not None]
        state_counts = {state: epistemic_states.count(state)
                       for state in set(epistemic_states)}

        # Metabolic statistics
        metabolic_stats = self.metabolic_manager.get_state_statistics()

        # Emotional statistics (Session 48)
        emotional_stats = {}
        if self.cycles and self.cycles[0].emotional_state:
            # Aggregate emotional metrics across cycles
            emotional_keys = ['curiosity', 'frustration', 'progress', 'engagement']
            for key in emotional_keys:
                values = [c.emotional_state.get(key, 0.0) for c in self.cycles
                         if c.emotional_state]
                if values:
                    emotional_stats[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values)
                    }

        # Integration statistics
        integration_stats = {
            'total_cycles': len(self.cycles),
            'total_errors': self.total_errors,
            'crisis_events': self.crisis_events,
            'focus_episodes': self.focus_episodes,
            'mean_processing_time': np.mean([c.processing_time for c in self.cycles]),
        }

        # Consolidation statistics (Session 50)
        consolidation_stats = {
            'total_consolidations': self.consolidation_count,
            'last_consolidation_cycle': self.last_consolidation_cycle,
            'stored_memories': len(self.consolidated_memories)
        }

        # Transfer Learning statistics (Session 51)
        transfer_stats = {}
        if self.pattern_retriever:
            # Count cycles with transfer learning applied
            cycles_with_learning = [c for c in self.cycles if c.learning_applied]
            total_patterns_retrieved = sum(c.patterns_retrieved for c in self.cycles)

            transfer_stats = {
                'cycles_with_patterns': len(cycles_with_learning),
                'total_patterns_retrieved': total_patterns_retrieved,
                'average_patterns_per_cycle': (
                    total_patterns_retrieved / len(self.cycles)
                    if self.cycles else 0.0
                ),
                'retriever_stats': self.pattern_retriever.get_statistics()
            }

        return {
            'quality': quality_stats,
            'epistemic_states': state_counts,
            'metabolic': metabolic_stats,
            'emotional': emotional_stats,  # Session 48
            'consolidation': consolidation_stats,  # Session 50
            'transfer_learning': transfer_stats,  # Session 51
            'integration': integration_stats
        }

    def get_recent_cycles(self, n: int = 10) -> List[ConsciousnessCycle]:
        """Get n most recent consciousness cycles"""
        return self.cycles[-n:] if len(self.cycles) >= n else self.cycles


def example_usage():
    """Example demonstrating unified consciousness integration"""
    print("Unified Consciousness Integration Demo")
    print("=" * 60)
    print()

    # Initialize unified consciousness
    consciousness = UnifiedConsciousnessManager(
        initial_atp=100.0,
        quality_atp_baseline=20.0,
        epistemic_atp_baseline=15.0
    )

    # Simulate consciousness cycles
    test_cases = [
        ("What is machine learning?",
         "Machine learning is a branch of artificial intelligence that enables systems to learn from data. It uses algorithms like neural networks (deep learning), decision trees, and support vector machines. Key paradigms include supervised learning (labeled data), unsupervised learning (pattern discovery), and reinforcement learning (reward-based). ML achieves 95%+ accuracy on many tasks through training on large datasets.",
         0.6),  # Normal salience

        ("Explain consciousness",
         "Consciousness is hard to define precisely. Maybe it involves awareness, or subjective experience, or something else. The problem is unclear.",
         0.8),  # High salience, low quality (should trigger FOCUS)

        ("What is 2+2?",
         "2+2 equals 4. This is basic arithmetic.",
         0.3),  # Low salience, good quality
    ]

    for i, (prompt, response, salience) in enumerate(test_cases, 1):
        print(f"\nCycle {i}:")
        print(f"Prompt: {prompt}")
        print(f"Salience: {salience:.1f}")

        cycle = consciousness.consciousness_cycle(
            prompt=prompt,
            response=response,
            task_salience=salience
        )

        print(f"Quality: {cycle.quality_score.normalized:.3f}")
        print(f"Epistemic State: {cycle.epistemic_state.value}")
        print(f"Metabolic State: {cycle.metabolic_state.value}")
        print(f"ATP Allocated: quality={cycle.quality_atp:.1f}, epistemic={cycle.epistemic_atp:.1f}")
        print(f"Processing: {cycle.processing_time*1000:.1f}ms")

    # Show statistics
    print("\n" + "=" * 60)
    print("Consciousness Statistics:")
    print("=" * 60)

    stats = consciousness.get_statistics()

    print("\nQuality Metrics:")
    print(f"  Mean: {stats['quality']['mean']:.3f}")
    print(f"  Range: [{stats['quality']['min']:.3f}, {stats['quality']['max']:.3f}]")

    print("\nEpistemic States:")
    for state, count in stats['epistemic_states'].items():
        print(f"  {state}: {count}")

    print("\nMetabolic Activity:")
    print(f"  Total cycles: {stats['integration']['total_cycles']}")
    print(f"  Focus episodes: {stats['integration']['focus_episodes']}")
    print(f"  Crisis events: {stats['integration']['crisis_events']}")

    print("\nMetabolic State Duration:")
    for state, duration in stats['metabolic']['state_durations'].items():
        print(f"  {state}: {duration:.1f}s")


if __name__ == '__main__':
    example_usage()
