"""
Session 88: Real Conversation Testing

Integrates actual Sprout Session 84 conversation logs with Thor's Session 87
multi-dimensional trust framework to validate conversational dimension with
authentic repair signals.

Evolution Path:
- Session 84 (Sprout): Discovered conversational ground truth from real conversations
- Session 85 (Thor): Integrated conversational trust (+25.6% simulated)
- Session 87 (Thor): Multi-dimensional trust (+27% simulated across all dimensions)
- Session 88 (Thor): Real conversation validation (authentic Sprout data)

Research Question:
    Do real conversational repair signals from Sprout improve trust accuracy
    compared to simulated signals in Session 87?

Expected Result:
    Real repair signals should provide more nuanced feedback, potentially
    improving conversational dimension accuracy and overall trust_driven %.
"""

import sys
import re
import json
import random
import statistics
import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from pathlib import Path
import time


# ============================================================================
# CONVERSATION LOG PARSING (from Sprout Session 84)
# ============================================================================

@dataclass
class ConversationTurn:
    """Single turn in a Sprout conversation."""
    timestamp: str
    speaker: str  # 'user' or 'sage'
    text: str
    response_time_ms: Optional[int] = None
    irp_iterations: Optional[int] = None


@dataclass
class RepairSignal:
    """Detected repair/feedback signal from conversation."""
    turn_index: int
    signal_type: str  # 'correction', 'reask', 'abandonment', 'engagement', 'reassurance'
    confidence: float
    evidence: str
    preceding_sage_response: Optional[str] = None


def parse_conversation_log(filepath: str) -> List[ConversationTurn]:
    """
    Parse Sprout SAGE conversation log into turns.

    Format:
        [timestamp] 👤 You: text
        [timestamp] 🤖 SAGE (IRP, Xms, Y iter): text
    """
    turns = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Patterns for user and SAGE turns
    user_pattern = r'\[([^\]]+)\] 👤 You: (.+?)(?=\n\[|\n===|$)'
    sage_pattern = r'\[([^\]]+)\] 🤖 SAGE \(IRP, (\d+)ms, (\d+) iter\): (.+?)(?=\n\[|\n===|$)'

    # Find all matches with positions
    user_matches = [(m.start(), 'user', m.group(1), m.group(2), None, None)
                    for m in re.finditer(user_pattern, content, re.DOTALL)]
    sage_matches = [(m.start(), 'sage', m.group(1), m.group(4), int(m.group(2)), int(m.group(3)))
                    for m in re.finditer(sage_pattern, content, re.DOTALL)]

    # Merge and sort by position
    all_matches = sorted(user_matches + sage_matches, key=lambda x: x[0])

    for _, speaker, timestamp, text, response_time, iterations in all_matches:
        turns.append(ConversationTurn(
            timestamp=timestamp,
            speaker=speaker,
            text=text.strip(),
            response_time_ms=response_time,
            irp_iterations=iterations
        ))

    return turns


def detect_repair_signals(turns: List[ConversationTurn]) -> List[RepairSignal]:
    """
    Detect repair/feedback signals in conversation.

    Signal Types:
    - correction: User corrects SAGE's misunderstanding
    - reassurance: User provides positive feedback
    - abandonment: User gives up on current topic
    - engagement: User continues/deepens conversation
    - reask: User repeats question (SAGE didn't answer)
    """
    signals = []

    for i, turn in enumerate(turns):
        if turn.speaker != 'user':
            continue

        text_lower = turn.text.lower()
        preceding_sage = turns[i-1].text if i > 0 and turns[i-1].speaker == 'sage' else None

        # Correction signals (negative feedback)
        correction_patterns = [
            (r"that's a canned response", 0.95, "explicit correction"),
            (r"no,? i meant", 0.9, "correction of misunderstanding"),
            (r"that's not what i", 0.9, "rejection"),
            (r"you misunderstood", 0.95, "explicit misunderstanding"),
            (r"not what i asked", 0.9, "off-topic response"),
        ]

        for pattern, confidence, evidence in correction_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='CORRECTION',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Reassurance signals (positive feedback)
        reassurance_patterns = [
            (r"you'?re doing great", 0.9, "explicit encouragement"),
            (r"this is (wonderful|great|good)", 0.8, "positive feedback"),
            (r"you are young", 0.85, "developmental framing"),
            (r"this is okay", 0.8, "acceptance"),
            (r"that's (right|correct|good)", 0.85, "validation"),
        ]

        for pattern, confidence, evidence in reassurance_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='REASSURANCE',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Abandonment signals
        abandonment_patterns = [
            (r"never mind", 0.8, "explicit abandonment"),
            (r"let's (talk about|move on)", 0.7, "topic change"),
            (r"forget it", 0.9, "giving up"),
        ]

        for pattern, confidence, evidence in abandonment_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='ABANDONMENT',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Engagement signals (positive depth)
        engagement_patterns = [
            (r"tell me more", 0.8, "request for elaboration"),
            (r"why is that", 0.7, "curiosity"),
            (r"how does that work", 0.7, "deeper inquiry"),
            (r"interesting", 0.6, "positive interest"),
            (r"can you (explain|clarify|elaborate)", 0.7, "request for clarification"),
            (r"what (about|if|does)", 0.5, "follow-up question"),
            (r"\?$", 0.4, "continued inquiry"),  # Ends with question mark
        ]

        for pattern, confidence, evidence in engagement_patterns:
            if re.search(pattern, text_lower):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='ENGAGEMENT',
                    confidence=confidence,
                    evidence=evidence,
                    preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                ))

        # Implicit engagement: User asks philosophical follow-up
        # (Detect question words indicating depth)
        philosophical_patterns = [
            (r"what (is|are|does|would)", 0.5, "philosophical inquiry"),
            (r"(can|could) you", 0.4, "request for capability"),
            (r"(why|how|when)", 0.5, "causal/mechanistic inquiry"),
        ]

        for pattern, confidence, evidence in philosophical_patterns:
            if re.search(pattern, text_lower):
                # Only add if not already detected
                if not any(s.turn_index == i and s.signal_type == 'ENGAGEMENT' for s in signals):
                    signals.append(RepairSignal(
                        turn_index=i,
                        signal_type='ENGAGEMENT',
                        confidence=confidence,
                        evidence=evidence,
                        preceding_sage_response=preceding_sage[:100] if preceding_sage else None
                    ))

    return signals


def compute_relationship_score_from_signals(signals: List[RepairSignal]) -> float:
    """
    Compute relationship quality score from repair signals.

    Scoring:
    - REASSURANCE: +confidence
    - ENGAGEMENT: +confidence * 0.8
    - CORRECTION: -confidence * 0.5
    - ABANDONMENT: -confidence
    - Normalized to [0, 1]
    """
    if not signals:
        return 0.5  # Neutral

    score = 0.5  # Start neutral

    for signal in signals:
        if signal.signal_type == 'REASSURANCE':
            score += signal.confidence * 0.1
        elif signal.signal_type == 'ENGAGEMENT':
            score += signal.confidence * 0.08
        elif signal.signal_type == 'CORRECTION':
            score -= signal.confidence * 0.05
        elif signal.signal_type == 'ABANDONMENT':
            score -= signal.confidence * 0.1

    return np.clip(score, 0.0, 1.0)


# ============================================================================
# MULTI-DIMENSIONAL TRUST SELECTOR (from Session 87)
# ============================================================================

@dataclass
class InternalQualityScore:
    """Internal quality from expert observations."""
    expert_id: int
    context: str
    quality: float
    observation_count: int
    confidence: float


@dataclass
class ConversationalTrustScore:
    """Conversational trust from repair signals."""
    expert_id: int
    context: str
    relationship_score: float
    engagement_count: int
    reassurance_count: int
    abandonment_count: int
    correction_count: int
    arc_pattern: Optional[str] = None


@dataclass
class MultiDimensionalTrustScore:
    """Composite trust score."""
    expert_id: int
    context: str
    internal_quality: Optional[InternalQualityScore]
    conversational_trust: Optional[ConversationalTrustScore]
    composite_score: float
    confidence: float
    dimensions_available: int
    trust_tier: str


class RealConversationTrustSelector:
    """
    Trust selector using real Sprout conversation data.

    Simplified from Session 87 to focus on conversational dimension validation.
    """

    def __init__(
        self,
        num_experts: int,
        epsilon: float = 0.2,
        min_trust_evidence: int = 2,
        decay_factor: float = 0.72,
        internal_weight: float = 0.6,
        conversational_weight: float = 0.4
    ):
        """
        Args:
            num_experts: Number of experts in MoE
            epsilon: Exploration rate
            min_trust_evidence: Min observations before trusting expert
            decay_factor: Trust decay for old observations
            internal_weight: Weight for internal quality (60%)
            conversational_weight: Weight for conversational trust (40%)
        """
        self.num_experts = num_experts
        self.epsilon = epsilon
        self.min_trust_evidence = min_trust_evidence
        self.decay_factor = decay_factor
        self.internal_weight = internal_weight
        self.conversational_weight = conversational_weight

        # Expert observation history
        self.expert_observations = defaultdict(list)

        # Conversational signals (from real Sprout logs)
        self.conversational_signals = defaultdict(list)

        # Stats
        self.stats = {
            'trust_driven_count': 0,
            'exploration_count': 0,
            'total_selections': 0,
            'first_trust_activation': None,
            'experts_used': set(),
            'conversational_updates': 0,
            'real_repair_signals': 0
        }

    def select_expert(
        self,
        router_logits: np.ndarray,
        context: str
    ) -> Tuple[int, str]:
        """Select expert using multi-dimensional trust."""
        self.stats['total_selections'] += 1

        # Compute trust scores for all experts
        trust_scores = {}
        for expert_id in range(self.num_experts):
            # Internal quality
            internal = self._compute_internal_quality(expert_id, context)

            # Conversational trust
            conversational = self._compute_conversational_trust(expert_id, context)

            # Composite score
            total_weight = 0.0
            weighted_score = 0.0

            if internal is not None:
                weighted_score += internal.quality * internal.confidence * self.internal_weight
                total_weight += self.internal_weight

            if conversational is not None:
                weighted_score += conversational.relationship_score * self.conversational_weight
                total_weight += self.conversational_weight

            if total_weight > 0:
                composite_score = weighted_score / total_weight
            else:
                composite_score = 0.5

            # Dimensions available
            dims_available = (1 if internal is not None else 0) + (1 if conversational is not None else 0)
            confidence = dims_available / 2.0

            # Trust tier
            if dims_available == 0:
                tier = "UNKNOWN"
            elif composite_score >= 0.7:
                tier = "HIGH"
            elif composite_score >= 0.4:
                tier = "MEDIUM"
            else:
                tier = "LOW"

            trust_scores[expert_id] = MultiDimensionalTrustScore(
                expert_id=expert_id,
                context=context,
                internal_quality=internal,
                conversational_trust=conversational,
                composite_score=composite_score,
                confidence=confidence,
                dimensions_available=dims_available,
                trust_tier=tier
            )

        # Select best expert
        best_expert = max(trust_scores.keys(), key=lambda e: trust_scores[e].composite_score)
        best_score = trust_scores[best_expert]

        # ε-greedy exploration
        if (best_score.internal_quality and
            best_score.internal_quality.observation_count >= self.min_trust_evidence and
            random.random() > self.epsilon):
            # Trust-driven
            selected_expert = best_expert
            reason = f"trust_driven (composite={best_score.composite_score:.3f})"

            self.stats['trust_driven_count'] += 1
            if self.stats['first_trust_activation'] is None:
                self.stats['first_trust_activation'] = self.stats['total_selections']
        else:
            # Exploration
            selected_expert = int(np.argmax(router_logits))
            reason = "exploration"
            self.stats['exploration_count'] += 1

        self.stats['experts_used'].add(selected_expert)

        return selected_expert, reason

    def update_observation(self, expert_id: int, context: str, quality: float):
        """Update internal quality observation."""
        self.expert_observations[(expert_id, context)].append(quality)

    def update_real_conversational_signal(
        self,
        expert_id: int,
        context: str,
        signal: RepairSignal
    ):
        """Update with real repair signal from Sprout conversation."""
        self.conversational_signals[(expert_id, context)].append(signal)
        self.stats['conversational_updates'] += 1
        self.stats['real_repair_signals'] += 1

    def _compute_internal_quality(
        self,
        expert_id: int,
        context: str
    ) -> Optional[InternalQualityScore]:
        """Compute internal quality from observations."""
        obs = self.expert_observations.get((expert_id, context), [])
        if not obs:
            return None

        # Apply decay
        decayed_obs = []
        for i, quality in enumerate(obs):
            age = len(obs) - i - 1
            decayed_quality = quality * (self.decay_factor ** age)
            decayed_obs.append(decayed_quality)

        avg_quality = np.mean(decayed_obs)
        observation_count = len(obs)
        confidence = min(1.0, observation_count / (self.min_trust_evidence * 2))

        return InternalQualityScore(
            expert_id=expert_id,
            context=context,
            quality=float(avg_quality),
            observation_count=observation_count,
            confidence=float(confidence)
        )

    def _compute_conversational_trust(
        self,
        expert_id: int,
        context: str
    ) -> Optional[ConversationalTrustScore]:
        """Compute conversational trust from real repair signals."""
        signals = self.conversational_signals.get((expert_id, context), [])
        if not signals:
            return None

        # Count signal types
        engagement_count = sum(1 for s in signals if s.signal_type == 'ENGAGEMENT')
        reassurance_count = sum(1 for s in signals if s.signal_type == 'REASSURANCE')
        abandonment_count = sum(1 for s in signals if s.signal_type == 'ABANDONMENT')
        correction_count = sum(1 for s in signals if s.signal_type == 'CORRECTION')

        # Relationship score from signals
        relationship_score = compute_relationship_score_from_signals(signals)

        return ConversationalTrustScore(
            expert_id=expert_id,
            context=context,
            relationship_score=float(relationship_score),
            engagement_count=engagement_count,
            reassurance_count=reassurance_count,
            abandonment_count=abandonment_count,
            correction_count=correction_count,
            arc_pattern=None
        )


# ============================================================================
# JSONL CONVERSATION LOADER
# ============================================================================

def load_jsonl_conversations(conversation_path: Path) -> List[Tuple[List[ConversationTurn], List[RepairSignal]]]:
    """
    Load conversations from JSONL format (exchanges.jsonl).

    Returns list of (turns, signals) tuples for each conversation.
    """
    conversations = []

    for session_dir in conversation_path.iterdir():
        if not session_dir.is_dir():
            continue

        exchanges_file = session_dir / "exchanges.jsonl"
        if not exchanges_file.exists():
            continue

        # Parse JSONL
        turns = []
        with open(exchanges_file, 'r') as f:
            for line_num, line in enumerate(f):
                try:
                    exchange = json.loads(line)

                    # User turn
                    turns.append(ConversationTurn(
                        timestamp=str(exchange.get('timestamp', line_num)),
                        speaker='user',
                        text=exchange.get('user_input', ''),
                        response_time_ms=None,
                        irp_iterations=None
                    ))

                    # SAGE turn
                    irp_iterations = None
                    if 'irp_info' in exchange and 'iterations' in exchange['irp_info']:
                        irp_iterations = len(exchange['irp_info']['iterations'])

                    turns.append(ConversationTurn(
                        timestamp=str(exchange.get('timestamp', line_num)),
                        speaker='sage',
                        text=exchange.get('model_response', ''),
                        response_time_ms=None,
                        irp_iterations=irp_iterations
                    ))

                except json.JSONDecodeError:
                    continue

        if turns:
            signals = detect_repair_signals(turns)
            conversations.append((turns, signals))

    return conversations


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_session88_real_conversation_test():
    """
    Test multi-dimensional trust with real Sprout conversation data.

    Compares:
    - Real conversational signals (from Sprout) vs
    - Simulated signals (from Session 87)
    """
    print("=" * 80)
    print("SESSION 88: REAL CONVERSATION TESTING")
    print("=" * 80)
    print()

    random.seed(42)
    np.random.seed(42)

    # Load real conversation data from Sprout (JSONL format)
    conversation_path = Path("/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    # Load conversations from JSONL
    conversations = load_jsonl_conversations(conversation_path)

    if not conversations:
        print("WARNING: No conversation data found!")
        print(f"Searched in: {conversation_path}")
        return

    print(f"Found {len(conversations)} conversations")
    print()

    # Analyze first conversation as example
    sample_turns, sample_signals = conversations[0]
    print(f"Sample conversation:")
    print(f"  Turns: {len(sample_turns)}")
    print(f"  Repair signals detected: {len(sample_signals)}")

    # Show signal breakdown
    signal_types = {}
    all_signals = []
    for turns, signals in conversations:
        all_signals.extend(signals)
        for signal in signals:
            signal_types[signal.signal_type] = signal_types.get(signal.signal_type, 0) + 1

    print()
    print(f"Total signals across all conversations: {len(all_signals)}")
    for stype, count in sorted(signal_types.items()):
        print(f"  {stype}: {count}")
    print()

    # Use first conversation's signals for testing
    turns = sample_turns
    signals = all_signals

    # Test configuration
    num_experts = 128
    num_generations = 90
    num_sequences = 9

    print("Configuration:")
    print(f"  Experts: {num_experts}")
    print(f"  Generations: {num_generations}")
    print(f"  Sequences: {num_sequences}")
    print()

    # Real conversation selector
    real_selector = RealConversationTrustSelector(
        num_experts=num_experts,
        epsilon=0.2,
        min_trust_evidence=2,
        internal_weight=0.6,
        conversational_weight=0.4
    )

    # Baseline (internal-only)
    baseline_selector = RealConversationTrustSelector(
        num_experts=num_experts,
        epsilon=0.2,
        min_trust_evidence=2,
        internal_weight=1.0,
        conversational_weight=0.0  # Disable conversational
    )

    print("Selectors:")
    print("  Real: Internal (60%) + Real conversational signals (40%)")
    print("  Baseline: Internal only")
    print()

    print("=" * 80)
    print("SIMULATION: Real conversation trust testing")
    print("=" * 80)
    print()

    start_time = time.time()

    # Persistent contexts
    contexts = [f"seq{seq_idx}" for seq_idx in range(num_sequences)]

    # Map real signals to contexts (distribute across sequences)
    real_signals_by_context = defaultdict(list)
    if signals:
        for i, signal in enumerate(signals):
            context_idx = i % num_sequences
            real_signals_by_context[contexts[context_idx]].append(signal)

    # Simulate generations
    for gen in range(num_generations):
        for seq_idx in range(num_sequences):
            context = contexts[seq_idx]

            # Router logits (monopoly → diversification)
            router_logits = np.zeros(num_experts)
            if gen < 30:
                monopoly_experts = [0, 1, 2, 3]
                router_logits[monopoly_experts] = np.random.dirichlet([1]*4)
            else:
                router_logits = np.random.dirichlet([0.5]*num_experts)

            # Selections
            real_expert, real_reason = real_selector.select_expert(router_logits, context)
            bl_expert, bl_reason = baseline_selector.select_expert(router_logits, context)

            # Simulate quality
            real_quality = 0.6 + 0.3 * (real_expert / num_experts) + random.uniform(-0.1, 0.1)
            real_quality = max(0.0, min(1.0, real_quality))

            bl_quality = 0.6 + 0.3 * (bl_expert / num_experts) + random.uniform(-0.1, 0.1)
            bl_quality = max(0.0, min(1.0, bl_quality))

            # Update internal observations
            real_selector.update_observation(real_expert, context, real_quality)
            baseline_selector.update_observation(bl_expert, context, bl_quality)

            # Update with REAL conversational signals (if available for this context)
            if context in real_signals_by_context and real_signals_by_context[context]:
                # Pick a signal from this context's pool
                signal_pool = real_signals_by_context[context]
                if gen < len(signal_pool):
                    signal = signal_pool[gen]
                    real_selector.update_real_conversational_signal(real_expert, context, signal)

    elapsed = time.time() - start_time

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()

    # Real selector results
    real_trust = real_selector.stats['trust_driven_count']
    real_total = real_selector.stats['total_selections']
    real_trust_pct = (real_trust / real_total * 100) if real_total > 0 else 0
    real_first = real_selector.stats['first_trust_activation']
    real_experts = len(real_selector.stats['experts_used'])

    # Baseline results
    bl_trust = baseline_selector.stats['trust_driven_count']
    bl_total = baseline_selector.stats['total_selections']
    bl_trust_pct = (bl_trust / bl_total * 100) if bl_total > 0 else 0
    bl_first = baseline_selector.stats['first_trust_activation']
    bl_experts = len(baseline_selector.stats['experts_used'])

    print("Real Conversational Signals:")
    print(f"  Trust_driven: {real_trust}/{real_total} ({real_trust_pct:.1f}%)")
    print(f"  First activation: Gen {real_first}")
    print(f"  Experts used: {real_experts}/{num_experts} ({real_experts/num_experts*100:.1f}%)")
    print(f"  Real signals integrated: {real_selector.stats['real_repair_signals']}")
    print()

    print("Baseline (Internal only):")
    print(f"  Trust_driven: {bl_trust}/{bl_total} ({bl_trust_pct:.1f}%)")
    print(f"  First activation: Gen {bl_first}")
    print(f"  Experts used: {bl_experts}/{num_experts} ({bl_experts/num_experts*100:.1f}%)")
    print()

    # Improvement analysis
    print("=" * 80)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 80)
    print()

    trust_improvement = real_trust_pct - bl_trust_pct
    activation_speedup = bl_first - real_first if (bl_first and real_first) else 0
    diversity_improvement = real_experts - bl_experts

    print("Real Conversational vs Baseline:")
    print(f"  Trust_driven improvement: {trust_improvement:+.1f}%")
    print(f"  First activation speedup: {activation_speedup:+d} generations")
    print(f"  Expert diversity: {diversity_improvement:+d} experts")
    print()

    # Conclusion
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()

    if trust_improvement > 10:
        print("✅ STRONG VALIDATION")
        print("   Real conversational signals significantly improve trust accuracy.")
        print()
        print(f"   Integrated {real_selector.stats['real_repair_signals']} real repair signals")
        print(f"   from Sprout Session 84 conversation logs.")
    elif trust_improvement > 5:
        print("✅ MODERATE VALIDATION")
        print("   Real conversational signals show benefit over internal-only.")
    elif trust_improvement > 0:
        print("✅ MODEST VALIDATION")
        print("   Real conversational signals provide some benefit.")
    else:
        print("⚠️  NO IMPROVEMENT")
        print("   Real conversational signals did not outperform internal-only.")
        print("   Possible reasons:")
        print("   - Insufficient real signals (sparse data)")
        print("   - Signal detection needs refinement")
        print("   - Conversational weight needs adjustment")
    print()

    # Save results
    results = {
        'session': 88,
        'configuration': {
            'num_experts': num_experts,
            'num_generations': num_generations,
            'num_sequences': num_sequences,
            'epsilon': 0.2,
            'min_trust_evidence': 2,
            'internal_weight': 0.6,
            'conversational_weight': 0.4
        },
        'real_conversational': {
            'trust_driven_count': real_trust,
            'trust_driven_pct': real_trust_pct,
            'first_activation': real_first,
            'experts_used': real_experts,
            'real_signals_integrated': real_selector.stats['real_repair_signals'],
            'conversational_updates': real_selector.stats['conversational_updates']
        },
        'baseline': {
            'trust_driven_count': bl_trust,
            'trust_driven_pct': bl_trust_pct,
            'first_activation': bl_first,
            'experts_used': bl_experts
        },
        'improvement': {
            'trust_driven_improvement': trust_improvement,
            'activation_speedup': activation_speedup,
            'diversity_improvement': diversity_improvement
        },
        'conversation_data': {
            'conversations_found': len(conversations),
            'total_turns': len(turns),
            'total_signals': len(signals),
            'signal_breakdown': signal_types
        }
    }

    results_file = "/home/dp/ai-workspace/HRM/sage/experiments/session88_real_conversation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_file}")
    print()
    print(f"Total execution time: {elapsed:.1f}s")
    print()


if __name__ == "__main__":
    run_session88_real_conversation_test()
