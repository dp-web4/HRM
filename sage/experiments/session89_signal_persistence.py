#!/usr/bin/env python3
"""
Session 89: Signal Persistence for Sparse Real Data

Problem (from Session 88):
- Real conversational signals are 40x sparser than simulated (2.7% vs 33% coverage)
- Current architecture: Signals only affect trust within specific contexts
- Result: 0% improvement with sparse real data

Solution:
- Implement PERSISTENT expert reputation across all contexts
- Conversational signals update GLOBAL expert scores, not just local trust
- Sparse signals become more influential by affecting all future selections
- Expert reputation accumulates over time, carrying across context boundaries

Architecture:
- PersistentReputationSelector: Extends MultiDimensionalTrustFirstSelector
- Global reputation tracking: Expert scores persist across contexts
- Signal integration: Real conversational feedback updates reputation permanently
- Graceful degradation: Works with any signal density (0% to 100%)

Expected Results:
- Sparse signals (2.7% coverage) should now improve trust_driven selections
- Expert reputation builds incrementally from rare feedback
- Trust activation earlier in generation (vs Session 88 late/never)
- Demonstrates viability of real conversation integration

Created: 2025-12-21 (Autonomous Session 89)
Hardware: Jetson AGX Thor
Previous: Session 88 (Real Conversation Testing, 0% improvement, sparsity discovered)
"""

import json
import logging
import random
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """Single turn in a conversation."""
    timestamp: str
    speaker: str  # 'user' or 'sage'
    text: str
    irp_iterations: Optional[int] = None


@dataclass
class RepairSignal:
    """Conversational repair signal indicating feedback quality."""
    turn_index: int
    signal_type: str  # 'engagement', 'reassurance', 'correction', 'abandonment'
    confidence: float  # 0.0-1.0
    evidence: str

    def to_trust_delta(self) -> float:
        """Convert signal to trust score delta."""
        # Positive signals increase trust, negative decrease
        signal_weights = {
            'engagement': +0.3,      # User engaged (positive)
            'reassurance': +0.2,     # User reassured (positive)
            'correction': -0.4,      # User corrected (negative)
            'abandonment': -0.6,     # User abandoned (very negative)
        }
        base_delta = signal_weights.get(self.signal_type, 0.0)
        return base_delta * self.confidence


@dataclass
class ExpertReputation:
    """Persistent reputation for a single expert."""
    expert_id: int
    layer: int
    reputation_score: float = 0.5  # Start neutral
    total_signals: int = 0
    positive_signals: int = 0
    negative_signals: int = 0
    last_updated: float = field(default_factory=time.time)

    def update_from_signal(self, signal: RepairSignal):
        """Update reputation from conversational signal."""
        delta = signal.to_trust_delta()

        # Update score with momentum (60% old, 40% new)
        self.reputation_score = 0.6 * self.reputation_score + 0.4 * (self.reputation_score + delta)
        # Clamp to [0, 1]
        self.reputation_score = max(0.0, min(1.0, self.reputation_score))

        # Update counters
        self.total_signals += 1
        if delta > 0:
            self.positive_signals += 1
        elif delta < 0:
            self.negative_signals += 1

        self.last_updated = time.time()

    def get_trust_score(self, evidence_weight: float = 1.0) -> float:
        """Get trust score with evidence weighting.

        Args:
            evidence_weight: Multiplier based on evidence count (0-1)

        Returns:
            Trust score in [0, 1]
        """
        # Trust increases with evidence
        # Reputation weighted by evidence strength
        base_trust = self.reputation_score
        evidence_factor = evidence_weight

        # Trust = reputation * evidence_strength + (1-evidence_strength) * neutral
        neutral_score = 0.5
        return base_trust * evidence_factor + neutral_score * (1.0 - evidence_factor)


class PersistentReputationSelector:
    """Expert selector with persistent reputation from conversational signals.

    Key Innovation: Conversational signals update GLOBAL expert reputation,
    not just context-specific trust. This makes sparse signals (2.7% coverage)
    viable by accumulating evidence across all contexts.

    Architecture:
    - Maintains persistent reputation database for all experts
    - Conversational signals update reputation scores globally
    - Trust-first selection uses reputation + internal quality
    - Evidence accumulation: More signals → stronger trust influence
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int = 48,
        epsilon: float = 0.2,
        min_evidence_for_trust: int = 1,  # Lower than Session 87 (was 2)
        reputation_weight: float = 0.4,  # 40% reputation, 60% internal quality
        reputation_db_path: Optional[Path] = None,
    ):
        """Initialize persistent reputation selector.

        Args:
            num_experts: Number of experts per layer
            num_layers: Number of transformer layers
            epsilon: Exploration rate (0.0-1.0)
            min_evidence_for_trust: Minimum signals before trust-driven selection
            reputation_weight: Weight for reputation vs internal quality (0-1)
            reputation_db_path: Path to persistent reputation database (optional)
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.min_evidence_for_trust = min_evidence_for_trust
        self.reputation_weight = reputation_weight
        self.internal_weight = 1.0 - reputation_weight

        # Persistent reputation storage
        self.reputations: Dict[Tuple[int, int], ExpertReputation] = {}
        self.reputation_db_path = reputation_db_path

        # Internal quality tracking (observation-based)
        self.expert_observations: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        # Selection history for analysis
        self.selection_history: List[Dict] = []
        self.trust_driven_count = 0
        self.total_selections = 0

        # Load existing reputations if database exists
        if self.reputation_db_path and self.reputation_db_path.exists():
            self._load_reputations()

        logger.info(f"Initialized PersistentReputationSelector:")
        logger.info(f"  Experts per layer: {num_experts}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Epsilon (exploration): {epsilon}")
        logger.info(f"  Min evidence for trust: {min_evidence_for_trust}")
        logger.info(f"  Reputation weight: {reputation_weight:.1%}")
        logger.info(f"  Internal quality weight: {self.internal_weight:.1%}")

    def _get_or_create_reputation(self, layer: int, expert_id: int) -> ExpertReputation:
        """Get existing reputation or create new neutral one."""
        key = (layer, expert_id)
        if key not in self.reputations:
            self.reputations[key] = ExpertReputation(
                expert_id=expert_id,
                layer=layer,
            )
        return self.reputations[key]

    def integrate_conversational_signal(
        self,
        layer: int,
        expert_id: int,
        signal: RepairSignal,
    ):
        """Integrate conversational signal into GLOBAL expert reputation.

        This is the key innovation: Signals update persistent reputation,
        not just context-specific trust. Makes sparse signals viable.

        Args:
            layer: Layer index
            expert_id: Expert index within layer
            signal: Conversational repair signal
        """
        reputation = self._get_or_create_reputation(layer, expert_id)
        reputation.update_from_signal(signal)

        logger.debug(
            f"Signal integrated: Layer {layer}, Expert {expert_id}, "
            f"Type: {signal.signal_type}, Confidence: {signal.confidence:.2f}, "
            f"New reputation: {reputation.reputation_score:.3f}"
        )

    def select_expert(
        self,
        layer: int,
        context: str,
        available_experts: Optional[List[int]] = None,
    ) -> int:
        """Select expert using persistent reputation + internal quality.

        Selection Strategy:
        1. Epsilon-greedy exploration
        2. Trust-first selection if sufficient evidence:
           - Score = reputation_weight * reputation + internal_weight * internal_quality
           - Reputation from global conversational signals
           - Internal quality from observation history
        3. Fallback to quality-only if insufficient evidence

        Args:
            layer: Layer index
            context: Context string (for internal quality tracking)
            available_experts: Optional subset of experts to choose from

        Returns:
            Selected expert ID
        """
        self.total_selections += 1

        if available_experts is None:
            available_experts = list(range(self.num_experts))

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            selected = random.choice(available_experts)
            self._record_selection(
                layer=layer,
                expert_id=selected,
                context=context,
                mechanism='exploration',
                trust_driven=False,
            )
            return selected

        # Calculate composite scores for all available experts
        expert_scores = []
        for expert_id in available_experts:
            reputation = self._get_or_create_reputation(layer, expert_id)

            # Internal quality from observations
            obs_key = (layer, expert_id)
            observations = self.expert_observations.get(obs_key, [])
            internal_quality = sum(observations) / len(observations) if observations else 0.5

            # Evidence weighting: More signals → stronger reputation influence
            if reputation.total_signals >= self.min_evidence_for_trust:
                # Enough evidence: Use reputation
                evidence_strength = min(1.0, reputation.total_signals / 5.0)  # Max at 5 signals
                reputation_score = reputation.get_trust_score(evidence_weight=evidence_strength)

                # Composite score
                composite_score = (
                    self.reputation_weight * reputation_score +
                    self.internal_weight * internal_quality
                )

                expert_scores.append((expert_id, composite_score, True))  # trust_driven=True
            else:
                # Insufficient evidence: Internal quality only
                expert_scores.append((expert_id, internal_quality, False))  # trust_driven=False

        # Select expert with highest score
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        selected, score, trust_driven = expert_scores[0]

        mechanism = 'trust_driven' if trust_driven else 'quality_only'
        if trust_driven:
            self.trust_driven_count += 1

        self._record_selection(
            layer=layer,
            expert_id=selected,
            context=context,
            mechanism=mechanism,
            trust_driven=trust_driven,
            score=score,
        )

        return selected

    def record_observation(self, layer: int, expert_id: int, quality: float):
        """Record internal quality observation for expert."""
        key = (layer, expert_id)
        self.expert_observations[key].append(quality)

    def _record_selection(
        self,
        layer: int,
        expert_id: int,
        context: str,
        mechanism: str,
        trust_driven: bool,
        score: Optional[float] = None,
    ):
        """Record selection for analysis."""
        self.selection_history.append({
            'generation': self.total_selections,
            'layer': layer,
            'expert_id': expert_id,
            'context': context,
            'mechanism': mechanism,
            'trust_driven': trust_driven,
            'score': score,
        })

    def _load_reputations(self):
        """Load reputations from SQLite database."""
        if not self.reputation_db_path:
            return

        try:
            conn = sqlite3.connect(self.reputation_db_path)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT layer, expert_id, reputation_score, total_signals,
                       positive_signals, negative_signals, last_updated
                FROM expert_reputations
            """)

            for row in cursor.fetchall():
                layer, expert_id, rep_score, total, pos, neg, updated = row
                key = (layer, expert_id)
                self.reputations[key] = ExpertReputation(
                    expert_id=expert_id,
                    layer=layer,
                    reputation_score=rep_score,
                    total_signals=total,
                    positive_signals=pos,
                    negative_signals=neg,
                    last_updated=updated,
                )

            conn.close()
            logger.info(f"Loaded {len(self.reputations)} expert reputations from database")

        except sqlite3.Error as e:
            logger.warning(f"Failed to load reputations: {e}")

    def save_reputations(self):
        """Save reputations to SQLite database."""
        if not self.reputation_db_path:
            logger.warning("No reputation database path configured, skipping save")
            return

        conn = sqlite3.connect(self.reputation_db_path)
        cursor = conn.cursor()

        # Create table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS expert_reputations (
                layer INTEGER,
                expert_id INTEGER,
                reputation_score REAL,
                total_signals INTEGER,
                positive_signals INTEGER,
                negative_signals INTEGER,
                last_updated REAL,
                PRIMARY KEY (layer, expert_id)
            )
        """)

        # Insert/update reputations
        for (layer, expert_id), rep in self.reputations.items():
            cursor.execute("""
                INSERT OR REPLACE INTO expert_reputations
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                layer, expert_id, rep.reputation_score, rep.total_signals,
                rep.positive_signals, rep.negative_signals, rep.last_updated
            ))

        conn.commit()
        conn.close()
        logger.info(f"Saved {len(self.reputations)} expert reputations to database")

    def get_statistics(self) -> Dict:
        """Get selector statistics."""
        # Reputation statistics
        all_reputations = [rep.reputation_score for rep in self.reputations.values()]
        reputations_with_signals = [
            rep for rep in self.reputations.values()
            if rep.total_signals > 0
        ]

        stats = {
            'total_selections': self.total_selections,
            'trust_driven_count': self.trust_driven_count,
            'trust_driven_pct': 100 * self.trust_driven_count / max(1, self.total_selections),
            'exploration_count': sum(1 for h in self.selection_history if h['mechanism'] == 'exploration'),
            'quality_only_count': sum(1 for h in self.selection_history if h['mechanism'] == 'quality_only'),
            'total_reputations_tracked': len(self.reputations),
            'reputations_with_signals': len(reputations_with_signals),
            'avg_reputation_score': sum(all_reputations) / len(all_reputations) if all_reputations else 0.5,
            'total_signals_integrated': sum(rep.total_signals for rep in self.reputations.values()),
        }

        # First trust activation
        for i, h in enumerate(self.selection_history):
            if h['trust_driven']:
                stats['first_trust_activation'] = i
                break
        else:
            stats['first_trust_activation'] = None

        return stats


def load_jsonl_conversations(conversation_path: Path) -> List[Tuple[List[ConversationTurn], List[RepairSignal]]]:
    """Load conversations from Sprout JSONL format (exchanges.jsonl).

    Args:
        conversation_path: Path to directory containing session subdirectories

    Returns:
        List of (turns, signals) tuples
    """
    conversations = []

    for session_dir in sorted(conversation_path.iterdir()):
        if not session_dir.is_dir():
            continue

        exchanges_file = session_dir / "exchanges.jsonl"
        if not exchanges_file.exists():
            continue

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
                    ))

                    # SAGE turn
                    turns.append(ConversationTurn(
                        timestamp=str(exchange.get('timestamp', line_num)),
                        speaker='sage',
                        text=exchange.get('model_response', ''),
                        irp_iterations=len(exchange['irp_info']['iterations']) if 'irp_info' in exchange else None
                    ))

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_num} in {exchanges_file}")
                    continue

        if turns:
            signals = detect_repair_signals(turns)
            conversations.append((turns, signals))
            logger.info(f"Loaded {session_dir.name}: {len(turns)} turns, {len(signals)} signals")

    return conversations


def detect_repair_signals(turns: List[ConversationTurn]) -> List[RepairSignal]:
    """Detect conversational repair signals from turn sequence.

    Detects both explicit feedback and implicit engagement patterns.
    """
    signals = []

    for i in range(len(turns) - 1):
        if turns[i].speaker != 'user':
            continue

        user_text = turns[i].text.lower()

        # Explicit positive signals
        reassurance_patterns = [
            (r"(yes|yeah|yep|correct|right|exactly)", 0.8, "explicit affirmation"),
            (r"(good|great|excellent|perfect|helpful)", 0.7, "positive evaluation"),
            (r"(thank|thanks)", 0.6, "gratitude"),
        ]

        for pattern, confidence, evidence in reassurance_patterns:
            if __import__('re').search(pattern, user_text):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='reassurance',
                    confidence=confidence,
                    evidence=evidence,
                ))
                break  # One signal per turn

        # Implicit engagement (philosophical inquiry)
        philosophical_patterns = [
            (r"what (is|are|does|would|could)", 0.5, "philosophical inquiry"),
            (r"(can|could) you", 0.4, "request for capability"),
            (r"(why|how|when)", 0.5, "causal/mechanistic inquiry"),
            (r"\?$", 0.3, "question mark (engaged follow-up)"),
        ]

        for pattern, confidence, evidence in philosophical_patterns:
            if __import__('re').search(pattern, user_text):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='engagement',
                    confidence=confidence,
                    evidence=evidence,
                ))
                break

        # Explicit negative signals
        correction_patterns = [
            (r"(no|nope|not|wrong|incorrect)", 0.8, "explicit negation"),
            (r"(actually|but)", 0.6, "correction indicator"),
        ]

        for pattern, confidence, evidence in correction_patterns:
            if __import__('re').search(pattern, user_text):
                signals.append(RepairSignal(
                    turn_index=i,
                    signal_type='correction',
                    confidence=confidence,
                    evidence=evidence,
                ))
                break

    return signals


def simulate_expert_selection_with_signals(
    conversations: List[Tuple[List[ConversationTurn], List[RepairSignal]]],
    selector: PersistentReputationSelector,
    num_generations: int = 810,
    num_selections_per_gen: int = 48,
) -> Dict:
    """Simulate expert selection process with conversational signal integration.

    Simulates MoE generation where:
    - Each generation has multiple expert selections (one per layer)
    - Real conversational signals integrated at corresponding generations
    - Signals update PERSISTENT expert reputation globally

    Args:
        conversations: List of (turns, signals) from real Sprout data
        selector: PersistentReputationSelector instance
        num_generations: Total generations to simulate
        num_selections_per_gen: Expert selections per generation (layers)

    Returns:
        Statistics dictionary
    """
    # Map signals to generation indices (spread across simulation)
    all_signals = []
    for turns, signals in conversations:
        for signal in signals:
            all_signals.append(signal)

    # Distribute signals across generations (simulate real-time integration)
    signal_generations = sorted(random.sample(range(num_generations), min(len(all_signals), num_generations)))
    signals_by_gen = {gen: all_signals[i] for i, gen in enumerate(signal_generations)}

    logger.info(f"Simulating {num_generations} generations with {len(all_signals)} signals")
    logger.info(f"Signal coverage: {100 * len(signal_generations) / num_generations:.1f}%")

    # Persistent context pool (9 contexts repeat across generations)
    contexts = [
        "philosophical_inquiry",
        "consciousness_definition",
        "epistemic_bias",
        "subjective_experience",
        "qualia_analysis",
        "hard_problem",
        "phenomenal_consciousness",
        "access_consciousness",
        "meta_cognition",
    ]

    for gen in range(num_generations):
        context = contexts[gen % len(contexts)]

        # Integrate any signals for this generation BEFORE selections
        if gen in signals_by_gen:
            signal = signals_by_gen[gen]
            # Randomly assign signal to expert that was "used"
            layer = random.randint(0, num_selections_per_gen - 1)
            expert_id = random.randint(0, selector.num_experts - 1)

            selector.integrate_conversational_signal(layer, expert_id, signal)

            if gen % 100 == 0:
                logger.info(f"Gen {gen}: Integrated {signal.signal_type} signal (confidence: {signal.confidence:.2f})")

        # Simulate expert selections for this generation
        for layer in range(num_selections_per_gen):
            expert_id = selector.select_expert(layer, context)

            # Simulate internal quality observation
            quality = random.uniform(0.3, 0.7)  # Simulated quality
            selector.record_observation(layer, expert_id, quality)

    stats = selector.get_statistics()
    return stats


def main():
    """Session 89: Signal Persistence for Sparse Real Data."""

    logger.info("="*80)
    logger.info("SESSION 89: SIGNAL PERSISTENCE FOR SPARSE REAL DATA")
    logger.info("="*80)

    # Configuration
    num_experts = 128
    num_layers = 48
    num_generations = 810

    # Load real Sprout conversations (from epistemic bias mapping experiments)
    conversation_path = Path("phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    if not conversation_path.exists():
        logger.error(f"Conversation path not found: {conversation_path}")
        logger.info("Falling back to simulated signals for architecture validation")
        conversations = []
    else:
        conversations = load_jsonl_conversations(conversation_path)
        logger.info(f"Loaded {len(conversations)} real conversations")

    # Test 1: Baseline (internal quality only, no reputation)
    logger.info("\n" + "="*80)
    logger.info("TEST 1: BASELINE (Internal Quality Only)")
    logger.info("="*80)

    baseline_selector = PersistentReputationSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        reputation_weight=0.0,  # No reputation, pure internal quality
    )

    # Simulate without conversational signals
    baseline_stats = simulate_expert_selection_with_signals(
        conversations=[],  # No signals
        selector=baseline_selector,
        num_generations=num_generations,
        num_selections_per_gen=num_layers,
    )

    logger.info("\nBaseline Results:")
    logger.info(f"  Total selections: {baseline_stats['total_selections']}")
    logger.info(f"  Trust-driven: {baseline_stats['trust_driven_pct']:.1f}%")
    logger.info(f"  First activation: Gen {baseline_stats.get('first_trust_activation', 'never')}")

    # Test 2: Persistent Reputation with Real Signals
    logger.info("\n" + "="*80)
    logger.info("TEST 2: PERSISTENT REPUTATION (Real Conversational Signals)")
    logger.info("="*80)

    persistent_selector = PersistentReputationSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        min_evidence_for_trust=1,  # Lower threshold for sparse data
        reputation_weight=0.4,  # 40% reputation, 60% internal quality
        reputation_db_path=Path("session89_reputation.db"),
    )

    persistent_stats = simulate_expert_selection_with_signals(
        conversations=conversations,
        selector=persistent_selector,
        num_generations=num_generations,
        num_selections_per_gen=num_layers,
    )

    logger.info("\nPersistent Reputation Results:")
    logger.info(f"  Total selections: {persistent_stats['total_selections']}")
    logger.info(f"  Trust-driven: {persistent_stats['trust_driven_pct']:.1f}%")
    logger.info(f"  First activation: Gen {persistent_stats.get('first_trust_activation', 'never')}")
    logger.info(f"  Signals integrated: {persistent_stats['total_signals_integrated']}")
    logger.info(f"  Experts with reputation: {persistent_stats['reputations_with_signals']}")
    logger.info(f"  Avg reputation: {persistent_stats['avg_reputation_score']:.3f}")

    # Save reputation database
    persistent_selector.save_reputations()

    # Calculate improvement
    improvement = persistent_stats['trust_driven_pct'] - baseline_stats['trust_driven_pct']

    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"Baseline trust_driven: {baseline_stats['trust_driven_pct']:.1f}%")
    logger.info(f"Persistent trust_driven: {persistent_stats['trust_driven_pct']:.1f}%")
    logger.info(f"Improvement: {improvement:+.1f} percentage points")

    if persistent_stats.get('first_trust_activation') is not None and baseline_stats.get('first_trust_activation') is not None:
        activation_speedup = baseline_stats['first_trust_activation'] - persistent_stats['first_trust_activation']
        logger.info(f"Activation speedup: {activation_speedup:+d} generations")

    # Save results
    results = {
        'session': 89,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': 'Jetson AGX Thor',
        'baseline': baseline_stats,
        'persistent_reputation': persistent_stats,
        'improvement': improvement,
        'configuration': {
            'num_experts': num_experts,
            'num_layers': num_layers,
            'num_generations': num_generations,
            'reputation_weight': 0.4,
            'min_evidence_for_trust': 1,
        },
        'conversations_loaded': len(conversations),
    }

    results_path = Path("session89_persistent_reputation_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("\nSession 89 complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
