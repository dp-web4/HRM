#!/usr/bin/env python3
"""
Session 90: Trust as Resource Permission - Hysteresis & Memory-Aware Routing

Synthesis from Sessions 74-89 + Nova feedback:
"Trust = permission to consume scarce shared resources"

Problem (from Sessions 88-89):
- Session 88: 0% improvement with 2.7% sparse signals
- Session 89: +0.1% improvement with 4% sparse signals (persistent reputation)
- Still below critical threshold (~10% coverage for meaningful impact)
- Missing: Switching cost, memory traffic cost, resource contention modeling

Nova's Key Insights:
1. Real bottleneck: TIME-TO-EXPERT, not just memory fit
2. Prevent "chaotic novelty": Hysteresis + budgeted exploration
3. Trust-based routing = MEMORY ARBITRATION
4. Fold memory traffic cost into trust score: "good AND cheap to touch"

Solution - Resource-Aware Trust Routing:
- Expert switching cost (prevents ping-pong thrashing)
- Memory traffic cost (bandwidth contention on UMA)
- LRU benefit (experts already loaded get trust boost)
- Composite permission score: expertise * persistence * cheapness

Expected:
- Sparse signals (4%) become more effective with resource-aware routing
- Reduced expert churn (stability metric)
- Better token/sec vs swap rate tradeoff
- Trust-driven selections that respect memory constraints

Architecture:
- ResourceAwareTrustSelector: Extends PersistentReputationSelector
- Hysteresis: Currently-loaded experts get +20% trust boost
- Memory cost: Swapping penalty proportional to expert size
- Switching budget: Track and limit expert churn per generation

Created: 2025-12-21 (Autonomous Session 90)
Hardware: Jetson AGX Thor
Previous: Session 89 (Persistent Reputation, +0.1% improvement)
Synthesis: Nova feedback + "trust = resource permission" principle
"""

import json
import logging
import random
import sqlite3
import sys
import time
from collections import defaultdict, deque
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
        signal_weights = {
            'engagement': +0.3,
            'reassurance': +0.2,
            'correction': -0.4,
            'abandonment': -0.6,
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
    last_used_generation: Optional[int] = None  # Track usage for LRU

    def update_from_signal(self, signal: RepairSignal):
        """Update reputation from conversational signal."""
        delta = signal.to_trust_delta()
        # Momentum-based update
        self.reputation_score = 0.6 * self.reputation_score + 0.4 * (self.reputation_score + delta)
        self.reputation_score = max(0.0, min(1.0, self.reputation_score))

        self.total_signals += 1
        if delta > 0:
            self.positive_signals += 1
        elif delta < 0:
            self.negative_signals += 1

        self.last_updated = time.time()

    def get_trust_score(self, evidence_weight: float = 1.0) -> float:
        """Get trust score with evidence weighting."""
        base_trust = self.reputation_score
        evidence_factor = evidence_weight
        neutral_score = 0.5
        return base_trust * evidence_factor + neutral_score * (1.0 - evidence_factor)


@dataclass
class ResourceCost:
    """Resource cost model for expert operations."""
    swap_in_cost: float = 1.0   # Cost to load expert from cold storage
    swap_out_cost: float = 0.5  # Cost to evict expert from hot memory
    memory_footprint: int = 1   # Relative memory size (experts may vary)
    bandwidth_cost: float = 1.0 # Memory traffic cost (UMA contention)


class ResourceAwareTrustSelector:
    """Expert selector with resource-aware trust routing.

    Key Innovation (from Nova feedback):
    "Trust = permission to consume scarce shared resources"

    Trust score incorporates:
    1. Expert reputation (conversational signals)
    2. Internal quality (observation history)
    3. Resource cost (memory traffic, swapping)
    4. Persistence benefit (hysteresis for loaded experts)

    Permission score = expertise * cheapness * persistence
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int = 48,
        epsilon: float = 0.2,
        min_evidence_for_trust: int = 1,
        reputation_weight: float = 0.4,
        max_hot_experts: int = 64,  # LRU cache size
        hysteresis_boost: float = 0.2,  # +20% trust for loaded experts
        switching_cost_weight: float = 0.3,  # How much switching cost matters
        memory_cost_weight: float = 0.2,  # How much memory traffic matters
        max_swaps_per_gen: int = 8,  # Budgeted exploration
        reputation_db_path: Optional[Path] = None,
    ):
        """Initialize resource-aware trust selector.

        Args:
            num_experts: Number of experts per layer
            num_layers: Number of transformer layers
            epsilon: Exploration rate
            min_evidence_for_trust: Minimum signals before trust-driven selection
            reputation_weight: Weight for reputation vs internal quality
            max_hot_experts: Maximum experts in hot memory (LRU cache size)
            hysteresis_boost: Trust boost for currently-loaded experts
            switching_cost_weight: Weight for switching cost in scoring
            memory_cost_weight: Weight for memory traffic cost in scoring
            max_swaps_per_gen: Maximum expert swaps per generation (prevent thrashing)
            reputation_db_path: Path to persistent reputation database
        """
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.min_evidence_for_trust = min_evidence_for_trust
        self.reputation_weight = reputation_weight
        self.internal_weight = 1.0 - reputation_weight

        # Resource management
        self.max_hot_experts = max_hot_experts
        self.hysteresis_boost = hysteresis_boost
        self.switching_cost_weight = switching_cost_weight
        self.memory_cost_weight = memory_cost_weight
        self.max_swaps_per_gen = max_swaps_per_gen

        # Hot expert cache (LRU)
        self.hot_experts: Set[Tuple[int, int]] = set()  # (layer, expert_id)
        self.lru_queue: deque = deque()  # Track access order

        # Persistent reputation storage
        self.reputations: Dict[Tuple[int, int], ExpertReputation] = {}
        self.reputation_db_path = reputation_db_path

        # Internal quality tracking
        self.expert_observations: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        # Resource tracking
        self.expert_costs: Dict[Tuple[int, int], ResourceCost] = {}
        self.swaps_this_generation: int = 0
        self.current_generation: int = 0

        # Selection history for analysis
        self.selection_history: List[Dict] = []
        self.trust_driven_count = 0
        self.total_selections = 0
        self.total_swaps = 0
        self.swap_denied_count = 0  # Swaps blocked by budget

        # Load existing reputations
        if self.reputation_db_path and self.reputation_db_path.exists():
            self._load_reputations()

        logger.info(f"Initialized ResourceAwareTrustSelector:")
        logger.info(f"  Experts per layer: {num_experts}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Epsilon (exploration): {epsilon}")
        logger.info(f"  Max hot experts: {max_hot_experts}")
        logger.info(f"  Hysteresis boost: {hysteresis_boost:.1%}")
        logger.info(f"  Switching cost weight: {switching_cost_weight:.1%}")
        logger.info(f"  Memory cost weight: {memory_cost_weight:.1%}")
        logger.info(f"  Max swaps/generation: {max_swaps_per_gen}")

    def _get_or_create_reputation(self, layer: int, expert_id: int) -> ExpertReputation:
        """Get existing reputation or create new neutral one."""
        key = (layer, expert_id)
        if key not in self.reputations:
            self.reputations[key] = ExpertReputation(
                expert_id=expert_id,
                layer=layer,
            )
        return self.reputations[key]

    def _get_or_create_cost(self, layer: int, expert_id: int) -> ResourceCost:
        """Get resource cost model for expert (could vary by expert size)."""
        key = (layer, expert_id)
        if key not in self.expert_costs:
            self.expert_costs[key] = ResourceCost()
        return self.expert_costs[key]

    def _is_hot(self, layer: int, expert_id: int) -> bool:
        """Check if expert is currently loaded in hot memory."""
        return (layer, expert_id) in self.hot_experts

    def _load_expert(self, layer: int, expert_id: int) -> float:
        """Load expert into hot memory (LRU eviction if needed).

        Returns:
            Resource cost of the operation
        """
        key = (layer, expert_id)

        # Already hot - just update LRU
        if key in self.hot_experts:
            self.lru_queue.remove(key)
            self.lru_queue.append(key)
            return 0.0  # No swap cost

        # Need to swap in
        cost_model = self._get_or_create_cost(layer, expert_id)
        swap_cost = cost_model.swap_in_cost + cost_model.bandwidth_cost

        # LRU eviction if cache full
        if len(self.hot_experts) >= self.max_hot_experts:
            # Evict least recently used
            evict_key = self.lru_queue.popleft()
            self.hot_experts.remove(evict_key)
            evict_cost = self._get_or_create_cost(*evict_key).swap_out_cost
            swap_cost += evict_cost

        # Load new expert
        self.hot_experts.add(key)
        self.lru_queue.append(key)
        self.total_swaps += 1
        self.swaps_this_generation += 1

        return swap_cost

    def integrate_conversational_signal(
        self,
        layer: int,
        expert_id: int,
        signal: RepairSignal,
    ):
        """Integrate conversational signal into global expert reputation."""
        reputation = self._get_or_create_reputation(layer, expert_id)
        reputation.update_from_signal(signal)

        logger.debug(
            f"Signal integrated: Layer {layer}, Expert {expert_id}, "
            f"Type: {signal.signal_type}, New reputation: {reputation.reputation_score:.3f}"
        )

    def select_expert(
        self,
        layer: int,
        context: str,
        available_experts: Optional[List[int]] = None,
    ) -> int:
        """Select expert using resource-aware trust routing.

        Selection incorporates:
        1. Expert reputation (from conversational signals)
        2. Internal quality (from observations)
        3. Resource cost (memory traffic, swapping)
        4. Hysteresis (bonus for already-loaded experts)
        5. Switching budget (prevent thrashing)

        Permission score = expertise * cheapness * persistence
        """
        self.total_selections += 1

        if available_experts is None:
            available_experts = list(range(self.num_experts))

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            selected = random.choice(available_experts)
            self._load_expert(layer, selected)
            self._record_selection(
                layer=layer,
                expert_id=selected,
                context=context,
                mechanism='exploration',
                trust_driven=False,
            )
            return selected

        # Calculate resource-aware permission scores
        expert_scores = []
        for expert_id in available_experts:
            reputation = self._get_or_create_reputation(layer, expert_id)

            # 1. Expertise score (reputation + internal quality)
            obs_key = (layer, expert_id)
            observations = self.expert_observations.get(obs_key, [])
            internal_quality = sum(observations) / len(observations) if observations else 0.5

            if reputation.total_signals >= self.min_evidence_for_trust:
                evidence_strength = min(1.0, reputation.total_signals / 5.0)
                reputation_score = reputation.get_trust_score(evidence_weight=evidence_strength)
                expertise = (
                    self.reputation_weight * reputation_score +
                    self.internal_weight * internal_quality
                )
                trust_driven = True
            else:
                expertise = internal_quality
                trust_driven = False

            # 2. Resource cost (cheapness)
            is_hot = self._is_hot(layer, expert_id)
            cost_model = self._get_or_create_cost(layer, expert_id)

            if is_hot:
                # Already loaded - minimal cost
                resource_cost = 0.0
            else:
                # Would need to swap
                if self.swaps_this_generation >= self.max_swaps_per_gen:
                    # Budget exhausted - heavily penalize swaps
                    resource_cost = 10.0  # Extreme penalty
                else:
                    # Normal swap cost
                    resource_cost = (
                        cost_model.swap_in_cost * self.switching_cost_weight +
                        cost_model.bandwidth_cost * self.memory_cost_weight
                    )

            # Cheapness score (inverse of cost)
            cheapness = 1.0 / (1.0 + resource_cost)

            # 3. Persistence (hysteresis)
            if is_hot:
                persistence = 1.0 + self.hysteresis_boost  # +20% bonus
            else:
                persistence = 1.0

            # Composite permission score
            permission = expertise * cheapness * persistence

            expert_scores.append((expert_id, permission, trust_driven, is_hot, resource_cost))

        # Select expert with highest permission score
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        selected, score, trust_driven, was_hot, cost = expert_scores[0]

        # Track swap denial if budget blocked better expert
        if not was_hot and self.swaps_this_generation >= self.max_swaps_per_gen:
            self.swap_denied_count += 1

        # Load expert (handles LRU, tracks swaps)
        swap_cost = self._load_expert(layer, selected)

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
            was_hot=was_hot,
            swap_cost=swap_cost,
        )

        # Update expert usage tracking
        reputation = self._get_or_create_reputation(layer, selected)
        reputation.last_used_generation = self.current_generation

        return selected

    def begin_generation(self):
        """Start new generation (reset swap budget)."""
        self.current_generation += 1
        self.swaps_this_generation = 0

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
        was_hot: Optional[bool] = None,
        swap_cost: Optional[float] = None,
    ):
        """Record selection for analysis."""
        self.selection_history.append({
            'generation': self.current_generation,
            'layer': layer,
            'expert_id': expert_id,
            'context': context,
            'mechanism': mechanism,
            'trust_driven': trust_driven,
            'score': score,
            'was_hot': was_hot,
            'swap_cost': swap_cost,
            'swaps_this_gen': self.swaps_this_generation,
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
        all_reputations = [rep.reputation_score for rep in self.reputations.values()]
        reputations_with_signals = [
            rep for rep in self.reputations.values()
            if rep.total_signals > 0
        ]

        # Calculate expert churn (swaps per selection)
        churn_rate = self.total_swaps / max(1, self.total_selections)

        # Hot cache hit rate
        cache_hits = sum(1 for h in self.selection_history if h.get('was_hot'))
        cache_hit_rate = cache_hits / max(1, self.total_selections)

        stats = {
            'total_selections': self.total_selections,
            'trust_driven_count': self.trust_driven_count,
            'trust_driven_pct': 100 * self.trust_driven_count / max(1, self.total_selections),
            'exploration_count': sum(1 for h in self.selection_history if h['mechanism'] == 'exploration'),
            'quality_only_count': sum(1 for h in self.selection_history if h['mechanism'] == 'quality_only'),
            'total_swaps': self.total_swaps,
            'expert_churn_rate': churn_rate,
            'cache_hit_rate': cache_hit_rate,
            'swap_denied_count': self.swap_denied_count,
            'total_reputations_tracked': len(self.reputations),
            'reputations_with_signals': len(reputations_with_signals),
            'avg_reputation_score': sum(all_reputations) / len(all_reputations) if all_reputations else 0.5,
            'total_signals_integrated': sum(rep.total_signals for rep in self.reputations.values()),
            'hot_experts_count': len(self.hot_experts),
        }

        # First trust activation
        for i, h in enumerate(self.selection_history):
            if h['trust_driven']:
                stats['first_trust_activation'] = h['generation']
                break
        else:
            stats['first_trust_activation'] = None

        return stats


def load_jsonl_conversations(conversation_path: Path) -> List[Tuple[List[ConversationTurn], List[RepairSignal]]]:
    """Load conversations from Sprout JSONL format."""
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

                    turns.append(ConversationTurn(
                        timestamp=str(exchange.get('timestamp', line_num)),
                        speaker='user',
                        text=exchange.get('user_input', ''),
                    ))

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
    """Detect conversational repair signals from turn sequence."""
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
                break

        # Implicit engagement
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
    selector: ResourceAwareTrustSelector,
    num_generations: int = 810,
    num_selections_per_gen: int = 48,
) -> Dict:
    """Simulate expert selection with conversational signal integration."""
    all_signals = []
    for turns, signals in conversations:
        for signal in signals:
            all_signals.append(signal)

    signal_generations = sorted(random.sample(range(num_generations), min(len(all_signals), num_generations)))
    signals_by_gen = {gen: all_signals[i] for i, gen in enumerate(signal_generations)}

    logger.info(f"Simulating {num_generations} generations with {len(all_signals)} signals")
    logger.info(f"Signal coverage: {100 * len(signal_generations) / num_generations:.1f}%")

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
        # Reset swap budget if resource-aware selector
        if hasattr(selector, 'begin_generation'):
            selector.begin_generation()
        context = contexts[gen % len(contexts)]

        # Integrate signals BEFORE selections
        if gen in signals_by_gen:
            signal = signals_by_gen[gen]
            layer = random.randint(0, num_selections_per_gen - 1)
            expert_id = random.randint(0, selector.num_experts - 1)
            selector.integrate_conversational_signal(layer, expert_id, signal)

            if gen % 100 == 0:
                logger.info(f"Gen {gen}: Integrated {signal.signal_type} signal")

        # Simulate expert selections
        for layer in range(num_selections_per_gen):
            expert_id = selector.select_expert(layer, context)
            quality = random.uniform(0.3, 0.7)
            selector.record_observation(layer, expert_id, quality)

    stats = selector.get_statistics()
    return stats


def main():
    """Session 90: Trust as Resource Permission."""

    logger.info("="*80)
    logger.info("SESSION 90: TRUST AS RESOURCE PERMISSION")
    logger.info("="*80)
    logger.info("Synthesis: 'Trust = permission to consume scarce shared resources'")
    logger.info("Innovation: Hysteresis + memory cost + switching budget")
    logger.info("="*80)

    num_experts = 128
    num_layers = 48
    num_generations = 810

    # Load real Sprout conversations
    conversation_path = Path("phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    if not conversation_path.exists():
        logger.error(f"Conversation path not found: {conversation_path}")
        conversations = []
    else:
        conversations = load_jsonl_conversations(conversation_path)
        logger.info(f"Loaded {len(conversations)} real conversations")

    # Test 1: Baseline (Session 89 architecture)
    logger.info("\n" + "="*80)
    logger.info("TEST 1: BASELINE (Session 89 - Persistent Reputation)")
    logger.info("="*80)

    from session89_signal_persistence import PersistentReputationSelector

    baseline_selector = PersistentReputationSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        min_evidence_for_trust=1,
        reputation_weight=0.4,
        reputation_db_path=Path("session90_baseline_reputation.db"),
    )

    baseline_stats = simulate_expert_selection_with_signals(
        conversations=conversations,
        selector=baseline_selector,
        num_generations=num_generations,
        num_selections_per_gen=num_layers,
    )

    logger.info("\nBaseline Results:")
    logger.info(f"  Trust-driven: {baseline_stats['trust_driven_pct']:.1f}%")
    logger.info(f"  First activation: Gen {baseline_stats.get('first_trust_activation', 'never')}")
    logger.info(f"  Signals integrated: {baseline_stats['total_signals_integrated']}")

    # Test 2: Resource-Aware Routing
    logger.info("\n" + "="*80)
    logger.info("TEST 2: RESOURCE-AWARE ROUTING (Hysteresis + Memory Cost)")
    logger.info("="*80)

    resource_selector = ResourceAwareTrustSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        min_evidence_for_trust=1,
        reputation_weight=0.4,
        max_hot_experts=64,  # LRU cache size
        hysteresis_boost=0.2,  # +20% for loaded experts
        switching_cost_weight=0.3,
        memory_cost_weight=0.2,
        max_swaps_per_gen=8,  # Budgeted exploration
        reputation_db_path=Path("session90_resource_aware_reputation.db"),
    )

    resource_stats = simulate_expert_selection_with_signals(
        conversations=conversations,
        selector=resource_selector,
        num_generations=num_generations,
        num_selections_per_gen=num_layers,
    )

    logger.info("\nResource-Aware Results:")
    logger.info(f"  Trust-driven: {resource_stats['trust_driven_pct']:.1f}%")
    logger.info(f"  First activation: Gen {resource_stats.get('first_trust_activation', 'never')}")
    logger.info(f"  Signals integrated: {resource_stats['total_signals_integrated']}")
    logger.info(f"  Expert churn rate: {resource_stats['expert_churn_rate']:.3f} swaps/selection")
    logger.info(f"  Cache hit rate: {resource_stats['cache_hit_rate']:.1%}")
    logger.info(f"  Swap denials: {resource_stats['swap_denied_count']}")

    resource_selector.save_reputations()

    # Calculate improvement
    improvement = resource_stats['trust_driven_pct'] - baseline_stats['trust_driven_pct']

    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"Baseline (S89): {baseline_stats['trust_driven_pct']:.1f}% trust_driven")
    logger.info(f"Resource-aware (S90): {resource_stats['trust_driven_pct']:.1f}% trust_driven")
    logger.info(f"Improvement: {improvement:+.1f} percentage points")

    if resource_stats.get('first_trust_activation') and baseline_stats.get('first_trust_activation'):
        speedup = baseline_stats['first_trust_activation'] - resource_stats['first_trust_activation']
        logger.info(f"Activation speedup: {speedup:+d} generations")

    logger.info(f"\nResource efficiency:")
    logger.info(f"  Cache hit rate: {resource_stats['cache_hit_rate']:.1%}")
    logger.info(f"  Expert churn: {resource_stats['expert_churn_rate']:.3f} swaps/selection")
    logger.info(f"  Swap budget utilization: {resource_stats['swap_denied_count']} denials")

    # Save results
    results = {
        'session': 90,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': 'Jetson AGX Thor',
        'synthesis': 'trust = permission to consume scarce shared resources',
        'baseline': baseline_stats,
        'resource_aware': resource_stats,
        'improvement': improvement,
        'configuration': {
            'num_experts': num_experts,
            'num_layers': num_layers,
            'num_generations': num_generations,
            'max_hot_experts': 64,
            'hysteresis_boost': 0.2,
            'switching_cost_weight': 0.3,
            'memory_cost_weight': 0.2,
            'max_swaps_per_gen': 8,
        },
        'conversations_loaded': len(conversations),
    }

    results_path = Path("session90_resource_aware_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("\nSession 90 complete!")
    logger.info("Nova feedback integrated: Hysteresis + memory cost + switching budget")

    return 0


if __name__ == "__main__":
    sys.exit(main())
