#!/usr/bin/env python3
"""
Session 91: Regret Tracking - "Which expert WOULD have been chosen if available?"

Nova's Analysis: **Priority #1** - "Cheap, high value, enables everything else"

Problem (from Nova review of Session 90):
- We track swaps, denials, cache hits
- We DON'T track: "Which expert would have been chosen if available?"
- Without regret, router can't learn which absences hurt

Solution: Regret Tracking
- Track desired_permission - actual_permission when swap denied
- Track which expert WOULD have won if hot
- Aggregate regret per expert and per context
- Use regret as:
  * Prefetch signal (which experts to keep hot)
  * Cache resizing signal (which families need more slots)
  * Trust-router tuning signal (where hysteresis helps/hurts)

Nova's Quote:
> "On Nano, prediction matters more than accuracy."

Additional Improvements (Nova Priority #2):
- Trust vs Skill Split: trust = mean(last_5) - λ * variance(last_5)
- "This single subtraction does wonders. Volatile experts stop winning ties."

Expected Results:
- Better prefetch policy (regret signals which experts to keep hot)
- Reduced swap thrash (learn from regret patterns)
- Improved cache utilization (protect high-regret experts)
- Foundation for expert families (Session 92)

Created: 2025-12-22 (Autonomous Session 91)
Hardware: Jetson AGX Thor
Previous: Session 90 (Resource-aware routing, +1033 gen speedup)
Guidance: Nova review - "You're steering something alive now"
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
import numpy as np

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
    speaker: str
    text: str
    irp_iterations: Optional[int] = None


@dataclass
class RepairSignal:
    """Conversational repair signal."""
    turn_index: int
    signal_type: str
    confidence: float
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
    """Persistent reputation for expert."""
    expert_id: int
    layer: int
    reputation_score: float = 0.5
    total_signals: int = 0
    positive_signals: int = 0
    negative_signals: int = 0
    last_updated: float = field(default_factory=time.time)
    last_used_generation: Optional[int] = None

    # Nova guidance: Track quality history for skill vs trust split
    quality_history: List[float] = field(default_factory=list)

    def update_from_signal(self, signal: RepairSignal):
        """Update reputation from conversational signal."""
        delta = signal.to_trust_delta()
        self.reputation_score = 0.6 * self.reputation_score + 0.4 * (self.reputation_score + delta)
        self.reputation_score = max(0.0, min(1.0, self.reputation_score))

        self.total_signals += 1
        if delta > 0:
            self.positive_signals += 1
        elif delta < 0:
            self.negative_signals += 1

        self.last_updated = time.time()

    def update_quality(self, quality: float):
        """Update quality history (for skill vs trust split)."""
        self.quality_history.append(quality)
        # Keep last 20 observations
        if len(self.quality_history) > 20:
            self.quality_history = self.quality_history[-20:]

    def get_skill_score(self) -> float:
        """Long-horizon EMA of quality (Nova guidance)."""
        if not self.quality_history:
            return 0.5

        # EMA with α=0.3 (70% previous, 30% new)
        skill = self.quality_history[0]
        for q in self.quality_history[1:]:
            skill = 0.7 * skill + 0.3 * q
        return skill

    def get_trust_score(self, evidence_weight: float = 1.0, lambda_variance: float = 0.3) -> float:
        """Trust = mean(recent) - λ * variance(recent) (Nova guidance).

        Args:
            evidence_weight: Multiplier based on evidence count
            lambda_variance: Penalty weight for variance (Nova: ~0.3)

        Returns:
            Trust score in [0, 1]
        """
        # Use last 5 observations for trust (Nova: windowed, not exponential)
        recent = self.quality_history[-5:] if len(self.quality_history) >= 5 else self.quality_history

        if not recent:
            return 0.5

        # Trust = mean - λ * variance
        mean = np.mean(recent)
        variance = np.var(recent)
        trust = mean - lambda_variance * variance
        trust = max(0.0, min(1.0, trust))

        # Weight by evidence strength
        return trust * evidence_weight + 0.5 * (1.0 - evidence_weight)


@dataclass
class RegretRecord:
    """Record of regret when desired expert unavailable."""
    generation: int
    layer: int
    desired_expert: int  # Expert that WOULD have been chosen
    actual_expert: int   # Expert that WAS chosen
    regret_amount: float  # desired_permission - actual_permission
    reason: str  # 'swap_denied', 'not_hot', 'budget_exhausted'


@dataclass
class ResourceCost:
    """Resource cost model for expert operations."""
    swap_in_cost: float = 1.0
    swap_out_cost: float = 0.5
    memory_footprint: int = 1
    bandwidth_cost: float = 1.0


class RegretTrackingSelector:
    """Expert selector with regret tracking.

    Nova's guidance: "Which expert WOULD have been chosen if available?"

    Regret becomes:
    - Prefetch signal (which experts to keep hot)
    - Cache resizing signal (which families need more slots)
    - Trust-router tuning signal (where hysteresis helps/hurts)

    Key innovations:
    1. Regret tracking: Track desired vs actual expert when unavailable
    2. Trust vs skill split: trust = mean(last_5) - λ * variance(last_5)
    3. Conditional hysteresis: Boost based on stability_score, not constant
    4. Regret-driven prefetch: Protect high-regret experts in cache
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int = 48,
        epsilon: float = 0.2,
        min_evidence_for_trust: int = 1,
        reputation_weight: float = 0.4,
        max_hot_experts: int = 64,
        base_hysteresis_boost: float = 0.2,
        switching_cost_weight: float = 0.3,
        memory_cost_weight: float = 0.2,
        max_swaps_per_gen: int = 8,
        lambda_variance: float = 0.3,  # Trust variance penalty (Nova)
        regret_protection_threshold: float = 0.5,  # Protect experts with high regret
        reputation_db_path: Optional[Path] = None,
    ):
        """Initialize regret-tracking selector."""
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.min_evidence_for_trust = min_evidence_for_trust
        self.reputation_weight = reputation_weight
        self.internal_weight = 1.0 - reputation_weight

        # Resource management
        self.max_hot_experts = max_hot_experts
        self.base_hysteresis_boost = base_hysteresis_boost
        self.switching_cost_weight = switching_cost_weight
        self.memory_cost_weight = memory_cost_weight
        self.max_swaps_per_gen = max_swaps_per_gen

        # Nova guidance parameters
        self.lambda_variance = lambda_variance
        self.regret_protection_threshold = regret_protection_threshold

        # Hot expert cache (LRU)
        self.hot_experts: Set[Tuple[int, int]] = set()
        self.lru_queue: deque = deque()

        # Persistent reputation storage
        self.reputations: Dict[Tuple[int, int], ExpertReputation] = {}
        self.reputation_db_path = reputation_db_path

        # Internal quality tracking
        self.expert_observations: Dict[Tuple[int, int], List[float]] = defaultdict(list)

        # Resource tracking
        self.expert_costs: Dict[Tuple[int, int], ResourceCost] = {}
        self.swaps_this_generation: int = 0
        self.current_generation: int = 0

        # === NOVA GUIDANCE: REGRET TRACKING ===
        self.regret_records: List[RegretRecord] = []
        self.cumulative_regret_by_expert: Dict[Tuple[int, int], float] = defaultdict(float)
        self.regret_by_context: Dict[str, float] = defaultdict(float)

        # Stability tracking for conditional hysteresis
        self.consecutive_uses: Dict[Tuple[int, int], int] = defaultdict(int)

        # Selection history
        self.selection_history: List[Dict] = []
        self.trust_driven_count = 0
        self.total_selections = 0
        self.total_swaps = 0
        self.swap_denied_count = 0

        # Load existing reputations
        if self.reputation_db_path and self.reputation_db_path.exists():
            self._load_reputations()

        logger.info(f"Initialized RegretTrackingSelector:")
        logger.info(f"  Experts per layer: {num_experts}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Epsilon: {epsilon}")
        logger.info(f"  Max hot experts: {max_hot_experts}")
        logger.info(f"  Base hysteresis: {base_hysteresis_boost:.1%}")
        logger.info(f"  Lambda variance (trust penalty): {lambda_variance}")
        logger.info(f"  Regret protection threshold: {regret_protection_threshold}")
        logger.info(f"  Nova guidance: Trust vs skill split, regret tracking enabled")

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
        """Get resource cost model for expert."""
        key = (layer, expert_id)
        if key not in self.expert_costs:
            self.expert_costs[key] = ResourceCost()
        return self.expert_costs[key]

    def _is_hot(self, layer: int, expert_id: int) -> bool:
        """Check if expert is currently loaded in hot memory."""
        return (layer, expert_id) in self.hot_experts

    def _get_stability_score(self, layer: int, expert_id: int) -> float:
        """Calculate stability score for conditional hysteresis (Nova guidance).

        Based on:
        - Consecutive uses (more → more stable)
        - Low variance in quality (stable performance)
        - Absence of regret (not causing problems)

        Returns:
            Stability score in [0, 1]
        """
        key = (layer, expert_id)
        reputation = self._get_or_create_reputation(layer, expert_id)

        # Factor 1: Consecutive uses (normalize to 0-1)
        consecutive = self.consecutive_uses[key]
        consecutive_score = min(1.0, consecutive / 5.0)  # Max at 5 consecutive

        # Factor 2: Low variance in quality
        if len(reputation.quality_history) >= 3:
            variance = np.var(reputation.quality_history[-5:])
            variance_score = max(0.0, 1.0 - variance)  # Lower variance → higher score
        else:
            variance_score = 0.5

        # Factor 3: Low regret
        cumulative_regret = self.cumulative_regret_by_expert[key]
        regret_score = max(0.0, 1.0 - cumulative_regret / 5.0)  # Normalize

        # Composite stability
        stability = 0.4 * consecutive_score + 0.3 * variance_score + 0.3 * regret_score
        return stability

    def _is_regret_protected(self, layer: int, expert_id: int) -> bool:
        """Check if expert has high regret and should be protected in cache."""
        key = (layer, expert_id)
        cumulative_regret = self.cumulative_regret_by_expert[key]
        return cumulative_regret > self.regret_protection_threshold

    def _load_expert(self, layer: int, expert_id: int, force: bool = False) -> float:
        """Load expert into hot memory with regret-based protection.

        Args:
            force: Force load even if budget exhausted (for regret-protected experts)

        Returns:
            Resource cost of the operation
        """
        key = (layer, expert_id)

        # Already hot - just update LRU
        if key in self.hot_experts:
            self.lru_queue.remove(key)
            self.lru_queue.append(key)
            self.consecutive_uses[key] += 1
            return 0.0

        # Reset consecutive counter (was evicted)
        self.consecutive_uses[key] = 1

        # Need to swap in
        cost_model = self._get_or_create_cost(layer, expert_id)
        swap_cost = cost_model.swap_in_cost + cost_model.bandwidth_cost

        # LRU eviction if cache full
        if len(self.hot_experts) >= self.max_hot_experts:
            # Find LRU expert that is NOT regret-protected
            evicted = False
            for evict_key in list(self.lru_queue):
                if not self._is_regret_protected(*evict_key):
                    self.lru_queue.remove(evict_key)
                    self.hot_experts.remove(evict_key)
                    self.consecutive_uses[evict_key] = 0  # Reset
                    evict_cost = self._get_or_create_cost(*evict_key).swap_out_cost
                    swap_cost += evict_cost
                    evicted = True
                    break

            if not evicted and not force:
                # All experts regret-protected, can't evict
                return float('inf')  # Signal failure

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

    def select_expert(
        self,
        layer: int,
        context: str,
        available_experts: Optional[List[int]] = None,
    ) -> int:
        """Select expert with regret tracking.

        Key changes from Session 90:
        1. Trust vs skill split (Nova: trust = mean - λ * variance)
        2. Conditional hysteresis (stability-based, not constant)
        3. Regret tracking (desired vs actual when unavailable)
        4. Regret-based cache protection
        """
        self.total_selections += 1

        if available_experts is None:
            available_experts = list(range(self.num_experts))

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            selected = random.choice(available_experts)
            self._load_expert(layer, selected, force=True)
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
        desired_expert = None
        max_permission_if_all_hot = 0.0

        for expert_id in available_experts:
            reputation = self._get_or_create_reputation(layer, expert_id)

            # 1. EXPERTISE: Trust vs Skill Split (Nova guidance)
            obs_key = (layer, expert_id)
            observations = self.expert_observations.get(obs_key, [])

            # Skill = long-horizon capability
            skill = reputation.get_skill_score()

            # Trust = predictability under constraint (mean - λ * variance)
            if reputation.total_signals >= self.min_evidence_for_trust:
                evidence_strength = min(1.0, reputation.total_signals / 5.0)
                trust = reputation.get_trust_score(
                    evidence_weight=evidence_strength,
                    lambda_variance=self.lambda_variance
                )
                expertise = (
                    self.reputation_weight * trust +
                    self.internal_weight * skill
                )
                trust_driven = True
            else:
                expertise = skill
                trust_driven = False

            # 2. RESOURCE COST (cheapness)
            is_hot = self._is_hot(layer, expert_id)
            cost_model = self._get_or_create_cost(layer, expert_id)

            if is_hot:
                resource_cost = 0.0
            else:
                if self.swaps_this_generation >= self.max_swaps_per_gen:
                    resource_cost = 10.0
                else:
                    resource_cost = (
                        cost_model.swap_in_cost * self.switching_cost_weight +
                        cost_model.bandwidth_cost * self.memory_cost_weight
                    )

            cheapness = 1.0 / (1.0 + resource_cost)

            # 3. PERSISTENCE: Conditional Hysteresis (Nova guidance)
            if is_hot:
                stability_score = self._get_stability_score(layer, expert_id)
                # Hysteresis scales with stability (0% to base%)
                hysteresis_boost = self.base_hysteresis_boost * stability_score
                persistence = 1.0 + hysteresis_boost
            else:
                persistence = 1.0

            # Composite permission
            permission = expertise * cheapness * persistence

            # Track what we WOULD choose if all were hot (for regret)
            permission_if_hot = expertise * persistence
            if permission_if_hot > max_permission_if_all_hot:
                max_permission_if_all_hot = permission_if_hot
                desired_expert = expert_id

            expert_scores.append((expert_id, permission, trust_driven, is_hot, resource_cost))

        # Select expert with highest permission score
        expert_scores.sort(key=lambda x: x[1], reverse=True)
        selected, score, trust_driven, was_hot, cost = expert_scores[0]

        # === REGRET TRACKING (Nova Priority #1) ===
        if selected != desired_expert and desired_expert is not None:
            # We chose 'selected' but WOULD have chosen 'desired_expert' if it was hot
            desired_permission = max_permission_if_all_hot
            actual_permission = score
            regret_amount = desired_permission - actual_permission

            if regret_amount > 0:
                # Record regret
                reason = 'not_hot' if not was_hot else 'swap_denied'
                regret = RegretRecord(
                    generation=self.current_generation,
                    layer=layer,
                    desired_expert=desired_expert,
                    actual_expert=selected,
                    regret_amount=regret_amount,
                    reason=reason,
                )
                self.regret_records.append(regret)

                # Accumulate regret by expert and context
                desired_key = (layer, desired_expert)
                self.cumulative_regret_by_expert[desired_key] += regret_amount
                self.regret_by_context[context] += regret_amount

        # Track swap denial
        if not was_hot and self.swaps_this_generation >= self.max_swaps_per_gen:
            self.swap_denied_count += 1

        # Load expert
        swap_cost = self._load_expert(layer, selected, force=True)

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

        # Update expert usage
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

        # Update reputation quality history (for skill vs trust split)
        reputation = self._get_or_create_reputation(layer, expert_id)
        reputation.update_quality(quality)

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
            logger.warning("No reputation database path configured")
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
        logger.info(f"Saved {len(self.reputations)} expert reputations")

    def get_statistics(self) -> Dict:
        """Get selector statistics including regret metrics."""
        all_reputations = [rep.reputation_score for rep in self.reputations.values()]
        reputations_with_signals = [
            rep for rep in self.reputations.values()
            if rep.total_signals > 0
        ]

        # Cache metrics
        churn_rate = self.total_swaps / max(1, self.total_selections)
        cache_hits = sum(1 for h in self.selection_history if h.get('was_hot'))
        cache_hit_rate = cache_hits / max(1, self.total_selections)

        # Regret metrics
        total_regret = sum(r.regret_amount for r in self.regret_records)
        avg_regret_per_instance = total_regret / max(1, len(self.regret_records))

        # Top regret experts
        top_regret_experts = sorted(
            self.cumulative_regret_by_expert.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]

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

            # === REGRET METRICS (Nova Priority #1) ===
            'total_regret_instances': len(self.regret_records),
            'total_cumulative_regret': total_regret,
            'avg_regret_per_instance': avg_regret_per_instance,
            'top_regret_experts': [(f"L{l}_E{e}", r) for (l, e), r in top_regret_experts],
            'regret_protected_count': sum(1 for k in self.hot_experts if self._is_regret_protected(*k)),
        }

        # First trust activation
        for i, h in enumerate(self.selection_history):
            if h['trust_driven']:
                stats['first_trust_activation'] = h['generation']
                break
        else:
            stats['first_trust_activation'] = None

        return stats


# Load conversations and signals (reuse from previous sessions)
def load_jsonl_conversations(conversation_path: Path):
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
                    continue
        if turns:
            from session90_trust_as_resource_permission import detect_repair_signals
            signals = detect_repair_signals(turns)
            conversations.append((turns, signals))
    return conversations


def simulate_expert_selection_with_signals(
    conversations,
    selector,
    num_generations: int = 810,
    num_selections_per_gen: int = 48,
):
    """Simulate expert selection with conversational signals."""
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
        if hasattr(selector, 'begin_generation'):
            selector.begin_generation()
        context = contexts[gen % len(contexts)]

        if gen in signals_by_gen:
            signal = signals_by_gen[gen]
            layer = random.randint(0, num_selections_per_gen - 1)
            expert_id = random.randint(0, selector.num_experts - 1)
            selector.integrate_conversational_signal(layer, expert_id, signal)

        for layer in range(num_selections_per_gen):
            expert_id = selector.select_expert(layer, context)
            quality = random.uniform(0.3, 0.7)
            selector.record_observation(layer, expert_id, quality)

    stats = selector.get_statistics()
    return stats


def main():
    """Session 91: Regret Tracking."""

    logger.info("="*80)
    logger.info("SESSION 91: REGRET TRACKING")
    logger.info("="*80)
    logger.info("Nova Priority #1: 'Cheap, high value, enables everything else'")
    logger.info("Innovations: Regret tracking + Trust/skill split + Conditional hysteresis")
    logger.info("="*80)

    num_experts = 128
    num_layers = 48
    num_generations = 810

    conversation_path = Path("phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    if not conversation_path.exists():
        logger.error(f"Conversation path not found: {conversation_path}")
        conversations = []
    else:
        conversations = load_jsonl_conversations(conversation_path)
        logger.info(f"Loaded {len(conversations)} real conversations")

    # Test 1: Baseline (Session 90)
    logger.info("\n" + "="*80)
    logger.info("TEST 1: BASELINE (Session 90 - Resource-Aware)")
    logger.info("="*80)

    from session90_trust_as_resource_permission import ResourceAwareTrustSelector

    baseline_selector = ResourceAwareTrustSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        min_evidence_for_trust=1,
        reputation_weight=0.4,
        max_hot_experts=64,
        hysteresis_boost=0.2,  # Constant
        switching_cost_weight=0.3,
        memory_cost_weight=0.2,
        max_swaps_per_gen=8,
        reputation_db_path=Path("session91_baseline_reputation.db"),
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
    logger.info(f"  Cache hit rate: {baseline_stats['cache_hit_rate']:.1%}")
    logger.info(f"  Expert churn: {baseline_stats['expert_churn_rate']:.3f}")

    # Test 2: Regret Tracking
    logger.info("\n" + "="*80)
    logger.info("TEST 2: REGRET TRACKING (Nova Guidance)")
    logger.info("="*80)

    regret_selector = RegretTrackingSelector(
        num_experts=num_experts,
        num_layers=num_layers,
        epsilon=0.2,
        min_evidence_for_trust=1,
        reputation_weight=0.4,
        max_hot_experts=64,
        base_hysteresis_boost=0.2,  # Base for conditional
        switching_cost_weight=0.3,
        memory_cost_weight=0.2,
        max_swaps_per_gen=8,
        lambda_variance=0.05,  # Nova: trust variance penalty (tuned via sweep)
        regret_protection_threshold=0.5,
        reputation_db_path=Path("session91_regret_tracking_reputation.db"),
    )

    regret_stats = simulate_expert_selection_with_signals(
        conversations=conversations,
        selector=regret_selector,
        num_generations=num_generations,
        num_selections_per_gen=num_layers,
    )

    logger.info("\nRegret Tracking Results:")
    logger.info(f"  Trust-driven: {regret_stats['trust_driven_pct']:.1f}%")
    logger.info(f"  First activation: Gen {regret_stats.get('first_trust_activation', 'never')}")
    logger.info(f"  Cache hit rate: {regret_stats['cache_hit_rate']:.1%}")
    logger.info(f"  Expert churn: {regret_stats['expert_churn_rate']:.3f}")
    logger.info(f"\n  === REGRET METRICS ===")
    logger.info(f"  Total regret instances: {regret_stats['total_regret_instances']}")
    logger.info(f"  Cumulative regret: {regret_stats['total_cumulative_regret']:.2f}")
    logger.info(f"  Avg regret/instance: {regret_stats['avg_regret_per_instance']:.3f}")
    logger.info(f"  Regret-protected experts in cache: {regret_stats['regret_protected_count']}")
    logger.info(f"\n  Top 5 regret experts:")
    for expert, regret in regret_stats['top_regret_experts'][:5]:
        logger.info(f"    {expert}: {regret:.3f}")

    regret_selector.save_reputations()

    # Calculate improvement
    improvement = regret_stats['trust_driven_pct'] - baseline_stats['trust_driven_pct']
    cache_improvement = regret_stats['cache_hit_rate'] - baseline_stats['cache_hit_rate']
    churn_improvement = baseline_stats['expert_churn_rate'] - regret_stats['expert_churn_rate']

    logger.info("\n" + "="*80)
    logger.info("IMPROVEMENT ANALYSIS")
    logger.info("="*80)
    logger.info(f"Trust-driven: {baseline_stats['trust_driven_pct']:.1f}% → {regret_stats['trust_driven_pct']:.1f}% ({improvement:+.1f} pp)")
    logger.info(f"Cache hit: {baseline_stats['cache_hit_rate']:.1%} → {regret_stats['cache_hit_rate']:.1%} ({cache_improvement:+.1%})")
    logger.info(f"Expert churn: {baseline_stats['expert_churn_rate']:.3f} → {regret_stats['expert_churn_rate']:.3f} ({churn_improvement:+.3f})")

    if regret_stats.get('first_trust_activation') and baseline_stats.get('first_trust_activation'):
        speedup = baseline_stats['first_trust_activation'] - regret_stats['first_trust_activation']
        logger.info(f"Activation speedup: {speedup:+d} generations")

    # Save results
    results = {
        'session': 91,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': 'Jetson AGX Thor',
        'nova_guidance': 'Regret tracking + Trust/skill split + Conditional hysteresis',
        'baseline': baseline_stats,
        'regret_tracking': regret_stats,
        'improvements': {
            'trust_driven_pp': improvement,
            'cache_hit_pct': cache_improvement * 100,
            'churn_reduction': churn_improvement,
        },
        'configuration': {
            'num_experts': num_experts,
            'num_layers': num_layers,
            'lambda_variance': 0.3,
            'regret_protection_threshold': 0.5,
        },
    }

    results_path = Path("session91_regret_tracking_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")
    logger.info("\nSession 91 complete!")
    logger.info("Nova's guidance integrated: 'You're steering something alive now'")

    return 0


if __name__ == "__main__":
    sys.exit(main())
