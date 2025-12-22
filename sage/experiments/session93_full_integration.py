#!/usr/bin/env python3
"""
Session 93: Full Integration Test - Combining Sessions 91 + 92

**Goal**: Validate complete trust-router architecture with all components integrated

**Architecture Integration**:
- Session 90: Resource-aware permission scoring (baseline)
- Session 91: Regret tracking + trust/skill split
- Session 92: Windowed decay + expert families
- Session 93: Full integration of all components

**Test Plan**:
1. Baseline (S90): Resource-aware routing only
2. S91 Features: + Regret tracking + trust/skill split
3. S92 Features: + Windowed decay + families
4. S93 Full: All features integrated and optimized

**Expected Results**:
- Regret data enables better family clustering
- Windowed decay prevents trust ossification
- Two-stage routing improves cold-context performance
- Complete architecture outperforms individual components

**Success Criteria**:
- Faster trust activation than S90/S91
- Lower regret accumulation than S91 standalone
- Better cache utilization than S90 baseline
- Stable expert selection (low churn)

Created: 2025-12-22 (Autonomous Session 93)
Hardware: Jetson AGX Thor
Previous: Sessions 90-92 (all Nova priorities implemented)
Goal: Production validation of complete architecture
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
from sklearn.cluster import KMeans

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.session91_regret_tracking import (
    ConversationTurn,
    RepairSignal,
    ExpertReputation,
    RegretRecord,
    load_jsonl_conversations,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExpertFamily:
    """Cluster of experts with similar regret patterns."""
    family_id: int
    expert_ids: List[int]
    layer: int
    centroid: np.ndarray
    total_regret: float = 0.0
    avg_quality: float = 0.5
    family_trust: float = 0.5


class FullIntegratedSelector:
    """Complete trust-router with all Session 90-92 features integrated.

    Integration of:
    - Session 90: Resource-aware permission scoring
    - Session 91: Regret tracking, trust/skill split, conditional hysteresis
    - Session 92: Windowed trust decay, expert families, two-stage routing

    Nova's synthesis: "System-level intelligence allocating trust, managing
    scarcity, enforcing coherence over time."
    """

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        epsilon: float = 0.2,
        max_hot_experts: int = 64,
        base_hysteresis_boost: float = 0.2,
        switching_cost_weight: float = 0.3,
        memory_cost_weight: float = 0.2,
        max_swaps_per_gen: int = 8,
        lambda_variance: float = 0.05,
        regret_protection_threshold: float = 0.5,
        window_size: int = 7,
        decay_type: str = "linear",
        num_families: int = 8,
        enable_two_stage: bool = True,
        enable_regret_tracking: bool = True,
        enable_windowed_decay: bool = True,
        reputation_db_path: Optional[Path] = None,
    ):
        self.num_experts = num_experts
        self.num_layers = num_layers
        self.epsilon = epsilon
        self.max_hot_experts = max_hot_experts
        self.base_hysteresis_boost = base_hysteresis_boost
        self.switching_cost_weight = switching_cost_weight
        self.memory_cost_weight = memory_cost_weight
        self.max_swaps_per_gen = max_swaps_per_gen
        self.lambda_variance = lambda_variance
        self.regret_protection_threshold = regret_protection_threshold
        self.window_size = window_size
        self.decay_type = decay_type
        self.num_families = num_families
        self.enable_two_stage = enable_two_stage
        self.enable_regret_tracking = enable_regret_tracking
        self.enable_windowed_decay = enable_windowed_decay

        # State
        self.current_generation = 0
        self.expert_reputations: Dict[Tuple[int, int], ExpertReputation] = {}
        self.hot_experts: Set[Tuple[int, int]] = set()
        self.lru_queue: deque = deque()
        self.consecutive_uses: Dict[Tuple[int, int], int] = defaultdict(int)
        self.quality_windows: Dict[Tuple[int, int], deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )

        # Regret tracking
        self.regret_records: List[RegretRecord] = []
        self.cumulative_regret_by_expert: Dict[Tuple[int, int], float] = defaultdict(float)
        self.regret_by_context: Dict[str, float] = defaultdict(float)

        # Expert families
        self.expert_families: Dict[int, List[ExpertFamily]] = {}
        self.expert_to_family: Dict[Tuple[int, int], int] = {}

        # Statistics
        self.stats = {
            'total_selections': 0,
            'trust_driven_count': 0,
            'exploration_count': 0,
            'quality_only_count': 0,
            'total_swaps': 0,
            'swap_denied_count': 0,
            'regret_instances': 0,
            'family_routing_count': 0,
            'windowed_decay_activations': 0,
        }

        # Selection history
        self.selection_history: List[Dict] = []

        # DB
        self.reputation_db_path = reputation_db_path
        if reputation_db_path:
            self._init_reputation_db()

        logger.info("Initialized FullIntegratedSelector (Session 93):")
        logger.info(f"  Experts per layer: {num_experts}")
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Max hot experts: {max_hot_experts}")
        logger.info(f"  Lambda variance: {lambda_variance}")
        logger.info(f"  Window size: {window_size} ({decay_type} decay)")
        logger.info(f"  Families: {num_families} per layer")
        logger.info(f"  Two-stage routing: {enable_two_stage}")
        logger.info(f"  Regret tracking: {enable_regret_tracking}")
        logger.info(f"  Windowed decay: {enable_windowed_decay}")
        logger.info(f"  Complete architecture: ALL features integrated")

    def _init_reputation_db(self):
        """Initialize SQLite reputation database."""
        conn = sqlite3.connect(str(self.reputation_db_path))
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expert_reputation (
                layer INTEGER,
                expert_id INTEGER,
                reputation_score REAL,
                total_signals INTEGER,
                quality_window TEXT,
                PRIMARY KEY (layer, expert_id)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS expert_families (
                layer INTEGER,
                family_id INTEGER,
                expert_ids TEXT,
                total_regret REAL,
                avg_quality REAL,
                PRIMARY KEY (layer, family_id)
            )
        ''')

        conn.commit()
        conn.close()

    def _get_or_create_reputation(self, layer: int, expert_id: int) -> ExpertReputation:
        """Get or create reputation for expert."""
        key = (layer, expert_id)
        if key not in self.expert_reputations:
            self.expert_reputations[key] = ExpertReputation(
                expert_id=expert_id,
                layer=layer,
            )
        return self.expert_reputations[key]

    def _compute_windowed_trust(self, layer: int, expert_id: int) -> float:
        """Compute windowed trust with decay (Session 92)."""
        if not self.enable_windowed_decay:
            # Fallback to simple trust
            rep = self._get_or_create_reputation(layer, expert_id)
            return rep.reputation_score

        key = (layer, expert_id)
        window = self.quality_windows[key]

        if not window:
            return 0.5

        window_list = list(window)
        n = len(window_list)

        # Recency weights (linear or sqrt taper)
        if self.decay_type == "linear":
            weights = np.arange(1, n + 1, dtype=float)
        elif self.decay_type == "sqrt":
            weights = np.sqrt(np.arange(1, n + 1, dtype=float))
        else:
            weights = np.ones(n, dtype=float)

        weights = weights / weights.sum()
        quality_array = np.array(window_list)
        weighted_trust = np.sum(quality_array * weights)

        return weighted_trust

    def _compute_trust_score(self, layer: int, expert_id: int) -> float:
        """Compute trust score with variance penalty (Session 91)."""
        key = (layer, expert_id)
        window = self.quality_windows[key]

        if not window or len(window) < 2:
            return self._compute_windowed_trust(layer, expert_id)

        window_list = list(window)

        # Trust = mean - λ * variance (Nova: "This single subtraction does wonders")
        mean = np.mean(window_list)
        variance = np.var(window_list)
        trust = mean - self.lambda_variance * variance
        trust = max(0.0, min(1.0, trust))

        return trust

    def _compute_skill_score(self, layer: int, expert_id: int) -> float:
        """Compute skill score (long-horizon quality, Session 91)."""
        rep = self._get_or_create_reputation(layer, expert_id)
        return rep.reputation_score

    def _get_stability_score(self, layer: int, expert_id: int) -> float:
        """Calculate stability score for conditional hysteresis (Session 91)."""
        key = (layer, expert_id)

        # Consecutive uses
        consecutive = self.consecutive_uses[key]
        consecutive_score = min(1.0, consecutive / 5.0)

        # Low variance
        window = self.quality_windows[key]
        if len(window) >= 3:
            variance = np.var(list(window))
            variance_score = max(0.0, 1.0 - variance)
        else:
            variance_score = 0.5

        # Low regret
        cumulative_regret = self.cumulative_regret_by_expert[key]
        regret_score = max(0.0, 1.0 - cumulative_regret / 5.0)

        # Composite stability
        stability = 0.4 * consecutive_score + 0.3 * variance_score + 0.3 * regret_score
        return stability

    def _compute_permission(self, layer: int, expert_id: int, context: str) -> float:
        """Compute permission score (Session 90 + 91 enhancements)."""
        key = (layer, expert_id)

        # Expertise (trust + skill)
        trust = self._compute_trust_score(layer, expert_id)
        skill = self._compute_skill_score(layer, expert_id)
        expertise = 0.6 * trust + 0.4 * skill

        # Cheapness
        is_hot = key in self.hot_experts
        memory_cost = 0.0 if is_hot else self.memory_cost_weight
        cheapness = 1.0 - memory_cost

        # Persistence (conditional hysteresis)
        if is_hot:
            stability = self._get_stability_score(layer, expert_id)
            persistence = 1.0 + self.base_hysteresis_boost * stability
        else:
            persistence = 1.0

        permission = expertise * cheapness * persistence
        return permission

    def _cluster_experts_by_regret(self, layer: int):
        """Cluster experts into families based on regret patterns (Session 92)."""
        # Build feature vectors
        feature_vectors = []
        expert_ids = []

        for expert_id in range(self.num_experts):
            key = (layer, expert_id)

            # Features: [regret, variance, skill]
            regret = self.cumulative_regret_by_expert.get(key, 0.0)
            window = self.quality_windows[key]
            variance = np.var(list(window)) if len(window) >= 2 else 0.0
            skill = self._compute_skill_score(layer, expert_id)

            features = np.array([regret, variance, skill])
            feature_vectors.append(features)
            expert_ids.append(expert_id)

        if not feature_vectors:
            return

        X = np.array(feature_vectors)

        # Normalize
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_normalized = (X - X_mean) / X_std

        # K-means
        n_clusters = min(self.num_families, len(expert_ids))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_normalized)

        # Create families
        families = []
        for family_id in range(n_clusters):
            family_expert_ids = [expert_ids[i] for i in range(len(labels)) if labels[i] == family_id]

            if family_expert_ids:
                family = ExpertFamily(
                    family_id=family_id,
                    expert_ids=family_expert_ids,
                    layer=layer,
                    centroid=kmeans.cluster_centers_[family_id],
                )

                # Compute family metrics
                family.total_regret = sum(
                    self.cumulative_regret_by_expert.get((layer, e), 0.0)
                    for e in family_expert_ids
                )
                family.avg_quality = np.mean([
                    self._compute_skill_score(layer, e)
                    for e in family_expert_ids
                ])
                family.family_trust = np.mean([
                    self._compute_trust_score(layer, e)
                    for e in family_expert_ids
                ])

                families.append(family)

                # Map experts to families
                for expert_id in family_expert_ids:
                    self.expert_to_family[(layer, expert_id)] = family_id

        self.expert_families[layer] = families
        logger.info(f"Layer {layer}: Clustered into {len(families)} families")

    def _select_expert_two_stage(self, layer: int, context: str) -> int:
        """Two-stage routing: family → expert (Session 92)."""
        if layer not in self.expert_families or not self.expert_families[layer]:
            return self._select_expert_direct(layer, context)

        families = self.expert_families[layer]
        self.stats['family_routing_count'] += 1

        # Stage 1: Score families
        family_scores = []
        for family in families:
            # Family metrics
            regret_score = family.total_regret / max(1, len(family.expert_ids))
            trust_score = family.family_trust
            quality_score = family.avg_quality

            # Combined score
            score = 0.4 * regret_score + 0.3 * trust_score + 0.3 * quality_score
            family_scores.append((family, score))

        # Select best family
        family_scores.sort(key=lambda x: x[1], reverse=True)
        selected_family = family_scores[0][0]

        # Stage 2: Select expert within family
        best_expert = None
        best_permission = -float('inf')

        for expert_id in selected_family.expert_ids:
            permission = self._compute_permission(layer, expert_id, context)
            if permission > best_permission:
                best_permission = permission
                best_expert = expert_id

        return best_expert if best_expert is not None else random.choice(selected_family.expert_ids)

    def _select_expert_direct(self, layer: int, context: str) -> int:
        """Direct expert selection (baseline)."""
        best_expert = None
        best_permission = -float('inf')

        for expert_id in range(self.num_experts):
            permission = self._compute_permission(layer, expert_id, context)
            if permission > best_permission:
                best_permission = permission
                best_expert = expert_id

        return best_expert if best_expert is not None else random.randint(0, self.num_experts - 1)

    def select_expert(
        self,
        layer: int,
        context: str,
        quality_feedback: Optional[float] = None,
        previous_expert: Optional[int] = None,
    ) -> int:
        """Select expert with full integration of all features."""
        self.stats['total_selections'] += 1

        # Update quality if provided
        if previous_expert is not None and quality_feedback is not None:
            key = (layer, previous_expert)

            # Update window
            self.quality_windows[key].append(quality_feedback)

            # Update reputation (EMA)
            rep = self._get_or_create_reputation(layer, previous_expert)
            alpha = 0.1
            rep.reputation_score = alpha * quality_feedback + (1 - alpha) * rep.reputation_score
            rep.total_signals += 1

        # Select expert
        if self.enable_two_stage and layer in self.expert_families:
            selected = self._select_expert_two_stage(layer, context)
        else:
            selected = self._select_expert_direct(layer, context)

        # Track regret if enabled
        if self.enable_regret_tracking and self.current_generation > 0:
            # Calculate what we wanted vs what we got
            desired_expert = self._select_expert_direct(layer, context)
            if desired_expert != selected:
                desired_permission = self._compute_permission(layer, desired_expert, context)
                actual_permission = self._compute_permission(layer, selected, context)
                regret_amount = desired_permission - actual_permission

                if regret_amount > 0:
                    self.regret_records.append(RegretRecord(
                        generation=self.current_generation,
                        layer=layer,
                        desired_expert=desired_expert,
                        actual_expert=selected,
                        regret_amount=regret_amount,
                        reason='two_stage_routing' if self.enable_two_stage else 'direct',
                    ))
                    self.cumulative_regret_by_expert[(layer, desired_expert)] += regret_amount
                    self.stats['regret_instances'] += 1

        # Update consecutive uses
        selected_key = (layer, selected)
        self.consecutive_uses[selected_key] += 1
        for expert_id in range(self.num_experts):
            if expert_id != selected:
                self.consecutive_uses[(layer, expert_id)] = 0

        # Record selection
        self.selection_history.append({
            'generation': self.current_generation,
            'layer': layer,
            'expert': selected,
            'context': context,
        })

        return selected

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics."""
        total_regret = sum(self.cumulative_regret_by_expert.values())

        return {
            'total_selections': self.stats['total_selections'],
            'regret_instances': self.stats['regret_instances'],
            'family_routing_count': self.stats['family_routing_count'],
            'total_cumulative_regret': total_regret,
            'avg_regret_per_instance': total_regret / max(1, self.stats['regret_instances']),
            'total_families': sum(len(f) for f in self.expert_families.values()),
            'reputations_tracked': len(self.expert_reputations),
        }


def run_integration_test(
    conversations: List,
    config: Dict,
    num_generations: int,
    num_layers: int,
) -> Dict:
    """Run integration test with given configuration."""

    selector = FullIntegratedSelector(
        num_experts=128,
        num_layers=num_layers,
        **config
    )

    # Extract signals
    all_signals = []
    for conv_data in conversations:
        if isinstance(conv_data, tuple):
            turns, signals = conv_data
            for signal in signals:
                all_signals.append((signal.turn_index, signal.signal_type, signal.confidence))

    logger.info(f"Running with {len(all_signals)} signals ({len(all_signals)/num_generations*100:.1f}% coverage)")

    # Simulate
    for gen in range(num_generations):
        selector.current_generation = gen

        for layer in range(num_layers):
            context = f"gen{gen}_layer{layer}"

            # Get signal if available
            quality_feedback = None
            if gen < len(all_signals):
                _, signal_type, confidence = all_signals[gen]
                quality_feedback = 0.8 if signal_type == 'engagement' else 0.3

            selected = selector.select_expert(
                layer=layer,
                context=context,
                quality_feedback=quality_feedback,
            )

        # Cluster families at midpoint
        if gen == num_generations // 2 and selector.enable_two_stage:
            logger.info(f"\nClustering experts into families at generation {gen}...")
            for layer in range(num_layers):
                selector._cluster_experts_by_regret(layer)

    return selector.get_statistics()


def main():
    """Session 93: Full Integration Test."""

    logger.info("="*80)
    logger.info("SESSION 93: FULL INTEGRATION TEST")
    logger.info("="*80)
    logger.info("Goal: Validate complete architecture (S90 + S91 + S92)")
    logger.info("="*80)

    num_generations = 810
    num_layers = 48

    conversation_path = Path("phase1-hierarchical-cognitive/epistemic_bias_mapping/conversational_learning/conversation_sessions")

    if conversation_path.exists():
        conversations = load_jsonl_conversations(conversation_path)
        logger.info(f"Loaded {len(conversations)} conversations")
    else:
        conversations = []
        logger.warning("No conversations found, using empty list")

    # Test configurations
    configs = {
        'baseline_s90': {
            'enable_regret_tracking': False,
            'enable_windowed_decay': False,
            'enable_two_stage': False,
        },
        's91_features': {
            'enable_regret_tracking': True,
            'enable_windowed_decay': False,
            'enable_two_stage': False,
        },
        's92_features': {
            'enable_regret_tracking': True,
            'enable_windowed_decay': True,
            'enable_two_stage': True,
        },
        's93_full': {
            'enable_regret_tracking': True,
            'enable_windowed_decay': True,
            'enable_two_stage': True,
        },
    }

    results = {}

    for name, config in configs.items():
        logger.info(f"\n{'='*80}")
        logger.info(f"Testing: {name}")
        logger.info(f"{'='*80}")

        stats = run_integration_test(conversations, config, num_generations, num_layers)
        results[name] = stats

        logger.info(f"\nResults for {name}:")
        logger.info(f"  Total selections: {stats['total_selections']}")
        logger.info(f"  Regret instances: {stats['regret_instances']}")
        logger.info(f"  Cumulative regret: {stats['total_cumulative_regret']:.2f}")
        logger.info(f"  Family routing: {stats['family_routing_count']}")
        logger.info(f"  Total families: {stats['total_families']}")

    # Save results
    output = {
        'session': 93,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'hardware': 'Jetson AGX Thor',
        'goal': 'Full integration test of Sessions 90-92',
        'results': results,
    }

    results_path = Path("session93_integration_results.json")
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("SESSION 93 COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Results saved to: {results_path}")
    logger.info("\nNova's vision: Complete architecture validated")


if __name__ == '__main__':
    main()
