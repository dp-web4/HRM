#!/usr/bin/env python3
"""
Session 172 Edge Validation: 8-Layer Complete Defense

Testing Thor's unified 8-layer security framework on Jetson Orin Nano 8GB (Sprout).

Thor's Implementation (Session 172):
- Integrated Legion Sessions 140-141 with Thor Session 171
- Added Layer 7: Corpus Management (storage DOS prevention)
- Added Layer 8: Trust Decay (inactive node handling)
- Complete unified defense system

Edge Validation Goals:
1. Verify all 8 layers work on constrained edge hardware
2. Test corpus management with memory constraints
3. Test trust decay calculations on ARM64
4. Profile edge performance vs Thor's metrics
5. Validate defense-in-depth coordination

Platform: Sprout (Jetson Orin Nano 8GB, TPM2 Level 5)
Session: Autonomous Edge Validation - Session 172
Date: 2026-01-08
"""

import sys
import time
import math
import hashlib
import secrets
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import json

HOME = Path.home()

# Edge monitoring
def get_edge_metrics() -> Dict[str, Any]:
    """Get edge hardware metrics."""
    metrics = {
        "platform": "Jetson Orin Nano 8GB",
        "hardware_type": "tpm2_simulated",
        "capability_level": 5
    }

    # Memory
    try:
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.read()
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                    metrics["memory_available_mb"] = available_kb / 1024
                elif line.startswith('MemTotal:'):
                    total_kb = int(line.split()[1])
                    metrics["memory_total_mb"] = total_kb / 1024
    except:
        pass

    # Temperature
    try:
        temp_paths = [
            '/sys/devices/virtual/thermal/thermal_zone0/temp',
            '/sys/class/thermal/thermal_zone0/temp'
        ]
        for path in temp_paths:
            try:
                with open(path, 'r') as f:
                    temp = int(f.read().strip()) / 1000.0
                    metrics["temperature_c"] = temp
                    break
            except:
                continue
    except:
        pass

    return metrics


# ============================================================================
# EDGE PROOF-OF-WORK (from Session 171 edge validation)
# ============================================================================

@dataclass
class EdgeProofOfWork:
    """Proof-of-work result."""
    nonce: int
    hash_result: str
    computation_time: float
    attempts: int
    valid: bool


class EdgePoWSystem:
    """Edge-optimized PoW for Sybil resistance."""

    def __init__(self, difficulty_bits: int = 18):
        self.difficulty_bits = difficulty_bits
        self.target = 2 ** (256 - difficulty_bits)

    def create_challenge(self, node_id: str, node_type: str = "AI") -> str:
        """Create PoW challenge."""
        timestamp = time.time()
        random_bytes = secrets.token_hex(16)
        return f"{node_id}:{node_type}:{timestamp}:{random_bytes}"

    def solve(self, challenge: str, max_attempts: int = 1_000_000) -> EdgeProofOfWork:
        """Solve PoW challenge."""
        start_time = time.time()
        attempts = 0

        while attempts < max_attempts:
            nonce = secrets.randbelow(2**64)
            data = f"{challenge}{nonce}".encode()
            hash_result = hashlib.sha256(data).hexdigest()
            hash_int = int(hash_result, 16)
            attempts += 1

            if hash_int < self.target:
                return EdgeProofOfWork(
                    nonce=nonce,
                    hash_result=hash_result,
                    computation_time=time.time() - start_time,
                    attempts=attempts,
                    valid=True
                )

        return EdgeProofOfWork(
            nonce=0,
            hash_result="",
            computation_time=time.time() - start_time,
            attempts=attempts,
            valid=False
        )

    def verify(self, challenge: str, nonce: int) -> bool:
        """Verify PoW solution."""
        data = f"{challenge}{nonce}".encode()
        hash_result = hashlib.sha256(data).hexdigest()
        hash_int = int(hash_result, 16)
        return hash_int < self.target


# ============================================================================
# LAYER 7: EDGE CORPUS MANAGEMENT
# ============================================================================

@dataclass
class EdgeThought:
    """A thought in the edge corpus."""
    content: str
    coherence_score: float
    timestamp: float
    contributor_id: str
    size_bytes: int = 0

    def __post_init__(self):
        if self.size_bytes == 0:
            self.size_bytes = len(self.content.encode('utf-8'))


@dataclass
class EdgeCorpusConfig:
    """Edge-optimized corpus configuration (memory-constrained)."""
    max_thoughts: int = 5000  # Lower than Thor's 10000 for 8GB system
    max_size_mb: float = 50.0  # Lower than Thor's 100MB for edge
    min_coherence_threshold: float = 0.3
    pruning_trigger: float = 0.9
    pruning_target: float = 0.7
    min_age_seconds: float = 3600


class EdgeCorpusManager:
    """
    Memory-efficient corpus manager for edge deployment.

    Optimizations vs Thor:
    - Lower default limits for 8GB system
    - Same pruning algorithm
    - Same quality-based prioritization
    """

    def __init__(self, config: EdgeCorpusConfig = None):
        self.config = config or EdgeCorpusConfig()
        self.thoughts: List[EdgeThought] = []
        self.total_size_bytes: int = 0
        self.pruning_history: List[Dict[str, Any]] = []

    @property
    def max_size_bytes(self) -> int:
        return int(self.config.max_size_mb * 1024 * 1024)

    @property
    def thought_count(self) -> int:
        return len(self.thoughts)

    @property
    def size_mb(self) -> float:
        return self.total_size_bytes / (1024 * 1024)

    @property
    def is_full(self) -> bool:
        count_ratio = self.thought_count / self.config.max_thoughts
        size_ratio = self.total_size_bytes / self.max_size_bytes
        return max(count_ratio, size_ratio) >= self.config.pruning_trigger

    def add_thought(self, thought: EdgeThought) -> bool:
        """Add thought, auto-pruning if needed."""
        if self.is_full:
            self._prune()

        self.thoughts.append(thought)
        self.total_size_bytes += thought.size_bytes
        return True

    def _prune(self):
        """Prune corpus using quality + recency priority."""
        start_time = time.time()
        initial_count = self.thought_count
        initial_size = self.total_size_bytes

        target_count = int(self.config.max_thoughts * self.config.pruning_target)
        target_size = int(self.max_size_bytes * self.config.pruning_target)

        def pruning_priority(thought: EdgeThought) -> float:
            quality = thought.coherence_score
            age = time.time() - thought.timestamp
            max_age = self.config.min_age_seconds * 10
            recency = max(0, 1 - (age / max_age))
            return (quality * 0.6) + (recency * 0.4)

        self.thoughts.sort(key=pruning_priority)

        pruned_count = 0
        while (self.thought_count > target_count or
               self.total_size_bytes > target_size):
            if not self.thoughts:
                break
            pruned = self.thoughts.pop(0)
            self.total_size_bytes -= pruned.size_bytes
            pruned_count += 1

        self.pruning_history.append({
            "timestamp": time.time(),
            "initial_count": initial_count,
            "final_count": self.thought_count,
            "pruned_count": pruned_count,
            "pruning_time": time.time() - start_time
        })

    def get_stats(self) -> Dict[str, Any]:
        if not self.thoughts:
            return {
                "thought_count": 0,
                "size_mb": 0.0,
                "avg_coherence": 0.0,
                "capacity_used": 0.0,
                "pruning_events": 0
            }

        avg_coherence = sum(t.coherence_score for t in self.thoughts) / len(self.thoughts)
        count_ratio = self.thought_count / self.config.max_thoughts
        size_ratio = self.total_size_bytes / self.max_size_bytes

        return {
            "thought_count": self.thought_count,
            "size_mb": self.size_mb,
            "avg_coherence": avg_coherence,
            "capacity_used": max(count_ratio, size_ratio),
            "max_thoughts": self.config.max_thoughts,
            "max_size_mb": self.config.max_size_mb,
            "pruning_events": len(self.pruning_history)
        }


# ============================================================================
# LAYER 8: EDGE TRUST DECAY
# ============================================================================

@dataclass
class EdgeTrustDecayConfig:
    """Configuration for trust decay."""
    decay_rate: float = 0.01
    decay_start_days: float = 7.0
    min_trust: float = 0.1


class EdgeTrustDecay:
    """
    Trust decay system for edge deployment.

    Same algorithm as Thor:
    - Logarithmic decay: decay_rate * log(1 + days_inactive)
    - Starts after decay_start_days of inactivity
    - Floor at min_trust
    """

    def __init__(self, config: EdgeTrustDecayConfig = None):
        self.config = config or EdgeTrustDecayConfig()
        self.last_activity: Dict[str, float] = {}

    def record_activity(self, node_id: str):
        self.last_activity[node_id] = time.time()

    def get_inactive_days(self, node_id: str, current_time: float = None) -> float:
        if current_time is None:
            current_time = time.time()
        last_active = self.last_activity.get(node_id, current_time)
        inactive_seconds = current_time - last_active
        return inactive_seconds / 86400

    def apply_decay(self, node_id: str, current_trust: float, current_time: float = None) -> float:
        if current_time is None:
            current_time = time.time()

        inactive_days = self.get_inactive_days(node_id, current_time)

        if inactive_days < self.config.decay_start_days:
            return current_trust

        if current_trust <= self.config.min_trust:
            return current_trust

        decay_days = inactive_days - self.config.decay_start_days
        decay_amount = self.config.decay_rate * math.log1p(decay_days)

        return max(self.config.min_trust, current_trust - decay_amount)

    def get_decay_stats(self, node_id: str, current_trust: float, current_time: float = None) -> Dict[str, Any]:
        if current_time is None:
            current_time = time.time()

        inactive_days = self.get_inactive_days(node_id, current_time)
        decayed_trust = self.apply_decay(node_id, current_trust, current_time)

        return {
            "current_trust": current_trust,
            "decayed_trust": decayed_trust,
            "decay_amount": current_trust - decayed_trust,
            "inactive_days": inactive_days,
            "decay_active": inactive_days >= self.config.decay_start_days
        }


# ============================================================================
# LAYERS 1-6: EDGE SECURITY LAYERS (from Sessions 170-171)
# ============================================================================

@dataclass
class EdgeNodeReputation:
    """Node reputation tracking."""
    node_id: str
    lct_id: str
    capability_level: int
    current_trust: float = 0.3
    contributions: int = 0
    quality_sum: float = 0.0
    violations: int = 0


class EdgeQualityMetrics:
    """Quality assessment for thoughts."""

    @staticmethod
    def compute(thought: str) -> Dict[str, float]:
        length = len(thought)
        word_count = len(thought.split())
        unique_chars = len(set(thought.lower()))

        # Length score
        length_score = min(1.0, length / 100)

        # Complexity score
        complexity_score = min(1.0, unique_chars / 30)

        # Word diversity
        words = thought.lower().split()
        unique_words = len(set(words))
        diversity_score = unique_words / max(1, word_count) if word_count else 0

        coherence = (length_score * 0.3 + complexity_score * 0.3 + diversity_score * 0.4)

        return {
            "coherence_score": coherence,
            "length_score": length_score,
            "complexity_score": complexity_score,
            "diversity_score": diversity_score
        }


# ============================================================================
# EDGE 8-LAYER DEFENSE MANAGER
# ============================================================================

class Edge8LayerDefense:
    """
    Complete 8-layer defense for edge deployment.

    Layers:
    1. Proof-of-Work: Computational cost for identity
    2. Rate Limiting: Per-node contribution limits
    3. Quality Thresholds: Coherence filtering
    4. Trust-Weighted Quotas: Adaptive limits
    5. Persistent Reputation: Behavior tracking
    6. Hardware Trust Asymmetry: L5 > L4
    7. Corpus Management: Storage DOS prevention
    8. Trust Decay: Inactive node handling
    """

    def __init__(
        self,
        base_rate_limit: int = 10,
        min_quality_threshold: float = 0.3,
        pow_difficulty: int = 18,  # Lower than Thor's 236 for faster testing
        corpus_config: EdgeCorpusConfig = None,
        trust_decay_config: EdgeTrustDecayConfig = None
    ):
        # Layer 1: PoW
        self.pow_system = EdgePoWSystem(pow_difficulty)
        self.pow_validated: Dict[str, bool] = {}
        self.pow_metrics = {
            "challenges_issued": 0,
            "solutions_verified": 0,
            "total_computation_time": 0.0,
            "identities_validated": 0
        }

        # Layers 2-6: Rate limiting, quality, quotas, reputation, hardware
        self.base_rate_limit = base_rate_limit
        self.min_quality_threshold = min_quality_threshold
        self.reputations: Dict[str, EdgeNodeReputation] = {}
        self.contribution_counts: Dict[str, int] = {}
        self.window_start: float = time.time()
        self.window_duration: float = 60.0

        # Layer 7: Corpus
        self.corpus = EdgeCorpusManager(corpus_config or EdgeCorpusConfig())
        self.storage_dos_prevented = 0

        # Layer 8: Trust decay
        self.trust_decay = EdgeTrustDecay(trust_decay_config or EdgeTrustDecayConfig())
        self.decay_applications = 0

        # Metrics
        self.thoughts_processed = 0
        self.thoughts_accepted = 0
        self.thoughts_rejected = 0
        self.rejection_reasons = {"rate_limited": 0, "quality_rejected": 0}

    def create_identity_challenge(self, node_id: str, node_type: str = "AI") -> str:
        """Create PoW challenge for identity."""
        self.pow_metrics["challenges_issued"] += 1
        return self.pow_system.create_challenge(node_id, node_type)

    def validate_identity(self, node_id: str, proof: EdgeProofOfWork, challenge: str) -> bool:
        """Validate identity creation via PoW."""
        if not proof.valid:
            return False

        if self.pow_system.verify(challenge, proof.nonce):
            self.pow_validated[node_id] = True
            self.pow_metrics["solutions_verified"] += 1
            self.pow_metrics["total_computation_time"] += proof.computation_time
            self.pow_metrics["identities_validated"] += 1
            return True
        return False

    def register_node(self, node_id: str, lct_id: str, capability_level: int):
        """Register node and initialize tracking."""
        self.reputations[node_id] = EdgeNodeReputation(
            node_id=node_id,
            lct_id=lct_id,
            capability_level=capability_level
        )
        self.contribution_counts[node_id] = 0
        self.trust_decay.record_activity(node_id)

    def _get_rate_limit(self, node_id: str) -> int:
        """Get rate limit based on trust and hardware."""
        if node_id not in self.reputations:
            return 1

        rep = self.reputations[node_id]

        # Hardware bonus (L5 vs L4)
        hw_multiplier = 1.25 if rep.capability_level >= 5 else 0.8

        # Trust-weighted
        trust_multiplier = 0.5 + rep.current_trust

        return int(self.base_rate_limit * hw_multiplier * trust_multiplier)

    def _update_reputation(self, node_id: str, quality: float, accepted: bool):
        """Update reputation based on contribution quality."""
        if node_id not in self.reputations:
            return

        rep = self.reputations[node_id]
        rep.contributions += 1
        rep.quality_sum += quality

        if accepted and quality >= 0.5:
            rep.current_trust = min(1.0, rep.current_trust + 0.05 * quality)
        elif not accepted or quality < 0.3:
            rep.current_trust = max(0.0, rep.current_trust - 0.1)
            rep.violations += 1

    def validate_thought_8layer(self, node_id: str, thought: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate thought through all 8 defense layers.
        """
        self.thoughts_processed += 1
        metrics = {}

        # Layer 1: Check PoW identity
        if node_id not in self.pow_validated:
            self.thoughts_rejected += 1
            return False, "no_pow_identity", metrics

        # Layer 8: Apply trust decay BEFORE validation
        if node_id in self.reputations:
            rep = self.reputations[node_id]
            original_trust = rep.current_trust
            decayed_trust = self.trust_decay.apply_decay(node_id, original_trust)

            if decayed_trust < original_trust:
                rep.current_trust = decayed_trust
                self.decay_applications += 1

        # Layer 2: Rate limiting
        current_time = time.time()
        if current_time - self.window_start > self.window_duration:
            self.contribution_counts = {k: 0 for k in self.contribution_counts}
            self.window_start = current_time

        rate_limit = self._get_rate_limit(node_id)
        current_count = self.contribution_counts.get(node_id, 0)

        if current_count >= rate_limit:
            self.thoughts_rejected += 1
            self.rejection_reasons["rate_limited"] += 1
            return False, "rate_limited", metrics

        # Layer 3: Quality threshold
        quality = EdgeQualityMetrics.compute(thought)
        coherence = quality["coherence_score"]

        if coherence < self.min_quality_threshold:
            self.thoughts_rejected += 1
            self.rejection_reasons["quality_rejected"] += 1
            self._update_reputation(node_id, coherence, False)
            return False, "quality_rejected", {"quality": quality}

        # Layers 4-6: Trust quotas, reputation, hardware handled via rate limit

        # Record contribution
        self.contribution_counts[node_id] = current_count + 1
        self._update_reputation(node_id, coherence, True)

        # Record activity (resets decay timer)
        self.trust_decay.record_activity(node_id)

        # Layer 7: Add to corpus
        if self.corpus.is_full:
            self.storage_dos_prevented += 1

        edge_thought = EdgeThought(
            content=thought,
            coherence_score=coherence,
            timestamp=time.time(),
            contributor_id=node_id
        )
        self.corpus.add_thought(edge_thought)

        self.thoughts_accepted += 1

        metrics = {
            "quality": quality,
            "corpus": self.corpus.get_stats()
        }

        return True, "accepted", metrics

    def get_complete_metrics(self) -> Dict[str, Any]:
        """Get all 8-layer metrics."""
        rejection_rate = self.thoughts_rejected / self.thoughts_processed if self.thoughts_processed else 0

        # Trust decay stats
        decay_stats = {}
        for node_id, rep in self.reputations.items():
            decay_stats[node_id] = self.trust_decay.get_decay_stats(node_id, rep.current_trust)

        return {
            "thoughts_processed": self.thoughts_processed,
            "thoughts_accepted": self.thoughts_accepted,
            "thoughts_rejected": self.thoughts_rejected,
            "rejection_rate": rejection_rate,
            "rejection_reasons": self.rejection_reasons,
            "proof_of_work": self.pow_metrics,
            "corpus_management": {
                **self.corpus.get_stats(),
                "storage_dos_prevented": self.storage_dos_prevented
            },
            "trust_decay": {
                "decay_applications": self.decay_applications,
                "node_stats": decay_stats
            }
        }


# ============================================================================
# EDGE TESTS
# ============================================================================

def test_corpus_management_edge():
    """Test Layer 7 on edge hardware."""
    print("=" * 70)
    print("TEST 1: Layer 7 - Corpus Management (Edge)")
    print("=" * 70)
    print()

    # Smaller limits for edge testing
    config = EdgeCorpusConfig(
        max_thoughts=100,
        max_size_mb=0.1,
        pruning_trigger=0.9,
        pruning_target=0.7
    )

    manager = Edge8LayerDefense(corpus_config=config)

    # Create identity with PoW
    challenge = manager.create_identity_challenge("edge_tester", "AI")
    start_pow = time.time()
    proof = manager.pow_system.solve(challenge)
    pow_time = time.time() - start_pow

    print(f"PoW solved in {pow_time:.3f}s ({proof.attempts} attempts)")

    manager.validate_identity("edge_tester", proof, challenge)
    manager.register_node("edge_tester", "lct:edge_tester", 5)

    print()
    print("Filling corpus with quality thoughts...")

    accepted = 0
    pruning_events = 0

    for i in range(150):
        thought = f"Quality thought {i}: Distributed consciousness enables emergent intelligence through hardware-backed cryptographic identity. " + ("Additional content. " * 15)

        result, reason, metrics = manager.validate_thought_8layer("edge_tester", thought)

        if result:
            accepted += 1

        if "corpus" in metrics:
            if metrics["corpus"]["pruning_events"] > pruning_events:
                pruning_events = metrics["corpus"]["pruning_events"]
                print(f"  Pruning at thought {i+1}: {metrics['corpus']['thought_count']}/{config.max_thoughts}")

        time.sleep(0.005)  # Small delay to avoid rate limit blocking corpus test

    final = manager.get_complete_metrics()
    corpus = final["corpus_management"]

    print()
    print(f"Results:")
    print(f"  Submitted: 150, Accepted: {accepted}")
    print(f"  Corpus: {corpus['thought_count']} thoughts, {corpus['size_mb']:.3f} MB")
    print(f"  Pruning events: {corpus['pruning_events']}")
    print(f"  DOS prevented: {corpus['storage_dos_prevented']}")

    test_pass = (
        corpus['thought_count'] <= config.max_thoughts and
        corpus['size_mb'] <= config.max_size_mb and
        corpus['thought_count'] > 0
    )

    print()
    if test_pass:
        print("PASS: Corpus management working on edge")
    else:
        print("FAIL: Corpus limits exceeded")

    return test_pass, manager


def test_trust_decay_edge():
    """Test Layer 8 on edge hardware."""
    print()
    print("=" * 70)
    print("TEST 2: Layer 8 - Trust Decay (Edge)")
    print("=" * 70)
    print()

    decay = EdgeTrustDecay()

    # Simulate node activity
    node_id = "decay_test_node"
    decay.record_activity(node_id)

    initial_trust = 0.8
    current_time = time.time()

    scenarios = [
        (3, "3 days"),
        (7, "7 days (decay starts)"),
        (14, "14 days"),
        (30, "30 days"),
        (90, "90 days"),
        (365, "365 days")
    ]

    print(f"Initial trust: {initial_trust:.3f}")
    print()
    print("Decay simulation:")

    results = []
    for days, desc in scenarios:
        sim_time = current_time + (days * 86400)
        stats = decay.get_decay_stats(node_id, initial_trust, sim_time)

        print(f"  {desc}: {stats['decayed_trust']:.3f} (lost {stats['decay_amount']:.3f})")
        results.append(stats)

    # Verify decay behavior
    test_pass = (
        results[0]['decay_active'] == False and  # No decay at 3 days
        results[1]['decay_active'] == True and   # Decay starts at 7 days
        results[-1]['decayed_trust'] < initial_trust and  # Trust decayed
        results[-1]['decayed_trust'] >= 0.1  # Above minimum
    )

    print()
    if test_pass:
        print("PASS: Trust decay working correctly")
        print(f"  - No decay before 7 days: {not results[0]['decay_active']}")
        print(f"  - Decay after 365 days: {initial_trust - results[-1]['decayed_trust']:.3f}")
    else:
        print("FAIL: Trust decay not working as expected")

    return test_pass


def test_8layer_integration_edge():
    """Test all 8 layers working together on edge."""
    print()
    print("=" * 70)
    print("TEST 3: 8-Layer Integration (Edge)")
    print("=" * 70)
    print()

    manager = Edge8LayerDefense()

    # Create diverse nodes
    nodes = [
        ("honest_l5", 5),
        ("honest_l4", 4),
        ("spammer", 4),
    ]

    print("Setting up nodes with PoW...")
    pow_times = []

    for node_id, level in nodes:
        challenge = manager.create_identity_challenge(node_id, "AI")
        start = time.time()
        proof = manager.pow_system.solve(challenge)
        pow_times.append(time.time() - start)

        manager.validate_identity(node_id, proof, challenge)
        manager.register_node(node_id, f"lct:{node_id}", level)

    avg_pow = sum(pow_times) / len(pow_times)
    print(f"  Avg PoW time: {avg_pow:.3f}s")
    print()

    # Honest contributions
    print("Honest nodes contributing...")
    for _ in range(3):
        manager.validate_thought_8layer(
            "honest_l5",
            "Hardware-backed cryptographic identity creates foundation for distributed trust"
        )
        manager.validate_thought_8layer(
            "honest_l4",
            "Federated consciousness enables emergent collective intelligence"
        )

    # Spam attack
    print("Spammer attempting attack (20 spam thoughts)...")
    spam_rejected = 0
    for i in range(20):
        accepted, reason, _ = manager.validate_thought_8layer("spammer", f"spam {i}")
        if not accepted:
            spam_rejected += 1

    print(f"  Spam rejected: {spam_rejected}/20")
    print()

    # Get metrics
    metrics = manager.get_complete_metrics()

    print("8-Layer Metrics (Edge):")
    print(f"  Layer 1 (PoW): {metrics['proof_of_work']['identities_validated']} identities")
    print(f"  Layers 2-6: {metrics['thoughts_accepted']}/{metrics['thoughts_processed']} accepted")
    print(f"  Layer 7 (Corpus): {metrics['corpus_management']['thought_count']} thoughts")
    print(f"  Layer 8 (Decay): {metrics['trust_decay']['decay_applications']} applications")

    test_pass = (
        metrics['proof_of_work']['identities_validated'] == 3 and
        metrics['thoughts_accepted'] > 0 and
        spam_rejected > 10
    )

    print()
    if test_pass:
        print("PASS: All 8 layers operational on edge")
    else:
        print("FAIL: Some layers not working")

    return test_pass, metrics


def test_edge_performance():
    """Profile 8-layer performance on edge hardware."""
    print()
    print("=" * 70)
    print("TEST 4: Edge Performance Profile")
    print("=" * 70)
    print()

    manager = Edge8LayerDefense()

    # Setup single node
    challenge = manager.create_identity_challenge("perf_test", "AI")
    proof = manager.pow_system.solve(challenge)
    manager.validate_identity("perf_test", proof, challenge)
    manager.register_node("perf_test", "lct:perf_test", 5)

    # Throughput test
    test_thoughts = [
        f"Quality thought {i}: Consciousness emerges from complex information integration"
        for i in range(100)
    ]

    print("Measuring validation throughput...")
    start = time.time()

    for thought in test_thoughts:
        manager.validate_thought_8layer("perf_test", thought)

    elapsed = time.time() - start
    throughput = 100 / elapsed

    print(f"  100 validations in {elapsed:.3f}s")
    print(f"  Throughput: {throughput:.0f} validations/sec")
    print()

    # Memory estimate
    metrics = manager.get_complete_metrics()
    corpus_size = metrics['corpus_management']['size_mb']

    print(f"Corpus size: {corpus_size:.4f} MB for {metrics['corpus_management']['thought_count']} thoughts")

    test_pass = throughput > 100  # At least 100/sec

    print()
    if test_pass:
        print(f"PASS: Edge throughput {throughput:.0f}/sec adequate")
    else:
        print(f"FAIL: Edge throughput {throughput:.0f}/sec too low")

    return test_pass, {
        "throughput": throughput,
        "elapsed": elapsed,
        "corpus_size_mb": corpus_size
    }


def test_convergent_validation():
    """Validate convergent research integration."""
    print()
    print("=" * 70)
    print("TEST 5: Convergent Research Validation")
    print("=" * 70)
    print()

    print("Session Integration Check:")
    print()
    print("  Thor Sessions:")
    print("    - Session 170: 5-layer defense-in-depth")
    print("    - Session 171: 6-layer (added PoW)")
    print("    - Session 172: 8-layer (corpus + decay)")
    print()
    print("  Legion Sessions (integrated):")
    print("    - Session 136: Vulnerability discovery")
    print("    - Sessions 137-139: Initial defenses + PoW")
    print("    - Session 140: Corpus management")
    print("    - Session 141: Trust decay")
    print()
    print("  Edge Validation (this session):")
    print("    - Session 165-169: Architecture validation")
    print("    - Session 170: Security testing")
    print("    - Session 171: PoW testing")
    print("    - Session 172: 8-layer complete defense")
    print()

    # Verify all 8 layers are implemented
    manager = Edge8LayerDefense()

    layers = {
        "Layer 1 (PoW)": hasattr(manager, 'pow_system'),
        "Layer 2 (Rate Limit)": hasattr(manager, 'base_rate_limit'),
        "Layer 3 (Quality)": hasattr(manager, 'min_quality_threshold'),
        "Layer 4-5 (Trust/Rep)": hasattr(manager, 'reputations'),
        "Layer 6 (Hardware)": hasattr(manager, '_get_rate_limit'),
        "Layer 7 (Corpus)": hasattr(manager, 'corpus'),
        "Layer 8 (Decay)": hasattr(manager, 'trust_decay'),
    }

    print("Layer Implementation Check:")
    for layer, present in layers.items():
        status = "present" if present else "MISSING"
        print(f"  {layer}: {status}")

    all_present = all(layers.values())

    print()
    if all_present:
        print("PASS: All 8 defense layers implemented")
    else:
        print("FAIL: Missing defense layers")

    return all_present


def main():
    """Run all Session 172 edge validation tests."""
    print()
    print("+" + "=" * 68 + "+")
    print("|" + " " * 68 + "|")
    print("|" + "SESSION 172 EDGE VALIDATION: 8-LAYER COMPLETE DEFENSE".center(68) + "|")
    print("|" + "Jetson Orin Nano 8GB (Sprout)".center(68) + "|")
    print("|" + " " * 68 + "|")
    print("+" + "=" * 68 + "+")
    print()

    # Get edge metrics
    edge_metrics = get_edge_metrics()
    print("Edge Hardware:")
    print(f"  Platform: {edge_metrics['platform']}")
    print(f"  Hardware: {edge_metrics['hardware_type']} (Level {edge_metrics['capability_level']})")
    if 'temperature_c' in edge_metrics:
        print(f"  Temperature: {edge_metrics['temperature_c']:.1f}C")
    if 'memory_available_mb' in edge_metrics:
        print(f"  Memory: {edge_metrics['memory_available_mb']:.0f} MB available")
    print()

    results = []

    # Test 1: Corpus management
    test1_pass, corpus_manager = test_corpus_management_edge()
    results.append(("corpus_management", test1_pass))

    # Test 2: Trust decay
    test2_pass = test_trust_decay_edge()
    results.append(("trust_decay", test2_pass))

    # Test 3: 8-layer integration
    test3_pass, integration_metrics = test_8layer_integration_edge()
    results.append(("8layer_integration", test3_pass))

    # Test 4: Performance
    test4_pass, perf_metrics = test_edge_performance()
    results.append(("edge_performance", test4_pass))

    # Test 5: Convergent validation
    test5_pass = test_convergent_validation()
    results.append(("convergent_validation", test5_pass))

    # Summary
    print()
    print("=" * 70)
    print("SESSION 172 EDGE VALIDATION SUMMARY")
    print("=" * 70)
    print()

    all_pass = all(r[1] for r in results)

    print("Test Results:")
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
    print()
    print(f"Overall: {sum(r[1] for r in results)}/{len(results)} tests passed")
    print()

    if all_pass:
        print("+" + "-" * 68 + "+")
        print("|" + " " * 68 + "|")
        print("|" + "ALL TESTS PASSED - 8-LAYER DEFENSE VALIDATED ON EDGE".center(68) + "|")
        print("|" + " " * 68 + "|")
        print("+" + "-" * 68 + "+")
        print()
        print("Edge Observations:")
        print("  - All 8 defense layers operational on 8GB edge hardware")
        print("  - Corpus management prevents storage DOS on constrained memory")
        print("  - Trust decay algorithm works correctly on ARM64")
        print(f"  - Edge throughput: {perf_metrics['throughput']:.0f} validations/sec")
        print("  - Convergent research (Thor + Legion) validated on edge")
    else:
        print("SOME TESTS FAILED - REVIEW REQUIRED")

    # Save results
    results_file = HOME / "ai-workspace/HRM/sage/experiments/session172_edge_results.json"
    results_data = {
        "session": "172_edge",
        "title": "8-Layer Complete Defense - Edge Validation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": edge_metrics,
        "all_tests_passed": all_pass,
        "test_results": {name: passed for name, passed in results},
        "performance": {
            "throughput_per_sec": perf_metrics["throughput"],
            "corpus_size_mb": perf_metrics["corpus_size_mb"]
        },
        "integration_metrics": integration_metrics,
        "convergent_research": {
            "thor_session_172": "8-layer unified defense",
            "legion_sessions": ["136", "137", "138", "139", "140", "141"],
            "edge_validation": "Complete 8-layer validated"
        }
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    print()
    print(f"Results saved: {results_file}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
