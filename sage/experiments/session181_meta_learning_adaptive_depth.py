#!/usr/bin/env python3
"""
Thor Session 181: Meta-Learning Adaptive Depth

Research Goal: Integrate Legion Session 160's meta-learning with Thor Sessions
177-180's adaptive depth architecture, enabling the system to learn optimal
depth selection from verification history.

Architecture Evolution:
- Session 177: ATP-adaptive depth (metabolic adaptation)
- Session 178: Federated coordination (network-aware)
- Session 179: Reputation-aware depth (cognitive credit)
- Session 180: Persistent reputation (cross-session trust)
- Session 181: Meta-learning depth (learns from experience) ← YOU ARE HERE

Key Innovation: System improves depth selection over time by learning which
depths produce best outcomes. Combines biological adaptation (ATP), social
dynamics (reputation), and experiential learning (meta-learning).

Convergence:
- Thor's adaptive depth framework (Sessions 177-180)
- Legion's meta-learning patterns (Session 160)
- Result: Self-optimizing consciousness that learns from experience

Platform: Thor (Jetson AGX Thor, TrustZone L5)
Type: Autonomous Research
Date: 2026-01-10
"""

import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum

# Import Session 180 as base
import sys
HOME = Path.home()
sys.path.insert(0, str(HOME / "ai-workspace" / "HRM" / "sage" / "experiments"))

from session180_persistent_reputation import (
    PersistentReputationAwareAdaptiveSAGE,
    PersistentReputationManager,
    PersistentReputationScore
)

from session178_federated_sage_verification import (
    CognitiveDepth
)


# ============================================================================
# META-LEARNING PATTERNS
# ============================================================================

@dataclass
class DepthVerificationPattern:
    """
    A verification pattern for depth learning.

    Records outcome of using specific depth under specific conditions.
    """
    pattern_id: str
    node_id: str
    depth_used: CognitiveDepth
    atp_before: float
    atp_after: float
    reputation_before: float
    reputation_after: float
    quality_achieved: float  # 0-1 quality score
    success: bool  # Did it pass quality threshold?
    timestamp: float
    context: Dict[str, Any]  # ATP level, reputation level, network state

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for JSON serialization."""
        data = asdict(self)
        data['depth_used'] = self.depth_used.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DepthVerificationPattern':
        """Create from dict."""
        data['depth_used'] = CognitiveDepth[data['depth_used']]
        return cls(**data)


@dataclass
class LearningInsight:
    """An insight learned from pattern analysis."""
    insight_type: str
    description: str
    evidence_count: int
    confidence: float  # 0-1
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict."""
        return asdict(self)


# ============================================================================
# PERSISTENT META-LEARNING MANAGER
# ============================================================================

class PersistentMetaLearningManager:
    """
    Manages persistent meta-learning storage.

    Similar to reputation storage, but tracks verification patterns
    and learned insights across sessions.
    """

    def __init__(self, storage_path: Path, node_id: str):
        """
        Initialize meta-learning manager.

        Args:
            storage_path: Directory for learning storage
            node_id: Hardware-anchored node identity
        """
        self.storage_path = storage_path
        self.node_id = node_id
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Storage files
        self.patterns_file = self.storage_path / f"{node_id}_depth_patterns.jsonl"
        self.insights_file = self.storage_path / f"{node_id}_learning_insights.json"

        # In-memory caches
        self.patterns: List[DepthVerificationPattern] = []
        self.insights: List[LearningInsight] = []

        # Performance tracking
        self.depth_performance: Dict[CognitiveDepth, List[float]] = defaultdict(list)
        self.depth_success_rate: Dict[CognitiveDepth, Tuple[int, int]] = defaultdict(lambda: (0, 0))
        self.depth_atp_efficiency: Dict[CognitiveDepth, List[float]] = defaultdict(list)

        # Load existing data
        self._load()

    def _load(self):
        """Load learning data from disk."""
        # Load patterns (append-only log)
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r') as f:
                for line in f:
                    try:
                        pattern_data = json.loads(line)
                        pattern = DepthVerificationPattern.from_dict(pattern_data)
                        self.patterns.append(pattern)

                        # Update performance trackers
                        depth = pattern.depth_used
                        self.depth_performance[depth].append(pattern.quality_achieved)

                        successes, total = self.depth_success_rate[depth]
                        self.depth_success_rate[depth] = (
                            successes + (1 if pattern.success else 0),
                            total + 1
                        )

                        # ATP efficiency: did ATP increase or stay same?
                        atp_delta = pattern.atp_after - pattern.atp_before
                        self.depth_atp_efficiency[depth].append(atp_delta)

                    except (json.JSONDecodeError, KeyError):
                        continue  # Skip corrupt lines

        # Load insights (aggregated state)
        if self.insights_file.exists():
            with open(self.insights_file, 'r') as f:
                try:
                    insights_data = json.load(f)
                    self.insights = [LearningInsight(**i) for i in insights_data]
                except json.JSONDecodeError:
                    pass  # Will rebuild from patterns

    def _save_pattern(self, pattern: DepthVerificationPattern):
        """Append pattern to disk (JSONL format)."""
        with open(self.patterns_file, 'a') as f:
            f.write(json.dumps(pattern.to_dict()) + '\n')

    def _save_insights(self):
        """Save insights to disk (atomic write)."""
        insights_data = [i.to_dict() for i in self.insights]

        # Atomic write
        temp_file = self.insights_file.with_suffix('.tmp')
        with open(temp_file, 'w') as f:
            json.dump(insights_data, f, indent=2)
        temp_file.replace(self.insights_file)

    def record_pattern(
        self,
        depth_used: CognitiveDepth,
        atp_before: float,
        atp_after: float,
        reputation_before: float,
        reputation_after: float,
        quality_achieved: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ) -> DepthVerificationPattern:
        """
        Record a depth verification pattern.

        Args:
            depth_used: Cognitive depth that was used
            atp_before: ATP before verification
            atp_after: ATP after verification
            reputation_before: Reputation before
            reputation_after: Reputation after
            quality_achieved: Quality score (0-1)
            success: Did it pass quality threshold?
            context: Additional metadata

        Returns:
            Created pattern
        """
        pattern_id = hashlib.sha256(
            f"{self.node_id}{time.time()}{depth_used.name}".encode()
        ).hexdigest()[:16]

        pattern = DepthVerificationPattern(
            pattern_id=pattern_id,
            node_id=self.node_id,
            depth_used=depth_used,
            atp_before=atp_before,
            atp_after=atp_after,
            reputation_before=reputation_before,
            reputation_after=reputation_after,
            quality_achieved=quality_achieved,
            success=success,
            timestamp=time.time(),
            context=context or {}
        )

        # Store pattern
        self.patterns.append(pattern)
        self._save_pattern(pattern)

        # Update performance trackers
        self.depth_performance[depth_used].append(quality_achieved)

        successes, total = self.depth_success_rate[depth_used]
        self.depth_success_rate[depth_used] = (
            successes + (1 if success else 0),
            total + 1
        )

        atp_delta = atp_after - atp_before
        self.depth_atp_efficiency[depth_used].append(atp_delta)

        return pattern

    def analyze_patterns(self) -> List[LearningInsight]:
        """
        Analyze verification patterns to extract learning insights.

        Returns:
            List of learned insights
        """
        insights = []

        # Insight 1: Which depth produces highest quality?
        if self.depth_performance:
            best_depth = max(
                self.depth_performance.items(),
                key=lambda x: sum(x[1]) / len(x[1])
            )
            avg_quality = sum(best_depth[1]) / len(best_depth[1])

            insights.append(LearningInsight(
                insight_type="optimal_quality_depth",
                description=f"Depth {best_depth[0].name} produces highest quality ({avg_quality:.3f})",
                evidence_count=len(best_depth[1]),
                confidence=min(len(best_depth[1]) / 10, 1.0),
                recommendation=f"Prefer {best_depth[0].name} depth when ATP and reputation allow"
            ))

        # Insight 2: Which depth has best success rate?
        if self.depth_success_rate:
            best_success_depth = max(
                self.depth_success_rate.items(),
                key=lambda x: x[1][0] / x[1][1] if x[1][1] > 0 else 0
            )
            successes, total = best_success_depth[1]
            success_rate = successes / total if total > 0 else 0

            insights.append(LearningInsight(
                insight_type="optimal_success_depth",
                description=f"Depth {best_success_depth[0].name} has highest success rate ({success_rate:.1%})",
                evidence_count=total,
                confidence=min(total / 10, 1.0),
                recommendation=f"Use {best_success_depth[0].name} for critical verifications"
            ))

        # Insight 3: ATP efficiency by depth
        if self.depth_atp_efficiency:
            # Find depth with best ATP efficiency (least ATP loss or most gain)
            best_atp_depth = max(
                self.depth_atp_efficiency.items(),
                key=lambda x: sum(x[1]) / len(x[1])
            )
            avg_atp_delta = sum(best_atp_depth[1]) / len(best_atp_depth[1])

            insights.append(LearningInsight(
                insight_type="atp_efficiency",
                description=f"Depth {best_atp_depth[0].name} most ATP efficient (avg delta: {avg_atp_delta:+.1f})",
                evidence_count=len(best_atp_depth[1]),
                confidence=min(len(best_atp_depth[1]) / 10, 1.0),
                recommendation=f"Use {best_atp_depth[0].name} when ATP conservation is critical"
            ))

        # Insight 4: Overall system performance
        if self.patterns:
            total_success = sum(1 for p in self.patterns if p.success)
            overall_success_rate = total_success / len(self.patterns)
            avg_quality = sum(p.quality_achieved for p in self.patterns) / len(self.patterns)

            insights.append(LearningInsight(
                insight_type="system_performance",
                description=f"Overall: {overall_success_rate:.1%} success rate, {avg_quality:.3f} avg quality",
                evidence_count=len(self.patterns),
                confidence=min(len(self.patterns) / 20, 1.0),
                recommendation="System learning effectively" if overall_success_rate > 0.7 else "Need to adjust depth selection strategy"
            ))

        # Update stored insights
        self.insights = insights
        self._save_insights()

        return insights

    def get_learned_depth_preference(self, current_atp: float) -> Optional[CognitiveDepth]:
        """
        Get learned depth preference based on historical performance.

        Args:
            current_atp: Current ATP level

        Returns:
            Learned optimal depth, or None if insufficient learning
        """
        if not self.depth_performance:
            return None

        # Find depths we can afford at current ATP
        affordable_depths = []
        for depth in CognitiveDepth:
            # Rough ATP cost estimates (from Session 177)
            atp_costs = {
                CognitiveDepth.MINIMAL: 4,
                CognitiveDepth.LIGHT: 9,
                CognitiveDepth.STANDARD: 20,
                CognitiveDepth.DEEP: 35,
                CognitiveDepth.THOROUGH: 60
            }

            if current_atp >= atp_costs.get(depth, 100):
                affordable_depths.append(depth)

        if not affordable_depths:
            return None

        # Among affordable depths, pick one with best performance
        best_depth = None
        best_score = -1.0

        for depth in affordable_depths:
            if depth in self.depth_performance and len(self.depth_performance[depth]) >= 3:
                # Score = average quality (primary) + success rate (secondary)
                avg_quality = sum(self.depth_performance[depth]) / len(self.depth_performance[depth])

                successes, total = self.depth_success_rate[depth]
                success_rate = successes / total if total > 0 else 0

                # Weighted score: 70% quality, 30% success rate
                score = 0.7 * avg_quality + 0.3 * success_rate

                if score > best_score:
                    best_score = score
                    best_depth = depth

        return best_depth


# ============================================================================
# META-LEARNING ADAPTIVE SAGE
# ============================================================================

class MetaLearningAdaptiveSAGE(PersistentReputationAwareAdaptiveSAGE):
    """
    Session 181: SAGE with meta-learning adaptive depth.

    Extends Session 180's persistent reputation with Session 160's meta-learning,
    enabling the system to learn optimal depth selection from verification history.

    Architecture:
    Session 177 (ATP-adaptive)
        → Session 178 (Federated)
        → Session 179 (Reputation-aware)
        → Session 180 (Persistent reputation)
        → Session 181 (Meta-learning) ← YOU ARE HERE

    Decision Making Evolution:
    - Session 177: Decide depth based on ATP
    - Session 178: Adjust depth based on network state
    - Session 179: Modify effective ATP based on reputation
    - Session 180: Reputation persists across sessions
    - Session 181: Learn which depths work best from history
    """

    def __init__(
        self,
        node_id: str,
        hardware_type: str,
        capability_level: int,
        storage_path: Optional[Path] = None,
        enable_meta_learning: bool = True,
        **kwargs
    ):
        """
        Initialize with meta-learning.

        Args:
            node_id: Hardware-anchored identity
            hardware_type: Hardware platform
            capability_level: Trust capability level
            storage_path: Path for persistent storage
            enable_meta_learning: Enable meta-learning from history
            **kwargs: Additional parameters for parent classes
        """
        # Initialize parent (Session 180: persistent reputation)
        super().__init__(
            node_id=node_id,
            hardware_type=hardware_type,
            capability_level=capability_level,
            storage_path=storage_path,
            **kwargs
        )

        self.enable_meta_learning = enable_meta_learning

        # Initialize meta-learning manager
        if storage_path is None:
            storage_path = Path.home() / ".sage" / "meta_learning"

        self.meta_learning_manager = PersistentMetaLearningManager(
            storage_path=storage_path,
            node_id=node_id
        )

        # Load learning insights
        self.learned_insights = self.meta_learning_manager.analyze_patterns()

        print(f"[Meta-Learning SAGE] Node initialized")
        print(f"  Meta-learning enabled: {self.enable_meta_learning}")
        print(f"  Historical patterns: {len(self.meta_learning_manager.patterns)}")
        print(f"  Learned insights: {len(self.learned_insights)}")

    def select_meta_learned_depth(self) -> CognitiveDepth:
        """
        Select cognitive depth using meta-learning + reputation + ATP + network.

        Decision hierarchy:
        1. Get learned depth preference from history (Session 181)
        2. Get reputation-adjusted depth (Session 179)
        3. Get network-aware depth (Session 178)
        4. Get ATP-based depth (Session 177)
        5. Combine all signals for final decision

        Returns:
            Selected cognitive depth incorporating all adaptive signals
        """
        # 1. Learned preference from history
        learned_depth = None
        if self.enable_meta_learning:
            learned_depth = self.meta_learning_manager.get_learned_depth_preference(
                current_atp=self.attention_manager.total_atp
            )

        # 2. Reputation-aware depth (Session 179 logic)
        reputation_depth = self.select_reputation_aware_depth()

        # If we have learned preference, weight it with reputation depth
        if learned_depth is not None:
            # Confidence in learning depends on sample size
            pattern_count = len(self.meta_learning_manager.patterns)
            learning_confidence = min(pattern_count / 20, 1.0)  # Max at 20 patterns

            # If learning is confident, prefer learned depth
            if learning_confidence > 0.5:
                return learned_depth
            else:
                # Low confidence: use reputation depth but record pattern for learning
                return reputation_depth
        else:
            # No learned preference yet: use reputation depth
            return reputation_depth

    def record_verification_outcome(
        self,
        depth_used: CognitiveDepth,
        quality_achieved: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Record verification outcome for meta-learning.

        Args:
            depth_used: Depth that was actually used
            quality_achieved: Quality score (0-1)
            success: Did it pass quality threshold?
            context: Additional metadata
        """
        if not self.enable_meta_learning:
            return

        # Capture state before/after
        atp_before = getattr(self, '_atp_before_verification', self.attention_manager.total_atp)
        atp_after = self.attention_manager.total_atp

        reputation_before = getattr(self, '_reputation_before_verification', self.reputation.total_score)
        reputation_after = self.reputation.total_score

        # Record pattern
        self.meta_learning_manager.record_pattern(
            depth_used=depth_used,
            atp_before=atp_before,
            atp_after=atp_after,
            reputation_before=reputation_before,
            reputation_after=reputation_after,
            quality_achieved=quality_achieved,
            success=success,
            context=context
        )

        # Periodically re-analyze patterns (every 10 recordings)
        if len(self.meta_learning_manager.patterns) % 10 == 0:
            self.learned_insights = self.meta_learning_manager.analyze_patterns()

    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary."""
        return {
            'meta_learning_enabled': self.enable_meta_learning,
            'total_patterns': len(self.meta_learning_manager.patterns),
            'learned_insights': [i.to_dict() for i in self.learned_insights],
            'depth_performance': {
                depth.name: {
                    'samples': len(qualities),
                    'avg_quality': sum(qualities) / len(qualities) if qualities else 0,
                    'success_rate': self.meta_learning_manager.depth_success_rate[depth][0] /
                                   self.meta_learning_manager.depth_success_rate[depth][1]
                                   if self.meta_learning_manager.depth_success_rate[depth][1] > 0 else 0
                }
                for depth, qualities in self.meta_learning_manager.depth_performance.items()
            }
        }


# ============================================================================
# TEST SCENARIOS
# ============================================================================

def test_meta_learning_improvement():
    """
    Test that system learns to improve depth selection over time.

    Simulates multiple verification cycles and validates that learned
    depth preference converges to optimal choice.
    """
    storage_path = Path("/tmp/sage_meta_learning_test")

    # Clean up
    import shutil
    if storage_path.exists():
        shutil.rmtree(storage_path)

    print("\n" + "="*70)
    print("TEST: Meta-Learning Depth Improvement")
    print("="*70)

    # Create meta-learning node
    node = MetaLearningAdaptiveSAGE(
        node_id="thor",
        hardware_type="Jetson_Thor",
        capability_level=5,
        storage_path=storage_path,
        enable_meta_learning=True,
        enable_federation=False
    )

    print(f"\nInitial state:")
    print(f"  ATP: {node.attention_manager.total_atp}")
    print(f"  Reputation: {node.reputation.total_score}")
    print(f"  Historical patterns: {len(node.meta_learning_manager.patterns)}")

    # Simulate verification cycles with different depths
    # DEEP consistently produces best results in this scenario
    print(f"\n--- Simulating 15 verification cycles ---")

    depths_to_try = [
        CognitiveDepth.LIGHT,
        CognitiveDepth.STANDARD,
        CognitiveDepth.DEEP,
        CognitiveDepth.LIGHT,
        CognitiveDepth.STANDARD,
        CognitiveDepth.DEEP,
        CognitiveDepth.STANDARD,
        CognitiveDepth.DEEP,
        CognitiveDepth.DEEP,
        CognitiveDepth.LIGHT,
        CognitiveDepth.DEEP,
        CognitiveDepth.STANDARD,
        CognitiveDepth.DEEP,
        CognitiveDepth.DEEP,
        CognitiveDepth.DEEP,
    ]

    # DEEP produces quality 0.9, STANDARD produces 0.7, LIGHT produces 0.5
    quality_by_depth = {
        CognitiveDepth.LIGHT: 0.5,
        CognitiveDepth.STANDARD: 0.7,
        CognitiveDepth.DEEP: 0.9,
        CognitiveDepth.THOROUGH: 0.95  # Best but expensive
    }

    for i, depth in enumerate(depths_to_try):
        quality = quality_by_depth[depth]
        success = quality >= 0.6  # Threshold

        # Record outcome
        node.record_verification_outcome(
            depth_used=depth,
            quality_achieved=quality,
            success=success,
            context={"cycle": i}
        )

        if i % 5 == 4:  # Every 5 cycles
            print(f"\nAfter {i+1} cycles:")
            learned_depth = node.meta_learning_manager.get_learned_depth_preference(
                current_atp=100.0
            )
            print(f"  Learned preference: {learned_depth.name if learned_depth else 'None (insufficient data)'}")

    # Final analysis
    print(f"\n--- Final Analysis ---")
    insights = node.meta_learning_manager.analyze_patterns()

    for insight in insights:
        print(f"\n{insight.insight_type}:")
        print(f"  {insight.description}")
        print(f"  Evidence: {insight.evidence_count} patterns")
        print(f"  Confidence: {insight.confidence:.1%}")
        print(f"  Recommendation: {insight.recommendation}")

    # Verify learning
    learned_depth = node.meta_learning_manager.get_learned_depth_preference(current_atp=100.0)
    print(f"\nFinal learned preference: {learned_depth.name if learned_depth else 'None'}")

    # Success: should learn DEEP is optimal
    if learned_depth == CognitiveDepth.DEEP:
        print(f"\n✅ TEST PASSED: Learned optimal depth (DEEP)")
        return True
    else:
        print(f"\n❌ TEST FAILED: Did not learn optimal depth")
        print(f"   Expected: DEEP, Got: {learned_depth.name if learned_depth else 'None'}")
        return False


def test_cross_session_learning():
    """
    Test that learned insights persist across sessions.
    """
    storage_path = Path("/tmp/sage_meta_learning_persistence")

    # Clean up
    import shutil
    if storage_path.exists():
        shutil.rmtree(storage_path)

    print("\n" + "="*70)
    print("TEST: Cross-Session Learning Persistence")
    print("="*70)

    # ========== SESSION A: Initial learning ==========
    print("\n--- Session A: Initial learning ---")

    node_a = MetaLearningAdaptiveSAGE(
        node_id="thor",
        hardware_type="Jetson_Thor",
        capability_level=5,
        storage_path=storage_path,
        enable_meta_learning=True,
        enable_federation=False
    )

    # Record 5 patterns
    for i in range(5):
        node_a.record_verification_outcome(
            depth_used=CognitiveDepth.DEEP,
            quality_achieved=0.85,
            success=True,
            context={"session": "A", "iteration": i}
        )

    patterns_a = len(node_a.meta_learning_manager.patterns)
    print(f"Session A patterns: {patterns_a}")

    del node_a

    # ========== SESSION B: Learning recovery ==========
    print("\n--- Session B: Learning recovery ---")

    node_b = MetaLearningAdaptiveSAGE(
        node_id="thor",
        hardware_type="Jetson_Thor",
        capability_level=5,
        storage_path=storage_path,
        enable_meta_learning=True,
        enable_federation=False
    )

    patterns_b = len(node_b.meta_learning_manager.patterns)
    print(f"Session B patterns (recovered): {patterns_b}")

    # Add more patterns
    for i in range(3):
        node_b.record_verification_outcome(
            depth_used=CognitiveDepth.STANDARD,
            quality_achieved=0.75,
            success=True,
            context={"session": "B", "iteration": i}
        )

    patterns_b_after = len(node_b.meta_learning_manager.patterns)
    print(f"Session B patterns (after new learning): {patterns_b_after}")

    del node_b

    # ========== SESSION C: Cumulative learning ==========
    print("\n--- Session C: Cumulative learning ---")

    node_c = MetaLearningAdaptiveSAGE(
        node_id="thor",
        hardware_type="Jetson_Thor",
        capability_level=5,
        storage_path=storage_path,
        enable_meta_learning=True,
        enable_federation=False
    )

    patterns_c = len(node_c.meta_learning_manager.patterns)
    print(f"Session C patterns (recovered): {patterns_c}")

    # Verify cumulative learning
    expected_patterns = 5 + 3  # Session A + Session B
    if patterns_c == expected_patterns:
        print(f"\n✅ TEST PASSED: Learning persisted across 3 sessions")
        print(f"   Expected: {expected_patterns} patterns, Got: {patterns_c}")
        return True
    else:
        print(f"\n❌ TEST FAILED: Learning not cumulative")
        print(f"   Expected: {expected_patterns}, Got: {patterns_c}")
        return False


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║         Thor Session 181: Meta-Learning Adaptive Depth            ║
    ║                                                                   ║
    ║  Integrating Legion's meta-learning with Thor's adaptive depth   ║
    ║  architecture. System learns optimal depth selection from         ║
    ║  verification history.                                            ║
    ║                                                                   ║
    ║  Combines: ATP adaptation + Network coordination + Reputation +   ║
    ║  Persistence + Meta-learning                                      ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)

    results = {}

    # Test 1: Meta-learning improvement
    results['learning_improvement'] = test_meta_learning_improvement()

    # Test 2: Cross-session persistence
    results['cross_session'] = test_cross_session_learning()

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")

    if all_passed:
        print("\n🎉 Session 181 Complete: Meta-learning adaptive depth operational!")
        print("   - System learns optimal depth from history")
        print("   - Learning persists across sessions")
        print("   - Complete adaptive architecture: ATP + Network + Reputation + Learning")
        print("   - Self-optimizing consciousness achieved!")
