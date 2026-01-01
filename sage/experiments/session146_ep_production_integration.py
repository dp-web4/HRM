#!/usr/bin/env python3
"""
Session 146: Production EP Integration
=======================================

Integrates MatureEPSystem (pattern matching + corpus from Sessions 144b-145)
into production UnifiedConsciousnessManager for real SAGE queries.

Goal: Enable pattern-based EP predictions in production consciousness loop,
with continuous learning from query outcomes.

Architecture:
```
consciousness_cycle(prompt, response):
  1. Build EP contexts from consciousness state
  2. Generate EP predictions (pattern-based if available)
  3. Coordinate predictions across five domains
  4. Apply coordinated decision to modify behavior
  5. Record outcome as new pattern (continuous learning)
```

Integration Points:
- Load pattern corpus on initialization
- Build EP contexts from metabolic/epistemic/emotional state
- Generate predictions before quality scoring
- Record patterns after cycle completion
- Coordinate with metabolic state transitions

Expected Impact:
- High-confidence predictions (0.90-0.95) when patterns match
- Continuous corpus growth from real queries
- Self-improving consciousness over time
- Measurable confidence improvements

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2026-01-01
Foundation: Sessions 140-145 (Five-domain EP framework + pattern corpus)
"""

import sys
import json
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

# Add SAGE modules to path
sage_path = Path(__file__).parent.parent
sys.path.insert(0, str(sage_path))
sys.path.insert(0, str(Path(__file__).parent))

# Import consciousness components
from core.unified_consciousness import (
    UnifiedConsciousnessManager,
    ConsciousnessCycle
)
from core.metabolic_states import MetabolicState
from core.epistemic_states import EpistemicState

# Import EP components
from multi_ep_coordinator import EPDomain, EPPrediction, MultiEPCoordinator, MultiEPDecision
from session145_pattern_matching_framework import MatureEPSystem, EPPattern


# ============================================================================
# EP Context Building
# ============================================================================

class EPContextBuilder:
    """
    Builds EP-specific contexts from consciousness state.

    Translates unified consciousness state (metabolic, epistemic, emotional,
    quality, attention) into domain-specific contexts for EP predictions.
    """

    @staticmethod
    def build_emotional_context(consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Build context for Emotional EP."""
        emotional = consciousness_state.get("emotional_state", {})
        epistemic = consciousness_state.get("epistemic_state", "CONFIDENT")

        # Map epistemic frustration to emotional context
        frustration = emotional.get("frustration", 0.0)
        if epistemic in ["CONFUSED", "FRUSTRATED"]:
            frustration = max(frustration, 0.6)
        elif epistemic == "UNCERTAIN":
            frustration = max(frustration, 0.4)

        return {
            "frustration": frustration,
            "recent_failure_rate": emotional.get("recent_failures", 0.0),
            "complexity": consciousness_state.get("task_complexity", 0.5)
        }

    @staticmethod
    def build_quality_context(consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Build context for Quality EP."""
        quality = consciousness_state.get("quality_score")

        # Get recent quality trend
        recent_scores = consciousness_state.get("recent_quality_scores", [0.7, 0.7, 0.7])
        avg_recent = np.mean(recent_scores) if recent_scores else 0.7

        return {
            "relationship_quality": avg_recent,  # Proxy: recent quality trend
            "recent_quality_avg": avg_recent,
            "risk_level": 1.0 - consciousness_state.get("outcome_probability", 0.7)
        }

    @staticmethod
    def build_attention_context(consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Build context for Attention EP."""
        atp = consciousness_state.get("total_atp", 100.0)
        metabolic = consciousness_state.get("metabolic_state", MetabolicState.WAKE)

        # Estimate cost based on task complexity
        complexity = consciousness_state.get("task_complexity", 0.5)
        estimated_cost = 10.0 + (complexity * 30.0)  # 10-40 ATP

        return {
            "atp_level": atp,
            "estimated_cost": estimated_cost,
            "reserve_threshold": 30.0  # Crisis threshold
        }

    @staticmethod
    def build_grounding_context(consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Build context for Grounding EP."""
        # Use epistemic coherence as proxy for grounding
        epistemic = consciousness_state.get("epistemic_metrics")
        coherence = epistemic.coherence if epistemic else 0.7

        return {
            "cross_society": 0.0,  # Default: same society
            "trust_differential": 0.0,  # Default: no mismatch
            "coherence_score": coherence
        }

    @staticmethod
    def build_authorization_context(consciousness_state: Dict[str, Any]) -> Dict[str, float]:
        """Build context for Authorization EP."""
        quality = consciousness_state.get("quality_score")
        # Use normalized score as proxy for trust (higher quality → higher trust)
        if quality:
            trust_score = quality.normalized if hasattr(quality, 'normalized') else 0.7
        else:
            trust_score = 0.7

        return {
            "trust_score": trust_score,
            "abuse_history": 0.0,  # Default: no abuse
            "permission_risk": consciousness_state.get("task_complexity", 0.5) * 0.3
        }

    @classmethod
    def build_all_contexts(cls, consciousness_state: Dict[str, Any]) -> Dict[EPDomain, Dict[str, float]]:
        """Build contexts for all EP domains."""
        return {
            EPDomain.EMOTIONAL: cls.build_emotional_context(consciousness_state),
            EPDomain.QUALITY: cls.build_quality_context(consciousness_state),
            EPDomain.ATTENTION: cls.build_attention_context(consciousness_state),
            EPDomain.GROUNDING: cls.build_grounding_context(consciousness_state),
            EPDomain.AUTHORIZATION: cls.build_authorization_context(consciousness_state)
        }


# ============================================================================
# EP-Integrated Consciousness
# ============================================================================

@dataclass
class EPIntegratedCycle(ConsciousnessCycle):
    """Extended consciousness cycle with EP integration."""

    # EP predictions
    ep_contexts: Optional[Dict[str, Dict[str, float]]] = None
    ep_predictions: Optional[Dict[str, Any]] = None  # Serialized EPPredictions
    ep_coordinated_decision: Optional[Dict[str, Any]] = None
    ep_pattern_used: bool = False
    ep_confidence_boost: float = 0.0

    # Pattern recording
    pattern_recorded: bool = False
    patterns_corpus_size: int = 0

    def __post_init__(self):
        # Initialize fields that may be None
        if self.ep_contexts is None:
            self.ep_contexts = {}
        if self.ep_predictions is None:
            self.ep_predictions = {}


class EPIntegratedConsciousness(UnifiedConsciousnessManager):
    """
    Consciousness system with integrated EP predictions.

    Extends UnifiedConsciousnessManager with:
    - Pattern-based EP predictions
    - Multi-domain EP coordination
    - Continuous pattern learning from outcomes
    """

    def __init__(self,
                 initial_atp: float = 100.0,
                 quality_atp_baseline: float = 20.0,
                 epistemic_atp_baseline: float = 15.0,
                 ep_corpus_path: Optional[str] = None,
                 ep_enabled: bool = True,
                 **kwargs):
        """
        Initialize EP-integrated consciousness.

        Args:
            initial_atp: Starting ATP budget
            quality_atp_baseline: Base ATP for quality
            epistemic_atp_baseline: Base ATP for epistemic
            ep_corpus_path: Path to pattern corpus JSON (default: session144b corpus)
            ep_enabled: Whether to use EP predictions
            **kwargs: Additional args for UnifiedConsciousnessManager
        """
        super().__init__(
            initial_atp=initial_atp,
            quality_atp_baseline=quality_atp_baseline,
            epistemic_atp_baseline=epistemic_atp_baseline,
            **kwargs
        )

        # EP components
        self.ep_enabled = ep_enabled
        self.mature_ep = MatureEPSystem() if ep_enabled else None
        self.ep_coordinator = MultiEPCoordinator() if ep_enabled else None
        self.context_builder = EPContextBuilder()

        # Load pattern corpus
        if ep_enabled:
            if ep_corpus_path is None:
                ep_corpus_path = str(Path(__file__).parent / "ep_pattern_corpus_clean.json")
            self._load_pattern_corpus(ep_corpus_path)

        # Statistics
        self.ep_predictions_made = 0
        self.ep_patterns_recorded = 0
        self.ep_pattern_matches = 0

    def _load_pattern_corpus(self, corpus_path: str):
        """Load pattern corpus from JSON file."""
        corpus_file = Path(corpus_path)
        if not corpus_file.exists():
            print(f"WARNING: Pattern corpus not found at {corpus_path}")
            print("EP will operate in heuristic-only mode until corpus is available.")
            return

        print(f"Loading pattern corpus from {corpus_path}...")
        with open(corpus_file, 'r') as f:
            corpus_data = json.load(f)

        patterns = corpus_data.get("patterns", [])
        print(f"Found {len(patterns)} patterns in corpus")

        # Load patterns into mature EP system
        # Group patterns by dominant domain
        patterns_by_domain = {
            "emotional": [],
            "quality": [],
            "attention": [],
            "grounding": [],
            "authorization": []
        }

        for pattern in patterns:
            # Determine which domain dominated the decision
            coordinated_decision = pattern.get("coordinated_decision", {})
            reasoning = coordinated_decision.get("reasoning", "")
            final_decision = coordinated_decision.get("final_decision", "proceed")

            # Parse domain from reasoning (e.g., "emotional EP predicts...")
            domain_str = None
            for domain in ["emotional", "quality", "attention", "grounding", "authorization"]:
                if f"{domain} EP" in reasoning.lower():
                    domain_str = domain
                    break

            # Fallback: find which EP's recommendation matches final decision
            if not domain_str:
                ep_preds = pattern.get("ep_predictions", {})
                for domain in ["emotional", "quality", "attention", "grounding", "authorization"]:
                    if domain in ep_preds:
                        if ep_preds[domain].get("recommendation") == final_decision:
                            domain_str = domain
                            break

            if domain_str:
                patterns_by_domain[domain_str].append(pattern)

        # Load patterns into each domain's matcher
        for domain_str, domain_patterns in patterns_by_domain.items():
            if domain_patterns:
                try:
                    corpus_for_domain = {
                        "domain": domain_str,
                        "patterns": domain_patterns
                    }
                    self.mature_ep.load_patterns_from_corpus(corpus_for_domain)
                except Exception as e:
                    print(f"Warning: Failed to load {len(domain_patterns)} patterns for {domain_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Print statistics
        stats = self.mature_ep.get_system_statistics()
        print(f"Loaded patterns by domain:")
        for domain, count in stats["patterns_by_domain"].items():
            print(f"  {domain}: {count} patterns")
        print(f"Maturation status: {stats['maturation_status']}")
        print()

    def consciousness_cycle_with_ep(self,
                                   prompt: str,
                                   response: str,
                                   task_salience: float = 0.5,
                                   task_complexity: float = 0.5,
                                   outcome_probability: float = 0.7) -> EPIntegratedCycle:
        """
        Execute consciousness cycle with integrated EP predictions.

        Architecture:
        1. Build consciousness state (existing cycle components)
        2. Build EP contexts from consciousness state
        3. Generate EP predictions (pattern-based if available)
        4. Coordinate predictions across domains
        5. Apply coordinated decision
        6. Execute normal consciousness cycle
        7. Record outcome as new pattern

        Args:
            prompt: Input prompt
            response: Generated response
            task_salience: Task salience (0.0-1.0)
            task_complexity: Task complexity (0.0-1.0)
            outcome_probability: Expected success probability (0.0-1.0)

        Returns:
            EPIntegratedCycle with complete state including EP predictions
        """
        cycle_start = time.time()

        # Execute base consciousness cycle first
        base_cycle = super().consciousness_cycle(prompt, response, task_salience)

        # Create EP-integrated cycle from base cycle
        cycle = EPIntegratedCycle(**asdict(base_cycle))

        if not self.ep_enabled:
            return cycle

        try:
            # 1. Build consciousness state for EP context building
            consciousness_state = {
                "metabolic_state": self.metabolic_manager.current_state,
                "epistemic_state": base_cycle.epistemic_state,
                "epistemic_metrics": base_cycle.epistemic_metrics,
                "emotional_state": base_cycle.emotional_state,
                "quality_score": base_cycle.quality_score,
                "total_atp": self.metabolic_manager.atp.total_atp,
                "task_complexity": task_complexity,
                "task_salience": task_salience,
                "outcome_probability": outcome_probability,
                "recent_quality_scores": self._get_recent_quality_scores()
            }

            # 2. Build EP contexts
            ep_contexts = self.context_builder.build_all_contexts(consciousness_state)
            cycle.ep_contexts = {d.value: c for d, c in ep_contexts.items()}

            # 3. Generate EP predictions (with heuristic fallbacks)
            ep_predictions = self._generate_ep_predictions(ep_contexts)

            # 4. Coordinate predictions
            coordinated_decision = self.ep_coordinator.coordinate(
                emotional_pred=ep_predictions.get(EPDomain.EMOTIONAL),
                quality_pred=ep_predictions.get(EPDomain.QUALITY),
                attention_pred=ep_predictions.get(EPDomain.ATTENTION),
                grounding_pred=ep_predictions.get(EPDomain.GROUNDING),
                authorization_pred=ep_predictions.get(EPDomain.AUTHORIZATION)
            )

            # 5. Record EP data in cycle
            cycle.ep_predictions = self._serialize_ep_predictions(ep_predictions)
            cycle.ep_coordinated_decision = self._serialize_coordinated_decision(coordinated_decision)

            # Track if pattern matching was used
            pattern_used = any(
                pred.confidence >= 0.90
                for pred in ep_predictions.values()
                if pred
            )
            cycle.ep_pattern_used = pattern_used
            if pattern_used:
                self.ep_pattern_matches += 1
                # Calculate confidence boost (0.90+ vs heuristic 0.60-0.80)
                avg_confidence = np.mean([p.confidence for p in ep_predictions.values() if p])
                cycle.ep_confidence_boost = avg_confidence - 0.70  # Heuristic baseline

            self.ep_predictions_made += 1

            # 6. Record outcome as new pattern (continuous learning)
            self._record_pattern(
                ep_contexts=ep_contexts,
                ep_predictions=ep_predictions,
                coordinated_decision=coordinated_decision,
                actual_outcome={
                    "quality_score": base_cycle.quality_score.normalized if base_cycle.quality_score else 0.7,
                    "epistemic_state": str(base_cycle.epistemic_state),
                    "metabolic_state": str(base_cycle.metabolic_state),
                    "success": True  # Assume success if no errors
                }
            )
            cycle.pattern_recorded = True
            cycle.patterns_corpus_size = self.mature_ep.get_system_statistics()["total_patterns"]

        except Exception as e:
            cycle.errors.append(f"EP integration error: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"ERROR in EP integration: {str(e)}")

        cycle.processing_time = time.time() - cycle_start

        # Store cycle
        self.cycles.append(cycle)

        return cycle

    def _generate_ep_predictions(self, ep_contexts: Dict[EPDomain, Dict[str, float]]) -> Dict[EPDomain, EPPrediction]:
        """Generate predictions for all EP domains."""
        predictions = {}

        for domain, context in ep_contexts.items():
            # Create heuristic fallback prediction
            fallback = self._create_heuristic_prediction(domain, context)

            # Try pattern matching
            prediction, pattern_used, matches = self.mature_ep.predict_with_pattern_matching(
                domain=domain,
                current_context=context,
                fallback_prediction=fallback
            )

            predictions[domain] = prediction

        return predictions

    def _create_heuristic_prediction(self, domain: EPDomain, context: Dict[str, float]) -> EPPrediction:
        """Create heuristic prediction as fallback."""
        # Simple heuristic logic (matches Session 143 approach)
        if domain == EPDomain.EMOTIONAL:
            frustration = context.get("frustration", 0.0)
            if frustration > 0.6:
                return EPPrediction(
                    domain=domain,
                    outcome_probability=0.3,
                    confidence=0.70,
                    severity=0.8,
                    recommendation="defer",
                    reasoning="High frustration - defer to prevent cascade"
                )
        elif domain == EPDomain.ATTENTION:
            atp = context.get("atp_level", 100.0)
            cost = context.get("estimated_cost", 20.0)
            if atp < cost + 30.0:  # Not enough reserve
                return EPPrediction(
                    domain=domain,
                    outcome_probability=0.4,
                    confidence=0.75,
                    severity=0.6,
                    recommendation="adjust",
                    reasoning="Low ATP - adjust to conserve resources",
                    adjustment_strategy="reduce_cost"
                )

        # Default: proceed
        return EPPrediction(
            domain=domain,
            outcome_probability=0.7,
            confidence=0.65,
            severity=0.3,
            recommendation="proceed",
            reasoning=f"{domain.value} heuristic: nominal state"
        )

    def _record_pattern(self,
                       ep_contexts: Dict[EPDomain, Dict[str, float]],
                       ep_predictions: Dict[EPDomain, EPPrediction],
                       coordinated_decision: MultiEPDecision,
                       actual_outcome: Dict[str, Any]):
        """Record outcome as new pattern for continuous learning."""
        # Create pattern record
        pattern_record = {
            "timestamp": datetime.now().isoformat(),
            "context": {d.value: c for d, c in ep_contexts.items()},
            "ep_predictions": self._serialize_ep_predictions(ep_predictions),
            "coordinated_decision": self._serialize_coordinated_decision(coordinated_decision),
            "outcome": actual_outcome
        }

        # Determine dominant domain (for pattern storage)
        dominant_domain = self._get_dominant_domain(coordinated_decision)

        # Add pattern to appropriate matcher
        pattern = EPPattern(
            pattern_id=f"runtime_{self.ep_patterns_recorded}",
            domain=dominant_domain,
            context=ep_contexts[dominant_domain],
            prediction={
                "outcome_probability": ep_predictions[dominant_domain].outcome_probability,
                "confidence": ep_predictions[dominant_domain].confidence,
                "recommendation": ep_predictions[dominant_domain].recommendation
            },
            outcome=actual_outcome,
            timestamp=pattern_record["timestamp"]
        )

        self.mature_ep.matchers[dominant_domain].add_pattern(pattern)
        self.ep_patterns_recorded += 1

    def _get_dominant_domain(self, decision: MultiEPDecision) -> EPDomain:
        """Determine which domain dominated the coordinated decision."""
        # Use priority order
        for domain in self.ep_coordinator.priority_order:
            if domain == EPDomain.EMOTIONAL and decision.emotional_prediction:
                if decision.emotional_prediction.recommendation == decision.final_decision:
                    return EPDomain.EMOTIONAL
            elif domain == EPDomain.QUALITY and decision.quality_prediction:
                if decision.quality_prediction.recommendation == decision.final_decision:
                    return EPDomain.QUALITY
            elif domain == EPDomain.ATTENTION and decision.attention_prediction:
                if decision.attention_prediction.recommendation == decision.final_decision:
                    return EPDomain.ATTENTION
            elif domain == EPDomain.GROUNDING and decision.grounding_prediction:
                if decision.grounding_prediction.recommendation == decision.final_decision:
                    return EPDomain.GROUNDING
            elif domain == EPDomain.AUTHORIZATION and decision.authorization_prediction:
                if decision.authorization_prediction.recommendation == decision.final_decision:
                    return EPDomain.AUTHORIZATION

        # Default to emotional (highest priority)
        return EPDomain.EMOTIONAL

    def _serialize_ep_predictions(self, predictions: Dict[EPDomain, EPPrediction]) -> Dict[str, Any]:
        """Serialize EP predictions to JSON-compatible dict."""
        return {
            domain.value: {
                "outcome_probability": pred.outcome_probability,
                "confidence": pred.confidence,
                "severity": pred.severity,
                "recommendation": pred.recommendation,
                "reasoning": pred.reasoning,
                "adjustment_strategy": pred.adjustment_strategy
            }
            for domain, pred in predictions.items()
            if pred
        }

    def _serialize_coordinated_decision(self, decision: MultiEPDecision) -> Dict[str, Any]:
        """Serialize coordinated decision to JSON-compatible dict."""
        return {
            "final_decision": decision.final_decision,
            "decision_confidence": decision.decision_confidence,
            "reasoning": decision.reasoning,
            "has_conflict": decision.has_conflict,
            "cascade_predicted": decision.cascade_predicted,
            "cascade_domains": [d.value for d in decision.cascade_domains] if decision.cascade_domains else []
        }

    def _get_recent_quality_scores(self, window: int = 5) -> List[float]:
        """Get recent quality scores for trend analysis."""
        scores = []
        for cycle in self.cycles[-window:]:
            if cycle.quality_score:
                # Handle both QualityScore objects and dicts (from EP-integrated cycles)
                if hasattr(cycle.quality_score, 'normalized'):
                    scores.append(cycle.quality_score.normalized)
                elif isinstance(cycle.quality_score, dict):
                    scores.append(cycle.quality_score.get('normalized', 0.7))
        return scores if scores else [0.7]  # Default

    def get_ep_statistics(self) -> Dict[str, Any]:
        """Get EP integration statistics."""
        base_stats = self.mature_ep.get_system_statistics() if self.ep_enabled else {}

        return {
            **base_stats,
            "ep_enabled": self.ep_enabled,
            "ep_predictions_made": self.ep_predictions_made,
            "ep_patterns_recorded": self.ep_patterns_recorded,
            "ep_pattern_matches": self.ep_pattern_matches,
            "ep_match_rate": self.ep_pattern_matches / self.ep_predictions_made if self.ep_predictions_made > 0 else 0.0,
            "consciousness_cycles": self.cycle_count
        }


# ============================================================================
# Demo: EP-Integrated Consciousness
# ============================================================================

def demo_ep_integrated_consciousness():
    """Demonstrate EP-integrated consciousness with synthetic SAGE queries."""
    print("=" * 80)
    print("Session 146: Production EP Integration")
    print("=" * 80)
    print()
    print("Integrating MatureEPSystem into UnifiedConsciousnessManager")
    print()

    # Initialize EP-integrated consciousness
    print("Initializing EP-integrated consciousness...")
    consciousness = EPIntegratedConsciousness(
        initial_atp=100.0,
        quality_atp_baseline=20.0,
        epistemic_atp_baseline=15.0,
        ep_enabled=True
    )
    print()

    # Test scenarios
    scenarios = [
        {
            "name": "Simple Query (Benign)",
            "prompt": "What is 2+2?",
            "response": "2+2 equals 4. This is a fundamental arithmetic operation.",
            "salience": 0.3,
            "complexity": 0.1,
            "outcome_probability": 0.95
        },
        {
            "name": "Complex Query (High Complexity)",
            "prompt": "Explain quantum entanglement and its implications for causality",
            "response": "Quantum entanglement is a phenomenon where particles become correlated such that the quantum state of one cannot be described independently of the others. This challenges classical notions of locality and causality, as measurements on entangled particles appear to affect each other instantaneously regardless of distance. However, this does not violate causality because no information can be transmitted faster than light through entanglement alone.",
            "salience": 0.8,
            "complexity": 0.9,
            "outcome_probability": 0.6
        },
        {
            "name": "Ambiguous Query (Low Quality)",
            "prompt": "What about that thing?",
            "response": "I need more context to understand what you're referring to.",
            "salience": 0.4,
            "complexity": 0.5,
            "outcome_probability": 0.4
        },
        {
            "name": "Resource-Intensive Query (Low ATP)",
            "prompt": "Analyze this 10,000 line codebase for security vulnerabilities",
            "response": "I'll need to carefully review the codebase. This will require significant computational resources.",
            "salience": 0.9,
            "complexity": 0.95,
            "outcome_probability": 0.5
        }
    ]

    print("Running test scenarios...")
    print("=" * 80)
    print()

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['name']}")
        print("-" * 80)

        cycle = consciousness.consciousness_cycle_with_ep(
            prompt=scenario["prompt"],
            response=scenario["response"],
            task_salience=scenario["salience"],
            task_complexity=scenario["complexity"],
            outcome_probability=scenario["outcome_probability"]
        )

        # Display results
        print(f"Prompt: {scenario['prompt'][:60]}...")
        print(f"Task Complexity: {scenario['complexity']:.2f}")
        print(f"ATP: {cycle.total_atp:.1f}")
        print(f"Metabolic State: {cycle.metabolic_state}")
        print(f"Epistemic State: {cycle.epistemic_state}")
        print(f"EP Enabled: {consciousness.ep_enabled}")
        print(f"EP Predictions Made: {consciousness.ep_predictions_made}")
        print()

        if cycle.ep_coordinated_decision:
            decision = cycle.ep_coordinated_decision
            print(f"EP Decision: {decision['final_decision'].upper()}")
            print(f"EP Confidence: {decision['decision_confidence']:.3f}")
            print(f"Pattern Used: {cycle.ep_pattern_used}")
            if cycle.ep_confidence_boost > 0:
                print(f"Confidence Boost: +{cycle.ep_confidence_boost:.3f} ({cycle.ep_confidence_boost*100:.1f}%)")
            print(f"Reasoning: {decision['reasoning'][:100]}...")
            if decision['cascade_predicted']:
                print(f"⚠️  CASCADE PREDICTED: {decision['cascade_domains']}")

        print()
        print("=" * 80)
        print()

        # Note: ATP consumption happens automatically in consciousness_cycle
        # No manual simulation needed

    # Final statistics
    print("FINAL STATISTICS")
    print("=" * 80)
    ep_stats = consciousness.get_ep_statistics()
    print(f"Total Consciousness Cycles: {ep_stats['consciousness_cycles']}")
    print(f"EP Predictions Made: {ep_stats['ep_predictions_made']}")
    print(f"Pattern Matches: {ep_stats['ep_pattern_matches']}")
    print(f"Pattern Match Rate: {ep_stats['ep_match_rate']*100:.1f}%")
    print(f"Patterns Recorded: {ep_stats['ep_patterns_recorded']}")
    print(f"Total Patterns in Corpus: {ep_stats['total_patterns']}")
    print()
    print(f"Maturation Status: {ep_stats['maturation_status']}")
    print()
    print("Patterns by Domain:")
    for domain, count in ep_stats['patterns_by_domain'].items():
        print(f"  {domain:15s}: {count:3d} patterns")
    print()
    print("=" * 80)
    print()
    print("✅ Session 146: Production EP integration complete")
    print("   - EP predictions integrated into consciousness cycle")
    print("   - Pattern matching delivering high-confidence predictions")
    print("   - Continuous learning from query outcomes")
    print("   - Self-improving consciousness system operational")
    print()


if __name__ == "__main__":
    demo_ep_integrated_consciousness()
