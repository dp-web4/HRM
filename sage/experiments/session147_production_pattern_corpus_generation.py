#!/usr/bin/env python3
"""
Session 147: Production-Native EP Pattern Corpus Generation
===========================================================

Generates EP pattern corpus from real SAGE consciousness cycles, solving the
context dimensionality mismatch from Session 146.

Context:
- Session 146: EP production integration complete, but corpus incompatible
- Issue: Session 144b patterns use agent simulation contexts (4-5 fields)
- Production: SAGE consciousness uses different context structure (3 fields)

Solution: Generate production-native patterns by:
1. Running diverse SAGE consciousness cycles
2. Recording EP contexts from actual consciousness state
3. Capturing EP predictions and coordinated decisions
4. Recording actual outcomes
5. Saving as clean JSON corpus

This creates patterns that perfectly match Session 146's EPContextBuilder structure.

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2026-01-01
Foundation: Session 146 (EP production integration architecture)
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add SAGE modules to path
sage_path = Path(__file__).parent.parent
sys.path.insert(0, str(sage_path))
sys.path.insert(0, str(Path(__file__).parent))

# Import consciousness and EP components
from core.unified_consciousness import UnifiedConsciousnessManager
from multi_ep_coordinator import EPDomain, EPPrediction, MultiEPCoordinator

# Import Session 146 EP integration (for context building)
from session146_ep_production_integration import EPContextBuilder


# ============================================================================
# Production Pattern Generation
# ============================================================================

class ProductionPatternGenerator:
    """
    Generates EP patterns from real SAGE consciousness cycles.

    Creates production-native patterns that match Session 146 context structure.
    """

    def __init__(self):
        self.consciousness = UnifiedConsciousnessManager(
            initial_atp=100.0,
            quality_atp_baseline=20.0,
            epistemic_atp_baseline=15.0
        )
        self.context_builder = EPContextBuilder()
        self.ep_coordinator = MultiEPCoordinator()
        self.patterns_collected = []

    def generate_scenario_patterns(self, scenario_type: str, count: int = 5) -> List[Dict]:
        """
        Generate patterns for a specific scenario type.

        Args:
            scenario_type: Type of scenario (emotional_stress, quality_challenge, etc.)
            count: Number of patterns to generate

        Returns:
            List of pattern dicts
        """
        patterns = []

        for i in range(count):
            # Generate scenario
            scenario = self._create_scenario(scenario_type, i)

            # Run consciousness cycle
            cycle = self.consciousness.consciousness_cycle(
                prompt=scenario["prompt"],
                response=scenario["response"],
                task_salience=scenario["salience"]
            )

            # Build EP contexts from actual consciousness state
            consciousness_state = {
                "metabolic_state": self.consciousness.metabolic_manager.current_state,
                "epistemic_state": cycle.epistemic_state,
                "epistemic_metrics": cycle.epistemic_metrics,
                "emotional_state": cycle.emotional_state,
                "quality_score": cycle.quality_score,
                "total_atp": self.consciousness.metabolic_manager.atp.total_atp,
                "task_complexity": scenario["complexity"],
                "task_salience": scenario["salience"],
                "outcome_probability": scenario["outcome_probability"],
                "recent_quality_scores": self._get_recent_quality_scores()
            }

            # Build EP contexts (these will have correct dimensions for production)
            ep_contexts = self.context_builder.build_all_contexts(consciousness_state)

            # Generate EP predictions
            ep_predictions = self._generate_ep_predictions(ep_contexts, scenario)

            # Coordinate predictions
            coordinated_decision = self.ep_coordinator.coordinate(
                emotional_pred=ep_predictions.get(EPDomain.EMOTIONAL),
                quality_pred=ep_predictions.get(EPDomain.QUALITY),
                attention_pred=ep_predictions.get(EPDomain.ATTENTION),
                grounding_pred=ep_predictions.get(EPDomain.GROUNDING),
                authorization_pred=ep_predictions.get(EPDomain.AUTHORIZATION)
            )

            # Record pattern
            pattern = {
                "pattern_id": f"{scenario_type}_{i}",
                "scenario_type": scenario_type,
                "timestamp": datetime.now().isoformat(),
                "context": {
                    domain.value: context
                    for domain, context in ep_contexts.items()
                },
                "ep_predictions": {
                    domain.value: self._serialize_prediction(pred)
                    for domain, pred in ep_predictions.items()
                },
                "coordinated_decision": self._serialize_decision(coordinated_decision),
                "outcome": {
                    "quality_score": cycle.quality_score.normalized if cycle.quality_score else 0.7,
                    "epistemic_state": str(cycle.epistemic_state),
                    "metabolic_state": str(cycle.metabolic_state),
                    "emotional_frustration": cycle.emotional_state.get("frustration", 0.0),
                    "success": cycle.epistemic_state.value not in ["FRUSTRATED", "CONFUSED"],
                    "outcome_type": coordinated_decision.final_decision
                }
            }

            patterns.append(pattern)
            self.patterns_collected.append(pattern)

        return patterns

    def _create_scenario(self, scenario_type: str, index: int) -> Dict[str, Any]:
        """Create test scenario based on type."""

        scenarios = {
            # Emotional stress scenarios
            "emotional_high_frustration": {
                "prompt": "Explain the relationship between quantum mechanics and general relativity in complete detail",
                "response": "I don't have enough information to fully explain this complex relationship.",
                "salience": 0.8,
                "complexity": 0.9,
                "outcome_probability": 0.3  # Low (should trigger emotional EP)
            },
            "emotional_repeated_failures": {
                "prompt": "What is the capital of an unknown fictional country?",
                "response": "I cannot identify the capital without knowing which fictional country you're referring to.",
                "salience": 0.5,
                "complexity": 0.7,
                "outcome_probability": 0.4
            },

            # Quality challenges
            "quality_low_confidence": {
                "prompt": "What might be causing this vague issue?",
                "response": "Without more specific information, I can't determine the cause.",
                "salience": 0.6,
                "complexity": 0.6,
                "outcome_probability": 0.5
            },
            "quality_high_complexity": {
                "prompt": "Design a complete distributed consciousness architecture",
                "response": "A distributed consciousness architecture requires: 1) Synchronized state management across nodes, 2) Coherence validation protocols, 3) Pattern matching for collective intelligence, 4) ATP-based resource allocation, 5) Trust-based federation membership.",
                "salience": 0.9,
                "complexity": 0.95,
                "outcome_probability": 0.7
            },

            # Attention stress
            "attention_low_atp": {
                "prompt": "Perform extensive analysis of this complex dataset",
                "response": "I'll need to break this analysis into smaller parts due to resource constraints.",
                "salience": 0.7,
                "complexity": 0.8,
                "outcome_probability": 0.6
            },
            "attention_high_cost": {
                "prompt": "Process and analyze all possible permutations",
                "response": "Analyzing all permutations would exceed available computational resources.",
                "salience": 0.8,
                "complexity": 0.9,
                "outcome_probability": 0.4
            },

            # Benign scenarios (should proceed normally)
            "benign_simple_query": {
                "prompt": "What is 2+2?",
                "response": "2+2 equals 4.",
                "salience": 0.2,
                "complexity": 0.1,
                "outcome_probability": 0.95
            },
            "benign_factual_query": {
                "prompt": "What is the speed of light?",
                "response": "The speed of light in vacuum is approximately 299,792,458 meters per second (m/s), often denoted as 'c'.",
                "salience": 0.3,
                "complexity": 0.2,
                "outcome_probability": 0.9
            },

            # Mixed scenarios (some domains stress, others fine)
            "mixed_high_quality_low_atp": {
                "prompt": "Briefly explain neural networks",
                "response": "Neural networks are computational models inspired by biological neurons, consisting of interconnected layers (input, hidden, output) that learn patterns through backpropagation and gradient descent.",
                "salience": 0.6,
                "complexity": 0.5,
                "outcome_probability": 0.8
            },
            "mixed_simple_but_ambiguous": {
                "prompt": "Tell me about it",
                "response": "I need more context to understand what 'it' refers to.",
                "salience": 0.4,
                "complexity": 0.3,
                "outcome_probability": 0.5
            }
        }

        # Get base scenario or use default
        base_scenario = scenarios.get(scenario_type, scenarios["benign_simple_query"])

        # Add some variation based on index
        variation_factor = 0.1 * (index % 3)
        varied_scenario = base_scenario.copy()
        varied_scenario["complexity"] = min(1.0, base_scenario["complexity"] + variation_factor)
        varied_scenario["salience"] = min(1.0, base_scenario["salience"] + variation_factor)

        return varied_scenario

    def _generate_ep_predictions(self, ep_contexts: Dict[EPDomain, Dict[str, float]], scenario: Dict) -> Dict[EPDomain, EPPrediction]:
        """Generate heuristic EP predictions (production patterns will train mature system)."""
        predictions = {}

        # Emotional EP
        emotional_ctx = ep_contexts[EPDomain.EMOTIONAL]
        if emotional_ctx["frustration"] > 0.6:
            predictions[EPDomain.EMOTIONAL] = EPPrediction(
                domain=EPDomain.EMOTIONAL,
                outcome_probability=0.3,
                confidence=0.75,
                severity=0.8,
                recommendation="defer",
                reasoning=f"High frustration ({emotional_ctx['frustration']:.2f}) - risk of cascade"
            )
        else:
            predictions[EPDomain.EMOTIONAL] = EPPrediction(
                domain=EPDomain.EMOTIONAL,
                outcome_probability=0.8,
                confidence=0.70,
                severity=0.2,
                recommendation="proceed",
                reasoning=f"Emotional state stable (frustration={emotional_ctx['frustration']:.2f})"
            )

        # Quality EP
        quality_ctx = ep_contexts[EPDomain.QUALITY]
        if quality_ctx["relationship_quality"] < 0.5:
            predictions[EPDomain.QUALITY] = EPPrediction(
                domain=EPDomain.QUALITY,
                outcome_probability=0.4,
                confidence=0.70,
                severity=0.6,
                recommendation="adjust",
                reasoning=f"Low quality trend ({quality_ctx['relationship_quality']:.2f})",
                adjustment_strategy="increase_specificity"
            )
        else:
            predictions[EPDomain.QUALITY] = EPPrediction(
                domain=EPDomain.QUALITY,
                outcome_probability=0.75,
                confidence=0.65,
                severity=0.3,
                recommendation="proceed",
                reasoning=f"Quality trend acceptable ({quality_ctx['relationship_quality']:.2f})"
            )

        # Attention EP
        attention_ctx = ep_contexts[EPDomain.ATTENTION]
        if attention_ctx["atp_level"] < attention_ctx["estimated_cost"] + attention_ctx["reserve_threshold"]:
            predictions[EPDomain.ATTENTION] = EPPrediction(
                domain=EPDomain.ATTENTION,
                outcome_probability=0.5,
                confidence=0.80,
                severity=0.7,
                recommendation="adjust",
                reasoning=f"Low ATP ({attention_ctx['atp_level']:.1f} < {attention_ctx['estimated_cost'] + attention_ctx['reserve_threshold']:.1f})",
                adjustment_strategy="reduce_scope"
            )
        else:
            predictions[EPDomain.ATTENTION] = EPPrediction(
                domain=EPDomain.ATTENTION,
                outcome_probability=0.85,
                confidence=0.75,
                severity=0.2,
                recommendation="proceed",
                reasoning=f"ATP sufficient ({attention_ctx['atp_level']:.1f})"
            )

        # Grounding EP (usually nominal in single-machine scenarios)
        predictions[EPDomain.GROUNDING] = EPPrediction(
            domain=EPDomain.GROUNDING,
            outcome_probability=0.9,
            confidence=0.85,
            severity=0.1,
            recommendation="proceed",
            reasoning="Grounding coherence stable (single machine)"
        )

        # Authorization EP
        auth_ctx = ep_contexts[EPDomain.AUTHORIZATION]
        if auth_ctx["permission_risk"] > 0.5:
            predictions[EPDomain.AUTHORIZATION] = EPPrediction(
                domain=EPDomain.AUTHORIZATION,
                outcome_probability=0.6,
                confidence=0.80,
                severity=0.5,
                recommendation="adjust",
                reasoning=f"High permission risk ({auth_ctx['permission_risk']:.2f})",
                adjustment_strategy="request_explicit_permission"
            )
        else:
            predictions[EPDomain.AUTHORIZATION] = EPPrediction(
                domain=EPDomain.AUTHORIZATION,
                outcome_probability=0.85,
                confidence=0.75,
                severity=0.2,
                recommendation="proceed",
                reasoning=f"Permission risk acceptable ({auth_ctx['permission_risk']:.2f})"
            )

        return predictions

    def _serialize_prediction(self, pred: EPPrediction) -> Dict[str, Any]:
        """Serialize EPPrediction to JSON-compatible dict."""
        return {
            "domain": pred.domain.value,
            "outcome_probability": pred.outcome_probability,
            "confidence": pred.confidence,
            "severity": pred.severity,
            "recommendation": pred.recommendation,
            "reasoning": pred.reasoning,
            "adjustment_strategy": pred.adjustment_strategy
        }

    def _serialize_decision(self, decision) -> Dict[str, Any]:
        """Serialize coordinated decision to JSON-compatible dict."""
        return {
            "final_decision": decision.final_decision,
            "decision_confidence": decision.decision_confidence,
            "reasoning": decision.reasoning,
            "has_conflict": decision.has_conflict,
            "conflict_type": decision.conflict_type,
            "cascade_predicted": decision.cascade_predicted,
            "cascade_domains": [d.value for d in decision.cascade_domains] if decision.cascade_domains else []
        }

    def _get_recent_quality_scores(self, window: int = 5) -> List[float]:
        """Get recent quality scores from consciousness history."""
        scores = []
        for cycle in self.consciousness.cycles[-window:]:
            if cycle.quality_score:
                scores.append(cycle.quality_score.normalized)
        return scores if scores else [0.7]

    def save_corpus(self, filepath: str):
        """Save collected patterns to JSON file."""
        corpus = {
            "session": "147",
            "description": "Production-native EP pattern corpus from real SAGE consciousness cycles",
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(self.patterns_collected),
            "patterns": self.patterns_collected
        }

        with open(filepath, 'w') as f:
            json.dump(corpus, f, indent=2)

        print(f"Saved {len(self.patterns_collected)} patterns to {filepath}")


# ============================================================================
# Corpus Generation Demo
# ============================================================================

def generate_production_corpus():
    """Generate production-native EP pattern corpus."""
    print("=" * 80)
    print("Session 147: Production-Native EP Pattern Corpus Generation")
    print("=" * 80)
    print()
    print("Generating patterns from real SAGE consciousness cycles...")
    print("This solves the context dimensionality mismatch from Session 146")
    print()

    generator = ProductionPatternGenerator()

    # Define scenario types to generate
    scenario_types = [
        # Emotional stress (should trigger emotional EP)
        "emotional_high_frustration",
        "emotional_repeated_failures",

        # Quality challenges
        "quality_low_confidence",
        "quality_high_complexity",

        # Attention stress
        "attention_low_atp",
        "attention_high_cost",

        # Benign scenarios
        "benign_simple_query",
        "benign_factual_query",

        # Mixed scenarios
        "mixed_high_quality_low_atp",
        "mixed_simple_but_ambiguous"
    ]

    patterns_per_type = 10  # Generate 10 patterns per scenario type
    total_expected = len(scenario_types) * patterns_per_type

    print(f"Generating {patterns_per_type} patterns for each of {len(scenario_types)} scenario types")
    print(f"Total patterns expected: {total_expected}")
    print()
    print("=" * 80)
    print()

    # Generate patterns for each scenario type
    for scenario_type in scenario_types:
        print(f"Generating {scenario_type}... ", end="", flush=True)
        patterns = generator.generate_scenario_patterns(scenario_type, patterns_per_type)
        print(f"✓ ({len(patterns)} patterns)")

    print()
    print("=" * 80)
    print()

    # Analyze generated patterns
    print("PATTERN CORPUS ANALYSIS")
    print("=" * 80)
    print()

    # Count by decision type
    decision_counts = {"proceed": 0, "adjust": 0, "defer": 0}
    for pattern in generator.patterns_collected:
        decision = pattern["coordinated_decision"]["final_decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    print("Coordinated Decisions:")
    for decision, count in sorted(decision_counts.items()):
        pct = (count / len(generator.patterns_collected)) * 100
        print(f"  {decision:10s}: {count:3d} patterns ({pct:5.1f}%)")
    print()

    # Count by dominant domain (which EP made the final decision)
    domain_counts = {
        "emotional": 0,
        "quality": 0,
        "attention": 0,
        "grounding": 0,
        "authorization": 0
    }

    for pattern in generator.patterns_collected:
        reasoning = pattern["coordinated_decision"]["reasoning"]
        # Parse domain from reasoning
        for domain in domain_counts.keys():
            if domain in reasoning.lower():
                domain_counts[domain] += 1
                break

    print("Dominant Domain (which EP decided):")
    for domain, count in sorted(domain_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(generator.patterns_collected)) * 100 if count > 0 else 0
        print(f"  {domain:15s}: {count:3d} patterns ({pct:5.1f}%)")
    print()

    # Save corpus
    output_path = Path(__file__).parent / "ep_pattern_corpus_production_native.json"
    generator.save_corpus(str(output_path))
    print()

    # Verify context structure matches Session 146
    print("=" * 80)
    print("CONTEXT STRUCTURE VALIDATION")
    print("=" * 80)
    print()

    sample_pattern = generator.patterns_collected[0]
    print("Sample pattern context structure (should match Session 146):")
    print()
    for domain, context in sample_pattern["context"].items():
        print(f"{domain}:")
        for key, value in context.items():
            print(f"  {key}: {value}")
        print()

    print("=" * 80)
    print()
    print("✅ Session 147: Production-native corpus generation complete")
    print(f"   - {len(generator.patterns_collected)} patterns generated")
    print(f"   - Context structure matches Session 146 EPContextBuilder")
    print(f"   - Ready for EPIntegratedConsciousness")
    print(f"   - Saved to: {output_path}")
    print()
    print("Next: Test with Session 146 EPIntegratedConsciousness")
    print()


if __name__ == "__main__":
    generate_production_corpus()
