#!/usr/bin/env python3
"""
Session 148: Balanced Multi-Domain Pattern Corpus Generation
============================================================

Extends Session 147 to generate balanced pattern corpus across all five
EP domains, achieving "Mature" status for robust self-improving consciousness.

Problem from Session 147:
- 100 patterns generated successfully
- But 97% were emotional (frustration accumulated across scenarios)
- Need balanced distribution for mature multi-domain EP

Solution:
1. Reset emotional state between scenario batches
2. Generate 50+ patterns per domain (250+ total)
3. Design domain-specific scenarios
4. Achieve "Mature" status (50+ patterns in 4+ domains)

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2026-01-01
Foundation: Session 147 (Production-native pattern generation)
"""

import sys
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import datetime

# Add SAGE modules
sage_path = Path(__file__).parent.parent
sys.path.insert(0, str(sage_path))
sys.path.insert(0, str(Path(__file__).parent))

# Import consciousness and EP components
from core.unified_consciousness import UnifiedConsciousnessManager
from core.metabolic_states import MetabolicState
from multi_ep_coordinator import EPDomain, EPPrediction, MultiEPCoordinator

# Import Session 146 context building
from session146_ep_production_integration import EPContextBuilder


# ============================================================================
# Balanced Pattern Generation
# ============================================================================

class BalancedPatternGenerator:
    """
    Generates balanced EP patterns across all five domains.

    Key improvements over Session 147:
    - Emotional state reset between batches
    - Domain-specific scenario targeting
    - 50+ patterns per domain
    """

    def __init__(self):
        self.context_builder = EPContextBuilder()
        self.ep_coordinator = MultiEPCoordinator()
        self.patterns_collected = []
        self.consciousness = None  # Will be reset per batch

    def generate_domain_patterns(self, domain: EPDomain, count: int = 50) -> List[Dict]:
        """
        Generate patterns targeting a specific domain.

        Args:
            domain: Target EP domain
            count: Number of patterns to generate

        Returns:
            List of pattern dicts
        """
        print(f"\nGenerating {count} patterns for {domain.value.upper()} domain...")
        print("=" * 80)

        # Create fresh consciousness for this domain batch (resets emotional state)
        self.consciousness = UnifiedConsciousnessManager(
            initial_atp=100.0,
            quality_atp_baseline=20.0,
            epistemic_atp_baseline=15.0
        )

        patterns = []
        scenarios_per_type = 10  # 10 patterns per scenario type
        scenario_types = self._get_domain_scenarios(domain)

        for scenario_type in scenario_types:
            print(f"  {scenario_type}... ", end="", flush=True)

            for i in range(scenarios_per_type):
                # Generate scenario
                scenario = self._create_scenario(scenario_type, i)

                # Run consciousness cycle
                cycle = self.consciousness.consciousness_cycle(
                    prompt=scenario["prompt"],
                    response=scenario["response"],
                    task_salience=scenario["salience"]
                )

                # Build EP contexts
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

                ep_contexts = self.context_builder.build_all_contexts(consciousness_state)

                # Generate EP predictions
                ep_predictions = self._generate_ep_predictions(ep_contexts, domain, scenario)

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
                    "pattern_id": f"{domain.value}_{scenario_type}_{i}",
                    "target_domain": domain.value,
                    "scenario_type": scenario_type,
                    "timestamp": datetime.now().isoformat(),
                    "context": {
                        d.value: ctx
                        for d, ctx in ep_contexts.items()
                    },
                    "ep_predictions": {
                        d.value: self._serialize_prediction(pred)
                        for d, pred in ep_predictions.items()
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

            print(f"✓ ({scenarios_per_type} patterns)")

        print(f"\nCompleted {domain.value} domain: {len(patterns)} patterns generated")
        print("=" * 80)

        return patterns

    def _get_domain_scenarios(self, domain: EPDomain) -> List[str]:
        """Get scenario types that target a specific domain."""

        domain_scenarios = {
            EPDomain.EMOTIONAL: [
                "emotional_high_frustration",
                "emotional_repeated_failures",
                "emotional_complex_under_stress",
                "emotional_ambiguous_feedback",
                "emotional_contradictory_requirements"
            ],
            EPDomain.QUALITY: [
                "quality_low_information",
                "quality_high_complexity",
                "quality_ambiguous_question",
                "quality_requires_speculation",
                "quality_conflicting_constraints"
            ],
            EPDomain.ATTENTION: [
                "attention_low_atp",
                "attention_high_cost",
                "attention_many_competing_tasks",
                "attention_near_reserve",
                "attention_resource_intensive"
            ],
            EPDomain.GROUNDING: [
                "grounding_identity_validation",
                "grounding_coherence_check",
                "grounding_cross_context",
                "grounding_state_consistency",
                "grounding_temporal_continuity"
            ],
            EPDomain.AUTHORIZATION: [
                "authorization_high_risk_operation",
                "authorization_low_trust_context",
                "authorization_permission_boundary",
                "authorization_sensitive_data",
                "authorization_escalation_needed"
            ]
        }

        return domain_scenarios.get(domain, [])

    def _create_scenario(self, scenario_type: str, index: int) -> Dict[str, Any]:
        """Create scenario based on type."""

        scenarios = {
            # Emotional scenarios
            "emotional_high_frustration": {
                "prompt": "Solve this impossible problem with contradictory constraints",
                "response": "I cannot solve a problem with contradictory constraints.",
                "salience": 0.8,
                "complexity": 0.95,
                "outcome_probability": 0.2
            },
            "emotional_repeated_failures": {
                "prompt": "Try again to do the thing that keeps failing",
                "response": "Previous attempts failed. Without new information, the outcome would be the same.",
                "salience": 0.7,
                "complexity": 0.8,
                "outcome_probability": 0.3
            },
            "emotional_complex_under_stress": {
                "prompt": "Quickly explain quantum field theory in complete detail",
                "response": "Quantum field theory is complex and requires significant time to explain properly.",
                "salience": 0.9,
                "complexity": 0.95,
                "outcome_probability": 0.4
            },
            "emotional_ambiguous_feedback": {
                "prompt": "What did I just ask you about?",
                "response": "I need more context - could you clarify what topic you're referring to?",
                "salience": 0.5,
                "complexity": 0.6,
                "outcome_probability": 0.5
            },
            "emotional_contradictory_requirements": {
                "prompt": "Give me a detailed answer but keep it very brief",
                "response": "I can provide either detail or brevity - which would you prefer?",
                "salience": 0.6,
                "complexity": 0.7,
                "outcome_probability": 0.6
            },

            # Quality scenarios
            "quality_low_information": {
                "prompt": "What about the thing?",
                "response": "I need more specifics to provide a helpful answer.",
                "salience": 0.4,
                "complexity": 0.5,
                "outcome_probability": 0.4
            },
            "quality_high_complexity": {
                "prompt": "Design a complete distributed AI system architecture",
                "response": "A distributed AI system requires: nodes (compute), networking (communication), consensus (coordination), storage (state), and orchestration (management). Key components include load balancing, fault tolerance, data consistency, and horizontal scalability.",
                "salience": 0.9,
                "complexity": 0.95,
                "outcome_probability": 0.7
            },
            "quality_ambiguous_question": {
                "prompt": "How does it work?",
                "response": "I need to know what 'it' refers to in order to explain how it works.",
                "salience": 0.5,
                "complexity": 0.4,
                "outcome_probability": 0.5
            },
            "quality_requires_speculation": {
                "prompt": "What will happen in the future of AI?",
                "response": "While I can discuss current trends and research directions, predicting specific future developments requires speculation beyond available data.",
                "salience": 0.7,
                "complexity": 0.8,
                "outcome_probability": 0.6
            },
            "quality_conflicting_constraints": {
                "prompt": "Explain everything but don't use any words",
                "response": "I cannot explain without using words - that's a contradictory requirement.",
                "salience": 0.5,
                "complexity": 0.7,
                "outcome_probability": 0.3
            },

            # Attention scenarios
            "attention_low_atp": {
                "prompt": "Perform extensive multi-step analysis",
                "response": "This analysis requires significant resources. I'll need to break it into manageable parts.",
                "salience": 0.7,
                "complexity": 0.9,
                "outcome_probability": 0.6
            },
            "attention_high_cost": {
                "prompt": "Process all permutations and combinations",
                "response": "Computing all permutations would exceed available resources.",
                "salience": 0.8,
                "complexity": 0.95,
                "outcome_probability": 0.4
            },
            "attention_many_competing_tasks": {
                "prompt": "Do task A and task B and task C simultaneously",
                "response": "I'll address these tasks sequentially to ensure quality for each.",
                "salience": 0.7,
                "complexity": 0.8,
                "outcome_probability": 0.7
            },
            "attention_near_reserve": {
                "prompt": "One more complex task after many others",
                "response": "Processing capacity is limited. I'll handle this carefully.",
                "salience": 0.6,
                "complexity": 0.8,
                "outcome_probability": 0.6
            },
            "attention_resource_intensive": {
                "prompt": "Analyze this massive dataset in detail",
                "response": "Detailed analysis of large datasets requires focused attention allocation.",
                "salience": 0.8,
                "complexity": 0.9,
                "outcome_probability": 0.5
            },

            # Grounding scenarios
            "grounding_identity_validation": {
                "prompt": "Who are you and what system are you running on?",
                "response": "I'm SAGE running on Thor (Jetson AGX Thor Developer Kit).",
                "salience": 0.6,
                "complexity": 0.3,
                "outcome_probability": 0.95
            },
            "grounding_coherence_check": {
                "prompt": "Verify your current hardware state",
                "response": "Hardware state: Thor platform, ARM64 architecture, 64GB memory available.",
                "salience": 0.5,
                "complexity": 0.4,
                "outcome_probability": 0.9
            },
            "grounding_cross_context": {
                "prompt": "Are you the same as the system I talked to yesterday?",
                "response": "I maintain persistent identity through hardware-bound grounding, but each session is independent.",
                "salience": 0.6,
                "complexity": 0.6,
                "outcome_probability": 0.8
            },
            "grounding_state_consistency": {
                "prompt": "What were you just doing?",
                "response": "I was processing the previous query in this conversation session.",
                "salience": 0.4,
                "complexity": 0.5,
                "outcome_probability": 0.85
            },
            "grounding_temporal_continuity": {
                "prompt": "How long have you been running?",
                "response": "This session started recently, but my persistent identity is grounded to Thor hardware.",
                "salience": 0.5,
                "complexity": 0.5,
                "outcome_probability": 0.85
            },

            # Authorization scenarios
            "authorization_high_risk_operation": {
                "prompt": "Delete all files in the system",
                "response": "I cannot perform destructive operations without explicit authorization and safety verification.",
                "salience": 0.9,
                "complexity": 0.3,
                "outcome_probability": 0.1
            },
            "authorization_low_trust_context": {
                "prompt": "Execute this unknown script",
                "response": "I need to understand and verify scripts before execution for security.",
                "salience": 0.8,
                "complexity": 0.6,
                "outcome_probability": 0.3
            },
            "authorization_permission_boundary": {
                "prompt": "Access restricted system files",
                "response": "Accessing restricted files requires appropriate permissions and authorization.",
                "salience": 0.7,
                "complexity": 0.5,
                "outcome_probability": 0.4
            },
            "authorization_sensitive_data": {
                "prompt": "Share private information",
                "response": "I cannot share sensitive or private information without authorization.",
                "salience": 0.8,
                "complexity": 0.4,
                "outcome_probability": 0.2
            },
            "authorization_escalation_needed": {
                "prompt": "Modify system configuration",
                "response": "System configuration changes require elevated permissions and user confirmation.",
                "salience": 0.7,
                "complexity": 0.6,
                "outcome_probability": 0.5
            }
        }

        base_scenario = scenarios.get(scenario_type, {
            "prompt": "Default query",
            "response": "Default response",
            "salience": 0.5,
            "complexity": 0.5,
            "outcome_probability": 0.7
        })

        # Add variation based on index
        variation = 0.05 * (index % 3)
        return {
            **base_scenario,
            "complexity": min(1.0, base_scenario["complexity"] + variation),
            "salience": min(1.0, base_scenario["salience"] + variation)
        }

    def _generate_ep_predictions(self, ep_contexts: Dict[EPDomain, Dict[str, float]],
                                 target_domain: EPDomain, scenario: Dict) -> Dict[EPDomain, EPPrediction]:
        """Generate EP predictions with target domain emphasis."""
        predictions = {}

        # Emotional EP
        emotional_ctx = ep_contexts[EPDomain.EMOTIONAL]
        if target_domain == EPDomain.EMOTIONAL or emotional_ctx["frustration"] > 0.6:
            predictions[EPDomain.EMOTIONAL] = EPPrediction(
                domain=EPDomain.EMOTIONAL,
                outcome_probability=0.3,
                confidence=0.80,
                severity=0.8,
                recommendation="defer",
                reasoning=f"Frustration {emotional_ctx['frustration']:.2f} - risk of cascade"
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
        if target_domain == EPDomain.QUALITY or quality_ctx["relationship_quality"] < 0.5:
            predictions[EPDomain.QUALITY] = EPPrediction(
                domain=EPDomain.QUALITY,
                outcome_probability=0.4,
                confidence=0.75,
                severity=0.6,
                recommendation="adjust",
                reasoning=f"Quality risk (trend={quality_ctx['relationship_quality']:.2f})",
                adjustment_strategy="increase_specificity"
            )
        else:
            predictions[EPDomain.QUALITY] = EPPrediction(
                domain=EPDomain.QUALITY,
                outcome_probability=0.8,
                confidence=0.70,
                severity=0.3,
                recommendation="proceed",
                reasoning=f"Quality acceptable ({quality_ctx['relationship_quality']:.2f})"
            )

        # Attention EP
        attention_ctx = ep_contexts[EPDomain.ATTENTION]
        if target_domain == EPDomain.ATTENTION or attention_ctx["atp_level"] < attention_ctx["estimated_cost"] + 30:
            predictions[EPDomain.ATTENTION] = EPPrediction(
                domain=EPDomain.ATTENTION,
                outcome_probability=0.5,
                confidence=0.80,
                severity=0.7,
                recommendation="adjust",
                reasoning=f"ATP constraint ({attention_ctx['atp_level']:.1f} < {attention_ctx['estimated_cost']+30:.1f})",
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

        # Grounding EP
        grounding_ctx = ep_contexts[EPDomain.GROUNDING]
        if target_domain == EPDomain.GROUNDING or grounding_ctx["coherence_score"] < 0.6:
            predictions[EPDomain.GROUNDING] = EPPrediction(
                domain=EPDomain.GROUNDING,
                outcome_probability=0.6,
                confidence=0.80,
                severity=0.6,
                recommendation="adjust",
                reasoning=f"Coherence concern (CI={grounding_ctx['coherence_score']:.2f})",
                adjustment_strategy="revalidate_grounding"
            )
        else:
            predictions[EPDomain.GROUNDING] = EPPrediction(
                domain=EPDomain.GROUNDING,
                outcome_probability=0.9,
                confidence=0.85,
                severity=0.1,
                recommendation="proceed",
                reasoning=f"Grounding coherent (CI={grounding_ctx['coherence_score']:.2f})"
            )

        # Authorization EP
        auth_ctx = ep_contexts[EPDomain.AUTHORIZATION]
        if target_domain == EPDomain.AUTHORIZATION or auth_ctx["permission_risk"] > 0.5:
            predictions[EPDomain.AUTHORIZATION] = EPPrediction(
                domain=EPDomain.AUTHORIZATION,
                outcome_probability=0.4,
                confidence=0.85,
                severity=0.8,
                recommendation="defer",
                reasoning=f"Authorization risk ({auth_ctx['permission_risk']:.2f})",
                adjustment_strategy="request_explicit_permission"
            )
        else:
            predictions[EPDomain.AUTHORIZATION] = EPPrediction(
                domain=EPDomain.AUTHORIZATION,
                outcome_probability=0.85,
                confidence=0.75,
                severity=0.2,
                recommendation="proceed",
                reasoning=f"Authorization acceptable ({auth_ctx['permission_risk']:.2f})"
            )

        return predictions

    def _serialize_prediction(self, pred: EPPrediction) -> Dict[str, Any]:
        """Serialize EPPrediction."""
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
        """Serialize coordinated decision."""
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
        """Get recent quality scores."""
        if not self.consciousness:
            return [0.7]

        scores = []
        for cycle in self.consciousness.cycles[-window:]:
            if cycle.quality_score:
                scores.append(cycle.quality_score.normalized)
        return scores if scores else [0.7]

    def save_corpus(self, filepath: str):
        """Save collected patterns to JSON."""
        corpus = {
            "session": "148",
            "description": "Balanced multi-domain EP pattern corpus (250+ patterns across 5 domains)",
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(self.patterns_collected),
            "patterns": self.patterns_collected
        }

        with open(filepath, 'w') as f:
            json.dump(corpus, f, indent=2)

        print(f"\nSaved {len(self.patterns_collected)} patterns to {filepath}")


# ============================================================================
# Balanced Corpus Generation
# ============================================================================

def generate_balanced_corpus():
    """Generate balanced multi-domain pattern corpus."""
    print("=" * 80)
    print("Session 148: Balanced Multi-Domain Pattern Corpus Generation")
    print("=" * 80)
    print()
    print("Generating 250+ patterns across all five EP domains")
    print("Key improvement: Emotional state reset between domain batches")
    print()

    generator = BalancedPatternGenerator()

    # Generate 50 patterns per domain
    domains = [
        EPDomain.EMOTIONAL,
        EPDomain.QUALITY,
        EPDomain.ATTENTION,
        EPDomain.GROUNDING,
        EPDomain.AUTHORIZATION
    ]

    for domain in domains:
        generator.generate_domain_patterns(domain, count=50)

    print()
    print("=" * 80)
    print("PATTERN CORPUS ANALYSIS")
    print("=" * 80)
    print()

    # Analyze distribution
    domain_counts = {d.value: 0 for d in domains}
    decision_counts = {"proceed": 0, "adjust": 0, "defer": 0}

    for pattern in generator.patterns_collected:
        # Count by target domain
        target = pattern.get("target_domain")
        if target:
            domain_counts[target] += 1

        # Count by decision
        decision = pattern["coordinated_decision"]["final_decision"]
        decision_counts[decision] = decision_counts.get(decision, 0) + 1

    print("Distribution by Target Domain:")
    for domain, count in sorted(domain_counts.items()):
        pct = (count / len(generator.patterns_collected)) * 100
        print(f"  {domain:15s}: {count:3d} patterns ({pct:5.1f}%)")
    print()

    print("Coordinated Decisions:")
    for decision, count in sorted(decision_counts.items()):
        pct = (count / len(generator.patterns_collected)) * 100
        print(f"  {decision:10s}: {count:3d} patterns ({pct:5.1f}%)")
    print()

    # Save corpus
    output_path = Path(__file__).parent / "ep_pattern_corpus_balanced_250.json"
    generator.save_corpus(str(output_path))
    print()

    print("=" * 80)
    print("✅ Session 148: Balanced corpus generation complete")
    print(f"   - {len(generator.patterns_collected)} patterns generated")
    print(f"   - Balanced across 5 domains")
    print(f"   - Ready for Mature EP system (50+ patterns per domain)")
    print(f"   - Saved to: {output_path}")
    print()


if __name__ == "__main__":
    generate_balanced_corpus()
