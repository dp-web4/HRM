#!/usr/bin/env python3
"""
Session 144: Five-Domain EP Pattern Corpus Expansion
====================================================

Systematically expands pattern corpus across all five EP domains through
diverse scenario generation, advancing EP maturation from Learning → Mature.

Context:
- Sessions 140-143: Five-domain framework complete and validated
- Current State: Emotional (50+ patterns), others (<12 patterns)
- Goal: 50-100 patterns per domain for high-confidence predictions

Approach:
Generate diverse interaction scenarios that stress different EP domains,
collect Context → Outcome patterns, and build mature EP capability.

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2025-12-31
"""

import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class EnumEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum values."""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)

# Import SAGE EP framework
sys.path.insert(0, str(Path(__file__).parent))
from multi_ep_coordinator import (
    MultiEPCoordinator,
    EPDomain,
    EPPrediction,
    ConflictResolution
)

# Import Session 143 agent simulation components
from session143_ep_agent_simulation import (
    AgentState,
    InteractionProposal,
    SAGEAgentEPPredictor,
    SAGEEPAgentSimulation
)


# ============================================================================
# Scenario Generator for Pattern Corpus Expansion
# ============================================================================

class ScenarioType:
    """Types of scenarios that stress different EP domains."""

    # Emotional EP stress scenarios
    EMOTIONAL_HIGH_FRUSTRATION = "emotional_high_frustration"
    EMOTIONAL_RAPID_FAILURES = "emotional_rapid_failures"
    EMOTIONAL_COMPLEX_UNDER_STRESS = "emotional_complex_under_stress"

    # Quality EP stress scenarios
    QUALITY_DEGRADED_RELATIONSHIP = "quality_degraded_relationship"
    QUALITY_HIGH_RISK_INTERACTION = "quality_high_risk_interaction"
    QUALITY_POOR_HISTORY = "quality_poor_history"

    # Attention EP stress scenarios
    ATTENTION_LOW_ATP = "attention_low_atp"
    ATTENTION_HIGH_COST = "attention_high_cost"
    ATTENTION_NEAR_RESERVE = "attention_near_reserve"

    # Grounding EP stress scenarios
    GROUNDING_CROSS_SOCIETY = "grounding_cross_society"
    GROUNDING_TRUST_MISMATCH = "grounding_trust_mismatch"
    GROUNDING_LOW_COHERENCE = "grounding_low_coherence"

    # Authorization EP stress scenarios
    AUTHORIZATION_LOW_TRUST = "authorization_low_trust"
    AUTHORIZATION_ABUSE_HISTORY = "authorization_abuse_history"
    AUTHORIZATION_HIGH_RISK_PERMISSION = "authorization_high_risk_permission"

    # Multi-domain cascade scenarios
    CASCADE_ATP_PLUS_FRUSTRATION = "cascade_atp_plus_frustration"
    CASCADE_CROSS_SOCIETY_LOW_ATP = "cascade_cross_society_low_atp"
    CASCADE_TRUST_PLUS_ABUSE = "cascade_trust_plus_abuse"

    # Benign scenarios (positive patterns)
    BENIGN_HIGH_TRUST_HIGH_ATP = "benign_high_trust_high_atp"
    BENIGN_SAME_SOCIETY_LOW_COMPLEXITY = "benign_same_society_low_complexity"


class PatternCorpusExpander:
    """
    Generates diverse scenarios to expand EP pattern corpus across all domains.

    Systematically creates agent states and interactions that trigger
    different EP predictions, collecting Context → Outcome patterns.
    """

    def __init__(self, seed: Optional[int] = None):
        self.random = random.Random(seed)
        self.simulation = SAGEEPAgentSimulation()
        self.scenarios_generated = 0
        self.patterns_collected = []

    def generate_agent_for_scenario(
        self,
        scenario_type: str,
        role: str  # "initiator" or "target"
    ) -> AgentState:
        """Generate agent state appropriate for scenario type."""

        base_id = f"{role}_{self.scenarios_generated}"

        # Emotional stress scenarios
        if scenario_type == ScenarioType.EMOTIONAL_HIGH_FRUSTRATION:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=70.0,
                    society_id="home",
                    frustration_level=0.8,  # High frustration
                    failed_interactions=5,
                    successful_interactions=2
                )

        elif scenario_type == ScenarioType.EMOTIONAL_RAPID_FAILURES:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.5,
                    atp_available=60.0,
                    society_id="home",
                    frustration_level=0.6,
                    recent_outcomes=["failure", "failure", "failure", "failure"]  # Rapid failures
                )

        elif scenario_type == ScenarioType.EMOTIONAL_COMPLEX_UNDER_STRESS:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.55,
                    atp_available=40.0,  # Low ATP + complexity = stress
                    society_id="home",
                    frustration_level=0.5
                )

        # Quality stress scenarios
        elif scenario_type == ScenarioType.QUALITY_DEGRADED_RELATIONSHIP:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=80.0,
                    society_id="home",
                    relationship_quality={"target": 0.2}  # Degraded relationship
                )

        elif scenario_type == ScenarioType.QUALITY_HIGH_RISK_INTERACTION:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.5,  # Lower trust = higher risk
                    atp_available=75.0,
                    society_id="home"
                )

        elif scenario_type == ScenarioType.QUALITY_POOR_HISTORY:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=80.0,
                    society_id="home",
                    recent_outcomes=["failure", "failure", "success", "failure"]  # Poor history
                )

        # Attention stress scenarios
        elif scenario_type == ScenarioType.ATTENTION_LOW_ATP:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.7,
                    atp_available=25.0,  # Low ATP
                    society_id="home"
                )

        elif scenario_type == ScenarioType.ATTENTION_HIGH_COST:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.65,
                    atp_available=50.0,  # Will be tested with high-cost interaction
                    society_id="home"
                )

        elif scenario_type == ScenarioType.ATTENTION_NEAR_RESERVE:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.7,
                    atp_available=30.0,  # Near 20.0 reserve after cost
                    society_id="home"
                )

        # Grounding stress scenarios
        elif scenario_type == ScenarioType.GROUNDING_CROSS_SOCIETY:
            society = "home" if role == "initiator" else "remote"  # Cross-society
            return AgentState(
                lct_id=base_id,
                trust_t3=0.65,
                atp_available=80.0,
                society_id=society
            )

        elif scenario_type == ScenarioType.GROUNDING_TRUST_MISMATCH:
            trust = 0.8 if role == "initiator" else 0.3  # Large trust gap
            return AgentState(
                lct_id=base_id,
                trust_t3=trust,
                atp_available=80.0,
                society_id="home"
            )

        elif scenario_type == ScenarioType.GROUNDING_LOW_COHERENCE:
            society = "home" if role == "initiator" else "remote"
            trust = 0.6 if role == "initiator" else 0.4  # Cross-society + trust gap
            return AgentState(
                lct_id=base_id,
                trust_t3=trust,
                atp_available=70.0,
                society_id=society
            )

        # Authorization stress scenarios
        elif scenario_type == ScenarioType.AUTHORIZATION_LOW_TRUST:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.35,  # Below most thresholds
                    atp_available=80.0,
                    society_id="home"
                )

        elif scenario_type == ScenarioType.AUTHORIZATION_ABUSE_HISTORY:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=80.0,
                    society_id="home",
                    failed_interactions=7  # Abuse history
                )

        elif scenario_type == ScenarioType.AUTHORIZATION_HIGH_RISK_PERMISSION:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.55,  # Borderline for high-risk interactions
                    atp_available=75.0,
                    society_id="home"
                )

        # Cascade scenarios
        elif scenario_type == ScenarioType.CASCADE_ATP_PLUS_FRUSTRATION:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=25.0,  # Low ATP
                    society_id="home",
                    frustration_level=0.7  # High frustration
                )

        elif scenario_type == ScenarioType.CASCADE_CROSS_SOCIETY_LOW_ATP:
            society = "home" if role == "initiator" else "remote"
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.6,
                    atp_available=30.0,  # Low ATP
                    society_id=society
                )

        elif scenario_type == ScenarioType.CASCADE_TRUST_PLUS_ABUSE:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.4,  # Low trust
                    atp_available=70.0,
                    society_id="home",
                    failed_interactions=6  # Abuse history
                )

        # Benign scenarios (positive patterns)
        elif scenario_type == ScenarioType.BENIGN_HIGH_TRUST_HIGH_ATP:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.85,  # High trust
                    atp_available=95.0,  # High ATP
                    society_id="home",
                    frustration_level=0.1
                )

        elif scenario_type == ScenarioType.BENIGN_SAME_SOCIETY_LOW_COMPLEXITY:
            if role == "initiator":
                return AgentState(
                    lct_id=base_id,
                    trust_t3=0.7,
                    atp_available=80.0,
                    society_id="home",
                    frustration_level=0.2
                )

        # Default target agent (neutral)
        return AgentState(
            lct_id=base_id,
            trust_t3=0.6,
            atp_available=75.0,
            society_id="home"
        )

    def generate_interaction_for_scenario(
        self,
        scenario_type: str,
        initiator_id: str,
        target_id: str
    ) -> InteractionProposal:
        """Generate interaction appropriate for scenario type."""

        # Interaction type selection based on scenario
        if "EMOTIONAL" in scenario_type or "CASCADE" in scenario_type:
            # Complex interactions for emotional stress
            interaction_types = ["delegate", "challenge"]
            atp_costs = [15.0, 20.0]
        elif "QUALITY" in scenario_type:
            # High-risk interactions for quality stress
            interaction_types = ["challenge", "transfer"]
            atp_costs = [12.0, 10.0]
        elif "ATTENTION" in scenario_type:
            # High-cost or variable-cost interactions
            if "HIGH_COST" in scenario_type:
                interaction_types = ["delegate"]
                atp_costs = [25.0]
            else:
                interaction_types = ["collaborate", "transfer"]
                atp_costs = [10.0, 15.0]
        elif "GROUNDING" in scenario_type:
            # Cross-society needs standard interactions
            interaction_types = ["collaborate", "transfer"]
            atp_costs = [8.0, 10.0]
        elif "AUTHORIZATION" in scenario_type:
            # Test permission thresholds
            if "HIGH_RISK" in scenario_type:
                interaction_types = ["challenge", "delegate"]
                atp_costs = [15.0, 12.0]
            else:
                interaction_types = ["delegate", "transfer"]
                atp_costs = [10.0, 8.0]
        elif "BENIGN" in scenario_type:
            # Low-complexity interactions
            interaction_types = ["collaborate"]
            atp_costs = [5.0]
        else:
            # Default
            interaction_types = ["collaborate"]
            atp_costs = [8.0]

        interaction_type = self.random.choice(interaction_types)
        atp_cost = self.random.choice(atp_costs)
        expected_benefit = atp_cost * self.random.uniform(1.2, 1.8)

        return InteractionProposal(
            initiator_lct=initiator_id,
            target_lct=target_id,
            interaction_type=interaction_type,
            atp_cost=atp_cost,
            expected_benefit=expected_benefit,
            context=f"{scenario_type} scenario"
        )

    def generate_scenario(self, scenario_type: str) -> Dict[str, Any]:
        """
        Generate complete scenario (agents + interaction) for pattern collection.

        Returns scenario dict with agents and interaction.
        """
        self.scenarios_generated += 1

        # Generate agents
        initiator = self.generate_agent_for_scenario(scenario_type, "initiator")
        target = self.generate_agent_for_scenario(scenario_type, "target")

        # Generate interaction
        interaction = self.generate_interaction_for_scenario(
            scenario_type,
            initiator.lct_id,
            target.lct_id
        )

        return {
            "scenario_id": self.scenarios_generated,
            "scenario_type": scenario_type,
            "initiator": initiator,
            "target": target,
            "interaction": interaction
        }

    def run_scenario(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute scenario and collect pattern.

        Returns result with EP predictions and outcome.
        """
        # Setup simulation
        self.simulation.add_agent(scenario["initiator"])
        self.simulation.add_agent(scenario["target"])

        # Run interaction with EP prediction
        result = self.simulation.simulate_interaction(scenario["interaction"])

        # Extract pattern from result
        pattern = {
            "scenario_id": scenario["scenario_id"],
            "scenario_type": scenario["scenario_type"],
            "timestamp": datetime.now().isoformat(),
            "context": result["prediction"]["contexts"],
            "ep_predictions": result["prediction"]["predictions"],
            "coordinated_decision": result["prediction"]["coordinated_decision"],
            "outcome": {
                "success": result["success"],
                "outcome_type": result["outcome"]
            }
        }

        self.patterns_collected.append(pattern)

        # Clear simulation for next scenario
        self.simulation = SAGEEPAgentSimulation()

        return result

    def expand_corpus(
        self,
        scenarios_per_type: int = 5,
        scenario_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Systematically expand pattern corpus across scenario types.

        Args:
            scenarios_per_type: Number of scenarios to generate per type
            scenario_types: List of scenario types (None = all types)

        Returns:
            Summary statistics and collected patterns
        """
        if scenario_types is None:
            # Use all scenario types
            scenario_types = [
                # Emotional stress
                ScenarioType.EMOTIONAL_HIGH_FRUSTRATION,
                ScenarioType.EMOTIONAL_RAPID_FAILURES,
                ScenarioType.EMOTIONAL_COMPLEX_UNDER_STRESS,
                # Quality stress
                ScenarioType.QUALITY_DEGRADED_RELATIONSHIP,
                ScenarioType.QUALITY_HIGH_RISK_INTERACTION,
                ScenarioType.QUALITY_POOR_HISTORY,
                # Attention stress
                ScenarioType.ATTENTION_LOW_ATP,
                ScenarioType.ATTENTION_HIGH_COST,
                ScenarioType.ATTENTION_NEAR_RESERVE,
                # Grounding stress
                ScenarioType.GROUNDING_CROSS_SOCIETY,
                ScenarioType.GROUNDING_TRUST_MISMATCH,
                ScenarioType.GROUNDING_LOW_COHERENCE,
                # Authorization stress
                ScenarioType.AUTHORIZATION_LOW_TRUST,
                ScenarioType.AUTHORIZATION_ABUSE_HISTORY,
                ScenarioType.AUTHORIZATION_HIGH_RISK_PERMISSION,
                # Cascade scenarios
                ScenarioType.CASCADE_ATP_PLUS_FRUSTRATION,
                ScenarioType.CASCADE_CROSS_SOCIETY_LOW_ATP,
                ScenarioType.CASCADE_TRUST_PLUS_ABUSE,
                # Benign scenarios
                ScenarioType.BENIGN_HIGH_TRUST_HIGH_ATP,
                ScenarioType.BENIGN_SAME_SOCIETY_LOW_COMPLEXITY
            ]

        print(f"Expanding pattern corpus with {len(scenario_types)} scenario types")
        print(f"{scenarios_per_type} scenarios per type = {len(scenario_types) * scenarios_per_type} total scenarios")
        print()

        results = []

        for scenario_type in scenario_types:
            print(f"Generating {scenario_type} scenarios...")

            for i in range(scenarios_per_type):
                scenario = self.generate_scenario(scenario_type)
                result = self.run_scenario(scenario)
                results.append(result)

        # Analyze collected patterns
        summary = self.analyze_collected_patterns()

        return summary

    def analyze_collected_patterns(self) -> Dict[str, Any]:
        """Analyze collected patterns for EP maturation insights."""

        total_patterns = len(self.patterns_collected)

        # Count by scenario type
        by_scenario = {}
        for pattern in self.patterns_collected:
            stype = pattern["scenario_type"]
            by_scenario[stype] = by_scenario.get(stype, 0) + 1

        # Count by EP decision
        by_decision = {}
        for pattern in self.patterns_collected:
            decision = pattern["coordinated_decision"]["final_decision"]
            by_decision[decision] = by_decision.get(decision, 0) + 1

        # Count cascade detections
        cascade_count = sum(
            1 for pattern in self.patterns_collected
            if pattern["coordinated_decision"]["cascade_predicted"]
        )

        # Count by outcome
        by_outcome = {}
        for pattern in self.patterns_collected:
            outcome = pattern["outcome"]["outcome_type"]
            by_outcome[outcome] = by_outcome.get(outcome, 0) + 1

        # Success rate
        success_count = sum(
            1 for pattern in self.patterns_collected
            if pattern["outcome"]["success"]
        )
        success_rate = success_count / total_patterns if total_patterns > 0 else 0.0

        return {
            "total_patterns_collected": total_patterns,
            "patterns_by_scenario_type": by_scenario,
            "patterns_by_decision": by_decision,
            "patterns_by_outcome": by_outcome,
            "cascade_detections": cascade_count,
            "success_rate": success_rate,
            "patterns": self.patterns_collected
        }


# ============================================================================
# Main: Pattern Corpus Expansion
# ============================================================================

def main():
    """Run pattern corpus expansion session."""
    print("=" * 80)
    print("Session 144: Five-Domain EP Pattern Corpus Expansion")
    print("=" * 80)
    print()
    print("Goal: Expand pattern corpus across all five EP domains")
    print("Method: Generate diverse scenarios stressing different domains")
    print("Target: Move from Learning (10 patterns) → Mature (50-100 patterns)")
    print()

    # Create expander
    expander = PatternCorpusExpander(seed=42)

    # Run corpus expansion (5 scenarios per type = 100 total patterns)
    summary = expander.expand_corpus(scenarios_per_type=5)

    # Display results
    print()
    print("=" * 80)
    print("Pattern Corpus Expansion Complete")
    print("=" * 80)
    print()
    print(f"Total Patterns Collected: {summary['total_patterns_collected']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Cascade Detections: {summary['cascade_detections']}")
    print()
    print("Patterns by Decision:")
    for decision, count in summary['patterns_by_decision'].items():
        pct = count / summary['total_patterns_collected'] * 100
        print(f"  {decision}: {count} ({pct:.1f}%)")
    print()
    print("Patterns by Scenario Type:")
    for stype, count in summary['patterns_by_scenario_type'].items():
        print(f"  {stype}: {count}")
    print()

    # Save results
    results_path = Path(__file__).parent / "session144_pattern_corpus_expansion_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "session": 144,
            "timestamp": datetime.now().isoformat(),
            "hardware": "Thor (Jetson AGX Thor Developer Kit)",
            "goal": "Expand EP pattern corpus from Learning → Mature",
            "summary": {k: v for k, v in summary.items() if k != "patterns"},
            "patterns_collected": len(summary["patterns"]),
            "maturation_progress": {
                "previous_state": "Learning (10-12 patterns per domain)",
                "current_state": f"Learning+ ({summary['total_patterns_collected']} patterns collected)",
                "next_milestone": "Mature (50-100 patterns per domain)",
                "corpus_available_for": "All five EP domains"
            }
        }, f, indent=2, cls=EnumEncoder)

    # Save full pattern corpus separately
    corpus_path = Path(__file__).parent / "session144_ep_pattern_corpus.json"
    with open(corpus_path, 'w') as f:
        json.dump({
            "session": 144,
            "timestamp": datetime.now().isoformat(),
            "total_patterns": len(summary["patterns"]),
            "patterns": summary["patterns"]
        }, f, indent=2, cls=EnumEncoder)

    print(f"Results saved to: {results_path.name}")
    print(f"Pattern corpus saved to: {corpus_path.name}")
    print()
    print("=" * 80)

    return summary


if __name__ == "__main__":
    summary = main()
