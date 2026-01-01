#!/usr/bin/env python3
"""
Session 143: SAGE Five-Domain EP Agent Simulation
==================================================

Demonstrates SAGE's complete five-domain EP framework applied to agent
interaction prediction, comparing to Web4's simplified EP approach.

Context:
- Session 142: Five-domain EP validated at 373K decisions/sec on Thor
- Web4 Session 111: Simplified EP demo with heuristic risk scoring
- Opportunity: Show full EP framework in practical application

Approach:
Instead of simplified risk heuristics, use SAGE's complete Multi-EP Coordinator
to predict agent interaction outcomes across all consciousness dimensions:

1. Emotional EP: Frustration/overwhelm risk from interaction
2. Quality EP: Relationship quality impact prediction
3. Attention EP: ATP resource sufficiency check
4. Grounding EP: Society coherence validation
5. Authorization EP: Permission verification for interaction type

This creates pattern corpus entries (Context → Outcome) for EP learning while
demonstrating production EP framework beyond benchmarks.

Hardware: Thor (Jetson AGX Thor Developer Kit)
Date: 2025-12-31
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import SAGE EP framework
sys.path.insert(0, str(Path(__file__).parent))
from multi_ep_coordinator import (
    MultiEPCoordinator,
    EPDomain,
    EPPrediction,
    ConflictResolution
)


# ============================================================================
# Agent Simulation Models (simplified from Web4 game)
# ============================================================================

@dataclass
class AgentState:
    """Simple agent state for simulation."""
    lct_id: str
    trust_t3: float  # 0.0-1.0 composite trust
    atp_available: float  # Attention Token Pool
    society_id: str  # Which society agent belongs to
    interaction_count: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0

    # EP-specific tracking
    frustration_level: float = 0.0  # 0.0-1.0
    relationship_quality: Dict[str, float] = None  # lct_id -> quality
    recent_outcomes: List[str] = None  # Track pattern learning

    def __post_init__(self):
        if self.relationship_quality is None:
            self.relationship_quality = {}
        if self.recent_outcomes is None:
            self.recent_outcomes = []


@dataclass
class InteractionProposal:
    """Proposed interaction between agents."""
    initiator_lct: str
    target_lct: str
    interaction_type: str  # "collaborate", "transfer", "delegate", "challenge"
    atp_cost: float
    expected_benefit: float
    context: str


# ============================================================================
# SAGE EP Context Builders
# ============================================================================

class SAGEAgentEPContextBuilder:
    """Builds EP contexts from agent states for five-domain prediction."""

    @staticmethod
    def build_emotional_context(agent: AgentState, interaction: InteractionProposal) -> Dict[str, Any]:
        """Build Emotional EP context from agent state."""
        return {
            "current_frustration": agent.frustration_level,
            "recent_failure_rate": (
                agent.failed_interactions / max(1, agent.interaction_count)
            ),
            "atp_stress": 1.0 - (agent.atp_available / 100.0),  # Normalize to 0-1
            "interaction_complexity": {
                "collaborate": 0.2,
                "transfer": 0.3,
                "delegate": 0.5,
                "challenge": 0.8
            }.get(interaction.interaction_type, 0.5)
        }

    @staticmethod
    def build_quality_context(agent: AgentState, interaction: InteractionProposal) -> Dict[str, Any]:
        """Build Quality EP context from relationship state."""
        current_quality = agent.relationship_quality.get(interaction.target_lct, 0.5)

        return {
            "current_relationship_quality": current_quality,
            "recent_avg_outcome": (
                sum(1 if o == "success" else 0 for o in agent.recent_outcomes[-5:]) /
                max(1, len(agent.recent_outcomes[-5:]))
            ),
            "trust_alignment": agent.trust_t3,  # Higher trust = higher quality potential
            "interaction_risk_to_quality": {
                "collaborate": 0.1,  # Low risk to quality
                "transfer": 0.2,
                "delegate": 0.4,
                "challenge": 0.7  # High risk to quality
            }.get(interaction.interaction_type, 0.3)
        }

    @staticmethod
    def build_attention_context(agent: AgentState, interaction: InteractionProposal) -> Dict[str, Any]:
        """Build Attention EP context from ATP state."""
        return {
            "atp_available": agent.atp_available,
            "atp_cost": interaction.atp_cost,
            "atp_reserve_needed": 20.0,  # Minimum ATP to keep in reserve
            "interaction_count": agent.interaction_count,
            "expected_benefit": interaction.expected_benefit
        }

    @staticmethod
    def build_grounding_context(
        agent: AgentState,
        target: AgentState,
        interaction: InteractionProposal
    ) -> Dict[str, Any]:
        """Build Grounding EP context from society coherence."""
        return {
            "same_society": agent.society_id == target.society_id,
            "initiator_trust": agent.trust_t3,
            "target_trust": target.trust_t3,
            "trust_gap": abs(agent.trust_t3 - target.trust_t3),
            "society_coherence_index": 1.0 if agent.society_id == target.society_id else 0.3
        }

    @staticmethod
    def build_authorization_context(
        agent: AgentState,
        target: AgentState,
        interaction: InteractionProposal
    ) -> Dict[str, Any]:
        """Build Authorization EP context from permission model."""
        # Simple permission model based on trust and relationship
        base_permission = agent.trust_t3 >= 0.5  # Need minimum trust

        # Interaction-specific permission requirements
        permission_thresholds = {
            "collaborate": 0.4,
            "transfer": 0.5,
            "delegate": 0.6,
            "challenge": 0.7
        }

        required_trust = permission_thresholds.get(interaction.interaction_type, 0.5)
        has_permission = agent.trust_t3 >= required_trust

        return {
            "has_base_permission": base_permission,
            "required_trust_level": required_trust,
            "actual_trust_level": agent.trust_t3,
            "permission_granted": has_permission,
            "past_abuse_count": agent.failed_interactions,
            "interaction_type": interaction.interaction_type
        }


# ============================================================================
# Five-Domain EP Prediction for Agent Interactions
# ============================================================================

class SAGEAgentEPPredictor:
    """Uses SAGE's five-domain EP to predict agent interaction outcomes."""

    def __init__(self):
        self.coordinator = MultiEPCoordinator()
        self.context_builder = SAGEAgentEPContextBuilder()
        self.prediction_count = 0

    def predict_interaction(
        self,
        initiator: AgentState,
        target: AgentState,
        interaction: InteractionProposal
    ) -> Dict[str, Any]:
        """
        Predict interaction outcome using five-domain EP framework.

        Returns comprehensive prediction with all EP domain inputs.
        """
        self.prediction_count += 1

        # Build contexts for each EP domain
        emotional_ctx = self.context_builder.build_emotional_context(initiator, interaction)
        quality_ctx = self.context_builder.build_quality_context(initiator, interaction)
        attention_ctx = self.context_builder.build_attention_context(initiator, interaction)
        grounding_ctx = self.context_builder.build_grounding_context(initiator, target, interaction)
        authorization_ctx = self.context_builder.build_authorization_context(initiator, target, interaction)

        # Generate predictions for each domain
        emotional_pred = self._predict_emotional(emotional_ctx)
        quality_pred = self._predict_quality(quality_ctx)
        attention_pred = self._predict_attention(attention_ctx)
        grounding_pred = self._predict_grounding(grounding_ctx)
        authorization_pred = self._predict_authorization(authorization_ctx)

        # Coordinate predictions using Multi-EP Coordinator
        decision = self.coordinator.coordinate(
            emotional_pred=emotional_pred,
            quality_pred=quality_pred,
            attention_pred=attention_pred,
            grounding_pred=grounding_pred,
            authorization_pred=authorization_pred
        )

        return {
            "prediction_id": self.prediction_count,
            "timestamp": datetime.now().isoformat(),
            "interaction": asdict(interaction),
            "contexts": {
                "emotional": emotional_ctx,
                "quality": quality_ctx,
                "attention": attention_ctx,
                "grounding": grounding_ctx,
                "authorization": authorization_ctx
            },
            "predictions": {
                "emotional": asdict(emotional_pred),
                "quality": asdict(quality_pred),
                "attention": asdict(attention_pred),
                "grounding": asdict(grounding_pred),
                "authorization": asdict(authorization_pred)
            },
            "coordinated_decision": {
                "final_decision": decision.final_decision,
                "confidence": decision.decision_confidence,
                "reasoning": decision.reasoning,
                "has_conflict": decision.has_conflict,
                "conflict_type": decision.conflict_type,
                "cascade_predicted": decision.cascade_predicted,
                "cascade_domains": [d.value for d in decision.cascade_domains] if decision.cascade_domains else []
            }
        }

    def _predict_emotional(self, ctx: Dict[str, Any]) -> EPPrediction:
        """Predict emotional impact of interaction."""
        frustration = ctx["current_frustration"]
        failure_rate = ctx["recent_failure_rate"]
        atp_stress = ctx["atp_stress"]
        complexity = ctx["interaction_complexity"]

        # Calculate cascade risk
        stress_factors = frustration + failure_rate + atp_stress + complexity
        cascade_risk = min(1.0, stress_factors / 4.0)

        # Severity as float (how bad if fails)
        severity_float = cascade_risk

        # Outcome probability (success chance considering emotional state)
        outcome_probability = 1.0 - cascade_risk

        if cascade_risk > 0.7:
            recommendation = "defer"
            reasoning = f"High cascade risk ({cascade_risk:.2f}): frustration={frustration:.2f}, failures={failure_rate:.2f}"
        elif cascade_risk > 0.4:
            recommendation = "adjust"
            reasoning = f"Moderate stress ({cascade_risk:.2f}): reduce complexity or wait"
        else:
            recommendation = "proceed"
            reasoning = f"Low emotional risk ({cascade_risk:.2f})"

        return EPPrediction(
            domain=EPDomain.EMOTIONAL,
            outcome_probability=outcome_probability,
            confidence=0.8,
            severity=severity_float,
            recommendation=recommendation,
            reasoning=reasoning,
            adjustment_strategy="reduce_complexity" if cascade_risk > 0.4 else None
        )

    def _predict_quality(self, ctx: Dict[str, Any]) -> EPPrediction:
        """Predict relationship quality impact."""
        current_quality = ctx["current_relationship_quality"]
        recent_avg = ctx["recent_avg_outcome"]
        trust = ctx["trust_alignment"]
        risk = ctx["interaction_risk_to_quality"]

        # Predict quality trend
        quality_momentum = (recent_avg - 0.5) * 2  # -1 to +1
        expected_quality = current_quality + (quality_momentum * 0.1) - risk

        # Severity is risk of quality degradation
        severity_float = max(0.0, min(1.0, risk))

        # Outcome probability based on expected quality improvement
        quality_delta = expected_quality - current_quality
        outcome_probability = max(0.0, min(1.0, 0.5 + quality_delta))

        if expected_quality < current_quality - 0.2:
            recommendation = "defer"
            reasoning = f"Quality would drop from {current_quality:.2f} to {expected_quality:.2f}"
        elif expected_quality < current_quality:
            recommendation = "adjust"
            reasoning = f"Quality may decline slightly (risk={risk:.2f})"
        else:
            recommendation = "proceed"
            reasoning = f"Quality likely improves (current={current_quality:.2f}, trust={trust:.2f})"

        return EPPrediction(
            domain=EPDomain.QUALITY,
            outcome_probability=outcome_probability,
            confidence=0.7,
            severity=severity_float,
            recommendation=recommendation,
            reasoning=reasoning,
            adjustment_strategy="lower_risk_interaction" if expected_quality < current_quality else None
        )

    def _predict_attention(self, ctx: Dict[str, Any]) -> EPPrediction:
        """Predict ATP resource impact."""
        available = ctx["atp_available"]
        cost = ctx["atp_cost"]
        reserve = ctx["atp_reserve_needed"]
        benefit = ctx["expected_benefit"]

        remaining = available - cost

        # Severity based on how close to reserve threshold
        if remaining < reserve:
            severity_float = 0.9
        elif remaining < reserve * 1.5:
            severity_float = 0.5
        else:
            severity_float = 0.1

        # Outcome probability based on ATP sufficiency
        atp_ratio = remaining / reserve if reserve > 0 else 1.0
        outcome_probability = min(1.0, max(0.0, atp_ratio / 2.0))  # Scale to 0-1

        if remaining < reserve:
            recommendation = "defer"
            reasoning = f"ATP would drop to {remaining:.1f} (below reserve {reserve:.1f})"
        elif remaining < reserve * 1.5:
            recommendation = "adjust"
            reasoning = f"ATP tight: {remaining:.1f} remaining after {cost:.1f} cost"
        else:
            recommendation = "proceed"
            reasoning = f"ATP sufficient: {remaining:.1f} remaining (benefit={benefit:.1f})"

        return EPPrediction(
            domain=EPDomain.ATTENTION,
            outcome_probability=outcome_probability,
            confidence=0.9,  # ATP is objective, high confidence
            severity=severity_float,
            recommendation=recommendation,
            reasoning=reasoning,
            adjustment_strategy="reduce_atp_cost" if remaining < reserve * 1.5 else None
        )

    def _predict_grounding(self, ctx: Dict[str, Any]) -> EPPrediction:
        """Predict society coherence and trust alignment."""
        same_society = ctx["same_society"]
        trust_gap = ctx["trust_gap"]
        coherence = ctx["society_coherence_index"]

        # Severity based on coherence risk
        severity_float = trust_gap if not same_society else trust_gap * 0.5

        # Outcome probability based on coherence index
        outcome_probability = coherence

        if not same_society and trust_gap > 0.4:
            recommendation = "defer"
            reasoning = f"Cross-society with large trust gap ({trust_gap:.2f})"
        elif not same_society or trust_gap > 0.3:
            recommendation = "adjust"
            reasoning = f"Coherence risk: same_society={same_society}, trust_gap={trust_gap:.2f}"
        else:
            recommendation = "proceed"
            reasoning = f"Good coherence: CI={coherence:.2f}, trust_gap={trust_gap:.2f}"

        return EPPrediction(
            domain=EPDomain.GROUNDING,
            outcome_probability=outcome_probability,
            confidence=0.85,
            severity=severity_float,
            recommendation=recommendation,
            reasoning=reasoning,
            adjustment_strategy="verify_society_alignment" if not same_society or trust_gap > 0.3 else None
        )

    def _predict_authorization(self, ctx: Dict[str, Any]) -> EPPrediction:
        """Predict permission and authorization validity."""
        has_permission = ctx["permission_granted"]
        required = ctx["required_trust_level"]
        actual = ctx["actual_trust_level"]
        abuse_count = ctx["past_abuse_count"]

        # Severity based on permission violation risk
        if not has_permission:
            severity_float = 0.95
        elif abuse_count > 2:
            severity_float = 0.6
        else:
            severity_float = 0.1

        # Outcome probability based on authorization status
        if has_permission and abuse_count < 3:
            outcome_probability = 0.9
        elif has_permission:
            outcome_probability = 0.6
        else:
            outcome_probability = 0.1

        if not has_permission or abuse_count > 5:
            recommendation = "defer"
            reasoning = f"Permission denied: required_trust={required:.2f}, actual={actual:.2f}, abuse={abuse_count}"
        elif abuse_count > 2:
            recommendation = "adjust"
            reasoning = f"Permission granted but abuse history ({abuse_count} incidents)"
        else:
            recommendation = "proceed"
            reasoning = f"Authorized: trust={actual:.2f} exceeds requirement={required:.2f}"

        return EPPrediction(
            domain=EPDomain.AUTHORIZATION,
            outcome_probability=outcome_probability,
            confidence=0.95,  # Authorization is rule-based, high confidence
            severity=severity_float,
            recommendation=recommendation,
            reasoning=reasoning,
            adjustment_strategy="request_explicit_permission" if abuse_count > 2 else None
        )


# ============================================================================
# Agent Simulation with Five-Domain EP
# ============================================================================

class SAGEEPAgentSimulation:
    """Agent simulation using SAGE's five-domain EP framework."""

    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.predictor = SAGEAgentEPPredictor()
        self.interaction_log: List[Dict[str, Any]] = []
        self.pattern_corpus: List[Dict[str, Any]] = []  # For EP learning

    def add_agent(self, agent: AgentState):
        """Add agent to simulation."""
        self.agents[agent.lct_id] = agent

    def simulate_interaction(self, interaction: InteractionProposal) -> Dict[str, Any]:
        """
        Simulate an interaction using five-domain EP prediction.

        Returns outcome with prediction vs reality comparison.
        """
        initiator = self.agents[interaction.initiator_lct]
        target = self.agents[interaction.target_lct]

        # Get EP prediction
        prediction = self.predictor.predict_interaction(initiator, target, interaction)

        decision = prediction["coordinated_decision"]["final_decision"]

        # Execute based on EP decision
        if decision == "defer":
            outcome = "deferred"
            success = False
            initiator.recent_outcomes.append("deferred")
        elif decision == "adjust":
            # Adjustment reduces risk
            outcome = "adjusted_and_proceeded"
            success = True  # Adjustment improves success rate
            self._apply_interaction_effects(initiator, target, interaction, adjusted=True)
            initiator.recent_outcomes.append("success")
        else:  # proceed
            outcome = "proceeded"
            success = self._execute_interaction(initiator, target, interaction)
            initiator.recent_outcomes.append("success" if success else "failure")

        # Update agent states
        initiator.interaction_count += 1
        if success:
            initiator.successful_interactions += 1
        else:
            initiator.failed_interactions += 1

        # Log interaction with prediction and outcome
        log_entry = {
            "interaction": asdict(interaction),
            "prediction": prediction,
            "outcome": outcome,
            "success": success,
            "agents_post_state": {
                "initiator": asdict(initiator),
                "target": asdict(target)
            }
        }
        self.interaction_log.append(log_entry)

        # Add to pattern corpus (Context → Outcome for learning)
        pattern_entry = {
            "context": prediction["contexts"],
            "prediction": prediction["coordinated_decision"],
            "outcome": {
                "success": success,
                "outcome_type": outcome
            },
            "prediction_accuracy": decision == ("defer" if not success else "proceed")
        }
        self.pattern_corpus.append(pattern_entry)

        return log_entry

    def _execute_interaction(self, initiator: AgentState, target: AgentState, interaction: InteractionProposal) -> bool:
        """Execute interaction and return success/failure."""
        # Simple success model based on trust and ATP
        base_success_rate = (initiator.trust_t3 + target.trust_t3) / 2

        # Interaction type affects difficulty
        difficulty = {
            "collaborate": 0.9,  # Easy
            "transfer": 0.7,
            "delegate": 0.5,
            "challenge": 0.3  # Hard
        }.get(interaction.interaction_type, 0.6)

        success_probability = base_success_rate * difficulty

        # ATP shortage reduces success
        if initiator.atp_available < interaction.atp_cost:
            success_probability *= 0.3

        # Simulate success
        import random
        success = random.random() < success_probability

        if success:
            self._apply_interaction_effects(initiator, target, interaction, adjusted=False)
        else:
            # Failure increases frustration
            initiator.frustration_level = min(1.0, initiator.frustration_level + 0.1)
            initiator.atp_available -= interaction.atp_cost * 0.5  # Still costs ATP

        return success

    def _apply_interaction_effects(
        self,
        initiator: AgentState,
        target: AgentState,
        interaction: InteractionProposal,
        adjusted: bool
    ):
        """Apply successful interaction effects to agent states."""
        # ATP cost
        initiator.atp_available -= interaction.atp_cost

        # Trust improvement (small)
        initiator.trust_t3 = min(1.0, initiator.trust_t3 + 0.05)
        target.trust_t3 = min(1.0, target.trust_t3 + 0.03)

        # Relationship quality improvement
        current_quality = initiator.relationship_quality.get(interaction.target_lct, 0.5)
        quality_gain = 0.1 if adjusted else 0.05  # Adjustments create better outcomes
        initiator.relationship_quality[interaction.target_lct] = min(1.0, current_quality + quality_gain)

        # Frustration reduction
        initiator.frustration_level = max(0.0, initiator.frustration_level - 0.05)

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for simulation run."""
        if not self.interaction_log:
            return {"error": "No interactions logged"}

        total = len(self.interaction_log)
        successes = sum(1 for entry in self.interaction_log if entry["success"])
        deferred = sum(1 for entry in self.interaction_log if entry["outcome"] == "deferred")
        adjusted = sum(1 for entry in self.interaction_log if entry["outcome"] == "adjusted_and_proceeded")
        proceeded = sum(1 for entry in self.interaction_log if entry["outcome"] == "proceeded")

        # Cascade detection stats
        cascades = sum(1 for entry in self.interaction_log
                      if entry["prediction"]["coordinated_decision"]["cascade_predicted"])

        # Decision distribution
        decisions = [entry["prediction"]["coordinated_decision"]["final_decision"]
                    for entry in self.interaction_log]

        return {
            "total_interactions": total,
            "successful": successes,
            "failed": total - successes,
            "success_rate": successes / total if total > 0 else 0,
            "ep_decisions": {
                "deferred": deferred,
                "adjusted": adjusted,
                "proceeded": proceeded,
                "defer_rate": deferred / total if total > 0 else 0,
                "adjust_rate": adjusted / total if total > 0 else 0,
                "proceed_rate": proceeded / total if total > 0 else 0
            },
            "cascade_detections": cascades,
            "pattern_corpus_size": len(self.pattern_corpus),
            "agent_final_states": {lct: asdict(agent) for lct, agent in self.agents.items()}
        }


# ============================================================================
# Demo: Compare Web4 Simplified EP vs SAGE Five-Domain EP
# ============================================================================

def run_sage_ep_demo():
    """Run demo simulation with SAGE's five-domain EP."""
    print("=" * 80)
    print("Session 143: SAGE Five-Domain EP Agent Simulation")
    print("=" * 80)
    print()
    print("Comparing Web4's simplified EP to SAGE's complete five-domain framework")
    print()

    # Create simulation
    sim = SAGEEPAgentSimulation()

    # Create agents (similar to Web4 demo)
    alice = AgentState(
        lct_id="alice",
        trust_t3=0.65,
        atp_available=95.0,
        society_id="home_society"
    )

    bob = AgentState(
        lct_id="bob",
        trust_t3=0.55,
        atp_available=75.0,
        society_id="home_society"
    )

    sim.add_agent(alice)
    sim.add_agent(bob)

    # Define interaction sequence
    interactions = [
        InteractionProposal("alice", "bob", "collaborate", 5.0, 8.0, "Building shared project"),
        InteractionProposal("bob", "alice", "collaborate", 10.0, 12.0, "Contributing to project"),
        InteractionProposal("alice", "bob", "transfer", 10.0, 5.0, "Sharing resources"),
        InteractionProposal("bob", "alice", "delegate", 10.0, 15.0, "Requesting help"),
        InteractionProposal("alice", "bob", "collaborate", 7.5, 10.0, "Continuing work"),
        InteractionProposal("bob", "alice", "collaborate", 10.0, 8.0, "Joint effort"),
        InteractionProposal("alice", "bob", "transfer", 5.0, 3.0, "Minor resource share"),
        InteractionProposal("bob", "alice", "transfer", 10.0, 6.0, "Reciprocal share"),
        InteractionProposal("alice", "bob", "challenge", 12.5, 20.0, "Disagreement scenario"),
        InteractionProposal("bob", "alice", "collaborate", 10.0, 8.0, "Reconciliation attempt")
    ]

    print("Running 10 interactions with five-domain EP predictions...\n")

    for i, interaction in enumerate(interactions, 1):
        print(f"Interaction {i}: {interaction.initiator_lct} → {interaction.target_lct} ({interaction.interaction_type})")
        result = sim.simulate_interaction(interaction)

        decision = result["prediction"]["coordinated_decision"]
        print(f"  EP Decision: {decision['final_decision']} (confidence: {decision['confidence']:.2f})")
        print(f"  Reasoning: {decision['reasoning']}")
        print(f"  Outcome: {result['outcome']} (success: {result['success']})")

        if decision["cascade_predicted"]:
            domains = ', '.join(decision["cascade_domains"])
            print(f"  ⚠️  CASCADE DETECTED in: {domains}")

        print()

    # Generate summary
    stats = sim.get_summary_statistics()

    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print()
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print()
    print("EP Decision Distribution:")
    print(f"  Proceeded: {stats['ep_decisions']['proceeded']} ({stats['ep_decisions']['proceed_rate']:.1%})")
    print(f"  Adjusted: {stats['ep_decisions']['adjusted']} ({stats['ep_decisions']['adjust_rate']:.1%})")
    print(f"  Deferred: {stats['ep_decisions']['deferred']} ({stats['ep_decisions']['defer_rate']:.1%})")
    print()
    print(f"Cascade Detections: {stats['cascade_detections']}")
    print(f"Pattern Corpus Entries: {stats['pattern_corpus_size']}")
    print()
    print("Final Agent States:")
    for lct, state in stats['agent_final_states'].items():
        print(f"  {lct}:")
        print(f"    Trust T3: {state['trust_t3']:.3f}")
        print(f"    ATP: {state['atp_available']:.1f}")
        print(f"    Frustration: {state['frustration_level']:.3f}")
        print(f"    Interactions: {state['successful_interactions']}/{state['interaction_count']} successful")

    # Save results
    results_path = Path(__file__).parent / "session143_ep_agent_simulation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "session": 143,
            "hardware": "Thor (Jetson AGX Thor Developer Kit)",
            "timestamp": datetime.now().isoformat(),
            "framework": "SAGE Five-Domain EP Coordinator",
            "summary": stats,
            "comparison_to_web4": {
                "web4_session": 111,
                "web4_framework": "Simplified heuristic EP",
                "sage_advantage": "Complete five-domain consciousness framework with cascade detection"
            }
        }, f, indent=2)

    print()
    print(f"Results saved to: {results_path.name}")
    print()
    print("=" * 80)
    print("Session 143 Complete")
    print("=" * 80)

    return sim, stats


if __name__ == "__main__":
    sim, stats = run_sage_ep_demo()
