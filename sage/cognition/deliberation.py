#!/usr/bin/env python3
"""
Deliberation Engine for SAGE SNARC Cognition
=============================================

Plans ahead, evaluates alternatives, and makes reflective decisions.
Transforms reactive SNARC into deliberative cognitive system.

Track 3: SNARC Cognition - Component 3/4
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import time

# Import from other Track 3 components
try:
    from sage.cognition.working_memory import WorkingMemory, PlanStep
    from sage.memory.retrieval import MemoryRetrieval
except ModuleNotFoundError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from sage.cognition.working_memory import WorkingMemory, PlanStep
    from sage.memory.retrieval import MemoryRetrieval


@dataclass
class Alternative:
    """
    Candidate action being considered during deliberation

    Represents one possible choice with predicted outcomes
    """
    action_id: str
    action_description: str
    predicted_outcome: Dict[str, float]  # outcome_type -> probability
    expected_reward: float
    expected_cost: float
    confidence: float  # How certain is prediction? (0-1)
    risk: float  # Variance in outcome (0-1)

    def __post_init__(self):
        """Validate alternative"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
        if not 0.0 <= self.risk <= 1.0:
            raise ValueError(f"risk must be 0-1, got {self.risk}")

    def expected_utility(self) -> float:
        """Compute expected utility = reward - cost"""
        return self.expected_reward - self.expected_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'action_id': self.action_id,
            'action_description': self.action_description,
            'predicted_outcome': self.predicted_outcome,
            'expected_reward': self.expected_reward,
            'expected_cost': self.expected_cost,
            'expected_utility': self.expected_utility(),
            'confidence': self.confidence,
            'risk': self.risk
        }


@dataclass
class DeliberationResult:
    """
    Result of deliberation process

    Contains chosen action and reasoning
    """
    chosen_alternative: Alternative
    alternatives_considered: List[Alternative]
    reasoning: str
    confidence: float  # Overall confidence in decision (0-1)
    deliberation_time: float  # Time spent deliberating (seconds)

    meta_assessment: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result"""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'chosen_action': self.chosen_alternative.action_id,
            'action_description': self.chosen_alternative.action_description,
            'expected_utility': self.chosen_alternative.expected_utility(),
            'confidence': self.confidence,
            'deliberation_time': self.deliberation_time,
            'alternatives_count': len(self.alternatives_considered),
            'reasoning': self.reasoning,
            'meta_assessment': self.meta_assessment
        }


class DeliberationEngine:
    """
    Multi-step planning and alternative evaluation

    Core Responsibilities:
    - Evaluate multiple action alternatives
    - Predict outcomes using memory-based models
    - Generate multi-step plans
    - Assess confidence in decisions (meta-cognition)

    Integration:
    - Track 2 (Memory): Query for outcome prediction
    - Working Memory: Store and track plans
    - Goal Manager: Evaluate alternatives against goals
    """

    def __init__(
        self,
        memory_retrieval: Optional[MemoryRetrieval] = None,
        working_memory: Optional[WorkingMemory] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize Deliberation Engine

        Args:
            memory_retrieval: Memory system from Track 2
            working_memory: Working memory from Track 3
            device: Device for tensor operations
        """
        self.memory = memory_retrieval
        self.working_memory = working_memory
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Prediction models (learned from experience)
        self.outcome_predictors: Dict[str, Any] = {}

        # Deliberation history
        self.deliberation_history: List[DeliberationResult] = []

        # Default outcome values (can be customized per goal)
        self.default_outcome_values = {
            'success': 1.0,
            'failure': -0.5,
            'partial': 0.3,
            'blocked': -0.2
        }

        # Statistics
        self.total_deliberations: int = 0
        self.avg_deliberation_time: float = 0.0

    def deliberate(
        self,
        situation: Dict[str, Any],
        available_actions: List[str],
        goal: Any,
        time_budget: float = 0.1  # seconds
    ) -> DeliberationResult:
        """
        Deliberate over available actions and select best

        Process:
        1. Generate alternatives for each action
        2. Predict outcomes using memory
        3. Evaluate expected utility
        4. Select best alternative
        5. Assess confidence

        Args:
            situation: Current state (sensors, context)
            available_actions: List of possible actions
            goal: Current goal being pursued
            time_budget: Maximum time for deliberation (seconds)

        Returns:
            DeliberationResult with chosen action and reasoning
        """
        start_time = time.time()

        # Generate alternatives
        alternatives = []
        for action in available_actions:
            # Check time budget
            if time.time() - start_time > time_budget:
                break

            alternative = self._generate_alternative(action, situation, goal)
            alternatives.append(alternative)

        # If no alternatives generated, return default
        if not alternatives:
            # Emergency: no deliberation possible, return first action
            default_alt = Alternative(
                action_id=available_actions[0] if available_actions else "no_action",
                action_description="Default (no deliberation)",
                predicted_outcome={'unknown': 1.0},
                expected_reward=0.0,
                expected_cost=0.0,
                confidence=0.1,
                risk=0.9
            )

            elapsed = time.time() - start_time

            return DeliberationResult(
                chosen_alternative=default_alt,
                alternatives_considered=[default_alt],
                reasoning="No time for deliberation, default action",
                confidence=0.1,
                deliberation_time=elapsed
            )

        # Select best alternative
        best_alternative = max(alternatives, key=lambda alt: alt.expected_utility())

        # Generate reasoning
        reasoning = self._explain_choice(best_alternative, alternatives)

        # Compute overall confidence
        overall_confidence = self._compute_confidence(best_alternative, alternatives)

        # Create result
        elapsed = time.time() - start_time

        result = DeliberationResult(
            chosen_alternative=best_alternative,
            alternatives_considered=alternatives,
            reasoning=reasoning,
            confidence=overall_confidence,
            deliberation_time=elapsed
        )

        # Meta-cognition
        result.meta_assessment = self.meta_cognition(result)

        # Record history
        self.deliberation_history.append(result)
        self.total_deliberations += 1

        # Update average deliberation time
        self.avg_deliberation_time = (
            (self.avg_deliberation_time * (self.total_deliberations - 1) + elapsed)
            / self.total_deliberations
        )

        return result

    def _generate_alternative(
        self,
        action: str,
        situation: Dict[str, Any],
        goal: Any
    ) -> Alternative:
        """
        Generate alternative for an action

        Predicts outcome and evaluates utility
        """
        # Predict outcome
        predicted_outcome, confidence = self.predict_outcome(action, situation)

        # Evaluate expected reward
        expected_reward = self._evaluate_reward(predicted_outcome, goal)

        # Estimate cost
        expected_cost = self._estimate_cost(action, situation)

        # Estimate risk (variance in outcomes)
        risk = self._estimate_risk(predicted_outcome)

        return Alternative(
            action_id=action,
            action_description=action,
            predicted_outcome=predicted_outcome,
            expected_reward=expected_reward,
            expected_cost=expected_cost,
            confidence=confidence,
            risk=risk
        )

    def predict_outcome(
        self,
        action: str,
        situation: Dict[str, Any]
    ) -> Tuple[Dict[str, float], float]:
        """
        Predict outcome of action in current situation

        Strategy:
        1. Query memory for similar (situation, action) pairs
        2. Aggregate outcomes from past experiences
        3. Return probability distribution + confidence

        Args:
            action: Action to predict
            situation: Current situation

        Returns:
            (outcome_probabilities, confidence)
        """
        # TODO: Full integration would query Track 2 memory here
        # For now, use simple heuristic-based prediction

        # Check if learned predictor exists
        if action in self.outcome_predictors:
            predictor = self.outcome_predictors[action]
            # Would use learned model here
            pass

        # Heuristic prediction (placeholder for memory-based)
        # Real implementation would query similar situations from memory

        # Simple heuristic: Most actions succeed with some probability
        base_success_prob = 0.7

        # Adjust based on situation complexity
        complexity = situation.get('complexity', 0.5)
        adjusted_success = base_success_prob * (1.0 - 0.3 * complexity)

        outcome_probs = {
            'success': adjusted_success,
            'failure': 0.2,
            'partial': 0.1
        }

        # Normalize
        total = sum(outcome_probs.values())
        outcome_probs = {k: v / total for k, v in outcome_probs.items()}

        # Confidence based on "memory coverage" (placeholder)
        # Real: confidence = similarity to past experiences
        confidence = 0.6

        return outcome_probs, confidence

    def _evaluate_reward(
        self,
        predicted_outcome: Dict[str, float],
        goal: Any
    ) -> float:
        """
        Evaluate expected reward given predicted outcomes

        Expected reward = Σ P(outcome) * value(outcome, goal)
        """
        expected_reward = 0.0

        for outcome_type, probability in predicted_outcome.items():
            # Get value of this outcome for the goal
            value = self._outcome_value(outcome_type, goal)
            expected_reward += probability * value

        return expected_reward

    def _outcome_value(self, outcome_type: str, goal: Any) -> float:
        """
        Value of outcome for goal

        Higher value = better for achieving goal
        """
        # Check if goal specifies custom values
        if hasattr(goal, 'outcome_values') and outcome_type in goal.outcome_values:
            return goal.outcome_values[outcome_type]

        # Use default values
        return self.default_outcome_values.get(outcome_type, 0.0)

    def _estimate_cost(self, action: str, situation: Dict[str, Any]) -> float:
        """
        Estimate cost of action

        Cost includes time, energy, risk
        """
        # Simple heuristic cost model
        # Real implementation would learn from experience

        base_cost = 0.1  # All actions have some cost

        # Complex actions cost more
        if 'navigate' in action.lower() or 'manipulate' in action.lower():
            base_cost = 0.3

        return base_cost

    def _estimate_risk(self, predicted_outcome: Dict[str, float]) -> float:
        """
        Estimate risk (variance) in outcome

        Higher variance = higher risk
        """
        if not predicted_outcome:
            return 1.0

        # Simple: risk = entropy of outcome distribution
        entropy = -sum(
            p * np.log2(p + 1e-10) for p in predicted_outcome.values()
        )

        # Normalize to [0, 1]
        max_entropy = np.log2(len(predicted_outcome))
        risk = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(risk)

    def _compute_confidence(
        self,
        chosen: Alternative,
        alternatives: List[Alternative]
    ) -> float:
        """
        Compute overall confidence in deliberation

        Factors:
        - Confidence in outcome prediction
        - Margin between best and second-best
        - Number of alternatives considered
        """
        # Factor 1: Chosen alternative's prediction confidence
        prediction_conf = chosen.confidence

        # Factor 2: Utility margin (how much better is best vs second-best?)
        utilities = [alt.expected_utility() for alt in alternatives]
        utilities.sort(reverse=True)

        if len(utilities) >= 2:
            margin = utilities[0] - utilities[1]
            margin_conf = min(margin / 1.0, 1.0)  # Normalize
        else:
            margin_conf = 0.5

        # Factor 3: Coverage (did we consider enough alternatives?)
        coverage_conf = min(len(alternatives) / 3.0, 1.0)  # Target: 3+ alternatives

        # Weighted combination
        overall_confidence = (
            0.5 * prediction_conf +
            0.3 * margin_conf +
            0.2 * coverage_conf
        )

        return min(max(overall_confidence, 0.0), 1.0)

    def _explain_choice(
        self,
        chosen: Alternative,
        alternatives: List[Alternative]
    ) -> str:
        """Generate human-readable explanation of choice"""
        # Compare to other alternatives
        utilities = {alt.action_id: alt.expected_utility() for alt in alternatives}

        explanation = f"Chose '{chosen.action_id}' (utility={chosen.expected_utility():.3f})"

        if len(alternatives) > 1:
            # Show why it's better
            other_utils = [u for aid, u in utilities.items() if aid != chosen.action_id]
            if other_utils:
                max_other = max(other_utils)
                margin = chosen.expected_utility() - max_other
                explanation += f", {margin:.3f} better than next best"

        return explanation

    def generate_plan(
        self,
        goal: Any,
        situation: Dict[str, Any],
        max_steps: int = 5
    ) -> List[PlanStep]:
        """
        Generate multi-step plan to achieve goal

        Strategy: Forward search with heuristic pruning
        1. Start from current situation
        2. Generate possible next actions
        3. Predict outcomes
        4. Recurse from predicted states (limited depth)
        5. Return best path

        Args:
            goal: Goal to achieve
            situation: Current situation
            max_steps: Maximum plan length

        Returns:
            List of PlanSteps
        """
        # Simplified greedy planning (full search too expensive for real-time)
        plan = []

        current_situation = situation.copy()

        for step_id in range(max_steps):
            # Get available actions for current situation
            available_actions = self._get_available_actions(current_situation)

            if not available_actions:
                break

            # Deliberate over actions
            result = self.deliberate(
                current_situation,
                available_actions,
                goal,
                time_budget=0.05  # Quick deliberation for each step
            )

            # Add chosen action to plan
            plan_step = PlanStep(
                step_id=step_id,
                action=result.chosen_alternative.action_id,
                preconditions=[],
                expected_outcome=str(result.chosen_alternative.predicted_outcome),
                status="pending"
            )

            plan.append(plan_step)

            # Simulate outcome to get next situation
            # (In real system, would actually execute or use learned model)
            current_situation = self._simulate_outcome(
                current_situation,
                result.chosen_alternative
            )

            # Check if goal achieved
            if self._goal_achieved(goal, current_situation):
                break

        return plan

    def _get_available_actions(self, situation: Dict[str, Any]) -> List[str]:
        """Get actions available in current situation"""
        # Placeholder: return generic actions
        # Real implementation would determine valid actions from situation
        return ['move_forward', 'turn_left', 'turn_right', 'stop']

    def _simulate_outcome(
        self,
        situation: Dict[str, Any],
        alternative: Alternative
    ) -> Dict[str, Any]:
        """
        Simulate outcome of action

        Returns predicted next situation
        """
        # Placeholder: return modified situation
        # Real implementation would use learned transition model
        next_situation = situation.copy()

        # Simple simulation: actions change situation incrementally
        next_situation['step'] = situation.get('step', 0) + 1

        return next_situation

    def _goal_achieved(self, goal: Any, situation: Dict[str, Any]) -> bool:
        """Check if goal is achieved in situation"""
        # Placeholder
        # Real implementation would check goal success condition
        if hasattr(goal, 'success_condition'):
            return goal.success_condition(situation)

        return False

    def meta_cognition(
        self,
        deliberation_result: DeliberationResult
    ) -> Dict[str, Any]:
        """
        Reflect on deliberation quality (meta-cognition)

        Questions:
        - Am I confident in this decision?
        - Did I consider enough alternatives?
        - Is my outcome prediction reliable?
        - Should I deliberate more or act now?

        Returns:
            Meta-cognitive assessment
        """
        assessment = {}

        # Confidence assessment
        assessment['confident'] = deliberation_result.confidence > 0.7
        assessment['confidence_level'] = 'high' if deliberation_result.confidence > 0.7 else 'medium' if deliberation_result.confidence > 0.4 else 'low'

        # Coverage assessment (enough alternatives?)
        alt_count = len(deliberation_result.alternatives_considered)
        assessment['sufficient_alternatives'] = alt_count >= 3
        assessment['alternatives_count'] = alt_count

        # Prediction reliability (based on chosen alternative)
        chosen = deliberation_result.chosen_alternative
        assessment['prediction_reliable'] = chosen.confidence > 0.6
        assessment['prediction_confidence'] = chosen.confidence

        # Time assessment (was deliberation thorough?)
        assessment['deliberation_thorough'] = deliberation_result.deliberation_time > 0.05
        assessment['deliberation_time'] = deliberation_result.deliberation_time

        # Risk assessment
        assessment['high_risk'] = chosen.risk > 0.7
        assessment['risk_level'] = chosen.risk

        # Overall recommendation
        if assessment['confident'] and assessment['prediction_reliable']:
            assessment['recommendation'] = 'act_now'
        elif not assessment['sufficient_alternatives'] and deliberation_result.deliberation_time < 0.1:
            assessment['recommendation'] = 'deliberate_more'
        else:
            assessment['recommendation'] = 'act_with_caution'

        return assessment

    def get_stats(self) -> Dict[str, Any]:
        """Get deliberation statistics"""
        return {
            'total_deliberations': self.total_deliberations,
            'avg_deliberation_time': self.avg_deliberation_time,
            'history_size': len(self.deliberation_history)
        }


def test_deliberation_engine():
    """Test Deliberation Engine"""
    print("\n" + "="*60)
    print("TESTING DELIBERATION ENGINE")
    print("="*60)

    # Create deliberation engine
    deliberation = DeliberationEngine()

    # Mock goal
    class MockGoal:
        def __init__(self):
            self.goal_type = 'navigation'
            self.outcome_values = {'success': 1.0, 'failure': -0.5}

    goal = MockGoal()

    # Test 1: Simple deliberation
    print("\n1. Deliberating over 3 actions...")
    situation = {'position': (0, 0), 'complexity': 0.3}
    actions = ['move_forward', 'turn_left', 'turn_right']

    result = deliberation.deliberate(situation, actions, goal, time_budget=0.1)

    print(f"   Chosen: {result.chosen_alternative.action_id}")
    print(f"   Utility: {result.chosen_alternative.expected_utility():.3f}")
    print(f"   Confidence: {result.confidence:.3f}")
    print(f"   Time: {result.deliberation_time*1000:.1f}ms")
    print(f"   Reasoning: {result.reasoning}")

    # Test 2: Outcome prediction
    print("\n2. Predicting outcome for 'move_forward'...")
    outcome, conf = deliberation.predict_outcome('move_forward', situation)
    print(f"   Predicted outcomes: {outcome}")
    print(f"   Confidence: {conf:.3f}")

    # Test 3: Generate multi-step plan
    print("\n3. Generating 3-step plan...")
    plan = deliberation.generate_plan(goal, situation, max_steps=3)
    print(f"   Generated {len(plan)} steps:")
    for step in plan:
        print(f"     {step.step_id}: {step.action}")

    # Test 4: Meta-cognition
    print("\n4. Meta-cognitive assessment...")
    meta = result.meta_assessment
    print(f"   Confident: {meta['confident']}")
    print(f"   Confidence level: {meta['confidence_level']}")
    print(f"   Sufficient alternatives: {meta['sufficient_alternatives']}")
    print(f"   Recommendation: {meta['recommendation']}")

    # Test 5: Statistics
    print("\n5. Deliberation statistics...")
    stats = deliberation.get_stats()
    print(f"   Total deliberations: {stats['total_deliberations']}")
    print(f"   Avg time: {stats['avg_deliberation_time']*1000:.1f}ms")

    print("\n" + "="*60)
    print("✅ DELIBERATION ENGINE TESTS PASSED")
    print("="*60)

    return deliberation


if __name__ == "__main__":
    test_deliberation_engine()
