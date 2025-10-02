#!/usr/bin/env python3
"""
Economic Reasoning Reward System
Society 4's Enhancement to Genesis SAGE Training

Extends Genesis's ReasoningReward with ATP/ADP economics.
Rewards efficiency alongside accuracy.
"""

import torch
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class EconomicRewardConfig:
    """Configuration for economic reward calculation"""
    # Base reward parameters (from Genesis)
    shortcut_penalty: float = -0.5
    reasoning_bonus: float = 1.0
    partial_credit: float = 0.3

    # Economic parameters (Society 4)
    efficiency_weight: float = 0.3      # Weight of efficiency in total reward
    atp_efficiency_threshold: float = 0.7  # Threshold for efficiency bonus
    max_efficiency_multiplier: float = 1.5  # Max reward multiplier for efficiency

    # Refund thresholds
    excellent_threshold: float = 0.8  # Reward score for excellent refund
    good_threshold: float = 0.5       # Reward score for good refund


class EconomicReasoningReward:
    """
    Reward function with ATP/ADP economics
    Society 4's contribution to SAGE training
    """

    def __init__(self, config: EconomicRewardConfig = None):
        self.config = config or EconomicRewardConfig()

    def calculate_reward(self,
                        prediction: torch.Tensor,
                        target: torch.Tensor,
                        reasoning_trace: Dict,
                        atp_spent: int) -> Tuple[float, int]:
        """
        Calculate reward AND ATP refund

        Args:
            prediction: Model prediction
            target: Ground truth
            reasoning_trace: Dict with h_ratio, salience, etc.
            atp_spent: ATP consumed for this inference

        Returns:
            (reward_score, atp_refund)
        """
        # Phase 1: Base Reasoning Reward (Genesis's original logic)
        base_reward = self._calculate_base_reward(
            prediction, target, reasoning_trace
        )

        # Phase 2: Economic Efficiency Assessment
        efficiency = self._calculate_efficiency(base_reward, atp_spent)

        # Phase 3: Combined Reward
        final_reward = self._combine_rewards(base_reward, efficiency)

        # Phase 4: ATP Refund Calculation
        atp_refund = self._calculate_refund(final_reward, atp_spent, efficiency)

        return final_reward, atp_refund

    def _calculate_base_reward(self,
                              prediction: torch.Tensor,
                              target: torch.Tensor,
                              reasoning_trace: Dict) -> float:
        """
        Base reasoning reward (Genesis's original logic)
        """
        # Check if answer is correct
        correct = torch.equal(prediction.argmax(dim=-1), target)

        # Base reward for correct answer
        reward = 1.0 if correct else 0.0

        # Penalize statistical shortcuts
        if self._is_statistical_shortcut(reasoning_trace):
            reward += self.config.shortcut_penalty

        # Reward actual reasoning steps
        reasoning_quality = self._evaluate_reasoning(reasoning_trace)
        reward += reasoning_quality * self.config.reasoning_bonus

        # Partial credit for good reasoning even if wrong
        if not correct and reasoning_quality > 0.5:
            reward += self.config.partial_credit

        return max(reward, 0.0)  # Ensure non-negative

    def _is_statistical_shortcut(self, trace: Dict) -> bool:
        """
        Detect if model used statistical shortcut
        (Genesis's original logic)
        """
        # Low strategic attention suggests shortcut
        h_ratio = trace.get('h_ratio', 0)
        if h_ratio < 0.2:
            return True

        # Check if salience is distributed (not focused)
        salience = trace.get('salience', None)
        if salience is not None:
            salience_std = salience.std().item() if torch.is_tensor(salience) else 0
            if salience_std < 0.1:  # Too uniform = not reasoning
                return True

        return False

    def _evaluate_reasoning(self, trace: Dict) -> float:
        """
        Evaluate quality of reasoning process
        (Genesis's original logic with Society 4 enhancements)
        """
        score = 0.0

        # Strategic attention usage (30%)
        h_ratio = trace.get('h_ratio', 0)
        score += min(h_ratio * 2, 1.0) * 0.3

        # Consciousness usage (20%)
        consciousness_size = trace.get('consciousness_size', 0)
        if consciousness_size > 0:
            score += 0.2

        # Salience distribution (30%)
        salience = trace.get('salience', None)
        if salience is not None:
            salience_std = salience.std().item() if torch.is_tensor(salience) else 0
            if 0.15 < salience_std < 0.35:  # Good selectivity
                score += 0.3

        # Iterative refinement (20%)
        if trace.get('iterations', 1) > 1:
            score += 0.2

        return min(score, 1.0)

    def _calculate_efficiency(self, reward: float, atp_spent: int) -> float:
        """
        Calculate economic efficiency: reward per ATP spent

        Args:
            reward: Base reasoning reward
            atp_spent: ATP consumed

        Returns:
            Efficiency score (0-1+)
        """
        if atp_spent == 0:
            return 0.0

        # Raw efficiency
        raw_efficiency = reward / atp_spent

        # Normalize to 0-1 range (assume optimal is ~0.1 reward/ATP)
        normalized_efficiency = min(raw_efficiency / 0.1, 1.0)

        return normalized_efficiency

    def _combine_rewards(self, base_reward: float, efficiency: float) -> float:
        """
        Combine base reward with efficiency

        Society 4's innovation: Reward models that solve tasks efficiently,
        not just correctly. Encourages strategic H-level use only when needed.
        """
        # Weighted combination
        combined = (
            base_reward * (1 - self.config.efficiency_weight) +
            efficiency * self.config.efficiency_weight
        )

        # Efficiency multiplier for high-efficiency solutions
        if efficiency > self.config.atp_efficiency_threshold:
            multiplier = 1.0 + (efficiency - self.config.atp_efficiency_threshold) * 0.5
            multiplier = min(multiplier, self.config.max_efficiency_multiplier)
            combined *= multiplier

        return combined

    def _calculate_refund(self,
                         reward: float,
                         atp_spent: int,
                         efficiency: float) -> int:
        """
        Calculate ATP refund based on performance

        Society 4's economic incentive:
        - Excellent reasoning: 50% refund
        - Good reasoning: 25% refund
        - High efficiency: +50% bonus

        This creates an economic pressure toward:
        1. Correct answers (high reward = high refund)
        2. Efficient solutions (efficiency bonus)
        3. Strategic H-level use (only when necessary)
        """
        refund = 0

        # Base refund on reasoning quality
        if reward >= self.config.excellent_threshold:
            refund = int(atp_spent * 0.5)  # 50% refund
        elif reward >= self.config.good_threshold:
            refund = int(atp_spent * 0.25)  # 25% refund

        # Efficiency bonus
        if efficiency > self.config.atp_efficiency_threshold:
            efficiency_bonus_ratio = 0.3 + (efficiency - self.config.atp_efficiency_threshold)
            efficiency_bonus = int(refund * min(efficiency_bonus_ratio, 0.5))
            refund += efficiency_bonus

        return refund

    def get_economic_metrics(self,
                            reward: float,
                            atp_spent: int,
                            atp_refunded: int) -> Dict:
        """
        Get economic analysis metrics for logging

        Returns:
            Dict with economic analysis
        """
        efficiency = self._calculate_efficiency(reward, atp_spent)
        net_cost = atp_spent - atp_refunded

        return {
            "reward": reward,
            "atp_spent": atp_spent,
            "atp_refunded": atp_refunded,
            "net_cost": net_cost,
            "efficiency": efficiency,
            "cost_per_reward": net_cost / max(reward, 0.01),
            "refund_ratio": atp_refunded / max(atp_spent, 1),
            "is_profitable": atp_refunded >= atp_spent
        }


class EconomicTrainingLogger:
    """
    Logs economic metrics during training
    Society 4's contribution to training observability
    """

    def __init__(self):
        self.episodes = []
        self.total_atp_spent = 0
        self.total_atp_refunded = 0
        self.total_reward = 0.0

    def log_episode(self, episode_num: int, metrics: Dict):
        """Log metrics for a training episode"""
        self.episodes.append({
            "episode": episode_num,
            **metrics
        })

        self.total_atp_spent += metrics["atp_spent"]
        self.total_atp_refunded += metrics["atp_refunded"]
        self.total_reward += metrics["reward"]

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        if not self.episodes:
            return {}

        return {
            "episodes": len(self.episodes),
            "total_atp_spent": self.total_atp_spent,
            "total_atp_refunded": self.total_atp_refunded,
            "net_atp_cost": self.total_atp_spent - self.total_atp_refunded,
            "total_reward": self.total_reward,
            "average_reward": self.total_reward / len(self.episodes),
            "average_efficiency": self.total_reward / max(self.total_atp_spent, 1),
            "average_refund_ratio": self.total_atp_refunded / max(self.total_atp_spent, 1),
            "profitable_episodes": sum(1 for e in self.episodes if e.get("is_profitable", False))
        }

    def save_log(self, filepath: str):
        """Save training log to file"""
        import json
        from pathlib import Path

        log_data = {
            "summary": self.get_summary(),
            "episodes": self.episodes
        }

        Path(filepath).write_text(json.dumps(log_data, indent=2))


# Example usage
if __name__ == "__main__":
    print("Economic Reasoning Reward System - Society 4")
    print("=" * 60)

    reward_system = EconomicReasoningReward()
    logger = EconomicTrainingLogger()

    # Simulate training episodes
    print("\nSimulating training episodes:")
    print("-" * 60)

    scenarios = [
        {
            "name": "Excellent + Efficient",
            "prediction": torch.tensor([[0.1, 0.1, 0.8]]),
            "target": torch.tensor([2]),
            "trace": {"h_ratio": 0.3, "consciousness_size": 100, "salience": torch.tensor([0.2, 0.5, 0.8])},
            "atp_spent": 8
        },
        {
            "name": "Excellent + Expensive",
            "prediction": torch.tensor([[0.1, 0.1, 0.8]]),
            "target": torch.tensor([2]),
            "trace": {"h_ratio": 0.9, "consciousness_size": 500, "salience": torch.tensor([0.2, 0.5, 0.8])},
            "atp_spent": 25
        },
        {
            "name": "Statistical Shortcut",
            "prediction": torch.tensor([[0.1, 0.1, 0.8]]),
            "target": torch.tensor([2]),
            "trace": {"h_ratio": 0.1, "consciousness_size": 0, "salience": torch.tensor([0.33, 0.33, 0.34])},
            "atp_spent": 2
        },
        {
            "name": "Wrong but Good Reasoning",
            "prediction": torch.tensor([[0.1, 0.8, 0.1]]),
            "target": torch.tensor([2]),
            "trace": {"h_ratio": 0.7, "consciousness_size": 300, "salience": torch.tensor([0.1, 0.6, 0.3])},
            "atp_spent": 15
        }
    ]

    for i, scenario in enumerate(scenarios):
        print(f"\nScenario {i+1}: {scenario['name']}")
        print(f"  ATP Spent: {scenario['atp_spent']}")

        reward, refund = reward_system.calculate_reward(
            scenario["prediction"],
            scenario["target"],
            scenario["trace"],
            scenario["atp_spent"]
        )

        metrics = reward_system.get_economic_metrics(reward, scenario["atp_spent"], refund)

        print(f"  Reward: {reward:.3f}")
        print(f"  ATP Refunded: {refund}")
        print(f"  Net Cost: {metrics['net_cost']}")
        print(f"  Efficiency: {metrics['efficiency']:.3f}")
        print(f"  Profitable: {metrics['is_profitable']}")

        logger.log_episode(i+1, metrics)

    # Show summary
    print("\n" + "=" * 60)
    print("Training Summary:")
    print("-" * 60)
    summary = logger.get_summary()
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("Key Insights:")
    print("-" * 60)
    print("  1. Efficient solutions get high rewards despite lower H-ratio")
    print("  2. Expensive H-level usage must justify cost with correctness")
    print("  3. Statistical shortcuts penalized even if answer is correct")
    print("  4. Good reasoning rewarded even for wrong answers")
    print("\nThis creates pressure toward economically sustainable reasoning.")
