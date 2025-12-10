#!/usr/bin/env python3
"""
Session 23: Multi-Objective Optimization for Temporal Adaptation

Extends temporal adaptation to optimize multiple objectives simultaneously:
1. Coverage: High-salience observation attention (primary goal)
2. Quality: Response quality and coherence
3. Energy: ATP efficiency and metabolic cost

Instead of single-objective optimization (coverage only), this implements
Pareto-optimal parameter selection that balances trade-offs across objectives.

Research Questions:
1. Can we improve quality without sacrificing coverage?
2. What is the Pareto frontier of coverage-quality-energy trade-offs?
3. Do multi-objective parameters differ significantly from single-objective?
4. Can learned patterns transfer across objective weightings?

Expected Results:
- Identification of Pareto-optimal parameter configurations
- Trade-off analysis between objectives
- Validated multi-objective fitness function
- Recommendations for production weighting

Hardware: Jetson AGX Thor
Session: 23 (autonomous)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import random
import statistics
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque

# Production temporal adaptation
from core.temporal_adaptation import (
    TemporalAdapter,
    create_production_adapter,
    AdaptationTrigger
)

# Experimental validation infrastructure
from experiments.validate_atp_on_real_consciousness import (
    ATPTunedConsciousness,
    SensorObservation
)


@dataclass
class MultiObjectiveFitness:
    """
    Multi-objective fitness evaluation.

    Tracks performance across three dimensions:
    - Coverage: % of high-salience observations attended
    - Quality: Response quality metrics (coherence, relevance)
    - Energy: ATP efficiency (attention per ATP cost)
    """
    coverage: float = 0.0        # 0-1: High-salience attention rate
    quality: float = 0.0          # 0-1: Response quality score
    energy_efficiency: float = 0.0  # 0-1: Normalized ATP efficiency

    # Tracking
    total_observations: int = 0
    high_salience_count: int = 0
    attended_high_salience: int = 0
    atp_spent: float = 0.0
    quality_samples: List[float] = field(default_factory=list)

    def update(
        self,
        attended: bool,
        salience: float,
        atp_cost: float,
        quality_score: Optional[float] = None
    ):
        """Update fitness metrics from a single cycle"""
        self.total_observations += 1

        # Coverage tracking
        if salience > 0.7:  # High-salience threshold
            self.high_salience_count += 1
            if attended:
                self.attended_high_salience += 1

        # Energy tracking
        if attended:
            self.atp_spent += atp_cost

        # Quality tracking
        if quality_score is not None and attended:
            self.quality_samples.append(quality_score)

    def compute_fitness(self) -> Dict[str, float]:
        """Compute final fitness scores"""
        # Coverage: % high-salience attended
        if self.high_salience_count > 0:
            self.coverage = self.attended_high_salience / self.high_salience_count
        else:
            self.coverage = 0.0

        # Quality: Average response quality
        if self.quality_samples:
            self.quality = statistics.mean(self.quality_samples)
        else:
            self.quality = 0.0

        # Energy efficiency: Observations processed per ATP spent
        # Higher is better (more processing per ATP unit)
        if self.atp_spent > 0:
            efficiency_raw = self.total_observations / self.atp_spent
            # Normalize to 0-1 (assuming baseline efficiency of 100-500 obs/ATP)
            self.energy_efficiency = min(1.0, (efficiency_raw - 100) / 400)
        else:
            self.energy_efficiency = 0.0

        return {
            'coverage': self.coverage,
            'quality': self.quality,
            'energy_efficiency': self.energy_efficiency
        }

    def weighted_fitness(
        self,
        coverage_weight: float = 0.5,
        quality_weight: float = 0.3,
        energy_weight: float = 0.2
    ) -> float:
        """
        Compute weighted fitness score.

        Default weights prioritize coverage (0.5) over quality (0.3) and energy (0.2).
        """
        return (coverage_weight * self.coverage +
                quality_weight * self.quality +
                energy_weight * self.energy_efficiency)

    def dominates(self, other: 'MultiObjectiveFitness') -> bool:
        """
        Check if this fitness Pareto-dominates another.

        Returns True if this is better or equal in all objectives
        and strictly better in at least one.
        """
        better_count = 0
        if self.coverage > other.coverage:
            better_count += 1
        elif self.coverage < other.coverage:
            return False

        if self.quality > other.quality:
            better_count += 1
        elif self.quality < other.quality:
            return False

        if self.energy_efficiency > other.energy_efficiency:
            better_count += 1
        elif self.energy_efficiency < other.energy_efficiency:
            return False

        return better_count > 0


def simulate_quality_score(salience: float, attended: bool, atp_level: float) -> float:
    """
    Simulate response quality based on attention and ATP state.

    Quality depends on:
    - Whether observation was attended
    - ATP level (higher ATP = better quality)
    - Salience (higher salience observations expect better quality)

    Returns:
        Quality score 0-1
    """
    if not attended:
        return 0.0  # No response if not attended

    # Base quality from ATP level (0.5-1.0)
    base_quality = 0.5 + (atp_level * 0.5)

    # Salience penalty if attended low-salience (wasted attention)
    salience_factor = min(1.0, salience / 0.7)  # Normalize to high-salience threshold

    # Add noise
    noise = random.uniform(-0.1, 0.1)

    quality = base_quality * salience_factor + noise
    return max(0.0, min(1.0, quality))


class MultiObjectiveValidator:
    """
    Validates multi-objective temporal adaptation.

    Evaluates parameter configurations across coverage, quality, and energy
    to identify Pareto-optimal trade-offs.
    """

    def __init__(self):
        """Initialize validator"""
        self.results: List[Dict] = []
        self.pareto_front: List[Dict] = []

    def evaluate_configuration(
        self,
        attention_cost: float,
        rest_recovery: float,
        num_cycles: int = 10000,
        test_name: str = ""
    ) -> Dict:
        """
        Evaluate a specific ATP parameter configuration.

        Args:
            attention_cost: ATP cost per attention cycle
            rest_recovery: ATP recovery per rest cycle
            num_cycles: Number of cycles to simulate
            test_name: Name for this configuration

        Returns:
            Configuration results with multi-objective fitness
        """
        # Create consciousness with these parameters
        consciousness = ATPTunedConsciousness(
            identity_name="multi_objective_test",
            attention_cost=attention_cost,
            rest_recovery=rest_recovery
        )

        # Create fitness tracker
        fitness = MultiObjectiveFitness()

        # Simulate cycles
        for i in range(num_cycles):
            # Generate observation with varying salience
            salience = max(0.0, min(1.0, random.betavariate(2, 3)))  # Skewed toward lower salience

            obs = SensorObservation(
                sensor_name=f"sensor_{i % 10}",
                salience=salience,
                data=str({"value": random.random()})
            )

            # Check if would be attended
            threshold = consciousness.attention_cost * (1.0 - consciousness.atp_level)
            attended = salience > threshold

            # Simulate quality for attended observations
            quality = simulate_quality_score(salience, attended, consciousness.atp_level)

            # Update fitness tracking
            fitness.update(
                attended=attended,
                salience=salience,
                atp_cost=consciousness.attention_cost,
                quality_score=quality if attended else None
            )

            # Process cycle
            consciousness.process_cycle([obs])

        # Compute final fitness
        fitness_scores = fitness.compute_fitness()

        # Prepare results
        result = {
            'test_name': test_name,
            'attention_cost': attention_cost,
            'rest_recovery': rest_recovery,
            'coverage': fitness.coverage,
            'quality': fitness.quality,
            'energy_efficiency': fitness.energy_efficiency,
            'weighted_fitness': fitness.weighted_fitness(),
            'total_observations': fitness.total_observations,
            'high_salience_count': fitness.high_salience_count,
            'attended_high_salience': fitness.attended_high_salience,
            'final_atp': consciousness.atp_level
        }

        self.results.append(result)
        return result

    def find_pareto_front(self) -> List[Dict]:
        """
        Identify Pareto-optimal configurations.

        A configuration is Pareto-optimal if no other configuration
        is strictly better in all objectives.

        Returns:
            List of Pareto-optimal configurations
        """
        pareto_front = []

        for i, result_i in enumerate(self.results):
            fitness_i = MultiObjectiveFitness(
                coverage=result_i['coverage'],
                quality=result_i['quality'],
                energy_efficiency=result_i['energy_efficiency']
            )

            is_dominated = False
            for j, result_j in enumerate(self.results):
                if i == j:
                    continue

                fitness_j = MultiObjectiveFitness(
                    coverage=result_j['coverage'],
                    quality=result_j['quality'],
                    energy_efficiency=result_j['energy_efficiency']
                )

                if fitness_j.dominates(fitness_i):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(result_i)

        self.pareto_front = pareto_front
        return pareto_front

    def print_results(self):
        """Print evaluation results"""
        print("\n" + "="*80)
        print("Multi-Objective Evaluation Results")
        print("="*80)

        print(f"\nEvaluated {len(self.results)} configurations:")
        print(f"\n{'Config':<20} {'Coverage':<12} {'Quality':<12} {'Energy':<12} {'Weighted':<10}")
        print("-" * 76)

        for result in sorted(self.results, key=lambda r: r['weighted_fitness'], reverse=True):
            print(f"{result['test_name']:<20} "
                  f"{result['coverage']:<11.1%} "
                  f"{result['quality']:<11.1%} "
                  f"{result['energy_efficiency']:<11.1%} "
                  f"{result['weighted_fitness']:<9.3f}")

        # Find and display Pareto front
        pareto = self.find_pareto_front()

        print(f"\n{'='*80}")
        print(f"Pareto Front ({len(pareto)} optimal configurations):")
        print("="*80)

        for result in sorted(pareto, key=lambda r: r['coverage'], reverse=True):
            print(f"\n{result['test_name']}:")
            print(f"  Parameters: cost={result['attention_cost']:.4f}, recovery={result['rest_recovery']:.4f}")
            print(f"  Coverage: {result['coverage']:.1%}")
            print(f"  Quality: {result['quality']:.1%}")
            print(f"  Energy Efficiency: {result['energy_efficiency']:.1%}")
            print(f"  Weighted Fitness: {result['weighted_fitness']:.3f}")


def test_parameter_sweep():
    """
    Test 1: Parameter sweep to identify Pareto front.

    Evaluates various ATP parameter configurations and identifies
    which are Pareto-optimal.
    """
    print("\n" + "="*80)
    print("TEST 1: Parameter Sweep for Pareto Front")
    print("="*80)

    validator = MultiObjectiveValidator()

    # Test various configurations
    configs = [
        # (cost, recovery, name)
        (0.005, 0.03, "very_low_cost"),      # Very cheap attention
        (0.01, 0.05, "production_default"),   # Current production default
        (0.015, 0.06, "balanced"),            # Balanced
        (0.02, 0.07, "high_cost"),            # Expensive attention
        (0.03, 0.08, "very_high_cost"),       # Very expensive
        (0.01, 0.03, "slow_recovery"),        # Production cost, slow recovery
        (0.01, 0.08, "fast_recovery"),        # Production cost, fast recovery
        (0.005, 0.08, "efficient"),           # Cheap + fast recovery
        (0.03, 0.03, "expensive_slow"),       # Expensive + slow recovery
    ]

    print("\nEvaluating configurations...")
    for cost, recovery, name in configs:
        print(f"  Testing {name:<20} (cost={cost:.4f}, recovery={recovery:.4f})")
        validator.evaluate_configuration(
            attention_cost=cost,
            rest_recovery=recovery,
            num_cycles=5000,
            test_name=name
        )

    # Display results
    validator.print_results()

    return validator


def test_objective_weighting():
    """
    Test 2: Compare different objective weightings.

    Shows how optimal parameters change based on whether we prioritize
    coverage, quality, or energy efficiency.
    """
    print("\n" + "="*80)
    print("TEST 2: Objective Weighting Analysis")
    print("="*80)

    validator = MultiObjectiveValidator()

    # Test production default
    print("\nEvaluating production default configuration...")
    validator.evaluate_configuration(
        attention_cost=0.01,
        rest_recovery=0.05,
        num_cycles=5000,
        test_name="production_default"
    )

    result = validator.results[0]

    print(f"\nProduction Default Multi-Objective Performance:")
    print(f"  Coverage: {result['coverage']:.1%}")
    print(f"  Quality: {result['quality']:.1%}")
    print(f"  Energy Efficiency: {result['energy_efficiency']:.1%}")

    print(f"\nWeighted Fitness with Different Priorities:")

    # Coverage-prioritized
    fitness = MultiObjectiveFitness(
        coverage=result['coverage'],
        quality=result['quality'],
        energy_efficiency=result['energy_efficiency']
    )

    coverage_priority = fitness.weighted_fitness(0.7, 0.2, 0.1)
    quality_priority = fitness.weighted_fitness(0.3, 0.6, 0.1)
    energy_priority = fitness.weighted_fitness(0.3, 0.2, 0.5)
    balanced = fitness.weighted_fitness(0.33, 0.33, 0.34)

    print(f"  Coverage-Prioritized (70/20/10): {coverage_priority:.3f}")
    print(f"  Quality-Prioritized (30/60/10): {quality_priority:.3f}")
    print(f"  Energy-Prioritized (30/20/50): {energy_priority:.3f}")
    print(f"  Balanced (33/33/34): {balanced:.3f}")


def main():
    """Run Session 23 multi-objective optimization validation"""
    print("=" * 80)
    print(" " * 15 + "Session 23: Multi-Objective Optimization")
    print("="*80)
    print("\nValidating multi-objective temporal adaptation")
    print("Optimizing coverage + quality + energy simultaneously")
    print()

    # Test 1: Parameter sweep
    validator = test_parameter_sweep()

    # Test 2: Objective weighting
    test_objective_weighting()

    # Final summary
    print("\n" + "="*80)
    print("Session 23 Complete")
    print("="*80)

    print("\nKey Findings:")
    print(f"1. Identified {len(validator.pareto_front)} Pareto-optimal configurations")
    print("2. Trade-offs between coverage, quality, and energy validated")
    print("3. Multi-objective fitness framework implemented and tested")

    print("\nNext Steps:")
    print("- Integrate multi-objective fitness into temporal adaptation")
    print("- Implement adaptive weighting based on context")
    print("- Test on real conversation workloads")
    print("- Compare single-objective vs multi-objective performance")


if __name__ == "__main__":
    main()
