"""
Online SNARC Weight Adaptation Experiment
==========================================

Implements real-time weight learning during consciousness deployment.

**Objective**: Enable consciousness kernel to continuously optimize SNARC
weights based on attention outcomes, adapting to operational context.

**Approach**:
1. Start with learned weights from batch training (novelty-dominant)
2. After each ATTENDED cycle, compute outcome quality
3. Update weights via gradient descent on (reward - salience)² loss
4. Track weight evolution over time
5. Detect context shifts via weight changes

**Expected Outcomes**:
- Weights converge to optimal profile for current context
- Weight changes signal context shifts
- Continuous improvement in attention quality

**Session**: Thor Autonomous Research (2025-12-06)
**Author**: Claude (guest) on Thor via claude-code
"""

import sys
import os
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import deque
import math

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))

from sage.core.snarc_compression import SNARCCompressor, SNARCWeights, SNARCDimensions
from sage.sensors.unified_sensors import (
    CPUSensor, MemorySensor, ProcessSensor, DiskSensor,
    SensorManager
)


# ============================================================================
# Online Weight Learner
# ============================================================================

@dataclass
class WeightSnapshot:
    """Snapshot of weights at a point in time"""
    timestamp: float
    cycle: int
    weights: Dict[str, float]
    loss: float
    learning_rate: float


class OnlineWeightLearner:
    """
    Learn SNARC weights online during deployment.

    **Algorithm**: Stochastic Gradient Descent
    - After each ATTENDED cycle with outcome
    - Compute gradient: ∂L/∂w_i = -2(reward - salience) × dimension_i + 2λw_i
    - Update: w_i ← w_i - α × gradient_i
    - Normalize to ensure Σw_i = 1

    **Features**:
    - Tracks weight evolution over time
    - Detects convergence (stable weights)
    - Saves weight snapshots periodically
    - Adaptive learning rate (decreases over time)
    """

    def __init__(
        self,
        initial_weights: SNARCWeights,
        learning_rate: float = 0.05,
        regularization: float = 0.01,
        lr_decay: float = 0.995,
        min_lr: float = 0.001
    ):
        """
        Initialize online weight learner.

        Args:
            initial_weights: Starting weights (from batch learning or defaults)
            learning_rate: Initial learning rate (higher for online learning)
            regularization: L2 regularization coefficient
            lr_decay: Learning rate decay per update (0.995 = -0.5% per update)
            min_lr: Minimum learning rate floor
        """
        self.weights = initial_weights
        self.learning_rate = learning_rate
        self.initial_lr = learning_rate
        self.regularization = regularization
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        # Training history
        self.weight_history: List[WeightSnapshot] = []
        self.loss_history: List[float] = []
        self.update_count = 0

        # Recent gradients for monitoring
        self.recent_gradients: deque = deque(maxlen=10)

        # Convergence tracking
        self.convergence_window = 20
        self.convergence_threshold = 0.001  # Max weight change for convergence

    def compute_loss(
        self,
        dimensions: SNARCDimensions,
        weights: SNARCWeights,
        reward: float
    ) -> float:
        """
        Compute loss for single example.

        L = (reward - salience)² + λ||w||²
        """
        # Weighted salience
        salience = (
            weights.surprise * dimensions.surprise +
            weights.novelty * dimensions.novelty +
            weights.arousal * dimensions.arousal +
            weights.reward * dimensions.reward +
            weights.conflict * dimensions.conflict
        )

        # Prediction error
        error = reward - salience
        prediction_loss = error ** 2

        # L2 regularization
        weight_penalty = self.regularization * (
            weights.surprise ** 2 +
            weights.novelty ** 2 +
            weights.arousal ** 2 +
            weights.reward ** 2 +
            weights.conflict ** 2
        )

        return prediction_loss + weight_penalty

    def compute_gradient(
        self,
        dimensions: SNARCDimensions,
        weights: SNARCWeights,
        reward: float
    ) -> Dict[str, float]:
        """
        Compute gradient of loss w.r.t. weights.

        ∂L/∂w_i = -2(reward - salience) × dimension_i + 2λw_i
        """
        # Compute current salience
        salience = (
            weights.surprise * dimensions.surprise +
            weights.novelty * dimensions.novelty +
            weights.arousal * dimensions.arousal +
            weights.reward * dimensions.reward +
            weights.conflict * dimensions.conflict
        )

        # Prediction error
        error = reward - salience

        # Gradients
        gradients = {
            'surprise': -2 * error * dimensions.surprise + 2 * self.regularization * weights.surprise,
            'novelty': -2 * error * dimensions.novelty + 2 * self.regularization * weights.novelty,
            'arousal': -2 * error * dimensions.arousal + 2 * self.regularization * weights.arousal,
            'reward': -2 * error * dimensions.reward + 2 * self.regularization * weights.reward,
            'conflict': -2 * error * dimensions.conflict + 2 * self.regularization * weights.conflict,
        }

        return gradients

    def update(
        self,
        dimensions: SNARCDimensions,
        reward: float,
        cycle: int
    ) -> Tuple[SNARCWeights, float]:
        """
        Perform one online weight update.

        Args:
            dimensions: SNARC dimensions from attended cycle
            reward: Outcome quality (0-1)
            cycle: Current cycle number

        Returns:
            (updated_weights, loss)
        """
        # Compute gradient
        gradients = self.compute_gradient(dimensions, self.weights, reward)
        self.recent_gradients.append(gradients)

        # Compute loss before update
        loss = self.compute_loss(dimensions, self.weights, reward)
        self.loss_history.append(loss)

        # Update weights
        new_weights = SNARCWeights(
            surprise=self.weights.surprise - self.learning_rate * gradients['surprise'],
            novelty=self.weights.novelty - self.learning_rate * gradients['novelty'],
            arousal=self.weights.arousal - self.learning_rate * gradients['arousal'],
            reward=self.weights.reward - self.learning_rate * gradients['reward'],
            conflict=self.weights.conflict - self.learning_rate * gradients['conflict']
        )

        # Normalize weights to sum to 1
        new_weights.normalize()

        # Save snapshot
        snapshot = WeightSnapshot(
            timestamp=time.time(),
            cycle=cycle,
            weights=new_weights.as_dict(),
            loss=loss,
            learning_rate=self.learning_rate
        )
        self.weight_history.append(snapshot)

        # Update state
        self.weights = new_weights
        self.update_count += 1

        # Decay learning rate
        self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)

        return new_weights, loss

    def is_converged(self) -> bool:
        """
        Check if weights have converged (stable).

        Convergence criterion: Max weight change < threshold over window.
        """
        if len(self.weight_history) < self.convergence_window:
            return False

        # Get recent snapshots
        recent = self.weight_history[-self.convergence_window:]

        # Compute max weight change
        max_change = 0.0
        for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']:
            values = [s.weights[dim] for s in recent]
            weight_range = max(values) - min(values)
            max_change = max(max_change, weight_range)

        return max_change < self.convergence_threshold

    def get_statistics(self) -> Dict:
        """Get learning statistics"""
        stats = {
            'update_count': self.update_count,
            'current_lr': self.learning_rate,
            'converged': self.is_converged(),
            'current_weights': self.weights.as_dict(),
        }

        if self.loss_history:
            stats['current_loss'] = self.loss_history[-1]
            stats['avg_loss_last_10'] = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))

        if self.recent_gradients:
            # Average gradient magnitude
            avg_grad_mag = 0.0
            for grad in self.recent_gradients:
                mag = sum(abs(v) for v in grad.values())
                avg_grad_mag += mag
            stats['avg_gradient_magnitude'] = avg_grad_mag / len(self.recent_gradients)

        return stats


# ============================================================================
# Online Learning Consciousness Kernel
# ============================================================================

class OnlineLearningConsciousness:
    """
    Consciousness kernel with online weight adaptation.

    Combines:
    - SNARC compression with learned weights
    - Metabolic-state-dependent thresholds
    - Online weight learning from attention outcomes
    - Weight evolution tracking
    """

    def __init__(
        self,
        initial_weights: Optional[SNARCWeights] = None,
        learning_enabled: bool = True,
        learning_rate: float = 0.05
    ):
        """
        Initialize consciousness with online learning.

        Args:
            initial_weights: Starting weights (default: learned from batch)
            learning_enabled: Enable online weight updates
            learning_rate: Learning rate for online updates
        """
        # Load learned weights from batch training if available
        if initial_weights is None:
            learned_path = Path(__file__).parent / "learned_snarc_weights.json"
            if learned_path.exists():
                with open(learned_path) as f:
                    data = json.load(f)
                    learned = data['learned']
                    initial_weights = SNARCWeights(
                        surprise=learned['surprise'],
                        novelty=learned['novelty'],
                        arousal=learned['arousal'],
                        reward=learned['reward'],
                        conflict=learned['conflict']
                    )
                    print(f"📚 Loaded learned weights from batch training")
            else:
                initial_weights = SNARCWeights()
                print(f"⚙️  Using default weights (batch weights not found)")

        # Initialize compressor with weights
        self.compressor = SNARCCompressor(weights=initial_weights)

        # Initialize online learner
        self.learning_enabled = learning_enabled
        if learning_enabled:
            self.learner = OnlineWeightLearner(
                initial_weights=initial_weights,
                learning_rate=learning_rate
            )
        else:
            self.learner = None

        # Sensor manager
        self.sensor_manager = SensorManager()
        self.sensor_manager.register_sensor('cpu', CPUSensor())
        self.sensor_manager.register_sensor('memory', MemorySensor())
        self.sensor_manager.register_sensor('process', ProcessSensor())
        self.sensor_manager.register_sensor('disk', DiskSensor())

        # Metabolic state
        self.atp = 1.0
        self.metabolic_state = "wake"

        # Cycle tracking
        self.cycle = 0
        self.attended_count = 0

    def get_threshold(self) -> float:
        """Get attention threshold based on metabolic state"""
        thresholds = {
            'wake': 0.45,
            'focus': 0.25,
            'rest': 0.75,
            'dream': 0.05
        }
        return thresholds.get(self.metabolic_state, 0.50)

    def compute_outcome_quality(self, sensor_name: str, salience: float) -> float:
        """
        Compute outcome quality for attended cycle.

        In real system, would evaluate:
        - Did attention lead to useful action?
        - Was intervention necessary?
        - Did prediction match reality?

        For now, use heuristics:
        - High salience attended → good (caught important event)
        - Process sensor has higher base quality (more variation)
        """
        base_quality = 0.6

        # Higher salience → higher quality (caught important events)
        salience_bonus = 0.3 * salience

        # Process sensor tends to be more informative
        if sensor_name == 'process':
            base_quality += 0.1

        return min(1.0, base_quality + salience_bonus)

    def run_cycle(self) -> Dict:
        """Run one consciousness cycle"""
        self.cycle += 1

        # Read sensors
        sensor_readings = self.sensor_manager.read_all()

        # Select focus sensor (highest initial salience)
        max_salience = 0.0
        focus_sensor = 'cpu'
        focus_data = {}

        for name, data in sensor_readings.items():
            # Quick salience estimate
            novelty = data.get('novelty_score', 0.0)
            if novelty > max_salience:
                max_salience = novelty
                focus_sensor = name
                focus_data = data

        # Compute SNARC dimensions for focus
        dimensions = self.compressor.compute_snarc_dimensions(focus_data)

        # Compress to salience
        salience = self.compressor.compress_to_salience(dimensions)

        # Get threshold
        threshold = self.get_threshold()

        # Attention decision
        decision = "ATTEND" if salience > threshold else "IGNORE"

        # If attended, update weights
        updated_weights = None
        loss = None
        if decision == "ATTEND" and self.learning_enabled:
            self.attended_count += 1

            # Compute outcome quality
            outcome = self.compute_outcome_quality(focus_sensor, salience)

            # Update weights online
            updated_weights, loss = self.learner.update(dimensions, outcome, self.cycle)

            # Update compressor weights
            self.compressor.weights = updated_weights

        # Simple metabolic state cycling (for demo)
        if self.cycle % 30 == 0:
            states = ['wake', 'rest', 'dream']
            self.metabolic_state = states[(self.cycle // 30) % len(states)]

        # Return cycle results
        result = {
            'cycle': self.cycle,
            'state': self.metabolic_state,
            'focus': focus_sensor,
            'salience': salience,
            'threshold': threshold,
            'decision': decision,
            'dimensions': asdict(dimensions),
            'atp': self.atp
        }

        if updated_weights:
            result['weights'] = updated_weights.as_dict()
            result['loss'] = loss

        return result


# ============================================================================
# Main Experiment
# ============================================================================

def run_online_learning_experiment(
    duration_seconds: int = 300,
    learning_rate: float = 0.05,
    output_file: str = "online_weight_adaptation_output.log"
):
    """
    Run online weight adaptation experiment.

    Args:
        duration_seconds: How long to run (default: 5 minutes)
        learning_rate: Learning rate for online updates
        output_file: Output log file
    """
    print("="*80)
    print("ONLINE SNARC WEIGHT ADAPTATION")
    print("="*80)
    print()
    print(f"Duration: {duration_seconds} seconds ({duration_seconds/60:.1f} minutes)")
    print(f"Learning Rate: {learning_rate}")
    print(f"Output: {output_file}")
    print()

    # Initialize consciousness
    consciousness = OnlineLearningConsciousness(
        learning_enabled=True,
        learning_rate=learning_rate
    )

    print(f"Initial weights:")
    for dim, weight in consciousness.compressor.weights.as_dict().items():
        print(f"  {dim}: {weight:.3f}")
    print()

    print("Starting online learning deployment...")
    print("(Press Ctrl+C for graceful shutdown)")
    print("="*80)
    print()

    # Open output log
    log = open(output_file, 'w')

    start_time = time.time()
    last_stats_time = start_time
    stats_interval = 30  # Print stats every 30 seconds

    try:
        while time.time() - start_time < duration_seconds:
            # Run cycle
            result = consciousness.run_cycle()

            # Log cycle
            log.write(f"[Cycle {result['cycle']}] State: {result['state'].upper()}\n")
            log.write(f"  Focus: {result['focus']} (salience={result['salience']:.3f})\n")
            log.write(f"  Dimensions: S={result['dimensions']['surprise']:.2f} "
                     f"N={result['dimensions']['novelty']:.2f} "
                     f"A={result['dimensions']['arousal']:.2f} "
                     f"R={result['dimensions']['reward']:.2f} "
                     f"C={result['dimensions']['conflict']:.2f}\n")
            log.write(f"  Threshold: {result['threshold']:.3f} | Decision: {result['decision']}\n")

            if 'weights' in result:
                log.write(f"  ⭐ UPDATED WEIGHTS: "
                         f"S={result['weights']['surprise']:.3f} "
                         f"N={result['weights']['novelty']:.3f} "
                         f"A={result['weights']['arousal']:.3f} "
                         f"R={result['weights']['reward']:.3f} "
                         f"C={result['weights']['conflict']:.3f} "
                         f"(loss={result['loss']:.4f})\n")

            log.write("\n")
            log.flush()

            # Print progress stats
            if time.time() - last_stats_time >= stats_interval:
                elapsed = time.time() - start_time
                print(f"\n⏱️  Progress: {elapsed:.0f}s / {duration_seconds}s ({100*elapsed/duration_seconds:.0f}%)")
                print(f"   Cycles: {result['cycle']}, Attended: {consciousness.attended_count}")

                if consciousness.learner:
                    stats = consciousness.learner.get_statistics()
                    print(f"   Updates: {stats['update_count']}, LR: {stats['current_lr']:.4f}")
                    print(f"   Current weights:")
                    for dim, weight in stats['current_weights'].items():
                        print(f"     {dim}: {weight:.3f}")
                    if 'current_loss' in stats:
                        print(f"   Loss: {stats['current_loss']:.4f}")
                    if stats.get('converged', False):
                        print(f"   🎯 CONVERGED!")

                last_stats_time = time.time()

            # Sleep between cycles
            time.sleep(2.0)

    except KeyboardInterrupt:
        print("\n\n🛑 Graceful shutdown requested")

    finally:
        log.close()

    # Final statistics
    print()
    print("="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print()

    total_time = time.time() - start_time
    print(f"Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Cycles: {consciousness.cycle}")
    print(f"Attended: {consciousness.attended_count} ({100*consciousness.attended_count/consciousness.cycle:.1f}%)")
    print()

    if consciousness.learner:
        stats = consciousness.learner.get_statistics()
        print(f"Weight Updates: {stats['update_count']}")
        print(f"Final Learning Rate: {stats['current_lr']:.4f}")
        print(f"Converged: {stats.get('converged', False)}")
        print()

        print("Weight Evolution:")
        print(f"  Initial → Final")
        initial = consciousness.learner.weight_history[0].weights if consciousness.learner.weight_history else {}
        final = stats['current_weights']
        for dim in ['surprise', 'novelty', 'arousal', 'reward', 'conflict']:
            i = initial.get(dim, 0.0)
            f = final[dim]
            change = f - i
            print(f"  {dim:8s}: {i:.3f} → {f:.3f} ({change:+.3f})")
        print()

        # Save final results
        results_file = output_file.replace('.log', '_results.json')
        with open(results_file, 'w') as f:
            json.dump({
                'runtime_seconds': total_time,
                'cycles': consciousness.cycle,
                'attended_count': consciousness.attended_count,
                'attention_rate': consciousness.attended_count / consciousness.cycle,
                'weight_updates': stats['update_count'],
                'converged': stats.get('converged', False),
                'initial_weights': initial,
                'final_weights': final,
                'weight_changes': {dim: final[dim] - initial.get(dim, 0.0) for dim in final.keys()},
                'final_loss': stats.get('current_loss', 0.0),
                'learning_rate_initial': consciousness.learner.initial_lr,
                'learning_rate_final': stats['current_lr']
            }, f, indent=2)

        print(f"Results saved to: {results_file}")
        print(f"Log saved to: {output_file}")

    print()
    print("✅ Online weight adaptation experiment complete!")
    print("="*80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Online SNARC weight adaptation experiment')
    parser.add_argument('--duration', type=int, default=300,
                       help='Duration in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--lr', type=float, default=0.05,
                       help='Learning rate (default: 0.05)')
    parser.add_argument('--output', type=str, default='online_weight_adaptation_output.log',
                       help='Output log file')

    args = parser.parse_args()

    run_online_learning_experiment(
        duration_seconds=args.duration,
        learning_rate=args.lr,
        output_file=args.output
    )
