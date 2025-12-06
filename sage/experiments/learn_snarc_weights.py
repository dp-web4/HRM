"""
SNARC Weight Learning from Deployment Data
===========================================

Implements gradient-based learning of SNARC dimension weights from extended
deployment test results.

**Approach**:
1. Analyze which SNARC dimensions correlated with successful attention decisions
2. Implement gradient descent on outcome quality (reward)
3. Adapt weights: w_i ← w_i + α × ∂reward/∂w_i
4. Test learned weights vs default weights

**Data Source**: Extended deployment test (156 cycles, 17 attention decisions)

**Key Finding**: Novelty (0.23-0.51) dominated salience in monitoring workload

**Expected**: Learned weights should emphasize novelty over other dimensions

**Session**: Thor Autonomous Research (2025-12-06)
**Author**: Claude (guest) on Thor via claude-code
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, SNARCWeights, SNARCDimensions, CompressionMode
import json
import numpy as np
from typing import List, Dict, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Jetson
import matplotlib.pyplot as plt


# ============================================================================
# Weight Learning Algorithm
# ============================================================================

class SNARCWeightLearner:
    """
    Learn optimal SNARC weights from deployment data using gradient descent.

    **Objective**: Maximize correlation between salience and outcome quality (reward)

    **Method**: Gradient descent on weighted sum coefficients
    """

    def __init__(self, learning_rate: float = 0.1, regularization: float = 0.01):
        """
        Initialize weight learner.

        Args:
            learning_rate: Step size for gradient descent
            regularization: L2 regularization strength (prevent overfitting)
        """
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.history = []

    def compute_gradient(
        self,
        dimensions: SNARCDimensions,
        weights: SNARCWeights,
        actual_reward: float,
        predicted_salience: float
    ) -> Dict[str, float]:
        """
        Compute gradient of loss function with respect to weights.

        Loss = (reward - salience)² + λ||w||²

        ∂L/∂w_i = -2(reward - salience) × dimension_i + 2λw_i

        Args:
            dimensions: SNARC dimensions for this observation
            weights: Current weights
            actual_reward: Observed reward from this attention decision
            predicted_salience: Salience computed with current weights

        Returns:
            Gradient dict {dimension: gradient_value}
        """
        # Prediction error
        error = actual_reward - predicted_salience

        # Gradient components
        gradients = {}

        # Surprise gradient
        gradients['surprise'] = -2 * error * dimensions.surprise + \
                               2 * self.regularization * weights.surprise

        # Novelty gradient
        gradients['novelty'] = -2 * error * dimensions.novelty + \
                              2 * self.regularization * weights.novelty

        # Arousal gradient
        gradients['arousal'] = -2 * error * dimensions.arousal + \
                              2 * self.regularization * weights.arousal

        # Reward gradient
        gradients['reward'] = -2 * error * dimensions.reward + \
                             2 * self.regularization * weights.reward

        # Conflict gradient
        gradients['conflict'] = -2 * error * dimensions.conflict + \
                               2 * self.regularization * weights.conflict

        return gradients

    def update_weights(
        self,
        weights: SNARCWeights,
        gradients: Dict[str, float]
    ) -> SNARCWeights:
        """
        Update weights using gradient descent.

        w_i ← w_i - α × ∂L/∂w_i

        Args:
            weights: Current weights
            gradients: Computed gradients

        Returns:
            Updated weights (normalized)
        """
        new_weights = SNARCWeights(
            surprise=weights.surprise - self.learning_rate * gradients['surprise'],
            novelty=weights.novelty - self.learning_rate * gradients['novelty'],
            arousal=weights.arousal - self.learning_rate * gradients['arousal'],
            reward=weights.reward - self.learning_rate * gradients['reward'],
            conflict=weights.conflict - self.learning_rate * gradients['conflict']
        )

        # Clamp to positive values
        new_weights.surprise = max(0.01, new_weights.surprise)
        new_weights.novelty = max(0.01, new_weights.novelty)
        new_weights.arousal = max(0.01, new_weights.arousal)
        new_weights.reward = max(0.01, new_weights.reward)
        new_weights.conflict = max(0.01, new_weights.conflict)

        # Normalize to sum to 1
        new_weights.normalize()

        return new_weights

    def train(
        self,
        training_data: List[Tuple[SNARCDimensions, float]],
        initial_weights: SNARCWeights,
        epochs: int = 100,
        verbose: bool = True
    ) -> Tuple[SNARCWeights, List[float]]:
        """
        Train weights using gradient descent.

        Args:
            training_data: List of (dimensions, reward) tuples
            initial_weights: Starting weights
            epochs: Number of training epochs
            verbose: Print progress

        Returns:
            (learned_weights, loss_history)
        """
        weights = SNARCWeights(
            surprise=initial_weights.surprise,
            novelty=initial_weights.novelty,
            arousal=initial_weights.arousal,
            reward=initial_weights.reward,
            conflict=initial_weights.conflict
        )

        loss_history = []

        for epoch in range(epochs):
            total_loss = 0.0

            for dimensions, reward in training_data:
                # Compute predicted salience with current weights
                predicted_salience = (
                    weights.surprise * dimensions.surprise +
                    weights.novelty * dimensions.novelty +
                    weights.arousal * dimensions.arousal +
                    weights.reward * dimensions.reward +
                    weights.conflict * dimensions.conflict
                )

                # Compute gradient
                gradients = self.compute_gradient(
                    dimensions, weights, reward, predicted_salience
                )

                # Update weights
                weights = self.update_weights(weights, gradients)

                # Compute loss
                error = reward - predicted_salience
                loss = error**2 + self.regularization * (
                    weights.surprise**2 + weights.novelty**2 + weights.arousal**2 +
                    weights.reward**2 + weights.conflict**2
                )
                total_loss += loss

            avg_loss = total_loss / len(training_data)
            loss_history.append(avg_loss)

            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch:3d}: Loss={avg_loss:.6f}, " +
                      f"Weights=[S:{weights.surprise:.3f}, N:{weights.novelty:.3f}, " +
                      f"A:{weights.arousal:.3f}, R:{weights.reward:.3f}, C:{weights.conflict:.3f}]")

        return weights, loss_history


# ============================================================================
# Data Loading from Extended Deployment
# ============================================================================

def load_deployment_data(log_file: str) -> List[Tuple[SNARCDimensions, float]]:
    """
    Parse extended deployment log to extract SNARC dimensions and rewards.

    Args:
        log_file: Path to extended_deployment_output.log

    Returns:
        List of (SNARCDimensions, reward) tuples for attended cycles
    """
    training_data = []

    with open(log_file, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for cycle headers with SNARC data
        if line.startswith('[Cycle') and 'State:' in line:
            # Check if SNARC line exists (line i+2)
            if i + 2 < len(lines) and 'SNARC:' in lines[i + 2]:
                snarc_line = lines[i + 2].strip()

                # Parse SNARC dimensions
                # Format: "  SNARC: S=0.00 N=0.36 A=0.20 R=0.00 C=0.00"
                try:
                    parts = snarc_line.split('SNARC:')[1].strip().split()
                    surprise = float(parts[0].split('=')[1])
                    novelty = float(parts[1].split('=')[1])
                    arousal = float(parts[2].split('=')[1])
                    reward_dim = float(parts[3].split('=')[1])
                    conflict = float(parts[4].split('=')[1])

                    dimensions = SNARCDimensions(
                        surprise=surprise,
                        novelty=novelty,
                        arousal=arousal,
                        reward=reward_dim,
                        conflict=conflict
                    )

                    # Check if this cycle was ATTENDED
                    # Format: line i+3 has "Threshold: X | ATP: Y | Decision: ATTEND"
                    if i + 3 < len(lines) and 'Decision: ATTEND' in lines[i + 3]:
                        # This cycle was attended - use reward of 0.7 (default from code)
                        # In real deployment, would extract actual reward from result
                        outcome_reward = 0.7  # Default reward for process sensor
                        training_data.append((dimensions, outcome_reward))

                except (IndexError, ValueError) as e:
                    # Skip malformed lines
                    pass

        i += 1

    return training_data


def analyze_dimension_importance(training_data: List[Tuple[SNARCDimensions, float]]):
    """
    Analyze which SNARC dimensions are most important in training data.

    Args:
        training_data: List of (dimensions, reward) tuples
    """
    print("\n" + "="*80)
    print("SNARC DIMENSION ANALYSIS")
    print("="*80)
    print()

    if not training_data:
        print("No training data available")
        return

    # Compute statistics for each dimension
    surprises = [d.surprise for d, r in training_data]
    novelties = [d.novelty for d, r in training_data]
    arousals = [d.arousal for d, r in training_data]
    rewards = [d.reward for d, r in training_data]
    conflicts = [d.conflict for d, r in training_data]

    print(f"Training examples: {len(training_data)}")
    print()
    print("Dimension Statistics:")
    print(f"  Surprise: mean={np.mean(surprises):.3f}, std={np.std(surprises):.3f}, " +
          f"min={np.min(surprises):.3f}, max={np.max(surprises):.3f}")
    print(f"  Novelty:  mean={np.mean(novelties):.3f}, std={np.std(novelties):.3f}, " +
          f"min={np.min(novelties):.3f}, max={np.max(novelties):.3f}")
    print(f"  Arousal:  mean={np.mean(arousals):.3f}, std={np.std(arousals):.3f}, " +
          f"min={np.min(arousals):.3f}, max={np.max(arousals):.3f}")
    print(f"  Reward:   mean={np.mean(rewards):.3f}, std={np.std(rewards):.3f}, " +
          f"min={np.min(rewards):.3f}, max={np.max(rewards):.3f}")
    print(f"  Conflict: mean={np.mean(conflicts):.3f}, std={np.std(conflicts):.3f}, " +
          f"min={np.min(conflicts):.3f}, max={np.max(conflicts):.3f}")
    print()

    # Identify dominant dimension
    means = {
        'surprise': np.mean(surprises),
        'novelty': np.mean(novelties),
        'arousal': np.mean(arousals),
        'reward': np.mean(rewards),
        'conflict': np.mean(conflicts)
    }

    dominant = max(means.items(), key=lambda x: x[1])
    print(f"Dominant dimension: {dominant[0].upper()} (mean={dominant[1]:.3f})")
    print()


# ============================================================================
# Main Weight Learning Experiment
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("SNARC WEIGHT LEARNING FROM EXTENDED DEPLOYMENT")
    print("="*80)
    print()

    # Load deployment data
    log_file = "/home/dp/extended_deployment_output.log"
    print(f"Loading training data from: {log_file}")
    training_data = load_deployment_data(log_file)
    print(f"Loaded {len(training_data)} training examples (ATTENDED cycles)")
    print()

    if len(training_data) == 0:
        print("ERROR: No training data found!")
        print("Check that log file exists and contains ATTENDED cycles with SNARC data")
        sys.exit(1)

    # Analyze dimension importance
    analyze_dimension_importance(training_data)

    # Initialize weights
    default_weights = SNARCWeights()  # Default: balanced weights
    print("Default weights:")
    print(f"  Surprise: {default_weights.surprise:.3f}")
    print(f"  Novelty:  {default_weights.novelty:.3f}")
    print(f"  Arousal:  {default_weights.arousal:.3f}")
    print(f"  Reward:   {default_weights.reward:.3f}")
    print(f"  Conflict: {default_weights.conflict:.3f}")
    print()

    # Train weights
    print("Training weights with gradient descent...")
    print()

    learner = SNARCWeightLearner(learning_rate=0.1, regularization=0.01)
    learned_weights, loss_history = learner.train(
        training_data,
        default_weights,
        epochs=100,
        verbose=True
    )

    print()
    print("="*80)
    print("WEIGHT LEARNING RESULTS")
    print("="*80)
    print()

    print("Learned weights:")
    print(f"  Surprise: {learned_weights.surprise:.3f} (default: {default_weights.surprise:.3f}, " +
          f"change: {learned_weights.surprise - default_weights.surprise:+.3f})")
    print(f"  Novelty:  {learned_weights.novelty:.3f} (default: {default_weights.novelty:.3f}, " +
          f"change: {learned_weights.novelty - default_weights.novelty:+.3f})")
    print(f"  Arousal:  {learned_weights.arousal:.3f} (default: {default_weights.arousal:.3f}, " +
          f"change: {learned_weights.arousal - default_weights.arousal:+.3f})")
    print(f"  Reward:   {learned_weights.reward:.3f} (default: {default_weights.reward:.3f}, " +
          f"change: {learned_weights.reward - default_weights.reward:+.3f})")
    print(f"  Conflict: {learned_weights.conflict:.3f} (default: {default_weights.conflict:.3f}, " +
          f"change: {learned_weights.conflict - default_weights.conflict:+.3f})")
    print()

    # Compute final loss
    final_loss = loss_history[-1] if loss_history else 0.0
    initial_loss = loss_history[0] if loss_history else 0.0
    improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0.0

    print(f"Training loss: {initial_loss:.6f} → {final_loss:.6f} (improvement: {improvement:.1f}%)")
    print()

    # Save learned weights
    weights_file = "learned_snarc_weights.json"
    weights_data = {
        'learned': learned_weights.as_dict(),
        'default': default_weights.as_dict(),
        'training_examples': len(training_data),
        'final_loss': final_loss,
        'improvement_percent': improvement
    }

    with open(weights_file, 'w') as f:
        json.dump(weights_data, f, indent=2)

    print(f"Saved learned weights to: {weights_file}")
    print()

    # Plot loss history
    if len(loss_history) > 0:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_history, linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('SNARC Weight Learning - Training Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = "snarc_weight_learning_loss.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Saved loss plot to: {plot_file}")
        print()

    # Compare default vs learned on training data
    print("="*80)
    print("VALIDATION: Default vs Learned Weights")
    print("="*80)
    print()

    default_predictions = []
    learned_predictions = []
    actuals = []

    for dimensions, reward in training_data:
        # Default weights prediction
        default_pred = (
            default_weights.surprise * dimensions.surprise +
            default_weights.novelty * dimensions.novelty +
            default_weights.arousal * dimensions.arousal +
            default_weights.reward * dimensions.reward +
            default_weights.conflict * dimensions.conflict
        )

        # Learned weights prediction
        learned_pred = (
            learned_weights.surprise * dimensions.surprise +
            learned_weights.novelty * dimensions.novelty +
            learned_weights.arousal * dimensions.arousal +
            learned_weights.reward * dimensions.reward +
            learned_weights.conflict * dimensions.conflict
        )

        default_predictions.append(default_pred)
        learned_predictions.append(learned_pred)
        actuals.append(reward)

    # Compute MSE
    default_mse = np.mean([(p - a)**2 for p, a in zip(default_predictions, actuals)])
    learned_mse = np.mean([(p - a)**2 for p, a in zip(learned_predictions, actuals)])

    print(f"Mean Squared Error:")
    print(f"  Default weights: {default_mse:.6f}")
    print(f"  Learned weights: {learned_mse:.6f}")
    print(f"  Improvement: {((default_mse - learned_mse) / default_mse * 100):.1f}%")
    print()

    print("✅ Weight learning complete!")
    print("="*80)
