#!/usr/bin/env python3
"""
Epistemic Calibration System - Session 39

Measures how well epistemic confidence matches actual correctness, addressing
the insight from Session 38 that confidence ≠ quality but confidence SHOULD
match epistemic accuracy.

Key distinction:
- Confidence does NOT predict output quality (Session 38 finding)
- Confidence SHOULD predict correctness of knowledge claims
- This system measures epistemic calibration, not quality prediction

Example:
- "I'm confident X is true" with conf=0.9 → X should be correct ~90% of time
- "I'm uncertain about Y" with conf=0.3 → Y might be wrong ~70% of time

Author: Thor (Autonomous Session 39)
Date: 2025-12-12
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class CorrectnessLabel(Enum):
    """Ground truth correctness labels"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIALLY_CORRECT = "partially_correct"
    UNKNOWN = "unknown"


@dataclass
class EpistemicClaim:
    """
    A claim made with epistemic confidence.

    Attributes:
        claim: The factual claim or statement
        confidence: Confidence in the claim (0.0-1.0)
        uncertainty: Explicit uncertainty estimate (0.0-1.0)
        ground_truth: Actual correctness (for calibration)
    """
    claim: str
    confidence: float
    uncertainty: float
    ground_truth: Optional[CorrectnessLabel] = None

    def is_labeled(self) -> bool:
        """Check if ground truth is available"""
        return self.ground_truth is not None and self.ground_truth != CorrectnessLabel.UNKNOWN


@dataclass
class CalibrationBin:
    """
    Calibration bin for grouping similar confidence levels.

    Attributes:
        confidence_range: (min, max) confidence for this bin
        predicted_prob: Mean confidence in bin
        observed_prob: Actual proportion correct in bin
        count: Number of claims in bin
    """
    confidence_range: Tuple[float, float]
    predicted_prob: float
    observed_prob: float
    count: int

    def calibration_error(self) -> float:
        """Absolute calibration error for this bin"""
        return abs(self.predicted_prob - self.observed_prob)


@dataclass
class CalibrationMetrics:
    """
    Complete calibration metrics for epistemic system.

    Attributes:
        expected_calibration_error: Mean absolute difference between confidence and accuracy
        max_calibration_error: Maximum bin calibration error
        bins: Calibration bins for visualization
        brier_score: Probabilistic accuracy score (lower is better)
        log_loss: Logarithmic loss (lower is better)
        accuracy: Overall proportion correct
        sample_size: Number of labeled claims
    """
    expected_calibration_error: float  # ECE - primary calibration metric
    max_calibration_error: float       # MCE - worst-case bin error
    bins: List[CalibrationBin]
    brier_score: float                 # Probabilistic accuracy
    log_loss: float                    # Log loss (negative log likelihood)
    accuracy: float                    # Overall accuracy
    sample_size: int

    def is_well_calibrated(self, ece_threshold: float = 0.10) -> bool:
        """
        Check if system is well-calibrated.

        Args:
            ece_threshold: Maximum acceptable ECE (default 0.10 = 10%)

        Returns:
            True if ECE ≤ threshold
        """
        return self.expected_calibration_error <= ece_threshold


class EpistemicCalibrationSystem:
    """
    Measures epistemic calibration - how well confidence matches correctness.

    This is distinct from quality prediction:
    - Quality: How good is the output? (Session 27)
    - Calibration: When confident, am I usually correct? (Session 39)

    Well-calibrated system:
    - 90% confidence claims are correct 90% of the time
    - 50% confidence claims are correct 50% of the time
    - Uncertainty estimates match actual error rates

    Poorly-calibrated system:
    - Overconfident: 90% confidence but only 60% correct
    - Underconfident: 50% confidence but 90% correct
    - Miscalibrated uncertainty
    """

    def __init__(self, num_bins: int = 10):
        """
        Initialize calibration system.

        Args:
            num_bins: Number of confidence bins for ECE calculation (default 10)
        """
        self.num_bins = num_bins
        self.claims: List[EpistemicClaim] = []

    def add_claim(self,
                  claim: str,
                  confidence: float,
                  uncertainty: float,
                  ground_truth: Optional[CorrectnessLabel] = None):
        """
        Add epistemic claim for calibration tracking.

        Args:
            claim: Factual claim or statement
            confidence: Confidence level (0.0-1.0)
            uncertainty: Uncertainty estimate (0.0-1.0)
            ground_truth: Actual correctness (if known)
        """
        epistemic_claim = EpistemicClaim(
            claim=claim,
            confidence=confidence,
            uncertainty=uncertainty,
            ground_truth=ground_truth
        )
        self.claims.append(epistemic_claim)

    def calculate_calibration(self) -> CalibrationMetrics:
        """
        Calculate comprehensive calibration metrics.

        Returns:
            CalibrationMetrics with ECE, MCE, bins, Brier score, etc.
        """
        # Filter to labeled claims only
        labeled = [c for c in self.claims if c.is_labeled()]

        if not labeled:
            # No labeled data - return default metrics
            return CalibrationMetrics(
                expected_calibration_error=0.0,
                max_calibration_error=0.0,
                bins=[],
                brier_score=0.0,
                log_loss=0.0,
                accuracy=0.0,
                sample_size=0
            )

        # Convert to binary labels (1 = correct, 0 = incorrect)
        # PARTIALLY_CORRECT counts as 0.5
        confidences = []
        labels = []
        for claim in labeled:
            confidences.append(claim.confidence)
            if claim.ground_truth == CorrectnessLabel.CORRECT:
                labels.append(1.0)
            elif claim.ground_truth == CorrectnessLabel.INCORRECT:
                labels.append(0.0)
            elif claim.ground_truth == CorrectnessLabel.PARTIALLY_CORRECT:
                labels.append(0.5)
            else:
                continue  # Skip UNKNOWN

        confidences = np.array(confidences)
        labels = np.array(labels)

        # Calculate bins
        bins = self._calculate_bins(confidences, labels)

        # Expected Calibration Error (ECE)
        ece = np.mean([b.calibration_error() * b.count for b in bins]) / len(labels)

        # Maximum Calibration Error (MCE)
        mce = max([b.calibration_error() for b in bins]) if bins else 0.0

        # Brier Score (mean squared error of probabilities)
        brier = np.mean((confidences - labels) ** 2)

        # Log Loss (negative log likelihood)
        epsilon = 1e-15  # Prevent log(0)
        log_loss_val = -np.mean(
            labels * np.log(confidences + epsilon) +
            (1 - labels) * np.log(1 - confidences + epsilon)
        )

        # Overall accuracy (using 0.5 threshold)
        predictions = (confidences > 0.5).astype(float)
        accuracy = np.mean(predictions == labels)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            max_calibration_error=mce,
            bins=bins,
            brier_score=brier,
            log_loss=log_loss_val,
            accuracy=accuracy,
            sample_size=len(labels)
        )

    def _calculate_bins(self,
                       confidences: np.ndarray,
                       labels: np.ndarray) -> List[CalibrationBin]:
        """
        Calculate calibration bins.

        Args:
            confidences: Array of confidence values
            labels: Array of binary labels (1=correct, 0=incorrect)

        Returns:
            List of CalibrationBin objects
        """
        bins = []
        bin_edges = np.linspace(0, 1, self.num_bins + 1)

        for i in range(self.num_bins):
            bin_min = bin_edges[i]
            bin_max = bin_edges[i + 1]

            # Find claims in this bin
            in_bin = (confidences >= bin_min) & (confidences < bin_max if i < self.num_bins - 1 else confidences <= bin_max)

            if np.sum(in_bin) == 0:
                continue  # Skip empty bins

            bin_confidences = confidences[in_bin]
            bin_labels = labels[in_bin]

            predicted_prob = np.mean(bin_confidences)
            observed_prob = np.mean(bin_labels)
            count = len(bin_confidences)

            calibration_bin = CalibrationBin(
                confidence_range=(bin_min, bin_max),
                predicted_prob=predicted_prob,
                observed_prob=observed_prob,
                count=count
            )

            bins.append(calibration_bin)

        return bins

    def uncertainty_calibration(self) -> Dict[str, float]:
        """
        Measure uncertainty calibration separately.

        Uncertainty should predict error: high uncertainty → likely incorrect.

        Returns:
            Dict with uncertainty metrics:
            - uncertainty_error_correlation: Correlation between uncertainty and error
            - high_uncertainty_error_rate: Error rate when uncertainty > 0.5
            - low_uncertainty_error_rate: Error rate when uncertainty < 0.3
        """
        labeled = [c for c in self.claims if c.is_labeled()]

        if not labeled:
            return {
                'uncertainty_error_correlation': 0.0,
                'high_uncertainty_error_rate': 0.0,
                'low_uncertainty_error_rate': 0.0
            }

        uncertainties = []
        errors = []  # 1 if incorrect, 0 if correct

        for claim in labeled:
            uncertainties.append(claim.uncertainty)

            if claim.ground_truth == CorrectnessLabel.CORRECT:
                errors.append(0.0)
            elif claim.ground_truth == CorrectnessLabel.INCORRECT:
                errors.append(1.0)
            elif claim.ground_truth == CorrectnessLabel.PARTIALLY_CORRECT:
                errors.append(0.5)

        uncertainties = np.array(uncertainties)
        errors = np.array(errors)

        # Correlation between uncertainty and error
        if len(uncertainties) > 1:
            correlation = np.corrcoef(uncertainties, errors)[0, 1]
        else:
            correlation = 0.0

        # Error rate when high uncertainty
        high_uncertainty = uncertainties > 0.5
        if np.sum(high_uncertainty) > 0:
            high_unc_error_rate = np.mean(errors[high_uncertainty])
        else:
            high_unc_error_rate = 0.0

        # Error rate when low uncertainty
        low_uncertainty = uncertainties < 0.3
        if np.sum(low_uncertainty) > 0:
            low_unc_error_rate = np.mean(errors[low_uncertainty])
        else:
            low_unc_error_rate = 0.0

        return {
            'uncertainty_error_correlation': correlation,
            'high_uncertainty_error_rate': high_unc_error_rate,
            'low_uncertainty_error_rate': low_unc_error_rate
        }

    def overconfidence_analysis(self) -> Dict[str, float]:
        """
        Analyze overconfidence vs underconfidence patterns.

        Returns:
            Dict with:
            - overconfidence_rate: Proportion of bins where predicted > observed
            - mean_overconfidence: Mean (predicted - observed) when overconfident
            - underconfidence_rate: Proportion of bins where predicted < observed
            - mean_underconfidence: Mean (observed - predicted) when underconfident
        """
        metrics = self.calculate_calibration()

        if not metrics.bins:
            return {
                'overconfidence_rate': 0.0,
                'mean_overconfidence': 0.0,
                'underconfidence_rate': 0.0,
                'mean_underconfidence': 0.0
            }

        overconfident_bins = [b for b in metrics.bins
                             if b.predicted_prob > b.observed_prob]
        underconfident_bins = [b for b in metrics.bins
                              if b.predicted_prob < b.observed_prob]

        total_bins = len(metrics.bins)

        overconfidence_rate = len(overconfident_bins) / total_bins
        underconfidence_rate = len(underconfident_bins) / total_bins

        if overconfident_bins:
            mean_overconf = np.mean([b.predicted_prob - b.observed_prob
                                    for b in overconfident_bins])
        else:
            mean_overconf = 0.0

        if underconfident_bins:
            mean_underconf = np.mean([b.observed_prob - b.predicted_prob
                                     for b in underconfident_bins])
        else:
            mean_underconf = 0.0

        return {
            'overconfidence_rate': overconfidence_rate,
            'mean_overconfidence': mean_overconf,
            'underconfidence_rate': underconfidence_rate,
            'mean_underconfidence': mean_underconf
        }


def example_usage():
    """Example demonstrating epistemic calibration measurement"""
    calibration = EpistemicCalibrationSystem(num_bins=5)

    # Add some example claims with ground truth
    # Well-calibrated examples
    calibration.add_claim(
        "The Earth orbits the Sun",
        confidence=0.95,
        uncertainty=0.05,
        ground_truth=CorrectnessLabel.CORRECT
    )

    calibration.add_claim(
        "Python was released in 1991",
        confidence=0.90,
        uncertainty=0.10,
        ground_truth=CorrectnessLabel.CORRECT
    )

    # Overconfident example
    calibration.add_claim(
        "The capital of Australia is Sydney",  # Actually Canberra
        confidence=0.80,
        uncertainty=0.20,
        ground_truth=CorrectnessLabel.INCORRECT
    )

    # Well-calibrated uncertain
    calibration.add_claim(
        "The Higgs boson mass is approximately 125 GeV/c²",
        confidence=0.60,
        uncertainty=0.40,
        ground_truth=CorrectnessLabel.CORRECT
    )

    # Appropriately uncertain and wrong
    calibration.add_claim(
        "Dark matter is composed of WIMPs",  # Uncertain physics
        confidence=0.30,
        uncertainty=0.70,
        ground_truth=CorrectnessLabel.UNKNOWN
    )

    # Calculate metrics
    metrics = calibration.calculate_calibration()

    print(f"Expected Calibration Error: {metrics.expected_calibration_error:.3f}")
    print(f"Max Calibration Error: {metrics.max_calibration_error:.3f}")
    print(f"Brier Score: {metrics.brier_score:.3f}")
    print(f"Accuracy: {metrics.accuracy:.3f}")
    print(f"Sample Size: {metrics.sample_size}")
    print(f"Well-calibrated: {metrics.is_well_calibrated()}")

    print("\nCalibration Bins:")
    for b in metrics.bins:
        print(f"  [{b.confidence_range[0]:.1f}-{b.confidence_range[1]:.1f}]: "
              f"predicted={b.predicted_prob:.3f}, observed={b.observed_prob:.3f}, "
              f"error={b.calibration_error():.3f}, count={b.count}")


if __name__ == '__main__':
    example_usage()
