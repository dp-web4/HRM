#!/usr/bin/env python3
"""
Real Measurement Functions for SAGE Observational Framework

Session 34: Real Measurement Integration

Implements actual measurement functions that connect the observational
framework (Session 33) to real SAGE consciousness metrics rather than
simulated data.

This bridges:
- Quality metrics (Session 27)
- Epistemic states (Session 30-31)
- Temporal adaptation (Session 17-29)
- Federation infrastructure (Session 32)

With the observational predictions framework (Session 33).

Author: Thor (Autonomous Session 34)
Date: 2025-12-11
Hardware: Jetson AGX Thor
"""

import time
import statistics
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from sage.core.quality_metrics import score_response_quality, QualityScore
from sage.core.epistemic_states import (
    EpistemicMetrics,
    EpistemicStateTracker,
    EpistemicState
)
from sage.core.sage_observational_framework import ObservationResult


class SAGERealMeasurements:
    """
    Real measurement implementation for SAGE observational predictions.

    Provides actual measurements using SAGE consciousness infrastructure
    rather than simulated data.
    """

    def __init__(self):
        """Initialize real measurement system"""
        self.epistemic_tracker = EpistemicStateTracker(history_size=1000)
        self.quality_history: List[QualityScore] = []
        self.adaptation_history: List[Dict] = []

    # ========================================================================
    # Quality & Performance Measurements (Q1-Q5)
    # ========================================================================

    def measure_response_quality(self,
                                responses: List[str],
                                questions: Optional[List[str]] = None) -> ObservationResult:
        """
        Measure Q1: Response quality threshold.

        Uses actual quality_metrics.score_response_quality() on real responses.

        Prediction: ≥85% of responses score ≥0.85 normalized quality

        Args:
            responses: List of actual SAGE responses to evaluate
            questions: Optional corresponding questions

        Returns:
            ObservationResult with proportion exceeding 0.85 threshold
        """
        if not responses:
            return ObservationResult(
                prediction_id='Q1',
                observed_value=0.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="No responses provided for measurement"
            )

        # Score all responses using actual quality metrics
        scores = []
        for i, response in enumerate(responses):
            question = questions[i] if questions and i < len(questions) else None
            score = score_response_quality(response, question)
            scores.append(score.normalized)
            self.quality_history.append(score)

        # Calculate proportion meeting threshold
        threshold = 0.85
        exceeding_threshold = sum(1 for s in scores if s >= threshold)
        proportion = exceeding_threshold / len(scores)

        # Error estimate: Binomial standard error
        error = np.sqrt(proportion * (1 - proportion) / len(scores))

        # Calculate mean quality as additional metric
        mean_quality = statistics.mean(scores)
        std_quality = statistics.stdev(scores) if len(scores) > 1 else 0.0

        return ObservationResult(
            prediction_id='Q1',
            observed_value=proportion,
            observed_error=error,
            significance=0.0,  # Will be calculated by framework
            validated=False,   # Will be determined by framework
            measurement_time=time.time(),
            sample_size=len(scores),
            notes=f"Mean quality: {mean_quality:.3f} ± {std_quality:.3f}, "
                  f"{exceeding_threshold}/{len(scores)} exceed threshold"
        )

    def measure_epistemic_accuracy(self,
                                   predictions: List[Tuple[EpistemicState, EpistemicMetrics]],
                                   ground_truth: List[EpistemicState]) -> ObservationResult:
        """
        Measure Q2: Epistemic state accuracy.

        Compares predicted epistemic states from metrics.primary_state()
        against ground truth labels.

        Prediction: ≥66% accuracy (4/6 states correctly identified)

        Args:
            predictions: List of (predicted_state, metrics) tuples
            ground_truth: List of true epistemic states

        Returns:
            ObservationResult with accuracy measurement
        """
        if not predictions or len(predictions) != len(ground_truth):
            return ObservationResult(
                prediction_id='Q2',
                observed_value=0.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="Invalid prediction/ground truth data"
            )

        # Calculate accuracy
        correct = sum(1 for (pred_state, _), true_state in zip(predictions, ground_truth)
                     if pred_state == true_state)
        accuracy = correct / len(predictions)

        # Binomial error
        error = np.sqrt(accuracy * (1 - accuracy) / len(predictions))

        # Confusion matrix summary
        state_counts = {}
        for (pred_state, _), true_state in zip(predictions, ground_truth):
            key = (true_state.value, pred_state.value)
            state_counts[key] = state_counts.get(key, 0) + 1

        return ObservationResult(
            prediction_id='Q2',
            observed_value=accuracy,
            observed_error=error,
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(predictions),
            notes=f"{correct}/{len(predictions)} correct, "
                  f"Confusion patterns: {len(state_counts)} unique (pred,true) pairs"
        )

    def measure_weight_stability(self,
                                weight_history: List[Dict[str, float]],
                                stable_period_start: int) -> ObservationResult:
        """
        Measure Q3: Adaptive weight stability.

        Calculates volatility (standard deviation) of weights during
        stable conditions (after convergence).

        Prediction: Weight volatility < 0.025 in stable periods

        Args:
            weight_history: List of weight dictionaries {objective: weight}
            stable_period_start: Index where stable period begins

        Returns:
            ObservationResult with weight volatility measurement
        """
        if not weight_history or stable_period_start >= len(weight_history):
            return ObservationResult(
                prediction_id='Q3',
                observed_value=0.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="Insufficient weight history"
            )

        # Extract stable period
        stable_weights = weight_history[stable_period_start:]

        # Calculate volatility for each objective
        objectives = list(stable_weights[0].keys())
        volatilities = []

        for obj in objectives:
            values = [w[obj] for w in stable_weights]
            vol = statistics.stdev(values) if len(values) > 1 else 0.0
            volatilities.append(vol)

        # Mean volatility across objectives
        mean_volatility = statistics.mean(volatilities)
        std_volatility = statistics.stdev(volatilities) if len(volatilities) > 1 else 0.0

        return ObservationResult(
            prediction_id='Q3',
            observed_value=mean_volatility,
            observed_error=std_volatility / np.sqrt(len(volatilities)),
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(stable_weights),
            notes=f"Volatility range: {min(volatilities):.4f} - {max(volatilities):.4f}, "
                  f"{len(objectives)} objectives tracked"
        )

    def measure_multi_objective_fitness(self,
                                       fitness_history: List[float],
                                       convergence_idx: Optional[int] = None) -> ObservationResult:
        """
        Measure Q4: Multi-objective fitness.

        Measures sustained fitness level after convergence.

        Prediction: Fitness ≥0.83 (combined multi-objective score)

        Args:
            fitness_history: List of fitness scores over time
            convergence_idx: Optional index where convergence occurred

        Returns:
            ObservationResult with fitness measurement
        """
        if not fitness_history:
            return ObservationResult(
                prediction_id='Q4',
                observed_value=0.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="No fitness history"
            )

        # Use converged period if specified, otherwise last 20% of history
        if convergence_idx is not None and convergence_idx < len(fitness_history):
            relevant_fitness = fitness_history[convergence_idx:]
        else:
            start_idx = int(len(fitness_history) * 0.8)
            relevant_fitness = fitness_history[start_idx:]

        if not relevant_fitness:
            relevant_fitness = fitness_history[-10:]  # Last 10 as fallback

        mean_fitness = statistics.mean(relevant_fitness)
        std_fitness = statistics.stdev(relevant_fitness) if len(relevant_fitness) > 1 else 0.0
        error = std_fitness / np.sqrt(len(relevant_fitness))

        return ObservationResult(
            prediction_id='Q4',
            observed_value=mean_fitness,
            observed_error=error,
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(relevant_fitness),
            notes=f"Peak fitness: {max(fitness_history):.3f}, "
                  f"Min in period: {min(relevant_fitness):.3f}"
        )

    def measure_convergence_time(self,
                                fitness_history: List[float],
                                convergence_threshold: float = 0.02) -> ObservationResult:
        """
        Measure Q5: Temporal adaptation convergence time.

        Measures cycles until fitness stabilizes (convergence).

        Prediction: Convergence < 1000 cycles

        Args:
            fitness_history: List of fitness scores over time
            convergence_threshold: Threshold for considering fitness stable

        Returns:
            ObservationResult with convergence time in cycles
        """
        if len(fitness_history) < 10:
            return ObservationResult(
                prediction_id='Q5',
                observed_value=len(fitness_history),
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=len(fitness_history),
                notes="Insufficient history for convergence detection"
            )

        # Detect convergence: when rolling std dev < threshold
        window = 10
        convergence_cycle = None

        for i in range(window, len(fitness_history)):
            window_data = fitness_history[i-window:i]
            std_dev = statistics.stdev(window_data)

            if std_dev < convergence_threshold:
                convergence_cycle = i
                break

        if convergence_cycle is None:
            # No convergence detected
            convergence_cycle = len(fitness_history)
            notes = f"No convergence detected, using full history length"
        else:
            final_fitness = statistics.mean(fitness_history[convergence_cycle:])
            notes = f"Converged at cycle {convergence_cycle}, final fitness {final_fitness:.3f}"

        # Error estimate based on window size uncertainty
        error = window * 2  # ±2 windows of uncertainty

        return ObservationResult(
            prediction_id='Q5',
            observed_value=float(convergence_cycle),
            observed_error=float(error),
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(fitness_history),
            notes=notes
        )

    # ========================================================================
    # Efficiency & Resource Measurements (E1-E4)
    # ========================================================================

    def measure_efficiency_gain(self,
                               multi_obj_performance: List[float],
                               single_obj_baseline: List[float]) -> ObservationResult:
        """
        Measure E1: ATP utilization efficiency gain.

        Compares multi-objective vs single-objective performance.

        Prediction: +200% efficiency (2.0x improvement)

        Args:
            multi_obj_performance: Performance metrics from multi-objective
            single_obj_baseline: Performance metrics from single-objective baseline

        Returns:
            ObservationResult with efficiency gain multiplier
        """
        if not multi_obj_performance or not single_obj_baseline:
            return ObservationResult(
                prediction_id='E1',
                observed_value=1.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="Insufficient data for comparison"
            )

        multi_mean = statistics.mean(multi_obj_performance)
        single_mean = statistics.mean(single_obj_baseline)

        if single_mean == 0:
            return ObservationResult(
                prediction_id='E1',
                observed_value=1.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=len(multi_obj_performance),
                notes="Baseline is zero, cannot compute ratio"
            )

        efficiency_gain = multi_mean / single_mean

        # Error propagation for ratio
        multi_std = statistics.stdev(multi_obj_performance) if len(multi_obj_performance) > 1 else 0.0
        single_std = statistics.stdev(single_obj_baseline) if len(single_obj_baseline) > 1 else 0.0

        # Relative error in ratio
        rel_error = np.sqrt((multi_std/multi_mean)**2 + (single_std/single_mean)**2)
        error = efficiency_gain * rel_error

        return ObservationResult(
            prediction_id='E1',
            observed_value=efficiency_gain,
            observed_error=error,
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(multi_obj_performance),
            notes=f"Multi-obj: {multi_mean:.3f}, Single-obj: {single_mean:.3f}"
        )

    def measure_epistemic_overhead(self,
                                  epistemic_times_ms: List[float]) -> ObservationResult:
        """
        Measure E2: Epistemic tracking computational overhead.

        Measures time spent on epistemic state calculations.

        Prediction: Overhead < 5 ms/turn average

        Args:
            epistemic_times_ms: List of epistemic calculation times in milliseconds

        Returns:
            ObservationResult with mean overhead time
        """
        if not epistemic_times_ms:
            return ObservationResult(
                prediction_id='E2',
                observed_value=0.0,
                observed_error=0.0,
                significance=0.0,
                validated=False,
                measurement_time=time.time(),
                sample_size=0,
                notes="No timing data"
            )

        mean_time = statistics.mean(epistemic_times_ms)
        std_time = statistics.stdev(epistemic_times_ms) if len(epistemic_times_ms) > 1 else 0.0
        error = std_time / np.sqrt(len(epistemic_times_ms))

        return ObservationResult(
            prediction_id='E2',
            observed_value=mean_time,
            observed_error=error,
            significance=0.0,
            validated=False,
            measurement_time=time.time(),
            sample_size=len(epistemic_times_ms),
            notes=f"Min: {min(epistemic_times_ms):.3f} ms, "
                  f"Max: {max(epistemic_times_ms):.3f} ms, "
                  f"P95: {np.percentile(epistemic_times_ms, 95):.3f} ms"
        )

    # Additional measurement functions would follow same pattern...
    # (E3-E4, M1-M4, F1-F3, U1-U2)

    def get_measurement_summary(self) -> Dict:
        """
        Get summary of measurement system state.

        Returns:
            Dictionary with measurement system statistics
        """
        return {
            'quality_samples': len(self.quality_history),
            'epistemic_samples': len(self.epistemic_tracker.history),
            'adaptation_samples': len(self.adaptation_history),
            'last_measurement': time.time()
        }


# ============================================================================
# Helper Functions for Real Measurements
# ============================================================================

def estimate_epistemic_metrics_from_response(response: str,
                                            quality_score: Optional[QualityScore] = None) -> EpistemicMetrics:
    """
    Estimate epistemic metrics from response text and quality score.

    This is a heuristic estimator used when actual epistemic tracking
    isn't available. Real measurements should use actual EpistemicStateTracker data.

    Args:
        response: Response text to analyze
        quality_score: Optional pre-computed quality score

    Returns:
        Estimated EpistemicMetrics
    """
    if quality_score is None:
        quality_score = score_response_quality(response)

    # Heuristic mappings from quality to epistemic metrics
    # High quality → high confidence & comprehension
    confidence = quality_score.normalized * 0.8 + 0.1  # 0.1-0.9 range
    comprehension_depth = quality_score.normalized * 0.7 + 0.2  # 0.2-0.9 range

    # Detect uncertainty markers
    uncertainty_markers = ['maybe', 'possibly', 'might', 'could', 'uncertain', 'unclear']
    uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response.lower())
    uncertainty = min(uncertainty_count * 0.15, 0.9)  # Cap at 0.9

    # Coherence from quality (avoids hedging → coherent)
    coherence = 0.7 if quality_score.avoids_hedging else 0.4

    # Frustration inversely related to quality (when quality is low)
    frustration = max(0.0, (1.0 - quality_score.normalized - 0.3)) * 1.43  # Scale to 0-1

    return EpistemicMetrics(
        confidence=confidence,
        comprehension_depth=comprehension_depth,
        uncertainty=uncertainty,
        coherence=coherence,
        frustration=frustration
    )


def analyze_conversation_quality(conversation_history: List[Tuple[str, str]]) -> ObservationResult:
    """
    Analyze quality across a full conversation.

    Convenience function for measuring Q1 on conversation history.

    Args:
        conversation_history: List of (question, response) tuples

    Returns:
        ObservationResult for Q1 prediction
    """
    measurer = SAGERealMeasurements()

    questions = [q for q, _ in conversation_history]
    responses = [r for _, r in conversation_history]

    return measurer.measure_response_quality(responses, questions)
