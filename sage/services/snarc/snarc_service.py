"""
SNARC System Service - "Sensor of Sensors"

Main orchestration class for 5D salience assessment.
Integrates all detectors and provides attention recommendations to SAGE kernel.
"""

from typing import Dict, List, Any, Optional
import time

from .data_structures import (
    SalienceReport,
    SalienceBreakdown,
    CognitiveStance,
    Outcome,
    SensorOutput,
    SNARCMemory
)
from .detectors import (
    SurpriseDetector,
    NoveltyDetector,
    ArousalDetector,
    RewardEstimator,
    ConflictDetector
)


class SNARCService:
    """
    SNARC System Service - Salience Assessment

    Observes entire sensor field (IRP stack outputs),
    computes 5D salience (Surprise, Novelty, Arousal, Reward, Conflict),
    recommends attention allocation to SAGE kernel.
    """

    def __init__(
        self,
        salience_weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.1
    ):
        """
        Initialize SNARC service

        Args:
            salience_weights: Weights for each salience dimension.
                            If None, uses equal weights (0.2 each).
            learning_rate: How fast to adapt weights from outcomes (0-1)
        """
        # Default equal weights
        if salience_weights is None:
            salience_weights = {
                'surprise': 0.2,
                'novelty': 0.2,
                'arousal': 0.2,
                'reward': 0.2,
                'conflict': 0.2
            }

        self.salience_weights = salience_weights
        self.learning_rate = learning_rate

        # Initialize all detectors
        self.surprise_detector = SurpriseDetector()
        self.novelty_detector = NoveltyDetector()
        self.arousal_detector = ArousalDetector()
        self.reward_estimator = RewardEstimator()
        self.conflict_detector = ConflictDetector()

        # Memory of past assessments and outcomes
        self.assessment_history: List[SNARCMemory] = []

    def assess_salience(
        self,
        sensor_outputs: Dict[str, Any]
    ) -> SalienceReport:
        """
        Compute 5D salience across sensor field

        Args:
            sensor_outputs: Dict mapping sensor_id -> sensor_output
                          Can be raw data or SensorOutput objects

        Returns:
            SalienceReport with attention recommendation
        """
        # Convert to SensorOutput if needed
        standardized_outputs = self._standardize_sensor_outputs(sensor_outputs)

        # Compute salience for each sensor
        sensor_salience = {}

        for sensor_id, sensor_output in standardized_outputs.items():
            # Extract data from SensorOutput
            data = sensor_output.data if hasattr(sensor_output, 'data') else sensor_output

            # Compute each dimension
            surprise = self.surprise_detector.compute(data, sensor_id)
            novelty = self.novelty_detector.compute(data, sensor_id)
            arousal = self.arousal_detector.compute(data, sensor_id)
            reward = self.reward_estimator.compute(data, sensor_id)
            conflict = self.conflict_detector.compute(
                {sid: (sout.data if hasattr(sout, 'data') else sout)
                 for sid, sout in standardized_outputs.items()},
                sensor_id
            )

            # Create breakdown
            breakdown = SalienceBreakdown(
                surprise=surprise,
                novelty=novelty,
                arousal=arousal,
                reward=reward,
                conflict=conflict
            )

            # Compute total salience
            total = breakdown.total(self.salience_weights)

            sensor_salience[sensor_id] = {
                'breakdown': breakdown,
                'total': total,
                'output': sensor_output
            }

        # Find highest salience sensor
        if not sensor_salience:
            # No sensors - return neutral report
            return self._neutral_report()

        focus_target_id = max(
            sensor_salience.keys(),
            key=lambda sid: sensor_salience[sid]['total']
        )

        focus_salience = sensor_salience[focus_target_id]

        # Suggest cognitive stance based on salience pattern
        suggested_stance = self._suggest_stance(focus_salience['breakdown'])

        # Retrieve relevant memories
        relevant_memories = self._retrieve_relevant_memories(
            focus_salience['breakdown'],
            top_k=5
        )

        # Compute confidence
        confidence = self._compute_confidence(focus_salience['breakdown'])

        # Create report
        report = SalienceReport(
            focus_target=focus_target_id,
            salience_score=focus_salience['total'],
            salience_breakdown=focus_salience['breakdown'],
            suggested_stance=suggested_stance,
            relevant_memories=relevant_memories,
            confidence=confidence,
            metadata={
                'timestamp': time.time(),
                'num_sensors': len(sensor_outputs),
                'all_sensor_salience': {
                    sid: data['total']
                    for sid, data in sensor_salience.items()
                }
            }
        )

        # Store assessment
        self._store_assessment(report, standardized_outputs)

        return report

    def update_from_outcome(self, assessment: SalienceReport, outcome: Outcome):
        """
        Learn from outcome feedback

        Adjusts salience weights based on whether high-salience
        events led to useful actions.

        Args:
            assessment: The salience assessment that was made
            outcome: Result of action taken
        """
        # Find corresponding memory entry
        for memory in reversed(self.assessment_history):
            if memory.assessment == assessment:
                memory.outcome = outcome
                break

        # Update weights if outcome was negative
        if not outcome.success or outcome.reward < 0.5:
            # This salience pattern didn't lead to good outcome
            # Slightly reduce weights of highest contributors
            breakdown = assessment.salience_breakdown

            # Find which dimensions were highest
            dimensions = {
                'surprise': breakdown.surprise,
                'novelty': breakdown.novelty,
                'arousal': breakdown.arousal,
                'reward': breakdown.reward,
                'conflict': breakdown.conflict
            }

            max_dim = max(dimensions, key=dimensions.get)

            # Reduce weight slightly, redistribute to others
            old_weight = self.salience_weights[max_dim]
            reduction = old_weight * self.learning_rate * 0.1  # Small adjustment

            self.salience_weights[max_dim] -= reduction

            # Redistribute to others
            other_dims = [d for d in dimensions.keys() if d != max_dim]
            for dim in other_dims:
                self.salience_weights[dim] += reduction / len(other_dims)

        # Update reward estimator with outcome
        if assessment.focus_target:
            # Find the sensor output that was focused on
            for memory in reversed(self.assessment_history):
                if memory.assessment == assessment:
                    if assessment.focus_target in memory.sensor_snapshot:
                        sensor_output = memory.sensor_snapshot[assessment.focus_target]
                        self.reward_estimator.update_from_outcome(
                            sensor_output,
                            assessment.focus_target,
                            outcome.reward
                        )
                    break

    def _standardize_sensor_outputs(
        self,
        sensor_outputs: Dict[str, Any]
    ) -> Dict[str, SensorOutput]:
        """Convert raw outputs to SensorOutput objects if needed"""
        standardized = {}

        for sensor_id, output in sensor_outputs.items():
            if isinstance(output, SensorOutput):
                standardized[sensor_id] = output
            else:
                # Wrap in SensorOutput
                standardized[sensor_id] = SensorOutput(
                    sensor_id=sensor_id,
                    timestamp=time.time(),
                    data=output,
                    sensor_type="unknown"
                )

        return standardized

    def _suggest_stance(self, breakdown: SalienceBreakdown) -> CognitiveStance:
        """
        Suggest cognitive stance based on salience pattern

        Different patterns suggest different approaches:
        - High surprise + novelty → CURIOUS_UNCERTAINTY (explore)
        - Low surprise + novelty, high reward → CONFIDENT_EXECUTION (exploit)
        - High conflict → SKEPTICAL_VERIFICATION (verify)
        - High arousal + moderate novelty → EXPLORATORY (investigate)
        - High reward → FOCUSED_ATTENTION (pursue goal)
        """
        # Normalize all values for comparison
        dims = {
            'surprise': breakdown.surprise,
            'novelty': breakdown.novelty,
            'arousal': breakdown.arousal,
            'reward': breakdown.reward,
            'conflict': breakdown.conflict
        }

        # Check for high conflict first (safety)
        if dims['conflict'] > 0.7:
            return CognitiveStance.SKEPTICAL_VERIFICATION

        # High surprise AND novelty → curious exploration
        if dims['surprise'] > 0.6 and dims['novelty'] > 0.6:
            return CognitiveStance.CURIOUS_UNCERTAINTY

        # High arousal with moderate novelty → active exploration
        if dims['arousal'] > 0.7 and dims['novelty'] > 0.4:
            return CognitiveStance.EXPLORATORY

        # High reward → focused goal pursuit
        if dims['reward'] > 0.7:
            return CognitiveStance.FOCUSED_ATTENTION

        # Low surprise and novelty → confident execution
        if dims['surprise'] < 0.3 and dims['novelty'] < 0.3:
            return CognitiveStance.CONFIDENT_EXECUTION

        # Default: curious but not urgent
        return CognitiveStance.CURIOUS_UNCERTAINTY

    def _compute_confidence(self, breakdown: SalienceBreakdown) -> float:
        """
        Compute confidence in salience assessment

        Higher confidence when:
        - Dimensions agree (all high or all low)
        - Have enough history to normalize properly
        """
        # Variance in dimensions
        dims = [
            breakdown.surprise,
            breakdown.novelty,
            breakdown.arousal,
            breakdown.reward,
            breakdown.conflict
        ]

        import numpy as np
        variance = np.var(dims)

        # Low variance = dimensions agree = high confidence
        # High variance = mixed signals = low confidence
        confidence = 1.0 - min(variance * 2.0, 1.0)

        return float(confidence)

    def _retrieve_relevant_memories(
        self,
        breakdown: SalienceBreakdown,
        top_k: int = 5
    ) -> List[SNARCMemory]:
        """
        Retrieve memories with similar salience patterns

        Useful for kernel to understand: "I've seen this pattern before"
        """
        if not self.assessment_history:
            return []

        # Compute similarity to each past assessment
        similarities = []
        for memory in self.assessment_history:
            past_breakdown = memory.assessment.salience_breakdown
            similarity = self._breakdown_similarity(breakdown, past_breakdown)
            similarities.append((similarity, memory))

        # Sort by similarity
        similarities.sort(reverse=True, key=lambda x: x[0])

        # Return top k
        return [memory for _, memory in similarities[:top_k]]

    def _breakdown_similarity(
        self,
        breakdown1: SalienceBreakdown,
        breakdown2: SalienceBreakdown
    ) -> float:
        """
        Compute similarity between salience patterns

        Uses cosine similarity of dimension vectors
        """
        import numpy as np

        vec1 = np.array([
            breakdown1.surprise,
            breakdown1.novelty,
            breakdown1.arousal,
            breakdown1.reward,
            breakdown1.conflict
        ])

        vec2 = np.array([
            breakdown2.surprise,
            breakdown2.novelty,
            breakdown2.arousal,
            breakdown2.reward,
            breakdown2.conflict
        ])

        # Cosine similarity
        dot = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)

        if norm_product == 0:
            return 0.0

        cos_sim = dot / norm_product
        return float((cos_sim + 1.0) / 2.0)  # Normalize to 0-1

    def _store_assessment(
        self,
        assessment: SalienceReport,
        sensor_snapshot: Dict[str, SensorOutput]
    ):
        """Store assessment in memory"""
        memory = SNARCMemory(
            assessment=assessment,
            outcome=None,  # Will be filled in later via update_from_outcome
            timestamp=time.time(),
            sensor_snapshot={
                sid: sout.data if hasattr(sout, 'data') else sout
                for sid, sout in sensor_snapshot.items()
            }
        )

        self.assessment_history.append(memory)

        # Limit history size
        if len(self.assessment_history) > 1000:
            self.assessment_history = self.assessment_history[-1000:]

    def _neutral_report(self) -> SalienceReport:
        """Return neutral report when no sensors available"""
        return SalienceReport(
            focus_target="none",
            salience_score=0.0,
            salience_breakdown=SalienceBreakdown(
                surprise=0.0,
                novelty=0.0,
                arousal=0.0,
                reward=0.0,
                conflict=0.0
            ),
            suggested_stance=CognitiveStance.CONFIDENT_EXECUTION,
            confidence=0.0
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics"""
        successful_outcomes = sum(
            1 for m in self.assessment_history
            if m.outcome and m.was_useful()
        )

        return {
            'num_assessments': len(self.assessment_history),
            'successful_outcomes': successful_outcomes,
            'success_rate': (
                successful_outcomes / len(self.assessment_history)
                if self.assessment_history else 0.0
            ),
            'current_weights': self.salience_weights.copy()
        }

    def reset(self):
        """Reset all detectors and history"""
        self.surprise_detector.reset_all()
        self.novelty_detector.reset_all()
        self.arousal_detector.reset_all()
        self.reward_estimator.reset_all()
        self.conflict_detector.reset_all()
        self.assessment_history.clear()
