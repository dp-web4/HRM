"""
SAGE Kernel - Main Orchestration Loop

Minimal consciousness kernel that demonstrates the SAGE architecture:
- Continuous inference loop
- SNARC-based salience assessment
- Attention allocation
- Resource management
- Outcome learning

This is a simplified version focused on demonstrating the core loop.
Full version will integrate with IRP plugins and metabolic states.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
import time

from ..services.snarc import SNARCService, SalienceReport, Outcome
from ..services.snarc.data_structures import CognitiveStance


class MetabolicState(Enum):
    """Kernel operational states"""
    WAKE = "wake"          # Normal operation
    FOCUS = "focus"        # High attention on specific target
    REST = "rest"          # Low activity, consolidation
    DREAM = "dream"        # Offline learning/memory consolidation
    CRISIS = "crisis"      # Emergency response mode


@dataclass
class ExecutionResult:
    """Result of executing action on focused target"""
    success: bool
    reward: float  # -1.0 to 1.0
    description: str
    outputs: Dict[str, Any]


class SAGEKernel:
    """
    Minimal SAGE Kernel

    Demonstrates core orchestration loop:
    while True:
        observations = gather_sensors()
        salience_report = snarc.assess(observations)
        focus_target = select_target(salience_report)
        result = execute_action(focus_target, salience_report.stance)
        learn_from_outcome(result)
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable[[], Any]],
        action_handlers: Optional[Dict[str, Callable[[Any, CognitiveStance], ExecutionResult]]] = None,
        enable_logging: bool = True
    ):
        """
        Initialize SAGE kernel

        Args:
            sensor_sources: Dict mapping sensor_id -> callable that returns sensor data
                           Example: {'vision': camera.capture, 'audio': mic.sample}
            action_handlers: Dict mapping sensor_id -> callable(data, stance) -> ExecutionResult
                            If None, uses default logging handler
            enable_logging: Whether to print cycle information
        """
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers or {}
        self.enable_logging = enable_logging

        # Initialize SNARC service
        self.snarc = SNARCService()

        # Kernel state
        self.metabolic_state = MetabolicState.WAKE
        self.cycle_count = 0
        self.running = False

        # History
        self.execution_history: List[Dict[str, Any]] = []

    def run(self, max_cycles: Optional[int] = None, cycle_delay: float = 0.1):
        """
        Main execution loop

        Args:
            max_cycles: If specified, runs for N cycles then stops. If None, runs forever.
            cycle_delay: Seconds to wait between cycles
        """
        self.running = True
        self.cycle_count = 0

        try:
            while self.running:
                # Check cycle limit
                if max_cycles is not None and self.cycle_count >= max_cycles:
                    break

                # Run one cycle
                self._cycle()

                # Delay between cycles
                time.sleep(cycle_delay)

                self.cycle_count += 1

        except KeyboardInterrupt:
            if self.enable_logging:
                print("\n[SAGE] Interrupted by user")
        finally:
            self.running = False
            if self.enable_logging:
                print(f"[SAGE] Completed {self.cycle_count} cycles")
                self._print_statistics()

    def stop(self):
        """Stop the execution loop"""
        self.running = False

    def _cycle(self):
        """Execute one SAGE cycle"""
        cycle_start = time.time()

        # 1. GATHER: Collect sensor observations
        observations = self._gather_observations()

        if not observations:
            if self.enable_logging:
                print(f"[SAGE Cycle {self.cycle_count}] No sensor data available")
            return

        # 2. ASSESS: Get salience assessment from SNARC
        salience_report = self.snarc.assess_salience(observations)

        # 3. DECIDE: Select action based on salience and stance
        focus_target = salience_report.focus_target
        suggested_stance = salience_report.suggested_stance
        salience_score = salience_report.salience_score

        if self.enable_logging:
            self._log_cycle_info(salience_report, observations)

        # 4. EXECUTE: Take action on focused target
        result = self._execute_action(
            focus_target,
            observations[focus_target],
            suggested_stance,
            salience_report
        )

        # 5. LEARN: Update SNARC based on outcome
        outcome = Outcome(
            success=result.success,
            reward=result.reward,
            description=result.description
        )
        self.snarc.update_from_outcome(salience_report, outcome)

        # 6. RECORD: Store execution history
        cycle_time = time.time() - cycle_start
        self.execution_history.append({
            'cycle': self.cycle_count,
            'focus_target': focus_target,
            'salience_score': salience_score,
            'stance': suggested_stance.value,
            'result': result,
            'cycle_time': cycle_time
        })

    def _gather_observations(self) -> Dict[str, Any]:
        """Gather observations from all sensors"""
        observations = {}

        for sensor_id, source_fn in self.sensor_sources.items():
            try:
                observation = source_fn()
                observations[sensor_id] = observation
            except Exception as e:
                if self.enable_logging:
                    print(f"[SAGE] Error reading sensor '{sensor_id}': {e}")

        return observations

    def _execute_action(
        self,
        focus_target: str,
        observation: Any,
        stance: CognitiveStance,
        salience_report: SalienceReport
    ) -> ExecutionResult:
        """Execute action on focused target"""

        # Use custom handler if available
        if focus_target in self.action_handlers:
            handler = self.action_handlers[focus_target]
            return handler(observation, stance)

        # Default: Log-only handler
        return self._default_action_handler(focus_target, observation, stance, salience_report)

    def _default_action_handler(
        self,
        focus_target: str,
        observation: Any,
        stance: CognitiveStance,
        salience_report: SalienceReport
    ) -> ExecutionResult:
        """
        Default action handler that just logs

        In full SAGE, this would invoke IRP plugins based on stance
        """
        # Simulate action based on stance
        if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            action = "Verify data integrity"
            reward = 0.3  # Moderate reward for safety check
        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            action = "Explore and learn"
            reward = 0.6  # Good reward for exploration
        elif stance == CognitiveStance.FOCUSED_ATTENTION:
            action = "Pursue goal aggressively"
            reward = 0.8  # High reward for goal pursuit
        elif stance == CognitiveStance.EXPLORATORY:
            action = "Investigate thoroughly"
            reward = 0.7  # Good reward for investigation
        elif stance == CognitiveStance.CONFIDENT_EXECUTION:
            action = "Execute routine efficiently"
            reward = 0.5  # Moderate reward for routine
        else:
            action = "Monitor passively"
            reward = 0.4

        return ExecutionResult(
            success=True,
            reward=reward,
            description=f"{action} on {focus_target}",
            outputs={'action': action}
        )

    def _log_cycle_info(self, report: SalienceReport, observations: Dict[str, Any]):
        """Log cycle information"""
        print(f"\n[SAGE Cycle {self.cycle_count}]")
        print(f"  Sensors: {list(observations.keys())}")
        print(f"  Focus: {report.focus_target}")
        print(f"  Salience: {report.salience_score:.3f}")
        print(f"  Breakdown: S={report.salience_breakdown.surprise:.2f} "
              f"N={report.salience_breakdown.novelty:.2f} "
              f"A={report.salience_breakdown.arousal:.2f} "
              f"R={report.salience_breakdown.reward:.2f} "
              f"C={report.salience_breakdown.conflict:.2f}")
        print(f"  Stance: {report.suggested_stance.value}")
        print(f"  Confidence: {report.confidence:.3f}")

    def _print_statistics(self):
        """Print execution statistics"""
        if not self.execution_history:
            return

        print("\n[SAGE Statistics]")
        print(f"  Total cycles: {len(self.execution_history)}")

        # Average reward
        avg_reward = sum(h['result'].reward for h in self.execution_history) / len(self.execution_history)
        print(f"  Average reward: {avg_reward:.3f}")

        # Average cycle time
        avg_time = sum(h['cycle_time'] for h in self.execution_history) / len(self.execution_history)
        print(f"  Average cycle time: {avg_time*1000:.1f}ms")

        # Stance distribution
        stance_counts = {}
        for h in self.execution_history:
            stance = h['stance']
            stance_counts[stance] = stance_counts.get(stance, 0) + 1

        print("  Stance distribution:")
        for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(self.execution_history)
            print(f"    {stance}: {count} ({pct:.1f}%)")

        # SNARC service stats
        snarc_stats = self.snarc.get_statistics()
        print(f"\n[SNARC Statistics]")
        print(f"  Assessments: {snarc_stats['num_assessments']}")
        print(f"  Successful outcomes: {snarc_stats['successful_outcomes']}")
        print(f"  Success rate: {snarc_stats['success_rate']:.1%}")
        print(f"  Current weights:")
        for dim, weight in snarc_stats['current_weights'].items():
            print(f"    {dim}: {weight:.3f}")

    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history"""
        return self.execution_history.copy()

    def reset(self):
        """Reset kernel state"""
        self.snarc.reset()
        self.execution_history.clear()
        self.cycle_count = 0
        self.metabolic_state = MetabolicState.WAKE
