"""
SAGE + SNARC Demonstration

Shows complete orchestration loop:
Sensors → SNARC Salience Assessment → SAGE Kernel → Actions → Learning

This demo creates a multi-sensor environment and runs SAGE kernel
to show how attention allocation works based on salience.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from sage.core import SAGEKernel, MetabolicState
from sage.services.snarc.tests.simulated_sensors import (
    PeriodicSensor,
    StepChangeSensor,
    RandomWalkSensor,
    AnomalySensor,
    MultiSensorEnvironment
)
from sage.services.snarc.data_structures import CognitiveStance
from sage.core.sage_kernel import ExecutionResult


def create_dynamic_environment():
    """
    Create environment with multiple sensor types

    - Temperature: Calm periodic signal
    - Pressure: Sudden step changes (surprising)
    - Motion: Random walk (novel)
    - Alert: Occasional anomalies (high arousal)
    """
    return MultiSensorEnvironment({
        'temperature': PeriodicSensor(
            'temperature',
            frequency=0.5,
            amplitude=2.0,
            noise_level=0.1
        ),
        'pressure': StepChangeSensor(
            'pressure',
            baseline=1.0,
            step_value=5.0,
            step_interval=15,  # Change every 15 steps
            noise_level=0.1
        ),
        'motion': RandomWalkSensor(
            'motion',
            step_size=0.8,
            bounds=(-5, 5)
        ),
        'alert': AnomalySensor(
            'alert',
            baseline_mean=0.0,
            baseline_std=0.5,
            anomaly_probability=0.1,
            anomaly_magnitude=8.0
        )
    })


def create_action_handlers():
    """
    Create custom action handlers for each sensor

    These would be replaced by IRP plugins in full SAGE
    """
    def handle_temperature(data, stance: CognitiveStance) -> ExecutionResult:
        """Temperature control action"""
        if stance == CognitiveStance.CONFIDENT_EXECUTION:
            action = "Maintain current temperature setting"
            reward = 0.5
        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            action = "Investigate temperature fluctuation"
            reward = 0.6
        else:
            action = "Monitor temperature"
            reward = 0.4

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'temp_value': data, 'action': action}
        )

    def handle_pressure(data, stance: CognitiveStance) -> ExecutionResult:
        """Pressure response action"""
        if stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            action = "Verify pressure sensors for malfunction"
            reward = 0.7
        elif stance == CognitiveStance.EXPLORATORY:
            action = "Investigate pressure change source"
            reward = 0.8
        elif abs(data) > 3.0:
            action = "Emergency pressure adjustment"
            reward = 0.9
        else:
            action = "Normal pressure monitoring"
            reward = 0.5

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'pressure_value': data, 'action': action}
        )

    def handle_motion(data, stance: CognitiveStance) -> ExecutionResult:
        """Motion tracking action"""
        if stance == CognitiveStance.FOCUSED_ATTENTION:
            action = "Track motion pattern closely"
            reward = 0.8
        elif stance == CognitiveStance.CURIOUS_UNCERTAINTY:
            action = "Learn new motion pattern"
            reward = 0.7
        else:
            action = "Passive motion monitoring"
            reward = 0.4

        return ExecutionResult(
            success=True,
            reward=reward,
            description=action,
            outputs={'motion_value': data, 'action': action}
        )

    def handle_alert(data, stance: CognitiveStance) -> ExecutionResult:
        """Alert/anomaly response"""
        if abs(data) > 5.0:
            action = "ALERT: Anomaly detected - investigate immediately"
            reward = 1.0
            success = True
        elif stance == CognitiveStance.SKEPTICAL_VERIFICATION:
            action = "Verify alert is not false alarm"
            reward = 0.6
            success = True
        else:
            action = "Normal alert monitoring"
            reward = 0.3
            success = True

        return ExecutionResult(
            success=success,
            reward=reward,
            description=action,
            outputs={'alert_value': data, 'action': action}
        )

    return {
        'temperature': handle_temperature,
        'pressure': handle_pressure,
        'motion': handle_motion,
        'alert': handle_alert
    }


def run_demo(num_cycles: int = 50, verbose: bool = True):
    """
    Run SAGE+SNARC demonstration

    Args:
        num_cycles: Number of inference cycles to run
        verbose: Whether to print detailed logs
    """
    print("=" * 70)
    print("SAGE + SNARC Demonstration")
    print("Sensors → Salience Assessment → Attention Allocation → Action")
    print("=" * 70)

    # Create environment
    env = create_dynamic_environment()

    # Create sensor sources (callables that return current sensor values)
    sensor_sources = {
        sensor_id: (lambda sid=sensor_id: env.sensors[sid].get_sample())
        for sensor_id in env.sensors.keys()
    }

    # Create action handlers
    action_handlers = create_action_handlers()

    # Initialize SAGE kernel
    sage = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=verbose
    )

    print(f"\nRunning {num_cycles} cycles...\n")

    # Run SAGE loop
    sage.run(max_cycles=num_cycles, cycle_delay=0.05)

    return sage


def analyze_results(sage: SAGEKernel):
    """
    Analyze SAGE execution results

    Shows attention allocation patterns and learning
    """
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    history = sage.get_history()

    if not history:
        print("No execution history available")
        return

    # Focus distribution
    focus_counts = {}
    for h in history:
        target = h['focus_target']
        focus_counts[target] = focus_counts.get(target, 0) + 1

    print("\nAttention Distribution:")
    print("(Which sensors received the most attention?)")
    for target, count in sorted(focus_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(history)
        print(f"  {target:15s}: {count:3d} cycles ({pct:5.1f}%)")

    # Average salience per sensor
    salience_by_sensor = {}
    for h in history:
        target = h['focus_target']
        salience = h['salience_score']
        if target not in salience_by_sensor:
            salience_by_sensor[target] = []
        salience_by_sensor[target].append(salience)

    print("\nAverage Salience When Focused:")
    for target in sorted(salience_by_sensor.keys()):
        avg_salience = sum(salience_by_sensor[target]) / len(salience_by_sensor[target])
        print(f"  {target:15s}: {avg_salience:.3f}")

    # Stance evolution
    print("\nCognitive Stance Over Time:")
    stance_timeline = [h['stance'] for h in history]

    # Show first 10 and last 10
    print("  First 10 cycles:", stance_timeline[:10])
    print("  Last 10 cycles:", stance_timeline[-10:])

    # Reward progression
    print("\nReward Progression:")
    rewards = [h['result'].reward for h in history]
    early_avg = sum(rewards[:10]) / 10 if len(rewards) >= 10 else sum(rewards) / len(rewards)
    late_avg = sum(rewards[-10:]) / 10 if len(rewards) >= 10 else sum(rewards) / len(rewards)

    print(f"  Early average (first 10): {early_avg:.3f}")
    print(f"  Late average (last 10): {late_avg:.3f}")
    print(f"  Improvement: {late_avg - early_avg:+.3f}")

    if late_avg > early_avg:
        print("  ✓ System is learning (rewards increasing)")
    else:
        print("  - Rewards stable or decreasing")


if __name__ == '__main__':
    # Run demonstration
    sage_kernel = run_demo(num_cycles=50, verbose=True)

    # Analyze results
    analyze_results(sage_kernel)

    print("\n" + "=" * 70)
    print("Demonstration complete!")
    print("=" * 70)
