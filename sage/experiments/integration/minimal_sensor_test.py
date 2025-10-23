#!/usr/bin/env python3
"""
Minimal Sensor Test
Experiment: How does SNARC respond to different minimal stimuli?

Three sensors, each testing a different aspect:
1. Time sensor - Completely predictable (always increasing)
2. Noise sensor - Completely unpredictable (random)
3. Heartbeat sensor - Predictable rhythm (0, 1, 0, 1...)

Questions:
- Does SNARC recognize predictability and reduce salience?
- Does it recognize novelty in random data?
- Does it detect patterns in rhythmic data?
- How quickly does trust evolve?
"""

import sys
import os
from pathlib import Path
import time
import random

# Add sage to path
hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.core.sage_kernel import SAGEKernel, ExecutionResult

class TimeSensor:
    """Returns current timestamp - perfectly predictable"""
    def __call__(self):
        return {'type': 'time', 'value': time.time()}

class NoiseSensor:
    """Returns random value - unpredictable"""
    def __call__(self):
        return {'type': 'noise', 'value': random.random()}

class HeartbeatSensor:
    """Returns alternating 0/1 - predictable rhythm"""
    def __init__(self):
        self.state = 0

    def __call__(self):
        self.state = 1 - self.state  # Toggle
        return {'type': 'heartbeat', 'value': self.state}

def simple_action_handler(observation, stance):
    """Log what we observed and what stance SNARC suggested"""
    obs_type = observation.get('type', 'unknown')
    obs_value = observation.get('value', None)

    return ExecutionResult(
        success=True,
        reward=0.5,  # Neutral reward
        description=f"Observed {obs_type}={obs_value:.4f}, stance={stance.value}",
        outputs={'observation': observation, 'stance': stance.value}
    )

def test_sensor(sensor_name, sensor_callable, num_cycles=30):
    """Test a single sensor and observe SNARC behavior"""
    print("\n" + "=" * 70)
    print(f"TESTING: {sensor_name.upper()}")
    print("=" * 70 + "\n")

    # Create kernel with this sensor
    sensor_sources = {sensor_name: sensor_callable}
    action_handlers = {sensor_name: simple_action_handler}

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    # Run
    kernel.run(max_cycles=num_cycles, cycle_delay=0.1)

    # Analyze history
    history = kernel.get_history()

    print("\n## ANALYSIS ##\n")

    if history:
        # Salience over time
        saliences = [h.get('salience_score', 0) for h in history]
        print(f"Salience evolution:")
        print(f"  First 5: {saliences[:5]}")
        print(f"  Last 5:  {saliences[-5:]}")
        print(f"  Average first half: {sum(saliences[:len(saliences)//2])/len(saliences[:len(saliences)//2]):.3f}")
        print(f"  Average second half: {sum(saliences[len(saliences)//2:])/len(saliences[len(saliences)//2:]):.3f}")

        # Stance distribution
        stances = [h.get('stance', 'unknown') for h in history]
        stance_counts = {}
        for s in stances:
            stance_counts[s] = stance_counts.get(s, 0) + 1

        print(f"\nStance distribution:")
        for stance, count in sorted(stance_counts.items(), key=lambda x: -x[1]):
            pct = 100 * count / len(stances)
            print(f"  {stance}: {count} ({pct:.1f}%)")

        # Rewards
        rewards = [h['result'].reward for h in history]
        print(f"\nAverage reward: {sum(rewards)/len(rewards):.3f}")
    else:
        print("No history recorded!")

    print("\n" + "=" * 70 + "\n")

    return kernel

def main():
    print("=" * 70)
    print("MINIMAL SENSOR EXPERIMENTS")
    print("=" * 70)
    print("\nTesting SNARC's response to three minimal stimuli:\n")
    print("1. Time sensor - Predictable (always increasing)")
    print("2. Noise sensor - Unpredictable (random)")
    print("3. Heartbeat sensor - Rhythmic (0, 1, 0, 1...)")
    print("\nLet's see what SNARC discovers...\n")

    # Test 1: Time sensor
    time_sensor = TimeSensor()
    kernel_time = test_sensor('time', time_sensor, num_cycles=30)

    # Test 2: Noise sensor
    noise_sensor = NoiseSensor()
    kernel_noise = test_sensor('noise', noise_sensor, num_cycles=30)

    # Test 3: Heartbeat sensor
    heartbeat_sensor = HeartbeatSensor()
    kernel_heartbeat = test_sensor('heartbeat', heartbeat_sensor, num_cycles=30)

    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print("=" * 70)
    print("\nKey Questions:")
    print("1. Did SNARC reduce salience for predictable inputs?")
    print("2. Did it maintain high salience for random inputs?")
    print("3. Did it detect the heartbeat pattern?")
    print("4. How did stance evolve in each case?")
    print("5. What does this tell us about adaptation?")
    print()

if __name__ == "__main__":
    main()
