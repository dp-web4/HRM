#!/usr/bin/env python3
"""
Multi-Sensor Attention Test
Experiment: Does SNARC learn to prioritize based on reward?

Setup:
- Three sensors simultaneously active
- Sensor A: High rewards (0.9)
- Sensor B: Low rewards (0.1)
- Sensor C: Variable rewards (random)

Questions:
- Does SNARC shift focus to high-reward sensors?
- How quickly does it learn prioritization?
- Does variable reward maintain attention?
- What happens to SNARC weights over time?
"""

import sys
import os
from pathlib import Path
import random

hrm_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(hrm_root))
os.chdir(hrm_root)

from sage.core.sage_kernel import SAGEKernel, ExecutionResult
from sage.services.snarc.data_structures import CognitiveStance

class HighRewardSensor:
    """Sensor that consistently produces high rewards"""
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return {'type': 'high_reward', 'value': self.value}

class LowRewardSensor:
    """Sensor that consistently produces low rewards"""
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return {'type': 'low_reward', 'value': self.value}

class VariableRewardSensor:
    """Sensor with random reward outcomes"""
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1
        return {'type': 'variable_reward', 'value': self.value}

def high_reward_handler(observation, stance):
    """Always returns high reward"""
    return ExecutionResult(
        success=True,
        reward=0.9,  # High reward
        description=f"High reward sensor: {observation['value']}",
        outputs=observation
    )

def low_reward_handler(observation, stance):
    """Always returns low reward"""
    return ExecutionResult(
        success=True,
        reward=0.1,  # Low reward
        description=f"Low reward sensor: {observation['value']}",
        outputs=observation
    )

def variable_reward_handler(observation, stance):
    """Returns random reward"""
    reward = random.random()
    return ExecutionResult(
        success=True,
        reward=reward,
        description=f"Variable reward sensor: {observation['value']} (reward={reward:.2f})",
        outputs=observation
    )

def main():
    print("=" * 70)
    print("MULTI-SENSOR ATTENTION TEST")
    print("=" * 70)
    print("\nThree sensors competing for attention:")
    print("  A) high_reward  → Always gives 0.9 reward")
    print("  B) low_reward   → Always gives 0.1 reward")
    print("  C) variable     → Random reward (0.0-1.0)")
    print("\nQuestion: Does SNARC learn to focus on high-reward sensor?")
    print()

    # Create sensors
    sensor_a = HighRewardSensor()
    sensor_b = LowRewardSensor()
    sensor_c = VariableRewardSensor()

    # Setup kernel
    sensor_sources = {
        'high_reward': sensor_a,
        'low_reward': sensor_b,
        'variable_reward': sensor_c
    }

    action_handlers = {
        'high_reward': high_reward_handler,
        'low_reward': low_reward_handler,
        'variable_reward': variable_reward_handler
    }

    kernel = SAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )

    print("Starting kernel with THREE active sensors...")
    print("Observing attention allocation...\n")

    # Run for more cycles to see learning
    kernel.run(max_cycles=50, cycle_delay=0.05)

    # Analysis
    history = kernel.get_history()

    print("\n" + "=" * 70)
    print("ATTENTION ANALYSIS")
    print("=" * 70 + "\n")

    # Count focus distribution
    focus_counts = {}
    for h in history:
        target = h['focus_target']
        focus_counts[target] = focus_counts.get(target, 0) + 1

    print("Focus distribution:")
    for sensor, count in sorted(focus_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / len(history)
        print(f"  {sensor:15s}: {count:2d} cycles ({pct:5.1f}%)")

    # Reward by sensor
    print("\nAverage reward by sensor:")
    reward_by_sensor = {s: [] for s in sensor_sources.keys()}
    for h in history:
        sensor = h['focus_target']
        reward = h['result'].reward
        reward_by_sensor[sensor].append(reward)

    for sensor, rewards in sorted(reward_by_sensor.items()):
        if rewards:
            avg = sum(rewards) / len(rewards)
            print(f"  {sensor:15s}: {avg:.3f}")

    # Evolution over time
    print("\nFocus evolution (first 10 vs last 10):")

    first_10 = history[:10]
    last_10 = history[-10:]

    def count_focus(subset):
        counts = {}
        for h in subset:
            counts[h['focus_target']] = counts.get(h['focus_target'], 0) + 1
        return counts

    first_counts = count_focus(first_10)
    last_counts = count_focus(last_10)

    print("\n  First 10 cycles:")
    for sensor in sensor_sources.keys():
        count = first_counts.get(sensor, 0)
        print(f"    {sensor:15s}: {count}")

    print("\n  Last 10 cycles:")
    for sensor in sensor_sources.keys():
        count = last_counts.get(sensor, 0)
        print(f"    {sensor:15s}: {count}")

    # Salience trends
    print("\nSalience scores by sensor:")
    salience_by_sensor = {s: [] for s in sensor_sources.keys()}
    for h in history:
        sensor = h['focus_target']
        salience = h['salience_score']
        salience_by_sensor[sensor].append(salience)

    for sensor, scores in sorted(salience_by_sensor.items()):
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  {sensor:15s}: {avg:.3f}")

    print("\n" + "=" * 70)
    print("\nKEY FINDINGS:")
    print("1. Did focus shift to high-reward sensor?")
    print("2. How quickly did adaptation occur?")
    print("3. What about variable reward - interesting or ignored?")
    print("4. Did SNARC weights evolve meaningfully?")
    print()

if __name__ == "__main__":
    main()
