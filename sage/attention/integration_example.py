#!/usr/bin/env python3
"""
Complete Compression-Action-Threshold Integration Example
===========================================================

Demonstrates the full pattern:
    Layer 1: Multi-dimensional sensor inputs
      ↓ Layer 2: SNARC compression to scalar salience
    Scalar salience [0, 1]
      ↓ Layer 3: Context-dependent threshold
    Binary decision: ATTEND or IGNORE
      ↓ Layer 4: Action (plugin invocation)

Uses:
- Existing SNARCCompressor (sage/core/snarc_compression.py)
- New threshold_decision (sage/attention/threshold_decision.py)

Author: Claude (Sonnet 4.5)
Date: 2025-12-07
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, CompressionMode
from threshold_decision import (
    MetabolicState,
    get_attention_threshold,
    make_attention_decision
)
import random
import time


# =============================================================================
# Simulated Sensors and Plugins
# =============================================================================

class SensorSimulator:
    """Simulates multi-dimensional sensor readings"""

    def __init__(self, name: str, base_salience: float = 0.5):
        self.name = name
        self.base_salience = base_salience
        self.tick = 0

    def read(self) -> dict:
        """Generate sensor data with controlled randomness"""
        self.tick += 1

        # Base values with some variation
        noise = random.uniform(-0.1, 0.1)
        value = 50.0 + 30.0 * (self.base_salience + noise)

        # Occasionally generate urgent/novel events
        urgent = 0
        novelty = self.base_salience + noise

        if random.random() < 0.1:  # 10% chance
            urgent = random.randint(1, 3)
            novelty += 0.3
            value += 20.0

        return {
            'sensor_name': self.name,
            'value': max(0.0, min(100.0, value)),
            'urgent_count': urgent,
            'novelty_score': max(0.0, min(1.0, novelty)),
            'tick': self.tick
        }


class PluginSimulator:
    """Simulates IRP plugin with ATP cost"""

    def __init__(self, name: str, atp_cost: float = 10.0):
        self.name = name
        self.atp_cost = atp_cost
        self.invocation_count = 0

    def invoke(self, sensor_data: dict) -> str:
        """Simulate plugin execution"""
        self.invocation_count += 1
        return f"{self.name} processed sensor {sensor_data['sensor_name']} " \
               f"(value={sensor_data['value']:.1f}, invocation #{self.invocation_count})"


# =============================================================================
# Complete Attention Loop
# =============================================================================

class AttentionLoop:
    """
    Complete compression-action-threshold pattern implementation.

    Orchestrates:
    - Sensor reading (Layer 1)
    - SNARC compression (Layer 2)
    - Threshold computation (Layer 3)
    - Attention decision (Layer 4)
    - Plugin invocation (Action)
    """

    def __init__(
        self,
        sensors: list,
        plugins: dict,
        initial_atp: float = 100.0,
        initial_state: MetabolicState = MetabolicState.WAKE,
        compression_mode: CompressionMode = CompressionMode.LINEAR
    ):
        self.sensors = sensors
        self.plugins = plugins
        self.atp_budget = initial_atp
        self.total_atp = initial_atp
        self.metabolic_state = initial_state
        self.task_criticality = 0.5

        # SNARC compressor (Layer 2)
        self.compressor = SNARCCompressor(compression_mode=compression_mode)

        # Stats
        self.attended_count = 0
        self.ignored_count = 0
        self.cycle_count = 0

    def run_cycle(self) -> dict:
        """
        Execute one complete attention cycle.

        Returns:
            Dict with cycle results
        """
        self.cycle_count += 1
        cycle_results = {
            'cycle': self.cycle_count,
            'state': self.metabolic_state.value,
            'atp': self.atp_budget,
            'decisions': []
        }

        # Layer 1: Read all sensors
        for sensor in self.sensors:
            sensor_data = sensor.read()

            # Layer 2: SNARC compression to salience
            salience, snarc_dims = self.compressor.compute_salience(sensor_data)

            # Layer 3: Context-dependent threshold
            threshold = get_attention_threshold(
                state=self.metabolic_state,
                atp_remaining=self.atp_budget / self.total_atp,
                task_criticality=self.task_criticality
            )

            # Get plugin for this sensor
            plugin = self.plugins.get(sensor.name)
            if not plugin:
                continue

            # Layer 4: Binary decision
            decision = make_attention_decision(
                salience=salience,
                threshold=threshold,
                plugin_name=plugin.name,
                atp_cost=plugin.atp_cost,
                atp_budget=self.atp_budget
            )

            # Record decision
            decision_record = {
                'sensor': sensor.name,
                'salience': salience,
                'threshold': threshold,
                'decision': 'ATTEND' if decision.should_attend else 'IGNORE',
                'reason': decision.reason,
                'snarc': {
                    'surprise': snarc_dims.surprise,
                    'novelty': snarc_dims.novelty,
                    'arousal': snarc_dims.arousal,
                    'reward': snarc_dims.reward,
                    'conflict': snarc_dims.conflict
                }
            }

            # Action: Invoke plugin if attending
            if decision.should_attend:
                result = plugin.invoke(sensor_data)
                decision_record['plugin_result'] = result
                self.atp_budget -= plugin.atp_cost
                self.attended_count += 1
            else:
                self.ignored_count += 1

            cycle_results['decisions'].append(decision_record)

        return cycle_results

    def set_metabolic_state(self, state: MetabolicState):
        """Change metabolic state"""
        self.metabolic_state = state

    def set_criticality(self, criticality: float):
        """Change task criticality"""
        self.task_criticality = max(0.0, min(1.0, criticality))

    def replenish_atp(self, amount: float):
        """Replenish ATP budget"""
        self.atp_budget = min(self.total_atp, self.atp_budget + amount)

    def get_stats(self) -> dict:
        """Get loop statistics"""
        total = self.attended_count + self.ignored_count
        return {
            'cycles': self.cycle_count,
            'attended': self.attended_count,
            'ignored': self.ignored_count,
            'total_decisions': total,
            'attend_rate': self.attended_count / total if total > 0 else 0.0,
            'atp_remaining': self.atp_budget,
            'atp_used': self.total_atp - self.atp_budget
        }


# =============================================================================
# Demo
# =============================================================================

def run_demo():
    """Run complete integration demo"""

    print("=" * 80)
    print("COMPRESSION-ACTION-THRESHOLD INTEGRATION DEMO")
    print("=" * 80)
    print()

    # Create sensors with different base salience levels
    sensors = [
        SensorSimulator("vision", base_salience=0.6),
        SensorSimulator("audio", base_salience=0.4),
        SensorSimulator("tactile", base_salience=0.7),
    ]

    # Create plugins
    plugins = {
        "vision": PluginSimulator("VisionPlugin", atp_cost=15.0),
        "audio": PluginSimulator("AudioPlugin", atp_cost=10.0),
        "tactile": PluginSimulator("TactilePlugin", atp_cost=12.0),
    }

    # Create attention loop
    loop = AttentionLoop(
        sensors=sensors,
        plugins=plugins,
        initial_atp=100.0,
        initial_state=MetabolicState.WAKE,
        compression_mode=CompressionMode.LINEAR
    )

    # Scenario 1: Normal operation (WAKE state)
    print("Scenario 1: Normal Operation (WAKE State)")
    print("-" * 40)

    for i in range(3):
        results = loop.run_cycle()
        print(f"\nCycle {results['cycle']} - {results['state'].upper()} - ATP: {results['atp']:.1f}")

        for dec in results['decisions']:
            print(f"  {dec['sensor']:8s} | Sal:{dec['salience']:.2f} Thr:{dec['threshold']:.2f} | "
                  f"{dec['decision']:6s} | {dec['reason']}")

    # Scenario 2: Switch to FOCUS mode (lower threshold)
    print("\n" + "=" * 80)
    print("Scenario 2: FOCUS Mode (Lower Threshold - More Attentive)")
    print("-" * 40)

    loop.set_metabolic_state(MetabolicState.FOCUS)

    for i in range(3):
        results = loop.run_cycle()
        print(f"\nCycle {results['cycle']} - {results['state'].upper()} - ATP: {results['atp']:.1f}")

        for dec in results['decisions']:
            print(f"  {dec['sensor']:8s} | Sal:{dec['salience']:.2f} Thr:{dec['threshold']:.2f} | "
                  f"{dec['decision']:6s} | {dec['reason']}")

    # Scenario 3: Switch to REST mode (higher threshold)
    print("\n" + "=" * 80)
    print("Scenario 3: REST Mode (Higher Threshold - Selective)")
    print("-" * 40)

    loop.set_metabolic_state(MetabolicState.REST)
    loop.replenish_atp(50.0)  # Replenish some ATP

    for i in range(3):
        results = loop.run_cycle()
        print(f"\nCycle {results['cycle']} - {results['state'].upper()} - ATP: {results['atp']:.1f}")

        for dec in results['decisions']:
            print(f"  {dec['sensor']:8s} | Sal:{dec['salience']:.2f} Thr:{dec['threshold']:.2f} | "
                  f"{dec['decision']:6s} | {dec['reason']}")

    # Scenario 4: CRISIS mode with high criticality
    print("\n" + "=" * 80)
    print("Scenario 4: CRISIS Mode + High Criticality")
    print("-" * 40)

    loop.set_metabolic_state(MetabolicState.CRISIS)
    loop.set_criticality(0.9)

    for i in range(3):
        results = loop.run_cycle()
        print(f"\nCycle {results['cycle']} - {results['state'].upper()} - ATP: {results['atp']:.1f} - Crit: 0.9")

        for dec in results['decisions']:
            print(f"  {dec['sensor']:8s} | Sal:{dec['salience']:.2f} Thr:{dec['threshold']:.2f} | "
                  f"{dec['decision']:6s}")
            # Show SNARC breakdown for crisis mode
            snarc = dec['snarc']
            print(f"              SNARC: Sur={snarc['surprise']:.2f} Nov={snarc['novelty']:.2f} "
                  f"Aro={snarc['arousal']:.2f} Rew={snarc['reward']:.2f} Con={snarc['conflict']:.2f}")

    # Final stats
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("-" * 40)

    stats = loop.get_stats()
    print(f"Total Cycles:      {stats['cycles']}")
    print(f"Decisions Made:    {stats['total_decisions']}")
    print(f"Attended:          {stats['attended']} ({stats['attend_rate']*100:.1f}%)")
    print(f"Ignored:           {stats['ignored']}")
    print(f"ATP Remaining:     {stats['atp_remaining']:.1f} / {loop.total_atp:.1f}")
    print(f"ATP Used:          {stats['atp_used']:.1f}")

    print()

    # Plugin invocation counts
    print("Plugin Invocation Counts:")
    for name, plugin in plugins.items():
        print(f"  {name:8s}: {plugin.invocation_count} invocations")

    print()
    print("=" * 80)
    print("KEY INSIGHTS")
    print("-" * 40)
    print("1. Same sensor salience triggers different decisions in different states")
    print("2. FOCUS mode (low threshold) attends more; REST mode (high threshold) ignores more")
    print("3. ATP depletion raises threshold dynamically (conserve energy)")
    print("4. High criticality lowers threshold (don't miss important signals)")
    print("5. Complete pattern: Sensors → SNARC → Threshold → Decision → Action")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
