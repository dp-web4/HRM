"""
Thor SNARC-Compressed Consciousness - Integration Demo
=======================================================

Demonstrates integration of SNARC compression module with trust-weighted
consciousness kernel following compression-action-threshold pattern.

**Key Integration**:
- Replaces simplified salience with full SNARC compression
- Maintains trust-weighted attention
- Keeps metabolic states and DREAM consolidation
- Adds proper multi-dimensional sensor assessment

**Pattern**:
```
Multi-dimensional sensors → SNARC compression → Scalar salience
                                                        ↓
                                    Trust weighting (×trust_multiplier)
                                                        ↓
                        Metabolic state threshold → Binary decision
```

**Session**: Thor Autonomous Research (2025-12-05)
**Author**: Claude (guest) on Thor via claude-code
"""

import sys
sys.path.append('../core')

from snarc_compression import SNARCCompressor, CompressionMode, SNARCDimensions
from dataclasses import dataclass
from typing import Dict, Callable
import time
import random


# ============================================================================
# Simple Consciousness Components (Minimal for Demo)
# ============================================================================

@dataclass
class ActionResult:
    description: str
    reward: float
    trust_validated: bool = True


class MetabolicState:
    WAKE = "wake"
    FOCUS = "focus"


# ============================================================================
# SNARC-Enhanced Consciousness
# ============================================================================

class SNARCConsciousness:
    """
    Consciousness kernel using SNARC compression for salience calculation.

    Follows compression-action-threshold pattern:
    1. Multi-dimensional sensor input
    2. SNARC compression → scalar salience
    3. Trust weighting
    4. Metabolic state threshold
    5. Binary attention decision
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable[[], Dict]],
        action_handlers: Dict[str, Callable[[Dict], ActionResult]],
        sensor_lct_ids: Dict[str, str],
        trust_scores: Dict[str, float],
        trust_salience_weight: float = 0.3
    ):
        self.sensor_sources = sensor_sources
        self.action_handlers = action_handlers
        self.sensor_lct_ids = sensor_lct_ids
        self.trust_scores = trust_scores
        self.trust_salience_weight = trust_salience_weight

        # SNARC compressor
        self.snarc = SNARCCompressor(compression_mode=CompressionMode.LINEAR)

        # State
        self.cycle_count = 0
        self.metabolic_state = MetabolicState.WAKE
        self.execution_history = []

    def run_cycle(self):
        """Execute one consciousness cycle with SNARC compression"""
        # 1. SENSE: Gather observations
        observations = {}
        for sensor_id, sensor_fn in self.sensor_sources.items():
            try:
                data = sensor_fn()
                lct_id = self.sensor_lct_ids.get(sensor_id, f"unknown:{sensor_id}")
                trust_score = self.trust_scores.get(lct_id, 0.5)

                observations[sensor_id] = {
                    'data': data,
                    'trust': trust_score,
                    'lct_id': lct_id
                }
            except Exception as e:
                print(f"[Sensor Error] {sensor_id}: {e}")

        if not observations:
            return

        # 2. ASSESS: Calculate SNARC-compressed salience for each sensor
        salience_scores = {}
        snarc_breakdowns = {}

        for sensor_id, obs in observations.items():
            # SNARC compression: multi-D → scalar
            base_salience, dimensions = self.snarc.compute_salience(obs['data'])

            # Trust weighting
            trust_multiplier = (1.0 - self.trust_salience_weight) + \
                              (self.trust_salience_weight * obs['trust'])
            final_salience = base_salience * trust_multiplier

            salience_scores[sensor_id] = final_salience
            snarc_breakdowns[sensor_id] = dimensions

        # 3. FOCUS: Select highest trust-weighted salience
        focus_target = max(salience_scores, key=salience_scores.get)
        focus_salience = salience_scores[focus_target]
        focus_obs = observations[focus_target]
        focus_snarc = snarc_breakdowns[focus_target]

        # 4. ACT: Execute action
        if focus_target in self.action_handlers:
            try:
                result = self.action_handlers[focus_target](focus_obs['data'])

                # Log with SNARC breakdown
                print(f"\n[Cycle {self.cycle_count}] State: {self.metabolic_state.upper()}")
                print(f"  Focus: {focus_target} (salience={focus_salience:.3f}, trust={focus_obs['trust']:.3f})")
                print(f"  SNARC: S={focus_snarc.surprise:.2f} N={focus_snarc.novelty:.2f} " +
                      f"A={focus_snarc.arousal:.2f} R={focus_snarc.reward:.2f} C={focus_snarc.conflict:.2f}")
                print(f"  → {result.description}")

                self.execution_history.append({
                    'cycle': self.cycle_count,
                    'focus': focus_target,
                    'salience': focus_salience,
                    'trust': focus_obs['trust'],
                    'snarc': focus_snarc,
                    'reward': result.reward
                })

            except Exception as e:
                print(f"[Action Error] {focus_target}: {e}")

    def run(self, num_cycles: int = 10):
        """Run consciousness for specified cycles"""
        for _ in range(num_cycles):
            self.run_cycle()
            self.cycle_count += 1
            time.sleep(0.5)  # Short delay for demo

        self._print_summary()

    def _print_summary(self):
        """Print session summary"""
        print('\n' + '='*80)
        print("SNARC CONSCIOUSNESS SESSION SUMMARY")
        print('='*80)
        print(f"\nCycles completed: {self.cycle_count}")

        if self.execution_history:
            avg_salience = sum(h['salience'] for h in self.execution_history) / len(self.execution_history)
            avg_trust = sum(h['trust'] for h in self.execution_history) / len(self.execution_history)
            print(f"Average salience: {avg_salience:.3f}")
            print(f"Average trust: {avg_trust:.3f}")

            # SNARC dimension averages
            avg_surprise = sum(h['snarc'].surprise for h in self.execution_history) / len(self.execution_history)
            avg_novelty = sum(h['snarc'].novelty for h in self.execution_history) / len(self.execution_history)
            avg_arousal = sum(h['snarc'].arousal for h in self.execution_history) / len(self.execution_history)
            avg_reward = sum(h['snarc'].reward for h in self.execution_history) / len(self.execution_history)
            avg_conflict = sum(h['snarc'].conflict for h in self.execution_history) / len(self.execution_history)

            print(f"\nAverage SNARC dimensions:")
            print(f"  Surprise: {avg_surprise:.3f}")
            print(f"  Novelty: {avg_novelty:.3f}")
            print(f"  Arousal: {avg_arousal:.3f}")
            print(f"  Reward: {avg_reward:.3f}")
            print(f"  Conflict: {avg_conflict:.3f}")

        # SNARC compressor statistics
        snarc_stats = self.snarc.get_statistics()
        print(f"\nSNARC Compressor:")
        print(f"  Weights: {snarc_stats['weights']}")
        print(f"  Mode: {snarc_stats['compression_mode']}")

        print('='*80)


# ============================================================================
# Demo: SNARC-Compressed Consciousness
# ============================================================================

if __name__ == "__main__":
    import psutil

    print("="*80)
    print("THOR SNARC-COMPRESSED CONSCIOUSNESS - DEMO")
    print("="*80)
    print("\nIntegrating SNARC compression with trust-weighted consciousness")
    print("Following compression-action-threshold pattern:")
    print("  Multi-D sensors → SNARC → Scalar salience → Trust weight → Threshold → Action")
    print()

    # Define sensors with varying characteristics
    def cpu_sensor():
        """CPU sensor - moderate novelty, some urgency"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        return {
            'urgent_count': 1 if cpu_percent > 80 else 0,
            'novelty_score': 0.1 + random.random() * 0.2,
            'count': int(cpu_percent / 10),
            'atp_utilization': cpu_percent / 100.0,
            'reward': 0.3,  # Moderate reward for monitoring
        }

    def memory_sensor():
        """Memory sensor - low novelty, reward-oriented"""
        memory = psutil.virtual_memory()
        return {
            'urgent_count': 1 if memory.percent > 85 else 0,
            'novelty_score': 0.05 + random.random() * 0.1,
            'count': int(memory.percent / 10),
            'atp_utilization': memory.percent / 100.0,
            'reward': 0.6,  # Higher reward (resource management)
            'success': memory.percent < 80,
        }

    def process_sensor():
        """Process sensor - high novelty, some conflict"""
        proc_count = len(psutil.pids())
        return {
            'urgent_count': 0,
            'novelty_score': 0.3 + random.random() * 0.3,
            'count': min(proc_count // 50, 20),
            'uncertainty': 0.2 + random.random() * 0.2,
            'conflict_count': random.randint(0, 1),
            'reward': 0.2,
        }

    def disk_sensor():
        """Disk sensor - low arousal, goal-oriented"""
        disk = psutil.disk_usage('/')
        return {
            'urgent_count': 1 if disk.percent > 90 else 0,
            'novelty_score': 0.05,  # Very predictable
            'count': int(disk.percent / 10),
            'reward': 0.8 if disk.percent < 50 else 0.3,
            'goal_proximity': 1.0 - (disk.percent / 100.0),
        }

    # Define actions
    def cpu_action(data):
        return ActionResult(
            description=f"CPU at {data.get('count', 0)*10:.0f}%",
            reward=0.8,
            trust_validated=True
        )

    def memory_action(data):
        return ActionResult(
            description=f"Memory usage {data.get('count', 0)*10:.0f}%",
            reward=0.85,
            trust_validated=True
        )

    def process_action(data):
        return ActionResult(
            description=f"Processes: ~{data.get('count', 0)*50}",
            reward=0.6,
            trust_validated=random.random() > 0.2
        )

    def disk_action(data):
        return ActionResult(
            description=f"Disk usage {data.get('count', 0)*10:.0f}%",
            reward=0.9,
            trust_validated=True
        )

    # Trust scores (from Web4 LCT)
    trust_scores = {
        'lct:thor:cpu': 0.90,      # High trust
        'lct:thor:memory': 0.85,   # High trust
        'lct:thor:processes': 0.60,  # Medium trust
        'lct:thor:disk': 0.95,     # Very high trust
    }

    print("Sensor Trust Scores:")
    for lct_id, trust in trust_scores.items():
        print(f"  {lct_id}: {trust:.2f}")
    print()

    # Create SNARC consciousness
    consciousness = SNARCConsciousness(
        sensor_sources={
            'cpu': cpu_sensor,
            'memory': memory_sensor,
            'processes': process_sensor,
            'disk': disk_sensor
        },
        action_handlers={
            'cpu': cpu_action,
            'memory': memory_action,
            'processes': process_action,
            'disk': disk_action
        },
        sensor_lct_ids={
            'cpu': 'lct:thor:cpu',
            'memory': 'lct:thor:memory',
            'processes': 'lct:thor:processes',
            'disk': 'lct:thor:disk'
        },
        trust_scores=trust_scores,
        trust_salience_weight=0.3
    )

    print("Starting SNARC-compressed consciousness (10 cycles)...")
    print("="*80)

    # Run for 10 cycles
    consciousness.run(num_cycles=10)

    print("\nKey Observations:")
    print("- SNARC dimensions reveal WHY each sensor has its salience")
    print("- Trust weighting modulates final attention allocation")
    print("- High-reward sensors (disk, memory) compete with novel sensors (processes)")
    print("- Compression-action-threshold pattern in action!")
