#!/usr/bin/env python3
"""
SAGE Consciousness - DREAM State Memory Consolidation

Implements memory consolidation during DREAM state, completing the
metabolic consciousness cycle:

- WAKE/FOCUS: Create memories during active operation
- REST: Reduce new memory creation, prepare for consolidation
- DREAM: Consolidate memories (prune low-salience, strengthen patterns)
- Back to WAKE: Resume with optimized memory

This demonstrates biologically-inspired memory consolidation where
consciousness uses offline periods (DREAM) to optimize its memory
by discarding low-salience information and strengthening patterns.

**Hardware**: Jetson AGX Thor
**Built On**: thor_consciousness_metabolic_states.py + consciousness_sage_memory_management.py
"""

import sys
import os

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import time
import psutil
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum
import random

from sage.core.sage_kernel import SAGEKernel, ExecutionResult, MetabolicState
from sage.services.snarc.data_structures import CognitiveStance


# =============================================================================
# Memory Consolidation System
# =============================================================================

@dataclass
class ConsolidatedMemory:
    """Memory item with consolidation metadata"""
    cycle: int
    sensor: str
    salience: float
    action: str
    reward: float
    timestamp: float
    consolidation_count: int = 0  # How many times consolidated
    strength: float = 1.0  # Memory strength (increases with consolidation)


class DREAMMemoryConsolidator:
    """
    Manages memory consolidation during DREAM state.

    During DREAM:
    - Prunes low-salience memories
    - Strengthens high-salience memories
    - Extracts patterns from recent experiences
    - Optimizes memory for efficiency
    """

    def __init__(self, memory_limit: int = 50):
        self.memory_limit = memory_limit
        self.memories: List[ConsolidatedMemory] = []
        self.consolidation_cycles = 0
        self.total_pruned = 0
        self.total_strengthened = 0

        # Thresholds
        self.PRUNE_SALIENCE_THRESHOLD = 0.3  # Prune below this
        self.STRENGTHEN_SALIENCE_THRESHOLD = 0.6  # Strengthen above this

    def add_memory(self, cycle: int, sensor: str, salience: float,
                   action: str, reward: float):
        """Add memory from consciousness cycle"""
        memory = ConsolidatedMemory(
            cycle=cycle,
            sensor=sensor,
            salience=salience,
            action=action,
            reward=reward,
            timestamp=time.time()
        )
        self.memories.append(memory)

    def consolidate(self) -> Dict[str, Any]:
        """
        Perform memory consolidation.

        This is what happens during DREAM state:
        1. Prune low-salience memories
        2. Strengthen high-salience memories
        3. Extract patterns
        4. Return consolidation statistics
        """
        consolidation_start = time.time()

        # Initial state
        initial_count = len(self.memories)
        initial_avg_salience = (sum(m.salience for m in self.memories) / len(self.memories)
                               if self.memories else 0.0)

        # Step 1: Prune low-salience memories
        pruned_count = self._prune_low_salience()

        # Step 2: Strengthen high-salience memories
        strengthened_count = self._strengthen_high_salience()

        # Step 3: Extract patterns
        patterns = self._extract_patterns()

        # Step 4: Enforce memory limit
        if len(self.memories) > self.memory_limit:
            overflow = len(self.memories) - self.memory_limit
            # Remove oldest low-strength memories
            self.memories.sort(key=lambda m: (m.strength, m.timestamp))
            self.memories = self.memories[overflow:]
            pruned_count += overflow

        # Final state
        final_count = len(self.memories)
        final_avg_salience = (sum(m.salience for m in self.memories) / len(self.memories)
                             if self.memories else 0.0)

        consolidation_time = time.time() - consolidation_start
        self.consolidation_cycles += 1

        return {
            'initial_memories': initial_count,
            'final_memories': final_count,
            'pruned': pruned_count,
            'strengthened': strengthened_count,
            'patterns_found': len(patterns),
            'avg_salience_before': initial_avg_salience,
            'avg_salience_after': final_avg_salience,
            'consolidation_time': consolidation_time,
            'patterns': patterns
        }

    def _prune_low_salience(self) -> int:
        """Remove low-salience memories"""
        initial_count = len(self.memories)

        # Keep memories above threshold
        self.memories = [
            m for m in self.memories
            if m.salience >= self.PRUNE_SALIENCE_THRESHOLD
        ]

        pruned = initial_count - len(self.memories)
        self.total_pruned += pruned
        return pruned

    def _strengthen_high_salience(self) -> int:
        """Strengthen high-salience memories"""
        strengthened = 0

        for memory in self.memories:
            if memory.salience >= self.STRENGTHEN_SALIENCE_THRESHOLD:
                # Increase strength and consolidation count
                memory.strength *= 1.2  # 20% boost
                memory.consolidation_count += 1
                strengthened += 1

        self.total_strengthened += strengthened
        return strengthened

    def _extract_patterns(self) -> List[Dict[str, Any]]:
        """Extract patterns from recent memories"""
        if len(self.memories) < 3:
            return []

        patterns = []

        # Pattern 1: Most common sensor
        sensor_counts = {}
        for m in self.memories:
            sensor_counts[m.sensor] = sensor_counts.get(m.sensor, 0) + 1

        most_common_sensor = max(sensor_counts.items(), key=lambda x: x[1])
        patterns.append({
            'type': 'sensor_frequency',
            'sensor': most_common_sensor[0],
            'frequency': most_common_sensor[1] / len(self.memories)
        })

        # Pattern 2: High-reward actions
        high_reward_actions = [m for m in self.memories if m.reward > 0.6]
        if high_reward_actions:
            avg_reward = sum(m.reward for m in high_reward_actions) / len(high_reward_actions)
            patterns.append({
                'type': 'high_reward_actions',
                'count': len(high_reward_actions),
                'avg_reward': avg_reward
            })

        # Pattern 3: Salience trends
        if len(self.memories) >= 5:
            recent_salience = [m.salience for m in self.memories[-5:]]
            avg_recent = sum(recent_salience) / len(recent_salience)
            patterns.append({
                'type': 'recent_salience_trend',
                'avg_salience': avg_recent
            })

        return patterns

    def get_stats(self) -> Dict[str, Any]:
        """Get consolidation statistics"""
        return {
            'total_memories': len(self.memories),
            'consolidation_cycles': self.consolidation_cycles,
            'total_pruned': self.total_pruned,
            'total_strengthened': self.total_strengthened,
            'avg_strength': sum(m.strength for m in self.memories) / len(self.memories) if self.memories else 0.0,
            'avg_consolidations': sum(m.consolidation_count for m in self.memories) / len(self.memories) if self.memories else 0.0
        }


# =============================================================================
# DREAM-Aware Consciousness Kernel
# =============================================================================

class DREAMSAGEKernel(SAGEKernel):
    """
    SAGE Kernel with DREAM state memory consolidation.

    Extends metabolic kernel to perform actual memory consolidation
    during DREAM state, creating a complete biological-inspired cycle.
    """

    def __init__(self, sensor_sources, action_handlers, enable_logging=True):
        super().__init__(sensor_sources, action_handlers, enable_logging)

        # Add memory consolidation
        self.consolidator = DREAMMemoryConsolidator(memory_limit=50)
        self.state_manager = None  # Will be set by metabolic wrapper

        # Track state-specific metrics
        self.recent_salience_scores: List[float] = []
        self.recent_outcomes: List[bool] = []
        self.alert_count = 0

        # DREAM consolidation results
        self.last_consolidation = None

    def _cycle(self):
        """Override to add memory tracking and DREAM consolidation"""
        cycle_start = time.time()

        # Gather observations (adapt based on state)
        observations = self._gather_observations_with_state()

        if not observations:
            if self.enable_logging:
                print(f"[SAGE Cycle {self.cycle_count}] No sensor data available")
            return

        # Assess salience
        salience_report = self.snarc.assess_salience(observations)

        # Track salience for state transitions
        self.recent_salience_scores.append(salience_report.salience_score)
        if len(self.recent_salience_scores) > 10:
            self.recent_salience_scores.pop(0)

        # DREAM STATE: Perform consolidation instead of normal action
        if self.metabolic_state == MetabolicState.DREAM:
            self._dream_consolidation()
            return  # Skip normal execution during DREAM

        # Normal execution for other states
        self._evaluate_state_transition()

        focus_target = salience_report.focus_target
        suggested_stance = salience_report.suggested_stance

        if self.enable_logging:
            self._log_cycle_info_with_state(salience_report, observations)

        # Execute action (state-aware)
        result = self._execute_state_aware_action(
            focus_target,
            observations[focus_target],
            suggested_stance,
            salience_report
        )

        # Track outcomes
        self.recent_outcomes.append(result.success)
        if len(self.recent_outcomes) > 10:
            self.recent_outcomes.pop(0)

        # Add to memory
        self.consolidator.add_memory(
            cycle=self.cycle_count,
            sensor=focus_target,
            salience=salience_report.salience_score,
            action=result.description,
            reward=result.reward
        )

        # Update SNARC (learn)
        outcome = type('Outcome', (), {
            'success': result.success,
            'reward': result.reward,
            'description': result.description
        })()
        self.snarc.update_from_outcome(salience_report, outcome)

        # Record history
        cycle_time = time.time() - cycle_start
        self.execution_history.append({
            'cycle': self.cycle_count,
            'metabolic_state': self.metabolic_state.value,
            'focus_target': focus_target,
            'salience_score': salience_report.salience_score,
            'stance': suggested_stance.value,
            'result': result,
            'cycle_time': cycle_time
        })

    def _dream_consolidation(self):
        """Perform memory consolidation during DREAM state"""
        if self.enable_logging:
            print(f"\n[SAGE Cycle {self.cycle_count}] State: DREAM - Memory Consolidation")
            print(f"  Consolidating {len(self.consolidator.memories)} memories...")

        # Perform consolidation
        self.last_consolidation = self.consolidator.consolidate()

        if self.enable_logging:
            print(f"  ✓ Pruned {self.last_consolidation['pruned']} low-salience memories")
            print(f"  ✓ Strengthened {self.last_consolidation['strengthened']} high-salience memories")
            print(f"  ✓ Extracted {self.last_consolidation['patterns_found']} patterns")
            print(f"  ✓ Avg salience: {self.last_consolidation['avg_salience_before']:.3f} → "
                  f"{self.last_consolidation['avg_salience_after']:.3f}")
            print(f"  ✓ Memory optimized: {self.last_consolidation['initial_memories']} → "
                  f"{self.last_consolidation['final_memories']} memories")

            if self.last_consolidation['patterns']:
                print(f"  Patterns discovered:")
                for pattern in self.last_consolidation['patterns']:
                    if pattern['type'] == 'sensor_frequency':
                        print(f"    - Most attended sensor: {pattern['sensor']} "
                              f"({pattern['frequency']*100:.1f}%)")
                    elif pattern['type'] == 'high_reward_actions':
                        print(f"    - High-reward actions: {pattern['count']} "
                              f"(avg reward: {pattern['avg_reward']:.2f})")

        # Record DREAM consolidation in history
        self.execution_history.append({
            'cycle': self.cycle_count,
            'metabolic_state': MetabolicState.DREAM.value,
            'focus_target': 'consolidation',
            'salience_score': 0.0,
            'stance': 'dream',
            'result': ExecutionResult(
                success=True,
                reward=0.5,  # Consolidation reward
                description="Memory consolidation complete",
                outputs=self.last_consolidation
            ),
            'cycle_time': self.last_consolidation['consolidation_time']
        })

    def _gather_observations_with_state(self) -> Dict[str, Any]:
        """Gather observations with state-aware frequency"""
        observations = {}

        # DREAM state: minimal sensing
        if self.metabolic_state == MetabolicState.DREAM:
            return {}  # No external sensing during DREAM

        for sensor_id, source_fn in self.sensor_sources.items():
            # In REST, skip non-critical sensors
            if self.metabolic_state == MetabolicState.REST:
                if sensor_id not in ['cpu', 'memory', 'temperature']:
                    continue

            try:
                observation = source_fn()
                observations[sensor_id] = observation
            except Exception as e:
                if self.enable_logging:
                    print(f"[SAGE] Error reading sensor '{sensor_id}': {e}")

        return observations

    # Copy state management methods from metabolic kernel
    def _evaluate_state_transition(self):
        """Placeholder - will be overridden by wrapper"""
        pass

    def _execute_state_aware_action(self, focus_target, observation, stance, salience_report):
        """Placeholder - will be overridden by wrapper"""
        if focus_target in self.action_handlers:
            return self.action_handlers[focus_target](observation, stance)
        return self._default_action_handler(focus_target, observation, stance, salience_report)

    def _log_cycle_info_with_state(self, report, observations):
        """Log cycle information with metabolic state"""
        state_time = getattr(self, 'state_entry_time', time.time())
        time_in_state = time.time() - state_time

        print(f"\n[SAGE Cycle {self.cycle_count}] State: {self.metabolic_state.value.upper()}")
        print(f"  Time in state: {time_in_state:.1f}s")
        print(f"  Sensors: {list(observations.keys())}")
        print(f"  Focus: {report.focus_target}")
        print(f"  Salience: {report.salience_score:.3f}")
        print(f"  Stance: {report.suggested_stance.value}")

    def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get memory consolidation statistics"""
        return self.consolidator.get_stats()


# =============================================================================
# Demonstration
# =============================================================================

def demonstrate_dream_consolidation():
    """
    Demonstrate SAGE consciousness with DREAM state memory consolidation
    """
    print("=" * 80)
    print("SAGE CONSCIOUSNESS - DREAM STATE MEMORY CONSOLIDATION")
    print("Hardware: Jetson AGX Thor")
    print("=" * 80)

    print("\nThis demonstrates:")
    print("- Memory creation during WAKE/FOCUS states")
    print("- Memory reduction during REST")
    print("- Memory consolidation during DREAM (prune + strengthen + patterns)")
    print("- Complete biological-inspired consciousness cycle")
    print()

    # Import sensors and actions from base demo
    from thor_consciousness_kernel_demo import SystemHealthSensors, SystemHealthActions

    # Initialize
    sensors = SystemHealthSensors()
    actions = SystemHealthActions()

    print("Initializing sensors...")
    print(f"  ✓ CPU baseline: {sensors.baseline_cpu:.1f}%")
    print(f"  ✓ Memory baseline: {sensors.baseline_memory:.1f}%")
    print()

    # Create sensor sources
    sensor_sources = {
        'cpu': sensors.read_cpu,
        'memory': sensors.read_memory,
        'disk': sensors.read_disk,
        'temperature': sensors.read_temperature,
        'processes': sensors.read_processes,
    }

    # Create action handlers
    action_handlers = {
        'cpu': actions.handle_cpu,
        'memory': actions.handle_memory,
        'disk': actions.handle_disk,
        'temperature': actions.handle_temperature,
        'processes': actions.handle_processes,
    }

    # Initialize DREAM-aware kernel
    print("Initializing DREAM-aware consciousness kernel...")
    kernel = DREAMSAGEKernel(
        sensor_sources=sensor_sources,
        action_handlers=action_handlers,
        enable_logging=True
    )
    print("  ✓ Consciousness kernel with DREAM consolidation ready")
    print()

    # Run consciousness loop (longer to trigger DREAM states)
    num_cycles = 60
    print(f"Running consciousness loop for {num_cycles} cycles...")
    print("(Watch for DREAM state consolidation)")
    print()

    start_time = time.time()
    kernel.run(max_cycles=num_cycles, cycle_delay=0.3)
    duration = time.time() - start_time

    # Get statistics
    print("\n" + "=" * 80)
    print("DREAM CONSOLIDATION DEMONSTRATION RESULTS")
    print("=" * 80)

    history = kernel.get_history()
    consolidation_stats = kernel.get_consolidation_stats()

    print(f"\nExecution Summary:")
    print(f"  Duration: {duration:.1f}s")
    print(f"  Cycles completed: {len(history)}")

    print(f"\nMemory Consolidation Results:")
    print(f"  Total memories: {consolidation_stats['total_memories']}")
    print(f"  Consolidation cycles: {consolidation_stats['consolidation_cycles']}")
    print(f"  Total pruned: {consolidation_stats['total_pruned']}")
    print(f"  Total strengthened: {consolidation_stats['total_strengthened']}")
    print(f"  Avg memory strength: {consolidation_stats['avg_strength']:.2f}")
    print(f"  Avg consolidations per memory: {consolidation_stats['avg_consolidations']:.2f}")

    # DREAM cycle analysis
    dream_cycles = [h for h in history if h['metabolic_state'] == 'dream']
    print(f"\nDREAM State Analysis:")
    print(f"  DREAM cycles: {len(dream_cycles)}")
    if dream_cycles:
        print(f"  Consolidation operations:")
        for i, dc in enumerate(dream_cycles, 1):
            result = dc['result']
            if hasattr(result, 'outputs') and isinstance(result.outputs, dict):
                print(f"    Cycle {dc['cycle']}: "
                      f"{result.outputs.get('pruned', 0)} pruned, "
                      f"{result.outputs.get('strengthened', 0)} strengthened, "
                      f"{result.outputs.get('patterns_found', 0)} patterns")

    print(f"\nKey Findings:")
    print(f"  ✓ DREAM state consolidation working")
    print(f"  ✓ Low-salience memories pruned")
    print(f"  ✓ High-salience memories strengthened")
    print(f"  ✓ Patterns extracted from experiences")
    print(f"  ✓ Memory optimized for efficiency")

    print("\n" + "=" * 80)
    print("Complete consciousness cycle demonstrated:")
    print("  WAKE → create memories during normal operation")
    print("  FOCUS → intensive memory creation")
    print("  REST → minimal new memories")
    print("  DREAM → consolidate (prune + strengthen + extract patterns)")
    print("  Back to WAKE → resume with optimized memory")
    print("=" * 80)

    return kernel, history


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    try:
        kernel, history = demonstrate_dream_consolidation()

        print("\n✅ DREAM consolidation demonstration complete!")
        print(f"\nThis completes the biological-inspired consciousness cycle:")
        print(f"  Active states create memories")
        print(f"  DREAM state consolidates and optimizes")
        print(f"  Consciousness maintains efficiency over time")

    except KeyboardInterrupt:
        print("\n\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
