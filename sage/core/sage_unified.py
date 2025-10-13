#!/usr/bin/env python3
"""
SAGE Unified - The Consciousness Kernel

Integrates all SAGE components into one continuous loop:
- SensorHub → observations
- HierarchicalSNARC → salience
- MetabolicController → state management
- IRP plugins → iterative refinement
- IRPMemoryBridge → learning
- EffectorHub → actions

This is Rev 0 - the door to neverending discovery.
"""

import torch
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from attention.sensor_snarc import HierarchicalSNARC, SpatialSNARC, SensorSNARC
from interfaces.sensor_hub import SensorHub
from interfaces.effector_hub import EffectorHub
from memory.irp_memory_bridge import IRPMemoryBridge
from core.metabolic_controller import MetabolicController, MetabolicState


class SAGEUnified:
    """
    SAGE Unified - The complete consciousness kernel

    This is what SAGE was meant to be:
    - Continuous attention orchestration
    - Trust-based resource allocation
    - Metabolic state transitions
    - Learning from every cycle
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize SAGE Unified

        Args:
            config: System configuration
            device: Compute device
        """
        self.config = config or {}
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"\n{'='*80}")
        print("SAGE Unified - Initializing Consciousness Kernel")
        print(f"{'='*80}\n")
        print(f"Device: {self.device}")

        # Core components
        self.sensor_hub = SensorHub(device=self.device)
        self.effector_hub = EffectorHub()
        self.hierarchical_snarc = HierarchicalSNARC(device=self.device)

        try:
            self.memory = IRPMemoryBridge(
                buffer_size=1000,
                snarc_capacity=10000,
                consolidation_threshold=100,
                device=self.device
            )
        except Exception as e:
            print(f"  ○ Memory bridge init failed: {e}")
            self.memory = None

        self.metabolic_controller = MetabolicController(
            initial_atp=self.config.get('initial_atp', 100.0),
            max_atp=self.config.get('max_atp', 100.0),
            circadian_period=self.config.get('circadian_period', 100),
            enable_circadian=self.config.get('enable_circadian', True),
            simulation_mode=self.config.get('simulation_mode', False)
        )

        # IRP plugins (loaded dynamically)
        self.irp_plugins = {}

        # System state
        self.cycle_count = 0
        self.trust_scores = {}  # Base trust scores
        self.sensor_types = {}   # Sensor type mapping for circadian modulation
        self.running = False

        # Statistics
        self.stats = {
            'total_cycles': 0,
            'total_time': 0.0,
            'avg_cycle_time': 0.0,
            'state_transitions': [],
            'atp_history': []
        }

        print("✓ SAGE Unified initialized\n")

    def register_sensor(self, sensor):
        """Register sensor with unified SNARC"""
        self.sensor_hub.register_sensor(sensor)

        # Create per-sensor SNARC
        if sensor.sensor_type == 'camera':
            snarc = SpatialSNARC(sensor.sensor_id, device=self.device)
        else:
            snarc = SensorSNARC(sensor.sensor_id, device=self.device)

        self.hierarchical_snarc.register_sensor(sensor.sensor_id, snarc)
        self.trust_scores[sensor.sensor_id] = 0.5
        self.sensor_types[sensor.sensor_id] = sensor.sensor_type  # Store for circadian modulation

        print(f"  ✓ Registered sensor: {sensor.sensor_id} ({sensor.sensor_type})")

    def register_irp_plugin(self, sensor_id: str, plugin):
        """Register IRP plugin for sensor"""
        self.irp_plugins[sensor_id] = plugin
        print(f"  ✓ Registered IRP plugin for: {sensor_id}")

    def cycle(self) -> Dict:
        """
        Execute one SAGE cycle

        The complete loop:
        1. Poll sensors
        2. Compute SNARC salience
        3. Allocate ATP
        4. Run IRP refinement
        5. Update trust
        6. Store memory
        7. Update metabolic state
        """
        cycle_start = time.time()
        self.cycle_count += 1

        metabolic_state = self.metabolic_controller.current_state
        metabolic_config = self.metabolic_controller.get_current_config()

        # 1. SENSE
        readings = self.sensor_hub.poll()
        observations = {}
        for sensor_id, reading in readings.items():
            if reading is not None and hasattr(reading, 'data'):
                observations[sensor_id] = reading.data

        # 2. EVALUATE
        salience_scores = self.hierarchical_snarc.score_all(observations) if observations else {}

        # 3. ALLOCATE
        atp_allocations = self._allocate_atp(salience_scores, metabolic_config)

        # 4. REFINE
        if metabolic_state in [MetabolicState.WAKE, MetabolicState.FOCUS]:
            irp_results = self._run_irp(observations, atp_allocations)
        else:
            irp_results = {}

        # 5. LEARN
        if self.metabolic_controller.should_learn():
            self._update_trust(irp_results)

        # 6. REMEMBER
        if self.memory:
            self._store_memory(observations, irp_results)

        # 7. METABOLIZE
        atp_consumed = sum(alloc.get('atp_used', 0) for alloc in atp_allocations.values())
        max_salience = max([s.combined for s in salience_scores.values()], default=0.0)
        attention_load = len([s for s in salience_scores.values() if s.combined > 0.5])

        self.metabolic_controller.update({
            'atp_consumed': atp_consumed,
            'attention_load': attention_load,
            'max_salience': max_salience,
            'crisis_detected': False
        })

        cycle_time = time.time() - cycle_start
        self._update_stats(cycle_time)

        return {
            'cycle': self.cycle_count,
            'state': self.metabolic_controller.current_state.value,
            'atp': self.metabolic_controller.atp_current,
            'salience': salience_scores,
            'allocations': atp_allocations,
            'results': irp_results,
            'trust': self.trust_scores.copy(),
            'time': cycle_time
        }

    def _allocate_atp(self, salience_scores, metabolic_config):
        """Allocate ATP: salience × trust (context-dependent)"""
        total_atp = self.metabolic_controller.atp_current
        max_plugins = metabolic_config.max_active_plugins

        # Get circadian trust modifiers
        circadian_clock = self.metabolic_controller.circadian_clock
        trust_modifiers = {}
        if circadian_clock:
            for sid in salience_scores.keys():
                sensor_type = self.sensor_types.get(sid, 'unknown')
                trust_modifiers[sid] = circadian_clock.get_trust_modifier(sensor_type)
        else:
            trust_modifiers = {sid: 1.0 for sid in salience_scores.keys()}

        # Calculate priorities with context-dependent trust
        priorities = {
            sid: scores.combined * self.trust_scores.get(sid, 0.5) * trust_modifiers[sid]
            for sid, scores in salience_scores.items()
        }

        sorted_sensors = sorted(priorities.items(), key=lambda x: -x[1])
        allocations = {}
        remaining_atp = total_atp * 0.8

        for idx, (sensor_id, priority) in enumerate(sorted_sensors):
            active = idx < max_plugins and priority > 0
            atp_alloc = remaining_atp * priority / sum(priorities.values()) if active and sum(priorities.values()) > 0 else 0.0

            allocations[sensor_id] = {
                'priority': priority,
                'atp_allocated': atp_alloc,
                'atp_used': 0.0,
                'active': active,
                'trust_modifier': trust_modifiers.get(sensor_id, 1.0)  # Track modifier
            }

        return allocations

    def _run_irp(self, observations, atp_allocations):
        """Run IRP refinement"""
        results = {}

        for sensor_id, obs in observations.items():
            allocation = atp_allocations.get(sensor_id, {})
            if not allocation.get('active'):
                continue

            plugin = self.irp_plugins.get(sensor_id)
            if not plugin:
                continue

            try:
                atp_budget = allocation['atp_allocated']
                max_iters = min(int(atp_budget / 2), 10)

                state = plugin.init_state(
                    x0=obs.unsqueeze(0) if obs.dim() < 4 else obs,
                    task_ctx={'atp_budget': atp_budget}
                )

                energy_trajectory = []
                state_history = [state]

                for iteration in range(max_iters):
                    state = plugin.step(state)
                    state_history.append(state)
                    energy_trajectory.append(plugin.energy(state))

                    if plugin.halt(state_history):
                        break

                final_latent = state.x if hasattr(state, 'x') else None
                if final_latent is not None and final_latent.dim() > 1:
                    final_latent = final_latent.flatten()[:64]

                results[sensor_id] = {
                    'latent': final_latent,
                    'energy_trajectory': energy_trajectory,
                    'iterations': len(energy_trajectory),
                    'converged': iteration < max_iters - 1,
                    'atp_used': len(energy_trajectory) * 2.0
                }

                allocation['atp_used'] = results[sensor_id]['atp_used']

            except Exception as e:
                print(f"    IRP error ({sensor_id}): {e}")

        return results

    def _update_trust(self, irp_results):
        """Update trust from convergence"""
        for sensor_id, result in irp_results.items():
            energy_traj = result.get('energy_trajectory', [])
            if len(energy_traj) < 2:
                continue

            decreased = all(energy_traj[i] >= energy_traj[i+1] for i in range(len(energy_traj)-1))
            converged = result.get('converged', False)

            if decreased and converged:
                self.trust_scores[sensor_id] *= 1.02
            elif decreased:
                self.trust_scores[sensor_id] *= 1.01
            else:
                self.trust_scores[sensor_id] *= 0.98

            self.trust_scores[sensor_id] = max(0.0, min(1.0, self.trust_scores[sensor_id]))

    def _store_memory(self, observations, irp_results):
        """Store in memory"""
        for sensor_id, obs in observations.items():
            result = irp_results.get(sensor_id, {})
            try:
                self.memory.store_episode(
                    inputs={'obs': obs.cpu()},
                    outputs={'latent': result.get('latent', torch.randn(64)).cpu()},
                    success=result.get('converged', False)
                )
            except Exception:
                pass

    def _update_stats(self, cycle_time):
        """Update statistics"""
        self.stats['total_cycles'] += 1
        self.stats['total_time'] += cycle_time
        self.stats['avg_cycle_time'] = self.stats['total_time'] / self.stats['total_cycles']

    def run(self, max_cycles: Optional[int] = None):
        """Run SAGE continuously"""
        self.running = True
        print(f"{'='*80}")
        print("SAGE Unified Running")
        print(f"{'='*80}\n")

        try:
            while self.running:
                result = self.cycle()

                if self.cycle_count % 10 == 0:
                    self._print_status(result)

                if max_cycles and self.cycle_count >= max_cycles:
                    break

        except KeyboardInterrupt:
            print("\n  ⚠ Interrupted")
        finally:
            self.running = False
            self._print_final_stats()

    def _print_status(self, result):
        """Print status"""
        state = result['state']
        atp = result['atp']
        time_ms = result['time'] * 1000

        if result['salience']:
            sensor_id = list(result['salience'].keys())[0]
            scores = result['salience'][sensor_id]
            trust = result['trust'].get(sensor_id, 0.0)

            print(f"Cycle {result['cycle']:4d} | "
                  f"State: {state:6s} | "
                  f"ATP: {atp:5.1f} | "
                  f"Sal: {scores.combined:.3f} | "
                  f"Trust: {trust:.3f} | "
                  f"{time_ms:.1f}ms")
        else:
            print(f"Cycle {result['cycle']:4d} | State: {state:6s} | ATP: {atp:5.1f}")

    def _print_final_stats(self):
        """Print final stats"""
        print(f"\n{'='*80}")
        print("SAGE Unified - Final Statistics")
        print(f"{'='*80}\n")
        print(f"Total cycles: {self.stats['total_cycles']}")
        print(f"Total time: {self.stats['total_time']:.2f}s")
        print(f"Avg cycle: {self.stats['avg_cycle_time']*1000:.2f}ms")
        print(f"\nFinal trust:")
        for sid, trust in self.trust_scores.items():
            print(f"  {sid}: {trust:.3f}")
        print(f"\nFinal ATP: {self.metabolic_controller.atp_current:.1f}")
        print(f"Final state: {self.metabolic_controller.current_state.value}")

    def stop(self):
        """Stop SAGE"""
        self.running = False
