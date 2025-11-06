#!/usr/bin/env python3
"""
Unified SAGE System - Complete Consciousness Integration

Extends SAGEKernel with full integration of all components:
- SAGECore H↔L reasoning
- HRMOrchestrator plugin management
- MetabolicController state management
- Unified memory interface
- Sensor and effector abstractions

This is the production consciousness loop that brings together all pieces.
"""

import asyncio
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from pathlib import Path
import sys
import torch

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from sage.core.sage_kernel import SAGEKernel, ExecutionResult, MetabolicState
from sage.services.snarc.data_structures import CognitiveStance, Outcome
from sage.core.metabolic_controller import MetabolicController as FullMetabolicController


@dataclass
class SensorOutput:
    """Standardized sensor output"""
    data: Any
    timestamp: float
    quality: float  # 0-1
    sensor_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectorResult:
    """Standardized effector execution result"""
    success: bool
    latency: float
    quality: float  # 0-1
    outputs: Dict[str, Any] = field(default_factory=dict)


class SensorInterface:
    """Base interface for all sensors"""

    def capture(self) -> Optional[SensorOutput]:
        """Capture current sensor state"""
        raise NotImplementedError

    def get_metadata(self) -> Dict[str, Any]:
        """Sensor capabilities and status"""
        return {
            'type': self.__class__.__name__,
            'available': True
        }


class EffectorInterface:
    """Base interface for all effectors"""

    def execute(self, action: Any) -> EffectorResult:
        """Execute action and return result"""
        raise NotImplementedError

    def can_execute(self, action: Any) -> bool:
        """Check if action is executable"""
        return True


class UnifiedMemoryInterface:
    """
    Unified interface to all 4 memory systems:
    - SNARC Memory (high salience)
    - IRP Memory Bridge (pattern library)
    - Circular Buffer (recent context)
    - Verbatim Storage (full fidelity)
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.circular_buffer = []
        self.max_buffer_size = self.config.get('buffer_size', 100)
        self.snarc_memory = []
        self.pattern_library = {}
        self.verbatim_storage = []

    def store(self,
              experience: Dict[str, Any],
              salience_score: float,
              plugin_results: Optional[List] = None):
        """Store experience across appropriate memory systems"""

        # Always store in circular buffer
        self.circular_buffer.append({
            'experience': experience,
            'timestamp': time.time(),
            'salience': salience_score
        })

        # Maintain buffer size
        if len(self.circular_buffer) > self.max_buffer_size:
            self.circular_buffer.pop(0)

        # Store in SNARC memory if salient
        if salience_score > 0.7:
            self.snarc_memory.append({
                'experience': experience,
                'salience': salience_score,
                'timestamp': time.time()
            })

        # Store successful patterns
        if plugin_results and len(plugin_results) > 0:
            for result in plugin_results:
                if hasattr(result, 'telemetry') and result.telemetry.get('converged', False):
                    pattern_id = f"{result.plugin_id}_{len(self.pattern_library)}"
                    self.pattern_library[pattern_id] = {
                        'result': result,
                        'timestamp': time.time()
                    }

        # Store verbatim
        self.verbatim_storage.append({
            'experience': experience,
            'timestamp': time.time()
        })

    def recall_context(self, query: Optional[Dict] = None) -> Dict[str, Any]:
        """Retrieve relevant context from all systems"""
        return {
            'recent': self.circular_buffer[-10:] if self.circular_buffer else [],
            'salient': self.snarc_memory[-5:] if self.snarc_memory else [],
            'patterns': list(self.pattern_library.values())[-5:],
            'verbatim_count': len(self.verbatim_storage)
        }


class UnifiedSAGESystem(SAGEKernel):
    """
    Complete SAGE consciousness system integrating all components.

    Architecture:
    1. Metabolic state controls resource policy
    2. Sensors provide observations
    3. SNARC assesses salience
    4. SAGECore does H↔L reasoning (when loaded)
    5. HRMOrchestrator manages plugins with ATP budgets
    6. Memory stores and recalls context
    7. Effectors execute actions
    8. Loop continues with outcome-based learning
    """

    def __init__(
        self,
        sensor_sources: Dict[str, Callable[[], Any]],
        action_handlers: Optional[Dict[str, Callable[[Any, CognitiveStance], ExecutionResult]]] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_logging: bool = True
    ):
        """
        Initialize unified SAGE system

        Args:
            sensor_sources: Dict of sensor_id -> callable returning sensor data
            action_handlers: Dict of sensor_id -> action handler (optional)
            config: System configuration
            enable_logging: Enable cycle logging
        """
        # Initialize base kernel
        super().__init__(
            sensor_sources=sensor_sources,
            action_handlers=action_handlers,
            enable_logging=enable_logging
        )

        self.config = config or {}

        # Initialize metabolic controller
        self.metabolic_controller = FullMetabolicController(
            initial_atp=self.config.get('initial_atp', 100.0),
            max_atp=self.config.get('max_atp', 100.0),
            device=self.config.get('device', 'cpu'),
            enable_circadian=self.config.get('enable_circadian', True)
        )

        # Initialize unified memory
        self.memory = UnifiedMemoryInterface(
            config=self.config.get('memory_config', {})
        )

        # Plugin orchestrator (will be initialized when needed)
        self.orchestrator = None
        self._orchestrator_config = self.config.get('orchestrator_config', {})

        # SAGECore (H↔L transformer) - optional, loaded on demand
        self.sage_core = None
        self._sage_core_config = self.config.get('sage_core_config', {})

        # Effectors
        self.effectors: Dict[str, EffectorInterface] = {}

        # Telemetry
        self.cycle_telemetry = []

        if self.enable_logging:
            print("[UnifiedSAGE] System initialized")
            print(f"  Initial ATP: {self.metabolic_controller.atp_current:.1f}")
            print(f"  Metabolic State: {self.metabolic_controller.current_state.value}")
            print(f"  Sensors: {list(sensor_sources.keys())}")

    def register_effector(self, effector_id: str, effector: EffectorInterface):
        """Register an effector with the system"""
        self.effectors[effector_id] = effector
        if self.enable_logging:
            print(f"[UnifiedSAGE] Registered effector: {effector_id}")

    def load_sage_core(self, model_path: Optional[str] = None):
        """Load SAGECore H↔L transformer (optional enhancement)"""
        # Import here to avoid dependency if not using SAGECore
        from sage.core.sage_core import SAGECoreModel
        from sage.core.sage_config import SAGEConfig

        config = SAGEConfig(**self._sage_core_config)
        self.sage_core = SAGECoreModel(config)

        if model_path and Path(model_path).exists():
            self.sage_core.load_state_dict(torch.load(model_path))
            if self.enable_logging:
                print(f"[UnifiedSAGE] Loaded SAGECore from {model_path}")
        else:
            if self.enable_logging:
                print("[UnifiedSAGE] Initialized SAGECore (untrained)")

    def load_orchestrator(self):
        """Load HRM orchestrator for plugin management"""
        # Import here to avoid dependency if not using plugins
        from sage.orchestrator.hrm_orchestrator import HRMOrchestrator

        self.orchestrator = HRMOrchestrator(
            initial_atp=self.metabolic_controller.atp_current,
            max_concurrent=self._orchestrator_config.get('max_concurrent', 4),
            device=torch.device(self.config.get('device', 'cpu'))
        )

        # Create default plugins if requested
        if self._orchestrator_config.get('create_default_plugins', True):
            self.orchestrator.create_default_plugins()

        if self.enable_logging:
            print(f"[UnifiedSAGE] Loaded orchestrator with {len(self.orchestrator.plugins)} plugins")

    def _cycle(self):
        """
        Execute one unified SAGE cycle

        Extended from SAGEKernel with:
        1. Metabolic state awareness
        2. Memory integration
        3. Optional SAGECore reasoning
        4. Optional plugin orchestration
        5. Effector execution
        """
        cycle_start = time.time()
        cycle_telemetry = {
            'cycle': self.cycle_count,
            'metabolic_state': self.metabolic_controller.current_state.value,
            'atp_level': self.metabolic_controller.atp_current
        }

        # STEP 1: Check metabolic state and get resource policy
        state_config = self.metabolic_controller.get_current_config()
        cycle_telemetry['max_plugins'] = state_config.max_active_plugins
        cycle_telemetry['learning_enabled'] = state_config.learning_enabled

        # STEP 2: Gather observations from sensors
        observations = self._gather_observations()

        if not observations:
            if self.enable_logging:
                print(f"[SAGE Cycle {self.cycle_count}] No sensor data")
            return

        # Add memory as temporal sensor
        memory_context = self.memory.recall_context()
        observations['memory'] = memory_context

        # STEP 3: SNARC salience assessment
        salience_report = self.snarc.assess_salience(observations)
        cycle_telemetry['salience_score'] = salience_report.salience_score
        cycle_telemetry['focus_target'] = salience_report.focus_target
        cycle_telemetry['stance'] = salience_report.suggested_stance.value

        if self.enable_logging:
            self._log_cycle_info(salience_report, observations)

        # STEP 4: Strategic reasoning (if SAGECore loaded)
        strategy = None
        if self.sage_core is not None:
            # TODO: Implement SAGECore forward pass
            # For now, use basic strategy
            strategy = {
                'focus': salience_report.focus_target,
                'stance': salience_report.suggested_stance
            }

        # STEP 5: Execute action (base kernel behavior or plugins)
        focus_target = salience_report.focus_target
        observation = observations[focus_target]

        if self.orchestrator is not None and focus_target != 'memory':
            # Use plugin orchestration
            result = self._execute_with_plugins(
                focus_target,
                observation,
                salience_report,
                state_config
            )
        else:
            # Use base kernel action handlers
            result = self._execute_action(
                focus_target,
                observation,
                salience_report.suggested_stance,
                salience_report
            )

        cycle_telemetry['result_success'] = result.success
        cycle_telemetry['result_reward'] = result.reward

        # STEP 6: Execute effector actions if available
        if result.outputs and 'actions' in result.outputs:
            effector_results = self._execute_effectors(result.outputs['actions'])
            cycle_telemetry['effector_results'] = effector_results

        # STEP 7: Learn from outcome
        outcome = Outcome(
            success=result.success,
            reward=result.reward,
            description=result.description
        )

        if state_config.learning_enabled:
            self.snarc.update_from_outcome(salience_report, outcome)

        # STEP 8: Store in memory
        self.memory.store(
            experience={
                'observations': observations,
                'salience_report': salience_report,
                'result': result,
                'outcome': outcome
            },
            salience_score=salience_report.salience_score
        )

        # STEP 9: Update metabolic state
        # ATP consumed by this cycle
        cycle_atp = 1.0  # Base cost
        if self.orchestrator and hasattr(result, 'telemetry'):
            cycle_atp += result.telemetry.get('atp_consumed', 0)

        # Update using MetabolicController's interface
        new_state = self.metabolic_controller.update({
            'atp_consumed': cycle_atp,
            'attention_load': 1 if salience_report.salience_score > 0.5 else 0,
            'max_salience': salience_report.salience_score,
            'crisis_detected': False
        })

        if new_state != self.metabolic_controller.current_state:
            if self.enable_logging:
                print(f"[UnifiedSAGE] State transition: {self.metabolic_controller.current_state.value} → {new_state.value}")

        # Record telemetry
        cycle_telemetry['cycle_time'] = time.time() - cycle_start
        self.cycle_telemetry.append(cycle_telemetry)

        # Add to execution history
        self.execution_history.append({
            'cycle': self.cycle_count,
            'focus_target': focus_target,
            'salience_score': salience_report.salience_score,
            'stance': salience_report.suggested_stance.value,
            'result': result,
            'cycle_time': cycle_telemetry['cycle_time'],
            'metabolic_state': cycle_telemetry['metabolic_state']
        })

    def _execute_with_plugins(
        self,
        focus_target: str,
        observation: Any,
        salience_report: Any,
        state_config: Any
    ) -> ExecutionResult:
        """Execute using plugin orchestration"""

        # For now, fall back to base execution
        # TODO: Implement async plugin orchestration
        return self._execute_action(
            focus_target,
            observation,
            salience_report.suggested_stance,
            salience_report
        )

    def _execute_effectors(self, actions: Dict[str, Any]) -> Dict[str, EffectorResult]:
        """Execute actions through registered effectors"""
        results = {}

        for effector_id, action in actions.items():
            if effector_id in self.effectors:
                try:
                    result = self.effectors[effector_id].execute(action)
                    results[effector_id] = result

                    if self.enable_logging:
                        status = "✓" if result.success else "✗"
                        print(f"  {status} Effector '{effector_id}': {result.latency:.3f}s, quality={result.quality:.2f}")

                except Exception as e:
                    if self.enable_logging:
                        print(f"  ✗ Effector '{effector_id}' error: {e}")
                    results[effector_id] = EffectorResult(
                        success=False,
                        latency=0,
                        quality=0,
                        outputs={'error': str(e)}
                    )

        return results

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'cycle_count': self.cycle_count,
            'metabolic_state': self.metabolic_controller.current_state.value,
            'atp_level': self.metabolic_controller.atp_current,
            'max_atp': self.metabolic_controller.atp_max,
            'memory_items': len(self.memory.circular_buffer),
            'salient_memories': len(self.memory.snarc_memory),
            'patterns_learned': len(self.memory.pattern_library),
            'orchestrator_loaded': self.orchestrator is not None,
            'sage_core_loaded': self.sage_core is not None,
            'effector_count': len(self.effectors),
            'sensor_count': len(self.sensor_sources)
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.cycle_telemetry:
            return {}

        cycle_times = [t['cycle_time'] for t in self.cycle_telemetry]
        salience_scores = [t['salience_score'] for t in self.cycle_telemetry]

        return {
            'total_cycles': len(self.cycle_telemetry),
            'avg_cycle_time': sum(cycle_times) / len(cycle_times),
            'min_cycle_time': min(cycle_times),
            'max_cycle_time': max(cycle_times),
            'avg_salience': sum(salience_scores) / len(salience_scores),
            'metabolic_states_visited': list(set(t['metabolic_state'] for t in self.cycle_telemetry))
        }


# Example usage patterns

def create_minimal_sage() -> UnifiedSAGESystem:
    """Create minimal SAGE with simulated sensors"""

    def dummy_sensor():
        return {'value': time.time(), 'type': 'dummy'}

    sage = UnifiedSAGESystem(
        sensor_sources={'dummy': dummy_sensor},
        config={'initial_atp': 100.0}
    )

    return sage


def create_audio_sage() -> UnifiedSAGESystem:
    """Create SAGE with audio I/O"""
    # This would use actual audio sensors and TTS effectors
    pass


def create_robot_sage() -> UnifiedSAGESystem:
    """Create SAGE for robot control"""
    # This would use vision, proprioception, motor control
    pass


if __name__ == "__main__":
    print("=" * 70)
    print("Unified SAGE System - Integration Test")
    print("=" * 70)

    # Create minimal system
    sage = create_minimal_sage()

    print("\nSystem Status:")
    status = sage.get_system_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    print("\nRunning 10 test cycles...")
    sage.run(max_cycles=10, cycle_delay=0.1)

    print("\nPerformance Summary:")
    perf = sage.get_performance_summary()
    for key, value in perf.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("Integration test complete!")
    print("=" * 70)
