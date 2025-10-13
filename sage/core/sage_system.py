"""
SAGESystem - Unified Consciousness Kernel

This is the main integration class that orchestrates all SAGE components into
a continuous loop. SAGE is not a model - it's a consciousness kernel for edge devices.

Architecture:
    - SAGE = The kernel (scheduler, resource manager, learner)
    - IRP = The API (standard interface for plugins/"apps")
    - VAE = Translation layer (shared latent spaces for cross-modal communication)

The main loop:
    while True:
        observations = gather_from_sensors()
        attention_targets = compute_what_matters(observations)  # SNARC salience
        required_resources = determine_needed_plugins(attention_targets)
        manage_resource_loading(required_resources)
        results = invoke_irp_plugins(attention_targets)  # Iterative refinement
        update_trust_and_memory(results)
        send_to_effectors(results)

Design Principles:
    - Clean separation of concerns
    - Each component has single responsibility
    - Data flows unidirectionally through loop
    - State is explicit and manageable
    - Easy to test components independently
    - Easy to swap implementations
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import time
import sys
from collections import deque

# Import handling for both module and script execution
try:
    from .sage_config import SAGEConfig
except ImportError:
    # Running as script
    sys.path.insert(0, str(Path(__file__).parent))
    from sage_config import SAGEConfig

# Optional imports (may not be available in all environments)
try:
    import torch
except ImportError:
    torch = None
    print("[WARNING] PyTorch not available - some features will be limited")

try:
    import numpy as np
except ImportError:
    import random
    # Minimal numpy fallback for testing
    class NumpyFallback:
        @staticmethod
        def random():
            class Random:
                @staticmethod
                def uniform(low, high):
                    return random.uniform(low, high)
                @staticmethod
                def randint(low, high):
                    return random.randint(low, high-1)
            return Random()
        @staticmethod
        def mean(arr):
            return sum(arr) / len(arr) if arr else 0.0
        @staticmethod
        def min(arr):
            return min(arr) if arr else 0.0
        @staticmethod
        def max(arr):
            return max(arr) if arr else 0.0
    np = NumpyFallback()


# =============================================================================
# Data Structures - Explicit State Containers
# =============================================================================

@dataclass
class Observation:
    """Single sensor observation"""
    modality: str  # 'vision', 'audio', 'proprioception', 'clock'
    data: Any  # Tensor, array, scalar, etc.
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SalienceScore:
    """SNARC-based salience evaluation for a modality"""
    modality: str
    surprise: float = 0.0  # Deviation from expected
    novelty: float = 0.0  # Unseen patterns
    arousal: float = 0.0  # Complexity/information density
    reward: float = 0.0  # Task success signal
    conflict: float = 0.0  # Ambiguity/uncertainty
    combined: float = 0.0  # Weighted combination

    def __post_init__(self):
        """Compute combined score if not provided"""
        if self.combined == 0.0:
            # Weighted SNARC combination
            weights = {
                'surprise': 1.0,
                'novelty': 0.8,
                'arousal': 0.6,
                'reward': 1.2,
                'conflict': 0.7
            }
            self.combined = (
                weights['surprise'] * self.surprise +
                weights['novelty'] * self.novelty +
                weights['arousal'] * self.arousal +
                weights['reward'] * self.reward +
                weights['conflict'] * self.conflict
            ) / sum(weights.values())


@dataclass
class AttentionTarget:
    """Attention allocation target with priority"""
    modality: str
    priority: float  # 0.0 to 1.0
    data: Any
    salience: SalienceScore
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginResult:
    """Result from IRP plugin execution"""
    plugin_id: str
    output: Any
    energy_trajectory: List[float]
    iterations: int
    atp_used: float
    converged: bool
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CycleState:
    """Complete state for one cycle iteration"""
    cycle_id: int
    timestamp: float
    observations: Dict[str, Observation]
    salience_scores: Dict[str, SalienceScore]
    attention_targets: List[AttentionTarget]
    required_plugins: List[str]
    plugin_results: Dict[str, PluginResult]
    actions: Dict[str, Any]
    metabolic_state: str
    energy_level: float
    atp_budget: float
    atp_remaining: float


# =============================================================================
# Component Interfaces - Clean Contracts
# =============================================================================

class ISensorHub:
    """Interface for sensor management"""

    def poll(self) -> Dict[str, Observation]:
        """Poll all active sensors and return observations"""
        raise NotImplementedError

    def get_available_sensors(self) -> List[str]:
        """Get list of available sensor modalities"""
        raise NotImplementedError

    def enable_sensor(self, modality: str):
        """Enable a sensor"""
        raise NotImplementedError

    def disable_sensor(self, modality: str):
        """Disable a sensor"""
        raise NotImplementedError


class ISNARCScorer:
    """Interface for SNARC salience evaluation"""

    def evaluate(self, observations: Dict[str, Observation],
                 predictions: Optional[Dict[str, Any]] = None) -> Dict[str, SalienceScore]:
        """Evaluate salience for all observations"""
        raise NotImplementedError

    def update_predictions(self, observations: Dict[str, Observation]):
        """Update predictive models with new observations"""
        raise NotImplementedError


class IAttentionAllocator:
    """Interface for attention allocation"""

    def allocate(self, salience_scores: Dict[str, SalienceScore],
                 metabolic_state: str,
                 atp_budget: float) -> List[AttentionTarget]:
        """Allocate attention based on salience and metabolic state"""
        raise NotImplementedError


class IResourcePlanner:
    """Interface for resource planning"""

    def plan(self, targets: List[AttentionTarget],
             active_resources: Dict[str, Any],
             memory_available: float) -> List[str]:
        """Determine required plugins for attention targets"""
        raise NotImplementedError


class IResourceManager:
    """Interface for dynamic resource management"""

    def update(self, required_plugins: List[str]) -> Dict[str, Any]:
        """Load/unload plugins as needed, return active plugins"""
        raise NotImplementedError

    def get_active_plugins(self) -> Dict[str, Any]:
        """Get currently loaded plugins"""
        raise NotImplementedError

    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        raise NotImplementedError


class IOrchestrator:
    """Interface for IRP plugin orchestration"""

    def run_cycle(self, targets: List[AttentionTarget],
                  active_plugins: Dict[str, Any],
                  atp_budget: float) -> Dict[str, PluginResult]:
        """Execute IRP plugins for attention targets"""
        raise NotImplementedError


class IMemorySystem:
    """Interface for memory management"""

    def update(self, cycle_state: CycleState):
        """Update all memory systems with cycle state"""
        raise NotImplementedError

    def retrieve_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve relevant context from memory"""
        raise NotImplementedError

    def consolidate(self):
        """Run memory consolidation (during DREAM state)"""
        raise NotImplementedError


class ITrustTracker:
    """Interface for trust scoring"""

    def update(self, results: Dict[str, PluginResult]):
        """Update trust scores based on plugin behavior"""
        raise NotImplementedError

    def get_trust_scores(self) -> Dict[str, float]:
        """Get current trust scores for all plugins"""
        raise NotImplementedError

    def adjust_atp_budget(self, base_budget: float) -> Dict[str, float]:
        """Adjust ATP allocation per plugin based on trust"""
        raise NotImplementedError


class IEffectorHub:
    """Interface for effector management"""

    def execute(self, actions: Dict[str, Any]):
        """Execute actions on effectors"""
        raise NotImplementedError

    def get_available_effectors(self) -> List[str]:
        """Get list of available effectors"""
        raise NotImplementedError


class IMetabolicController:
    """Interface for metabolic state management"""

    def update(self, energy_level: float, fatigue: float, stress: float) -> str:
        """Update metabolic state, return new state"""
        raise NotImplementedError

    def get_state(self) -> str:
        """Get current metabolic state"""
        raise NotImplementedError

    def get_attention_breadth(self) -> float:
        """Get attention breadth for current state (0.0 to 1.0)"""
        raise NotImplementedError


# =============================================================================
# Mock Implementations - For Initial Testing
# =============================================================================

class MockSensorHub(ISensorHub):
    """Mock sensor hub for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_sensors = set(config.get('enabled_sensors', ['vision', 'clock']))

    def poll(self) -> Dict[str, Observation]:
        """Return mock observations"""
        observations = {}

        if 'vision' in self.enabled_sensors:
            # Use torch if available, otherwise use list
            if torch:
                data = torch.randn(3, 480, 640)
            else:
                data = [[[np.random.uniform(-1, 1) for _ in range(640)]
                        for _ in range(480)] for _ in range(3)]

            observations['vision'] = Observation(
                modality='vision',
                data=data,
                timestamp=time.time(),
                metadata={'source': 'mock_camera'}
            )

        if 'audio' in self.enabled_sensors:
            # Use torch if available, otherwise use list
            if torch:
                data = torch.randn(16000)
            else:
                data = [np.random.uniform(-1, 1) for _ in range(16000)]

            observations['audio'] = Observation(
                modality='audio',
                data=data,
                timestamp=time.time(),
                metadata={'source': 'mock_microphone'}
            )

        if 'clock' in self.enabled_sensors:
            observations['clock'] = Observation(
                modality='clock',
                data=time.time(),
                timestamp=time.time(),
                metadata={'source': 'system_clock'}
            )

        return observations

    def get_available_sensors(self) -> List[str]:
        return ['vision', 'audio', 'clock']

    def enable_sensor(self, modality: str):
        self.enabled_sensors.add(modality)

    def disable_sensor(self, modality: str):
        self.enabled_sensors.discard(modality)


class MockSNARCScorer(ISNARCScorer):
    """Mock SNARC scorer for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.predictions = {}
        self.first_observation = True

    def evaluate(self, observations: Dict[str, Observation],
                 predictions: Optional[Dict[str, Any]] = None) -> Dict[str, SalienceScore]:
        """Return mock salience scores"""
        scores = {}

        for modality, obs in observations.items():
            if self.first_observation:
                # First cycle - everything is novel
                scores[modality] = SalienceScore(
                    modality=modality,
                    surprise=0.9,
                    novelty=1.0,
                    arousal=0.1,
                    reward=0.0,
                    conflict=0.0
                )
            else:
                # Subsequent cycles - lower salience
                scores[modality] = SalienceScore(
                    modality=modality,
                    surprise=np.random.uniform(0.1, 0.3),
                    novelty=np.random.uniform(0.1, 0.3),
                    arousal=np.random.uniform(0.1, 0.3),
                    reward=0.0,
                    conflict=np.random.uniform(0.0, 0.2)
                )

        self.first_observation = False
        return scores

    def update_predictions(self, observations: Dict[str, Observation]):
        """Store observations as predictions"""
        self.predictions.update(observations)


class MockAttentionAllocator(IAttentionAllocator):
    """Mock attention allocator for testing"""

    def allocate(self, salience_scores: Dict[str, SalienceScore],
                 metabolic_state: str,
                 atp_budget: float) -> List[AttentionTarget]:
        """Allocate attention based on salience"""
        # Sort by combined salience
        sorted_scores = sorted(salience_scores.items(),
                             key=lambda x: x[1].combined,
                             reverse=True)

        # Attention breadth based on metabolic state
        breadth = {
            'WAKE': 0.7,
            'FOCUS': 0.3,
            'REST': 0.1,
            'DREAM': 0.0,
            'CRISIS': 1.0
        }.get(metabolic_state, 0.5)

        # Allocate attention
        targets = []
        remaining_attention = 1.0

        for modality, salience in sorted_scores:
            if remaining_attention <= 0:
                break

            priority = min(salience.combined * breadth, remaining_attention)

            targets.append(AttentionTarget(
                modality=modality,
                priority=priority,
                data=None,  # Will be filled by orchestrator
                salience=salience
            ))

            remaining_attention -= priority

        return targets


class MockResourcePlanner(IResourcePlanner):
    """Mock resource planner for testing"""

    def plan(self, targets: List[AttentionTarget],
             active_resources: Dict[str, Any],
             memory_available: float) -> List[str]:
        """Determine required plugins"""
        # Map modalities to plugins
        modality_to_plugin = {
            'vision': 'vision_irp',
            'audio': 'audio_irp',
            'language': 'language_irp',
            'clock': None,  # Built-in
        }

        required = []
        for target in targets:
            plugin = modality_to_plugin.get(target.modality)
            if plugin and target.priority > 0.1:  # Only load if priority is significant
                required.append(plugin)

        # Always keep memory plugin
        if 'memory_irp' not in required:
            required.append('memory_irp')

        return list(set(required))  # Deduplicate


class MockResourceManager(IResourceManager):
    """Mock resource manager for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.active_plugins = {}
        self.memory_usage = 0.0

    def update(self, required_plugins: List[str]) -> Dict[str, Any]:
        """Mock plugin loading/unloading"""
        # In real implementation: load/unload plugins
        # For mock: just track what should be loaded

        # Unload plugins not in required
        to_unload = [p for p in self.active_plugins if p not in required_plugins]
        for plugin in to_unload:
            del self.active_plugins[plugin]
            self.memory_usage -= 0.5  # Mock: 0.5GB per plugin

        # Load new plugins
        to_load = [p for p in required_plugins if p not in self.active_plugins]
        for plugin in to_load:
            self.active_plugins[plugin] = f"Mock_{plugin}"
            self.memory_usage += 0.5  # Mock: 0.5GB per plugin

        return self.active_plugins

    def get_active_plugins(self) -> Dict[str, Any]:
        return self.active_plugins

    def get_memory_usage(self) -> float:
        return max(0.0, self.memory_usage)


class MockOrchestrator(IOrchestrator):
    """Mock orchestrator for testing"""

    def run_cycle(self, targets: List[AttentionTarget],
                  active_plugins: Dict[str, Any],
                  atp_budget: float) -> Dict[str, PluginResult]:
        """Mock IRP execution"""
        results = {}
        atp_per_plugin = atp_budget / max(len(targets), 1)

        for target in targets:
            # Mock iterative refinement
            iterations = np.random.randint(3, 8)
            energy_start = np.random.uniform(2.0, 3.0)

            # Mock energy trajectory (decreasing)
            trajectory = [energy_start]
            for i in range(iterations - 1):
                next_energy = trajectory[-1] * np.random.uniform(0.7, 0.95)
                trajectory.append(next_energy)

            results[target.modality] = PluginResult(
                plugin_id=f"{target.modality}_irp",
                output=None,  # Mock: no actual output
                energy_trajectory=trajectory,
                iterations=iterations,
                atp_used=iterations * 100,  # 100 ATP per iteration
                converged=True,
                timestamp=time.time()
            )

        return results


class MockMemorySystem(IMemorySystem):
    """Mock memory system for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.buffer = deque(maxlen=config.get('buffer_size', 100))

    def update(self, cycle_state: CycleState):
        """Store cycle state in buffer"""
        self.buffer.append(cycle_state)

    def retrieve_context(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Return mock context"""
        return {'recent_cycles': len(self.buffer)}

    def consolidate(self):
        """Mock consolidation"""
        pass


class MockTrustTracker(ITrustTracker):
    """Mock trust tracker for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trust_scores = {}

    def update(self, results: Dict[str, PluginResult]):
        """Update trust scores"""
        for plugin_id, result in results.items():
            # Compute trust metrics
            if len(result.energy_trajectory) > 1:
                # Monotonicity: fraction of decreasing steps
                dE = [result.energy_trajectory[i+1] - result.energy_trajectory[i]
                      for i in range(len(result.energy_trajectory)-1)]
                monotonic = sum(1 for d in dE if d < 0) / len(dE) if dE else 0.0

                # Update trust (exponential moving average)
                old_trust = self.trust_scores.get(plugin_id, 0.5)
                new_trust = 0.9 * old_trust + 0.1 * monotonic
                self.trust_scores[plugin_id] = new_trust

    def get_trust_scores(self) -> Dict[str, float]:
        return self.trust_scores

    def adjust_atp_budget(self, base_budget: float) -> Dict[str, float]:
        """Allocate ATP based on trust"""
        if not self.trust_scores:
            return {}

        total_trust = sum(self.trust_scores.values())
        if total_trust == 0:
            return {p: base_budget / len(self.trust_scores)
                    for p in self.trust_scores}

        return {
            plugin: (trust / total_trust) * base_budget
            for plugin, trust in self.trust_scores.items()
        }


class MockEffectorHub(IEffectorHub):
    """Mock effector hub for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def execute(self, actions: Dict[str, Any]):
        """Mock action execution"""
        pass

    def get_available_effectors(self) -> List[str]:
        return []


class MockMetabolicController(IMetabolicController):
    """Mock metabolic controller for testing"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.state = 'WAKE'
        self.fatigue = 0.0
        self.stress = 0.0

    def update(self, energy_level: float, fatigue: float, stress: float) -> str:
        """Update metabolic state"""
        self.fatigue = fatigue
        self.stress = stress

        # Simple state transitions
        if energy_level < 0.2:
            self.state = 'REST'
        elif stress > 0.9:
            self.state = 'CRISIS'
        elif fatigue > 0.8:
            self.state = 'DREAM'
        else:
            self.state = 'WAKE'

        return self.state

    def get_state(self) -> str:
        return self.state

    def get_attention_breadth(self) -> float:
        breadth_map = {
            'WAKE': 0.7,
            'FOCUS': 0.3,
            'REST': 0.1,
            'DREAM': 0.0,
            'CRISIS': 1.0
        }
        return breadth_map.get(self.state, 0.5)


# =============================================================================
# Main SAGESystem Class
# =============================================================================

class SAGESystem:
    """
    Unified SAGE consciousness kernel.

    This class integrates all SAGE components into a continuous loop that:
    1. Senses the environment
    2. Evaluates salience (what matters)
    3. Allocates attention
    4. Plans resources
    5. Executes IRP plugins
    6. Updates memory and trust
    7. Takes actions
    8. Manages metabolic state

    The loop runs continuously, with all state explicit and manageable.
    """

    def __init__(self, config: Optional[SAGEConfig] = None,
                 use_mock_components: bool = True):
        """
        Initialize SAGESystem.

        Args:
            config: Configuration object (defaults to SAGEConfig())
            use_mock_components: If True, use mock implementations for testing
        """
        self.config = config or SAGEConfig()
        self.use_mock = use_mock_components

        # System state
        self.cycle_count = 0
        self.running = False
        self.energy_level = 1.0
        self.fatigue = 0.0
        self.stress = 0.0

        # Initialize components
        self._init_components()

        # Telemetry
        self.cycle_times = deque(maxlen=100)
        self.error_count = 0

    def _init_components(self):
        """Initialize all system components"""
        config_dict = {
            'enabled_sensors': ['vision', 'clock'],
            'buffer_size': 100,
        }

        if self.use_mock:
            # Use mock implementations
            self.sensor_hub = MockSensorHub(config_dict)
            self.snarc_scorer = MockSNARCScorer(config_dict)
            self.attention_allocator = MockAttentionAllocator()
            self.resource_planner = MockResourcePlanner()
            self.resource_manager = MockResourceManager(config_dict)
            self.orchestrator = MockOrchestrator()
            self.memory_system = MockMemorySystem(config_dict)
            self.trust_tracker = MockTrustTracker(config_dict)
            self.effector_hub = MockEffectorHub(config_dict)
            self.metabolic_controller = MockMetabolicController(config_dict)
        else:
            # Use real implementations (to be implemented)
            raise NotImplementedError("Real implementations not yet available")

    def _cycle(self) -> CycleState:
        """
        Execute one consciousness cycle.

        Returns:
            CycleState with complete cycle information
        """
        cycle_start = time.time()

        try:
            # 1. SENSING: Gather observations from sensors
            observations = self.sensor_hub.poll()

            # 2. SALIENCE EVALUATION: Compute SNARC scores
            salience_scores = self.snarc_scorer.evaluate(observations)

            # 3. ATTENTION ALLOCATION: Determine priorities
            metabolic_state = self.metabolic_controller.get_state()
            atp_budget = self._compute_atp_budget()
            attention_targets = self.attention_allocator.allocate(
                salience_scores, metabolic_state, atp_budget
            )

            # 4. RESOURCE PLANNING: Determine needed plugins
            active_resources = self.resource_manager.get_active_plugins()
            memory_available = 8.0 - self.resource_manager.get_memory_usage()
            required_plugins = self.resource_planner.plan(
                attention_targets, active_resources, memory_available
            )

            # 5. RESOURCE LOADING: Load/unload plugins
            active_plugins = self.resource_manager.update(required_plugins)

            # 6. IRP PLUGIN EXECUTION: Run iterative refinement
            plugin_results = self.orchestrator.run_cycle(
                attention_targets, active_plugins, atp_budget
            )

            # 7. ACTION EXECUTION: Execute effector commands
            actions = self._extract_actions(plugin_results)
            self.effector_hub.execute(actions)

            # 8. TRUST UPDATE: Update trust scores
            self.trust_tracker.update(plugin_results)

            # 9. MEMORY UPDATE: Store cycle state
            atp_used = sum(r.atp_used for r in plugin_results.values())
            cycle_state = CycleState(
                cycle_id=self.cycle_count,
                timestamp=cycle_start,
                observations=observations,
                salience_scores=salience_scores,
                attention_targets=attention_targets,
                required_plugins=required_plugins,
                plugin_results=plugin_results,
                actions=actions,
                metabolic_state=metabolic_state,
                energy_level=self.energy_level,
                atp_budget=atp_budget,
                atp_remaining=atp_budget - atp_used
            )
            self.memory_system.update(cycle_state)

            # 10. METABOLIC STATE UPDATE: Update energy and state
            cycle_time = time.time() - cycle_start
            self._update_metabolic_state(cycle_state, cycle_time)

            # Update cycle count and telemetry
            self.cycle_count += 1
            self.cycle_times.append(cycle_time)

            return cycle_state

        except Exception as e:
            self.error_count += 1
            self.stress = min(1.0, self.stress + 0.1)
            raise RuntimeError(f"Cycle {self.cycle_count} failed: {e}") from e

    def _compute_atp_budget(self) -> float:
        """Compute ATP budget for this cycle"""
        # Base budget from metabolic state
        state_budgets = {
            'WAKE': 1000.0,
            'FOCUS': 1500.0,
            'REST': 500.0,
            'DREAM': 300.0,
            'CRISIS': 2000.0
        }
        base = state_budgets.get(self.metabolic_controller.get_state(), 1000.0)

        # Adjust based on energy level
        return base * self.energy_level

    def _extract_actions(self, plugin_results: Dict[str, PluginResult]) -> Dict[str, Any]:
        """Extract effector commands from plugin results"""
        # In real implementation: parse plugin outputs for action commands
        # For now: empty (no actions)
        return {}

    def _update_metabolic_state(self, cycle_state: CycleState, cycle_time: float):
        """Update energy, fatigue, and metabolic state"""
        # Energy consumption (proportional to ATP used)
        energy_consumed = cycle_state.atp_budget - cycle_state.atp_remaining
        self.energy_level = max(0.0, self.energy_level - energy_consumed / 10000.0)

        # Fatigue accumulation
        self.fatigue = min(1.0, self.fatigue + 0.01)

        # Stress from errors
        if self.error_count > 0:
            self.stress = min(1.0, self.stress + 0.05)
        else:
            self.stress = max(0.0, self.stress - 0.01)

        # Update metabolic state
        self.metabolic_controller.update(
            self.energy_level, self.fatigue, self.stress
        )

    def run(self, max_cycles: Optional[int] = None, log_interval: int = 10):
        """
        Run the continuous consciousness loop.

        Args:
            max_cycles: Maximum cycles to run (None = infinite)
            log_interval: Print status every N cycles
        """
        self.running = True
        print(f"[SAGE] Starting consciousness loop...")
        print(f"[SAGE] Metabolic state: {self.metabolic_controller.get_state()}")
        print(f"[SAGE] Energy: {self.energy_level:.1%}")

        try:
            while self.running:
                cycle_state = self._cycle()

                # Logging
                if self.cycle_count % log_interval == 0:
                    self._print_status(cycle_state)

                # Check termination
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n[SAGE] Reached max cycles ({max_cycles})")
                    break

                # Emergency stop on critical energy
                if self.energy_level < 0.05:
                    print(f"\n[SAGE] Critical energy level - shutting down")
                    break

        except KeyboardInterrupt:
            print(f"\n[SAGE] Interrupted by user")
        except Exception as e:
            print(f"\n[SAGE] Fatal error: {e}")
            raise
        finally:
            self.running = False
            self._print_final_stats()

    def _print_status(self, cycle_state: CycleState):
        """Print cycle status"""
        avg_time = np.mean(self.cycle_times) if self.cycle_times else 0
        trust_scores = self.trust_tracker.get_trust_scores()

        print(f"\n[SAGE] Cycle {cycle_state.cycle_id} | "
              f"{avg_time*1000:.0f}ms | "
              f"Energy: {self.energy_level:.0%} | "
              f"ATP: {cycle_state.atp_remaining:.0f}/{cycle_state.atp_budget:.0f} | "
              f"State: {cycle_state.metabolic_state}")

        # Print salience breakdown
        print(f"  Salience: ", end="")
        for modality, score in cycle_state.salience_scores.items():
            print(f"{modality}={score.combined:.2f} ", end="")
        print()

        # Print plugin results
        print(f"  Plugins: ", end="")
        for plugin_id, result in cycle_state.plugin_results.items():
            print(f"{plugin_id}({result.iterations}it, {result.atp_used:.0f}ATP) ", end="")
        print()

        # Print trust scores
        if trust_scores:
            print(f"  Trust: ", end="")
            for plugin, trust in trust_scores.items():
                print(f"{plugin}={trust:.2f} ", end="")
            print()

    def _print_final_stats(self):
        """Print final statistics"""
        print(f"\n[SAGE] Final Statistics:")
        print(f"  Total cycles: {self.cycle_count}")
        print(f"  Total errors: {self.error_count}")
        print(f"  Final energy: {self.energy_level:.1%}")
        print(f"  Final fatigue: {self.fatigue:.1%}")
        print(f"  Final stress: {self.stress:.1%}")

        if self.cycle_times:
            print(f"  Avg cycle time: {np.mean(self.cycle_times)*1000:.1f}ms")
            print(f"  Min cycle time: {np.min(self.cycle_times)*1000:.1f}ms")
            print(f"  Max cycle time: {np.max(self.cycle_times)*1000:.1f}ms")

        trust_scores = self.trust_tracker.get_trust_scores()
        if trust_scores:
            print(f"  Final trust scores:")
            for plugin, trust in trust_scores.items():
                print(f"    {plugin}: {trust:.3f}")

    def stop(self):
        """Stop the consciousness loop"""
        self.running = False


# =============================================================================
# Entry Point
# =============================================================================

def main():
    """Main entry point for testing"""
    print("=" * 60)
    print("SAGE System - Unified Consciousness Kernel")
    print("=" * 60)

    # Create system with mock components
    system = SAGESystem(use_mock_components=True)

    # Run for 20 cycles
    system.run(max_cycles=20, log_interval=5)


if __name__ == "__main__":
    main()
