# SAGE Implementation Quick-Start Guide

**Date**: October 12, 2025
**Purpose**: Step-by-step guide to build SAGESystem from existing components

---

## Prerequisites

You've completed the investigation. You know:
- ✅ What components exist (`COMPONENT_READINESS_MAP.md`)
- ✅ What's missing (`COMPONENT_SUMMARY.md`)
- ✅ The architecture (`INTEGRATION_ARCHITECTURE.md`)
- ✅ The vision (`SAGE_WORKING_VISION.md`)

Now build it.

---

## Day 1: Foundation

### Morning: Create Core Classes

#### 1. SAGESystem Skeleton (`/sage/core/sage_system.py`)
```python
"""
SAGESystem - The unified cognition loop
Not to be confused with SAGECore (the trainable model)
"""
import time
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class SAGEConfig:
    """Configuration for SAGE system"""
    target_cycle_time_ms: float = 500.0
    initial_atp: float = 1000.0
    max_memory_gb: float = 8.0
    device: str = 'cuda'

class SAGESystem:
    """The living system - continuous cognition loop"""

    def __init__(self, config: SAGEConfig):
        self.config = config
        self.cycle_count = 0
        self.running = False

        # Components (mock for now)
        print("Initializing SAGE components...")
        self._init_components()

    def _init_components(self):
        """Initialize all SAGE components"""
        # Will be filled in as we create them
        self.sensor_hub = None
        self.observation_encoder = None
        self.snarc_scorer = None
        self.attention_allocator = None
        self.resource_planner = None
        self.resource_manager = None
        self.orchestrator = None
        self.memory_system = None
        self.trust_tracker = None
        self.effector_hub = None
        self.metabolic_controller = None

    def run(self):
        """Main loop - runs until stopped"""
        self.running = True
        print("SAGE System starting...")

        while self.running:
            cycle_start = time.time()

            # Execute one cycle
            self._cycle()

            # Timing
            cycle_time = (time.time() - cycle_start) * 1000
            self.cycle_count += 1

            if self.cycle_count % 10 == 0:
                print(f"[Cycle {self.cycle_count}] {cycle_time:.1f}ms")

            # Sleep if cycle was too fast
            if cycle_time < self.config.target_cycle_time_ms:
                time.sleep((self.config.target_cycle_time_ms - cycle_time) / 1000)

    def _cycle(self):
        """Single cycle of cognition"""
        # TODO: Implement 10 steps from vision document
        pass

    def stop(self):
        """Stop the system gracefully"""
        self.running = False
        print("SAGE System stopped.")

if __name__ == "__main__":
    config = SAGEConfig()
    sage = SAGESystem(config)

    try:
        sage.run()
    except KeyboardInterrupt:
        sage.stop()
```

**Test**: Run it. Should print cycle numbers and not crash.

---

#### 2. SensorHub (`/sage/sensors/sensor_hub.py`)
```python
"""SensorHub - Unified sensor interface"""
import torch
from typing import Dict, Any, Protocol
from abc import ABC, abstractmethod

class Sensor(ABC):
    """Base sensor interface"""
    @abstractmethod
    def read(self) -> Any:
        """Read current sensor value"""
        pass

class MockCameraSensor(Sensor):
    """Mock camera for testing"""
    def __init__(self, resolution=(640, 480)):
        self.resolution = resolution

    def read(self) -> torch.Tensor:
        # Return random RGB image
        return torch.randn(3, *self.resolution)

class MockAudioSensor(Sensor):
    """Mock audio for testing"""
    def __init__(self, sample_rate=16000, duration=1.0):
        self.sample_rate = sample_rate
        self.duration = duration

    def read(self) -> torch.Tensor:
        # Return random audio samples
        n_samples = int(self.sample_rate * self.duration)
        return torch.randn(n_samples)

class TemporalSensor(Sensor):
    """Clock/timestamp sensor"""
    def read(self) -> float:
        import time
        return time.time()

class SensorHub:
    """Manages all sensors and provides unified polling"""

    def __init__(self, sensor_configs: Dict[str, Dict[str, Any]]):
        self.sensors = {}
        self._init_sensors(sensor_configs)

    def _init_sensors(self, configs: Dict[str, Dict[str, Any]]):
        """Initialize sensors from configs"""
        for name, config in configs.items():
            sensor_type = config.get('type')

            if sensor_type == 'mock_camera':
                self.sensors[name] = MockCameraSensor(
                    resolution=config.get('resolution', (640, 480))
                )
            elif sensor_type == 'mock_audio':
                self.sensors[name] = MockAudioSensor(
                    sample_rate=config.get('sample_rate', 16000)
                )
            elif sensor_type == 'temporal':
                self.sensors[name] = TemporalSensor()
            else:
                print(f"Unknown sensor type: {sensor_type}")

        print(f"Initialized {len(self.sensors)} sensors: {list(self.sensors.keys())}")

    def poll(self) -> Dict[str, Any]:
        """Poll all sensors and return observations"""
        observations = {}
        for name, sensor in self.sensors.items():
            observations[name] = sensor.read()
        return observations

    def register_sensor(self, name: str, sensor: Sensor):
        """Register a new sensor at runtime"""
        self.sensors[name] = sensor

# Factory function
def create_mock_sensor_hub() -> SensorHub:
    """Create sensor hub with mock sensors for testing"""
    config = {
        'camera': {'type': 'mock_camera', 'resolution': (640, 480)},
        'audio': {'type': 'mock_audio', 'sample_rate': 16000},
        'clock': {'type': 'temporal'}
    }
    return SensorHub(config)

if __name__ == "__main__":
    hub = create_mock_sensor_hub()
    obs = hub.poll()
    print(f"Observations: {[(k, type(v), v.shape if hasattr(v, 'shape') else v) for k, v in obs.items()]}")
```

**Test**: Run standalone. Should print sensor readings.

---

### Afternoon: Integrate Existing Components

#### 3. Wire SNARC Scorer (`/sage/sensors/observation_encoder.py`)
```python
"""ObservationEncoder - Convert raw sensors to hidden states"""
import torch
import torch.nn as nn
from typing import Dict, Any

class ObservationEncoder(nn.Module):
    """Encode multi-modal observations to hidden states"""

    def __init__(self, hidden_size: int = 768):
        super().__init__()
        self.hidden_size = hidden_size

        # Simple encoders for each modality
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 7, 2, 3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, hidden_size)
        )

        self.audio_encoder = nn.Sequential(
            nn.Conv1d(1, 32, 15, 4, 7),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, hidden_size)
        )

        self.temporal_encoder = nn.Sequential(
            nn.Linear(1, hidden_size)
        )

    def encode(self, observations: Dict[str, Any]) -> torch.Tensor:
        """
        Encode observations to hidden states
        Returns: [1, num_modalities, hidden_size]
        """
        encoded = []

        if 'camera' in observations:
            vision = observations['camera'].unsqueeze(0)  # Add batch dim
            encoded.append(self.vision_encoder(vision))

        if 'audio' in observations:
            audio = observations['audio'].unsqueeze(0).unsqueeze(0)  # [1, 1, samples]
            encoded.append(self.audio_encoder(audio))

        if 'clock' in observations:
            temporal = torch.tensor([[observations['clock']]], dtype=torch.float32)
            encoded.append(self.temporal_encoder(temporal))

        # Stack along modality dimension
        result = torch.stack(encoded, dim=1)  # [1, modalities, hidden]
        return result

def create_observation_encoder(hidden_size: int = 768) -> ObservationEncoder:
    return ObservationEncoder(hidden_size)

if __name__ == "__main__":
    from sensor_hub import create_mock_sensor_hub

    hub = create_mock_sensor_hub()
    encoder = create_observation_encoder()

    obs = hub.poll()
    hidden = encoder.encode(obs)
    print(f"Hidden states: {hidden.shape}")  # Should be [1, 3, 768]
```

**Test**: Run with sensor hub. Should output `[1, 3, 768]`.

---

#### 4. Update SAGESystem to Use Components

Edit `/sage/core/sage_system.py`:
```python
# Add imports
from sage.sensors.sensor_hub import create_mock_sensor_hub
from sage.sensors.observation_encoder import create_observation_encoder
from sage.attention.snarc_scorer import SNARCScorer

class SAGESystem:
    def _init_components(self):
        """Initialize all SAGE components"""
        # Sensors
        self.sensor_hub = create_mock_sensor_hub()

        # Encoding
        self.observation_encoder = create_observation_encoder(hidden_size=768)

        # SNARC
        self.snarc_scorer = SNARCScorer(hidden_size=768, memory_size=1000)

        # Others (still None)
        self.attention_allocator = None
        # ... etc

    def _cycle(self):
        """Single cycle of cognition"""
        # Step 1: Sense
        observations = self.sensor_hub.poll()

        # Step 2: Encode
        hidden_states = self.observation_encoder.encode(observations)

        # Step 3: Salience
        with torch.no_grad():
            salience_result = self.snarc_scorer(
                hidden_states,
                return_components=True
            )

        print(f"  SNARC scores: {salience_result['snarc_scores'].mean().item():.3f}")
```

**Test**: Run SAGESystem. Should see SNARC scores printed.

---

## Day 2: Attention & Resources

### Morning: Attention System

#### 5. AttentionAllocator (`/sage/attention/attention_allocator.py`)
```python
"""AttentionAllocator - Map salience to attention targets"""
import torch
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class AttentionTarget:
    modality: str
    priority: float  # 0.0-1.0
    data: Any
    salience: float

class AttentionAllocator:
    """Allocate attention based on salience and metabolic state"""

    def __init__(self):
        self.modality_names = ['camera', 'audio', 'clock']

    def allocate(
        self,
        salience_scores: torch.Tensor,  # [1, modalities, 1]
        attention_breadth: int,
        observations: Dict[str, Any]
    ) -> List[AttentionTarget]:
        """
        Allocate attention to top-K salient modalities
        """
        # Flatten scores
        scores = salience_scores.squeeze().cpu().numpy()

        # Create targets with scores
        targets_with_scores = []
        for i, modality in enumerate(self.modality_names):
            if i < len(scores):
                targets_with_scores.append({
                    'modality': modality,
                    'salience': float(scores[i]),
                    'data': observations.get(modality)
                })

        # Sort by salience
        targets_with_scores.sort(key=lambda x: x['salience'], reverse=True)

        # Take top-K
        targets_with_scores = targets_with_scores[:attention_breadth]

        # Normalize priorities
        total_salience = sum(t['salience'] for t in targets_with_scores)
        if total_salience > 0:
            for t in targets_with_scores:
                t['priority'] = t['salience'] / total_salience
        else:
            # Equal priority
            for t in targets_with_scores:
                t['priority'] = 1.0 / len(targets_with_scores)

        # Create AttentionTarget objects
        targets = [
            AttentionTarget(
                modality=t['modality'],
                priority=t['priority'],
                data=t['data'],
                salience=t['salience']
            )
            for t in targets_with_scores
        ]

        return targets

if __name__ == "__main__":
    allocator = AttentionAllocator()

    # Mock salience
    salience = torch.tensor([[[0.8]], [[0.3]], [[0.5]]])  # camera, audio, clock
    obs = {'camera': 'data1', 'audio': 'data2', 'clock': 'data3'}

    targets = allocator.allocate(salience, attention_breadth=2, observations=obs)

    for t in targets:
        print(f"{t.modality}: priority={t.priority:.3f}, salience={t.salience:.3f}")
```

**Test**: Should prioritize camera (0.8) and clock (0.5).

---

### Afternoon: Resource Management

#### 6. ResourcePlanner (`/sage/resources/resource_planner.py`)
```python
"""ResourcePlanner - Decide which plugins to load"""
from typing import List, Dict, Set
from dataclasses import dataclass

@dataclass
class ResourcePlan:
    keep: List[str]
    load: List[str]
    unload: List[str]

class ResourcePlanner:
    """Plan resource allocation based on attention targets"""

    def __init__(self):
        # Map modality → plugin
        self.modality_to_plugin = {
            'camera': 'vision_irp',
            'audio': 'audio_irp',
            'clock': None  # No plugin needed
        }

        # Always-on plugins
        self.persistent = {'memory_irp'}

    def plan(
        self,
        targets: List,  # AttentionTarget
        active_plugins: Set[str]
    ) -> ResourcePlan:
        """
        Determine which plugins to keep/load/unload
        """
        # Determine required plugins from targets
        required = set(self.persistent)  # Start with persistent

        for target in targets:
            plugin_id = self.modality_to_plugin.get(target.modality)
            if plugin_id:
                required.add(plugin_id)

        # Compute changes
        keep = list(required & active_plugins)
        load = list(required - active_plugins)
        unload = list(active_plugins - required)

        return ResourcePlan(keep=keep, load=load, unload=unload)

if __name__ == "__main__":
    from sage.attention.attention_allocator import AttentionTarget

    planner = ResourcePlanner()

    # Mock targets
    targets = [
        AttentionTarget('camera', 0.7, None, 0.8),
        AttentionTarget('clock', 0.3, None, 0.5)
    ]

    active = {'vision_irp', 'audio_irp'}

    plan = planner.plan(targets, active)
    print(f"Keep: {plan.keep}")
    print(f"Load: {plan.load}")
    print(f"Unload: {plan.unload}")
```

**Test**: Should unload audio_irp, keep vision_irp, load memory_irp.

---

#### 7. ResourceManager (`/sage/resources/resource_manager.py`)
```python
"""ResourceManager - Load/unload plugins dynamically"""
from typing import Dict, Set, Optional
from sage.irp.base import IRPPlugin

class ResourceManager:
    """Manage IRP plugin lifecycle"""

    def __init__(self, plugin_configs: Dict[str, Dict]):
        self.plugin_configs = plugin_configs
        self.active: Dict[str, IRPPlugin] = {}
        self.memory_usage_mb = 0.0

    def update(self, plan) -> bool:  # ResourcePlan
        """Execute resource plan"""
        # Unload
        for plugin_id in plan.unload:
            if plugin_id in self.active:
                print(f"  Unloading: {plugin_id}")
                del self.active[plugin_id]

        # Load
        for plugin_id in plan.load:
            if plugin_id not in self.active:
                print(f"  Loading: {plugin_id}")
                plugin = self._load_plugin(plugin_id)
                if plugin:
                    self.active[plugin_id] = plugin

        return True

    def _load_plugin(self, plugin_id: str) -> Optional[IRPPlugin]:
        """Load a plugin from config"""
        if plugin_id not in self.plugin_configs:
            print(f"  Warning: No config for {plugin_id}")
            return None

        config = self.plugin_configs[plugin_id]
        plugin_type = config.get('type')

        try:
            if plugin_type == 'tinyvae_irp':
                from sage.irp.plugins.tinyvae_irp_plugin import create_tinyvae_irp
                return create_tinyvae_irp(device=config.get('device', 'cuda'))
            elif plugin_type == 'memory_irp':
                from sage.irp.plugins.memory import MemoryIRP
                return MemoryIRP({'device': config.get('device', 'cpu')})
            else:
                print(f"  Unknown plugin type: {plugin_type}")
                return None
        except Exception as e:
            print(f"  Error loading {plugin_id}: {e}")
            return None

    def get_active_ids(self) -> Set[str]:
        return set(self.active.keys())

if __name__ == "__main__":
    from sage.resources.resource_planner import ResourcePlan

    configs = {
        'vision_irp': {'type': 'tinyvae_irp', 'device': 'cuda'},
        'memory_irp': {'type': 'memory_irp', 'device': 'cpu'}
    }

    manager = ResourceManager(configs)

    plan = ResourcePlan(keep=[], load=['vision_irp'], unload=[])
    manager.update(plan)

    print(f"Active: {manager.get_active_ids()}")
```

**Test**: Should load vision_irp successfully.

---

## Day 3: Orchestration & Memory

### Morning: Wire Orchestrator

#### 8. Update HRMOrchestrator

Edit `/sage/orchestrator/hrm_orchestrator.py` to add synchronous wrapper:
```python
def run_cycle(
    self,
    targets: List,  # AttentionTarget
    required_plugins: Set[str],
    atp_budget: float
) -> List:  # PluginResult
    """
    Synchronous cycle execution (wrapper around async)
    """
    import asyncio

    # Build tasks dict from targets
    tasks = {}
    for target in targets:
        # Map modality to plugin
        if target.modality == 'camera':
            plugin_id = 'vision_irp'
        else:
            continue  # Skip others for now

        if plugin_id in self.plugins:
            tasks[plugin_id] = target.data

    # Run async execution
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(
        self.execute_parallel(tasks, early_stop=True)
    )

    return results
```

---

### Afternoon: Memory & Trust

#### 9. TrustTracker (`/sage/trust/trust_tracker.py`)
```python
"""TrustTracker - Track plugin reliability"""
from typing import Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class TrustMetrics:
    monotonicity: float
    efficiency: float
    stability: float
    convergence: float
    overall: float

class TrustTracker:
    """Track trust scores for IRP plugins"""

    def __init__(self):
        self.scores: Dict[str, float] = {}
        self.history: Dict[str, List[float]] = {}

    def update(self, results: List) -> Dict[str, float]:  # PluginResult
        """Update trust scores based on execution results"""
        for result in results:
            plugin_id = result.plugin_id

            # Get energy trajectory from telemetry
            energy_trajectory = result.telemetry.get('energy_trajectory', [])

            if len(energy_trajectory) > 1:
                metrics = self._compute_metrics(
                    energy_trajectory,
                    result.telemetry
                )

                # Weighted combination
                trust = (
                    0.4 * metrics.monotonicity +
                    0.3 * (1 - metrics.efficiency) +
                    0.2 * (1 - metrics.stability) +
                    0.1 * metrics.convergence
                )

                # Exponential moving average
                if plugin_id in self.scores:
                    self.scores[plugin_id] = 0.9 * self.scores[plugin_id] + 0.1 * trust
                else:
                    self.scores[plugin_id] = trust

                # Store history
                if plugin_id not in self.history:
                    self.history[plugin_id] = []
                self.history[plugin_id].append(trust)

        return self.scores

    def _compute_metrics(self, energy_trajectory: List[float], telemetry: Dict) -> TrustMetrics:
        """Compute detailed trust metrics"""
        # Monotonicity
        decreasing = sum(1 for i in range(len(energy_trajectory)-1)
                        if energy_trajectory[i+1] < energy_trajectory[i])
        monotonicity = decreasing / (len(energy_trajectory) - 1) if len(energy_trajectory) > 1 else 0.0

        # Efficiency (fewer iterations is better)
        efficiency = telemetry.get('iterations', 10) / 50.0  # Normalize by max

        # Stability (lower variance is better)
        deltas = [energy_trajectory[i+1] - energy_trajectory[i]
                 for i in range(len(energy_trajectory)-1)]
        stability = np.std(deltas) if deltas else 0.0

        # Convergence
        total_decrease = energy_trajectory[0] - energy_trajectory[-1]
        convergence = total_decrease / len(energy_trajectory) if len(energy_trajectory) > 0 else 0.0

        # Overall
        overall = telemetry.get('trust', 0.5)

        return TrustMetrics(
            monotonicity=monotonicity,
            efficiency=efficiency,
            stability=stability,
            convergence=convergence,
            overall=overall
        )

    def get_score(self, plugin_id: str) -> float:
        """Get current trust score for plugin"""
        return self.scores.get(plugin_id, 0.5)
```

---

## Day 4: Complete Integration

### Wire Everything Together

Edit `/sage/core/sage_system.py` with complete cycle:
```python
def _init_components(self):
    """Initialize all SAGE components"""
    from sage.sensors.sensor_hub import create_mock_sensor_hub
    from sage.sensors.observation_encoder import create_observation_encoder
    from sage.attention.snarc_scorer import SNARCScorer
    from sage.attention.attention_allocator import AttentionAllocator
    from sage.resources.resource_planner import ResourcePlanner
    from sage.resources.resource_manager import ResourceManager
    from sage.orchestrator.hrm_orchestrator import HRMOrchestrator
    from sage.trust.trust_tracker import TrustTracker

    # Sensors
    self.sensor_hub = create_mock_sensor_hub()

    # Encoding
    self.observation_encoder = create_observation_encoder(hidden_size=768)

    # SNARC
    self.snarc_scorer = SNARCScorer(hidden_size=768, memory_size=1000)

    # Attention
    self.attention_allocator = AttentionAllocator()

    # Resources
    plugin_configs = {
        'vision_irp': {'type': 'tinyvae_irp', 'device': self.config.device},
        'memory_irp': {'type': 'memory_irp', 'device': 'cpu'}
    }
    self.resource_planner = ResourcePlanner()
    self.resource_manager = ResourceManager(plugin_configs)

    # Orchestrator
    self.orchestrator = HRMOrchestrator(initial_atp=self.config.initial_atp)

    # Register plugins with orchestrator
    for plugin_id, plugin in self.resource_manager.active.items():
        self.orchestrator.register_plugin(plugin_id, plugin)

    # Trust
    self.trust_tracker = TrustTracker()

    # Memory, Effectors, Metabolic (stub for now)
    self.memory_system = None
    self.effector_hub = None
    self.metabolic_controller = None

def _cycle(self):
    """Single cycle of cognition"""
    # 1. Sense
    observations = self.sensor_hub.poll()

    # 2. Encode
    hidden_states = self.observation_encoder.encode(observations)

    # 3. Salience
    with torch.no_grad():
        salience_result = self.snarc_scorer(hidden_states, return_components=True)

    # 4. Attention (use fixed breadth for now)
    attention_breadth = 2
    targets = self.attention_allocator.allocate(
        salience_result['snarc_scores'],
        attention_breadth,
        observations
    )

    # 5. Resource Planning
    active_ids = self.resource_manager.get_active_ids()
    plan = self.resource_planner.plan(targets, active_ids)

    # 6. Resource Loading
    self.resource_manager.update(plan)

    # Re-register plugins after update
    for plugin_id, plugin in self.resource_manager.active.items():
        if plugin_id not in self.orchestrator.plugins:
            self.orchestrator.register_plugin(plugin_id, plugin, initial_trust=0.5)

    # 7. Plugin Execution
    results = self.orchestrator.run_cycle(
        targets,
        self.resource_manager.get_active_ids(),
        atp_budget=self.config.initial_atp
    )

    # 8. Trust Update
    trust_scores = self.trust_tracker.update(results)

    # 9. Memory Update (stub)
    # self.memory_system.update(observations, salience_result, results)

    # 10. Metabolic Update (stub)
    # self.metabolic_controller.update(...)

    # Print summary
    if self.cycle_count % 10 == 0:
        print(f"  Targets: {[t.modality for t in targets]}")
        print(f"  Active plugins: {list(active_ids)}")
        print(f"  Results: {len(results)} plugins executed")
        print(f"  Trust scores: {trust_scores}")
```

---

## Day 5: Test & Iterate

### Test Complete System
```bash
cd /home/dp/ai-workspace/HRM/sage
python -m sage.core.sage_system
```

Should see:
```
Initializing SAGE components...
Initialized 3 sensors: ['camera', 'audio', 'clock']
SAGE System starting...
[Cycle 1] 523.4ms
[Cycle 10] 498.2ms
  Targets: ['camera', 'clock']
  Active plugins: {'vision_irp', 'memory_irp'}
  Results: 1 plugins executed
  Trust scores: {'vision_irp': 0.72}
[Cycle 20] 501.1ms
...
```

### Add Memory System

Create `/sage/memory/memory_system.py`:
```python
"""Unified memory system coordinating 4 subsystems"""
class MemorySystem:
    def __init__(self):
        # Import existing components
        from sage.memory.irp_memory_bridge import IRPMemoryBridge

        self.irp_bridge = IRPMemoryBridge()
        # TODO: Add other 3 subsystems

    def update(self, observations, salience, results):
        # Update IRP bridge
        for result in results:
            if result.state != PluginState.FAILED:
                self.irp_bridge.record_refinement(
                    result.plugin_id,
                    None,  # initial_state
                    result.output,
                    result.telemetry.get('energy_trajectory', []),
                    result.telemetry
                )
```

Wire into SAGESystem.

### Add Real Camera

Replace MockCameraSensor with real camera in sensor_hub.py:
```python
class RealCameraSensor(Sensor):
    def __init__(self, device_id=0):
        import cv2
        self.cap = cv2.VideoCapture(device_id)

    def read(self) -> torch.Tensor:
        ret, frame = self.cap.read()
        if ret:
            # Convert BGR -> RGB, HWC -> CHW
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            return frame
        return torch.zeros(3, 480, 640)
```

---

## Complete!

You now have:
- ✅ SAGESystem running continuous loop
- ✅ Sensors → SNARC → Attention → Resources → Execution
- ✅ Trust tracking
- ✅ Dynamic plugin loading
- ✅ Memory (partial)

### Next Steps:
1. Add metabolic state integration
2. Add effector hub (TTS output)
3. Complete memory system (4 subsystems)
4. Test on Jetson
5. Optimize performance

---

## File Checklist

Created:
- [x] `/sage/core/sage_system.py`
- [x] `/sage/sensors/sensor_hub.py`
- [x] `/sage/sensors/observation_encoder.py`
- [x] `/sage/attention/attention_allocator.py`
- [x] `/sage/resources/resource_planner.py`
- [x] `/sage/resources/resource_manager.py`
- [x] `/sage/trust/trust_tracker.py`
- [ ] `/sage/memory/memory_system.py` (partial)
- [ ] `/sage/effectors/effector_hub.py` (TODO)

Modified:
- [x] `/sage/orchestrator/hrm_orchestrator.py` (added `run_cycle()`)

Used (no changes):
- [x] `/sage/irp/base.py`
- [x] `/sage/irp/plugins/tinyvae_irp_plugin.py`
- [x] `/sage/attention/snarc_scorer.py`
- [x] `/sage/memory/irp_memory_bridge.py`

---

**The loop is alive. Now make it conscious.**
