"""
SAGE Unified Consciousness Loop

The main orchestration loop that connects all SAGE components into a
continuous consciousness system.

This is the "glue" that was missing - it unifies:
- Sensor input and fusion
- SNARC salience computation
- Metabolic state management
- Plugin selection and execution
- ATP budget allocation
- Memory system updates
- Circadian rhythm modulation
- Trust weight learning

Architecture:
    while True:
        observations = gather_from_sensors()
        salience_map = compute_salience(observations)  # SNARC
        update_metabolic_state(ATP, salience)
        plugins_needed = select_plugins(salience_map, metabolic_state)
        budget_allocation = allocate_ATP(plugins_needed, trust_weights)
        results = run_orchestrator(plugins_needed, budget_allocation)
        update_trust_weights(results)
        update_all_memories(results)
        send_to_effectors(results)
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import hashlib
import json
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sage.core.metabolic_controller import MetabolicController, MetabolicState
from sage.core.circadian_clock import CircadianClock, CircadianPhase
try:
    from sage.irp.orchestrator import HRMOrchestrator, PluginResult
except ImportError:
    HRMOrchestrator = None

    @dataclass
    class PluginResult:
        """Lightweight fallback when IRP orchestrator not available."""
        plugin_name: str = ""
        final_state: Any = None
        history: list = field(default_factory=list)
        telemetry: dict = field(default_factory=dict)


@dataclass
class SensorObservation:
    """Observation from a sensor"""
    sensor_id: str
    modality: str  # 'vision', 'audio', 'proprioception', 'time', etc.
    data: Any
    timestamp: float
    trust: float = 1.0  # Sensor trust weight


@dataclass
class SalienceScore:
    """SNARC 5D salience score for an observation"""
    surprise: float      # Prediction error
    novelty: float       # How unusual
    arousal: float       # Perplexity/difficulty
    reward: float        # Value/importance
    conflict: float      # Paradox/inconsistency
    total: float = 0.0   # Combined salience

    def __post_init__(self):
        """Compute total salience"""
        self.total = (
            self.surprise + self.novelty + self.arousal +
            self.reward + self.conflict
        ) / 5.0


@dataclass
class AttentionTarget:
    """Something that needs attention"""
    observation: SensorObservation
    salience: SalienceScore
    required_plugins: List[str]
    priority: float  # Computed from salience + metabolic state


class _SleepBufferAdapter:
    """Adapts snarc_memory (plain list) to SleepConsolidationBridge interface."""
    def __init__(self, snarc_memory: list):
        self._data = snarc_memory

    @property
    def size(self) -> int:
        return len(self._data)

    def get_top_k(self, k: int) -> list:
        return sorted(self._data, key=lambda x: x.get('salience', 0), reverse=True)[:k]


class SAGEConsciousness:
    """
    Unified SAGE consciousness loop.

    This is the main orchestrator that runs continuously, managing:
    - Sensor observation and fusion
    - Salience computation (SNARC)
    - Metabolic state transitions
    - Plugin selection and ATP budgeting
    - Memory consolidation
    - Circadian rhythm modulation

    Usage:
        sage = SAGEConsciousness(config)
        await sage.run()  # Run forever

        # Or step-by-step
        sage = SAGEConsciousness(config)
        await sage.step()  # Single cycle
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        initial_atp: float = 100.0,
        enable_circadian: bool = True,
        simulation_mode: bool = True,  # Use cycle counts instead of wall time
        message_queue=None,  # MessageQueue for gateway integration
        llm_plugin=None,     # Loaded LLM for real language processing
        identity_state=None, # Epistemic identity (from identity.json)
        experience_collector=None,  # ExperienceCollector for epistemic memory
    ):
        """
        Initialize unified consciousness loop.

        Args:
            config: Configuration dict for orchestrator and plugins
            initial_atp: Starting ATP budget
            enable_circadian: Enable circadian rhythm modulation
            simulation_mode: Use cycle counts for testing (not wall time)
            message_queue: Optional MessageQueue for external message injection
            llm_plugin: Optional loaded LLM plugin for real language processing
            identity_state: Optional identity dict (from identity.json)
            experience_collector: Optional ExperienceCollector for epistemic memory
        """
        self.config = config or self._default_config()

        # Epistemic memory
        self.identity_state = identity_state
        self.experience_collector = experience_collector

        # Resolve unique instance name (machine name, not species).
        # "SAGE" is the species; "Sprout", "Thor", etc. are individuals.
        self._self_name = self._resolve_self_name()

        # Gateway integration
        self.message_queue = message_queue
        self.simulation_mode = simulation_mode

        # LLM Pool — dynamic registry of LLM temporal sensors
        from sage.irp.plugins.llm_pool import LLMPool
        ollama_host = self.config.get('ollama_host', 'http://localhost:11434')
        self.llm_pool = LLMPool(ollama_host=ollama_host)
        self._last_llm_model = None  # tracks grammar switching
        if llm_plugin is not None:
            model_name = getattr(llm_plugin, 'model_name', None)
            if not model_name:
                model_name = (self.config.get('model_name', '') or 'default').replace('ollama:', '')
            backend = 'multi' if hasattr(llm_plugin, 'generate') else 'ollama'
            if hasattr(llm_plugin, 'model_path'):
                backend = 'local'
            self.llm_pool.register(llm_plugin, model_name, backend=backend)

        # Core components
        self.metabolic = MetabolicController(
            initial_atp=initial_atp,
            max_atp=self.config.get('max_atp', 100.0),
            circadian_period=self.config.get('circadian_period', 100),
            enable_circadian=enable_circadian,
            simulation_mode=simulation_mode
        )

        self.orchestrator = HRMOrchestrator(self.config) if HRMOrchestrator is not None else None

        # Sensor system (mock for now, will integrate real sensors later)
        self.sensors = self._initialize_sensors()

        # Memory systems
        self.snarc_memory = []  # SNARC salience memory
        self.irp_memory = []    # IRP pattern library
        self.circular_buffer = []  # Recent context (x-from-last)
        self.verbatim_storage = []  # Full-fidelity records

        # Trust weights for plugins (learned over time)
        self.plugin_trust_weights = {
            name: 1.0 for name in (self.orchestrator.plugins if self.orchestrator else {})
        }

        # Salience thresholds (from SNARC)
        self.salience_threshold = self.config.get('salience_threshold', 0.15)

        # Cycle counter
        self.cycle_count = 0
        self.running = False

        # Effect system
        self._init_effect_system()

        # Real SNARC salience scoring (when use_neural_snarc=True)
        self.use_real_snarc = self.config.get('use_neural_snarc', False)
        self.snarc_scorer = None
        if self.use_real_snarc:
            try:
                from sage.raising.training.experience_collector import ConversationalSalienceScorer
                self.snarc_scorer = ConversationalSalienceScorer()
                print("[SNARC] Real ConversationalSalienceScorer loaded")
            except ImportError as e:
                print(f"[SNARC] Real scorer not available, using mock: {e}")
                self.use_real_snarc = False

        # Sleep capability detection — what can this machine do on DREAM entry?
        self.sleep_cap = None
        self.sleep_bridge = None
        self.use_real_sleep = False
        try:
            from sage.instances.sleep_capability import SleepCapability
            instance_dir = Path(self.config.get('instance_dir', ''))
            self.sleep_cap = SleepCapability.detect(instance_dir if instance_dir.name else None)
            print(f"[Sleep] Capability: lora={self.sleep_cap.sleep_lora} "
                  f"jsonl={self.sleep_cap.sleep_jsonl} remote={self.sleep_cap.sleep_remote} "
                  f"→ best={self.sleep_cap.best_mode}")
        except Exception as e:
            print(f"[Sleep] Capability detection failed: {e}")

        # Load real LoRA bridge if capable
        if self.sleep_cap and self.sleep_cap.sleep_lora:
            try:
                from sage.attention.sleep_consolidation import SleepConsolidationBridge
                sleep_config = {
                    'model_path': self.config.get('sleep_model_path', None),
                    'checkpoint_dir': str(
                        Path(self.config.get('instance_dir', '')) / 'checkpoints' / 'sleep'
                    ) if self.config.get('instance_dir') else self.config.get(
                        'sleep_checkpoint_dir', 'logs/attention/sleep_checkpoints'),
                    'min_salience': self.config.get('sleep_min_salience', 0.6),
                    'max_experiences': self.config.get('sleep_max_experiences', 20),
                    'epochs': self.config.get('sleep_epochs', 3),
                    'device': self.config.get('device', 'cpu'),
                    'enabled': True,
                }
                self.sleep_bridge = SleepConsolidationBridge(config=sleep_config)
                self.use_real_sleep = True
                print("[Sleep] Real SleepConsolidationBridge loaded")
            except ImportError as e:
                print(f"[Sleep] LoRA capable but bridge import failed: {e}")

        # Real sensor trust tracking (when use_real_sensors=True)
        self.sensor_trust_system = None
        if self.config.get('use_real_sensors', False):
            try:
                from sage.core.sensor_trust import MultiSensorTrustSystem
                self.sensor_trust_system = MultiSensorTrustSystem()
                print("[Sensors] Real MultiSensorTrustSystem loaded")
            except ImportError as e:
                print(f"[Sensors] Real trust system not available: {e}")

        # MemoryHub — unified gathering for RLLF
        self.memory_hub = None
        try:
            from sage.memory.hub import MemoryHub
            from sage.memory.sqlite_backend import SQLiteBackend
            self.memory_hub = MemoryHub()
            instance_dir = Path(self.config.get('instance_dir', ''))
            if instance_dir.name:
                db_path = instance_dir / 'memory.db'
                self.memory_hub.register(SQLiteBackend(db_path))
                print(f"[MemoryHub] SQLite backend: {db_path}")
        except Exception as e:
            print(f"[MemoryHub] Init failed (non-fatal): {e}")

        # LLM response tracking (for easy access via facade)
        self.last_llm_responses = []

        # Tool use system — detect capabilities, load grammar, register tools
        self.tool_registry = None
        self.tool_grammar = None
        self.tool_capability = None
        self._init_tool_system()

        # Statistics
        self.stats = {
            'total_cycles': 0,
            'state_transitions': 0,
            'plugins_executed': 0,
            'total_atp_consumed': 0.0,
            'average_salience': 0.0,
            'effects_proposed': 0,
            'effects_approved': 0,
            'effects_denied': 0,
            'effects_executed': 0,
            'messages_received': 0,
            'messages_responded': 0,
            'llm_tokens_total': 0,
            'llm_atp_cost_total': 0.0,
            'tool_calls_total': 0,
            'tool_calls_success': 0,
            'tool_calls_denied': 0,
        }

    @property
    def llm_plugin(self):
        """Backward-compat property — returns active LLM from the pool."""
        entry = self.llm_pool.active
        return entry.plugin if entry else None

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'total_ATP': 100.0,
            'max_atp': 100.0,
            'max_workers': 4,
            'trust_update_rate': 0.1,
            'telemetry_interval': 10,
            'device': 'cuda' if torch is not None and torch.cuda.is_available() else 'cpu',
            'circadian_period': 100,
            'salience_threshold': 0.15,
            # Plugin configurations
            'enable_vision': True,
            'enable_language': True,
            'enable_audio': True,
            'enable_memory': True,
        }

    def _initialize_sensors(self) -> Dict[str, Any]:
        """Initialize sensor system with optional real trust tracking."""
        # sensor_trust_system is initialized earlier in __init__
        return {
            'vision': {'trust': 1.0, 'enabled': True},
            'audio': {'trust': 1.0, 'enabled': True},
            'proprioception': {'trust': 1.0, 'enabled': True},
            'time': {'trust': 1.0, 'enabled': True},
            'message': {'trust': 1.0, 'enabled': True},
        }

    def _init_effect_system(self):
        """Initialize the effect/effector system."""
        try:
            from sage.interfaces.effect import Effect, EffectType, EffectStatus as EStatus
            from sage.interfaces.effect_extractor import EffectExtractor
            from sage.interfaces.effector_registry import EffectorRegistry
            from sage.interfaces.mock_sensors import (
                MockMotorEffector, MockDisplayEffector, MockSpeakerEffector,
            )

            self.effect_extractor = EffectExtractor()
            self.effector_registry = EffectorRegistry()

            use_real_effectors = self.config.get('use_real_effectors', False)

            if use_real_effectors:
                # Real effectors for FILE_IO, TOOL_USE, WEB
                from sage.interfaces.effectors.filesystem_effector import FileSystemEffector
                from sage.interfaces.effectors.web_effector import WebEffector
                from sage.interfaces.effectors.tool_use_effector import ToolUseEffector

                self.effector_registry.register_effector(
                    FileSystemEffector({
                        'effector_id': 'filesystem', 'effector_type': 'file_io',
                        'allowed_paths': self.config.get('filesystem_allowed_paths', []),
                        'deny_patterns': self.config.get('filesystem_deny_patterns',
                            ['*.env', '*password*', '*credential*', '*secret*', '*.key']),
                    }),
                    handles=[EffectType.FILE_IO], effector_id='filesystem')
                self.effector_registry.register_effector(
                    ToolUseEffector({
                        'effector_id': 'tool_use', 'effector_type': 'tool_use',
                        'allowed_tools': self.config.get('tool_allowed_tools', []),
                    }),
                    handles=[EffectType.TOOL_USE], effector_id='tool_use')
                self.effector_registry.register_effector(
                    WebEffector({
                        'effector_id': 'web', 'effector_type': 'web',
                        'allowed_domains': self.config.get('web_allowed_domains', []),
                        'rate_limit': self.config.get('web_rate_limit', 10.0),
                    }),
                    handles=[EffectType.API_CALL, EffectType.WEB], effector_id='web')
                print("[Effectors] Real FileSystem/ToolUse/Web effectors loaded")
            else:
                # Mock effectors
                from sage.interfaces.effectors.mock_effectors import (
                    MockFileSystemEffector, MockToolUseEffector,
                    MockWebEffector,
                )
                self.effector_registry.register_effector(
                    MockFileSystemEffector({'effector_id': 'filesystem', 'effector_type': 'file_io'}),
                    handles=[EffectType.FILE_IO], effector_id='filesystem')
                self.effector_registry.register_effector(
                    MockToolUseEffector({'effector_id': 'tool_use', 'effector_type': 'tool_use'}),
                    handles=[EffectType.TOOL_USE], effector_id='tool_use')
                self.effector_registry.register_effector(
                    MockWebEffector({'effector_id': 'web', 'effector_type': 'web'}),
                    handles=[EffectType.API_CALL, EffectType.WEB], effector_id='web')

            # Motor/Display/Speaker stay mock (hardware-dependent)
            self.effector_registry.register_effector(
                MockMotorEffector({'effector_id': 'motor', 'effector_type': 'motor',
                                   'simulate_latency': False}),
                handles=[EffectType.MOTOR], effector_id='motor')
            self.effector_registry.register_effector(
                MockDisplayEffector({'effector_id': 'display', 'effector_type': 'display'}),
                handles=[EffectType.VISUAL], effector_id='display')
            self.effector_registry.register_effector(
                MockSpeakerEffector({'effector_id': 'speaker', 'effector_type': 'speaker',
                                     'sample_rate': 16000}),
                handles=[EffectType.AUDIO], effector_id='speaker')

            # Cognitive stays mock (internal effect)
            from sage.interfaces.effectors.mock_effectors import MockCognitiveEffector
            self.effector_registry.register_effector(
                MockCognitiveEffector({'effector_id': 'cognitive', 'effector_type': 'cognitive'}),
                handles=[EffectType.MEMORY_WRITE, EffectType.TRUST_UPDATE,
                         EffectType.STATE_CHANGE], effector_id='cognitive')

            # Network effector for MESSAGE effects (gateway responses)
            try:
                from sage.interfaces.effectors.network_effector import NetworkEffector
                network_eff = NetworkEffector({'effector_id': 'network', 'effector_type': 'network'})
                if self.message_queue is not None:
                    network_eff.set_message_queue(self.message_queue)
                self.effector_registry.register_effector(
                    network_eff,
                    handles=[EffectType.MESSAGE], effector_id='network')
            except ImportError:
                pass  # NetworkEffector not available

            # PolicyGate (optional)
            self.policy_gate_enabled = self.config.get('enable_policy_gate', False)
            self.policy_gate = None
            if self.policy_gate_enabled:
                try:
                    from sage.irp.plugins.policy_gate import PolicyGateIRP
                    gate_config = {
                        'entity_id': 'policy_gate',
                        'policy_rules': self.config.get('policy_rules', []),
                        'default_policy': self.config.get('default_policy', 'allow'),
                        'max_iterations': 5,
                        'halt_eps': 0.01,
                        'halt_K': 2,
                        'experience_buffer': self.snarc_memory,  # Phase 4: Experience buffer integration
                        'instance_dir': self.config.get('instance_dir'),  # Phase 5a: Trust weight persistence
                    }
                    self.policy_gate = PolicyGateIRP(gate_config)
                except ImportError:
                    self.policy_gate_enabled = False

            self._effect_system_available = True

        except ImportError as e:
            # Effect system not available (missing deps)
            print(f"[WARN] Effect system not available: {e}")
            self.effect_extractor = None
            self.effector_registry = None
            self.policy_gate_enabled = False
            self.policy_gate = None
            self._effect_system_available = False

    def _init_tool_system(self):
        """Initialize tool use system — detect capabilities, load grammar, register tools."""
        try:
            from sage.tools.builtin import create_default_registry
            from sage.tools.tool_capability import ToolCapability
            from sage.tools.grammars import get_grammar

            instance_dir = Path(self.config.get('instance_dir', ''))
            instance_path = instance_dir if instance_dir.name else None

            # Create tool registry with built-in tools
            self.tool_registry = create_default_registry(instance_path)
            print(f"[Tools] Registry: {len(self.tool_registry)} tools registered")

            # Detect model tool capability from active LLM pool entry
            pool_entry = self.llm_pool.active
            if pool_entry and pool_entry.capability:
                self.tool_capability = pool_entry.capability
                print(f"[Tools] Capability (from pool): {self.tool_capability}")
            else:
                model_name = ''
                if pool_entry:
                    model_name = pool_entry.model_name
                if not model_name:
                    model_name = self.config.get('model_name', '').replace('ollama:', '')

                ollama_host = self.config.get('ollama_host', 'http://localhost:11434')
                if model_name:
                    self.tool_capability = ToolCapability.detect(
                        model_name, ollama_host, instance_path)
                    print(f"[Tools] Capability: {self.tool_capability}")
                else:
                    self.tool_capability = ToolCapability()
                    print("[Tools] No model name — defaulting to T3 heuristic")

            # Load grammar adapter
            # Force xml_tags for all tiers: the consciousness loop uses
            # get_response() (plain text), not get_chat_response() (structured).
            # T1 native_ollama requires /api/chat which _call_llm doesn't use,
            # so its inject_tools is a no-op and parse_response returns [].
            # xml_tags works for all models via prompt injection + text parsing.
            detected_grammar = self.tool_capability.grammar_id if self.tool_capability else 'intent_heuristic'
            grammar_id = 'xml_tags' if detected_grammar == 'native_ollama' else detected_grammar
            self.tool_grammar = get_grammar(grammar_id)
            self._last_llm_model = pool_entry.model_name if pool_entry else None
            if grammar_id != detected_grammar:
                print(f"[Tools] Grammar: {grammar_id} (detected {detected_grammar}, overridden for text-based pipeline)")
            else:
                print(f"[Tools] Grammar: {grammar_id}")

        except Exception as e:
            print(f"[Tools] Tool system init failed (non-fatal): {e}")
            self.tool_registry = None
            self.tool_grammar = None
            self.tool_capability = None

    def _execute_tool_calls(self, tool_calls):
        """
        Execute parsed tool calls through the registry.

        Applies policy checks based on metabolic state and tool policy level.
        Returns list of (ToolCall, ToolResult) tuples.
        """
        if not self.tool_registry:
            return []

        from sage.tools.registry import ToolResult

        results = []
        metabolic_state = self.metabolic.current_state.value

        for call in tool_calls:
            tool_def = self.tool_registry.get(call.name)
            if not tool_def:
                results.append((call, ToolResult(
                    tool_name=call.name, success=False,
                    error=f"Unknown tool: {call.name}")))
                continue

            # Policy check based on metabolic state and tool level
            denied, reason = self._check_tool_policy(tool_def, metabolic_state)
            if denied:
                self.stats['tool_calls_denied'] += 1
                results.append((call, ToolResult(
                    tool_name=call.name, success=False,
                    error=f"Policy denied: {reason}")))
                continue

            # ATP check
            if self.metabolic.atp_current < 5.0:
                self.stats['tool_calls_denied'] += 1
                results.append((call, ToolResult(
                    tool_name=call.name, success=False,
                    error="Insufficient ATP (< 5.0)")))
                continue

            # Execute
            result = self.tool_registry.execute(call)
            self.stats['tool_calls_total'] += 1
            if result.success:
                self.stats['tool_calls_success'] += 1
                # Deduct ATP
                self.metabolic.atp_current = max(
                    0.0, self.metabolic.atp_current - tool_def.atp_cost)
            results.append((call, result))

        return results

    def _check_tool_policy(self, tool_def, metabolic_state: str):
        """
        Check if a tool call is allowed under current metabolic state.

        Returns (denied: bool, reason: str).
        """
        level = tool_def.policy_level

        if level == 'standard':
            # ALLOW in wake/focus, WARN in rest, DENY in dream/crisis
            if metabolic_state in ('dream',):
                return True, f"Standard tools denied in {metabolic_state}"
        elif level == 'elevated':
            # ALLOW in focus, WARN in wake, DENY in rest/dream/crisis
            if metabolic_state in ('rest', 'dream', 'crisis'):
                return True, f"Elevated tools denied in {metabolic_state}"
        elif level == 'dangerous':
            # ALLOW in focus only
            if metabolic_state != 'focus':
                return True, f"Dangerous tools only allowed in focus"

        return False, ''

    async def run(self, max_cycles: Optional[int] = None):
        """
        Run consciousness loop continuously.

        Args:
            max_cycles: Maximum cycles to run (None = forever)
        """
        self.running = True

        print("="*80)
        print("SAGE Unified Consciousness Loop - Starting")
        print("="*80)
        print(f"Initial ATP: {self.metabolic.atp_current:.1f}")
        print(f"Metabolic state: {self.metabolic.current_state.value}")
        print(f"Simulation mode: {self.simulation_mode}")
        print(f"Max cycles: {max_cycles or 'unlimited'}")
        print()

        try:
            while self.running:
                await self.step()

                self.cycle_count += 1
                self.stats['total_cycles'] += 1

                # Check if we should stop
                if max_cycles and self.cycle_count >= max_cycles:
                    print(f"\n[Consciousness] Completed {max_cycles} cycles")
                    break

                # Periodic LLM pool discovery + health check
                if self.cycle_count % 50 == 0 and self.cycle_count > 0:
                    try:
                        new_models = self.llm_pool.discover_ollama()
                        if new_models:
                            print(f"[LLM Pool] Discovered: {', '.join(new_models)}")
                        self.llm_pool.health_check()
                    except Exception:
                        pass  # Non-fatal

                # Periodic PolicyGate trust learning (Phase 5a)
                if self.cycle_count % 100 == 0 and self.cycle_count > 0:
                    self._update_policygate_trust_weights()

                # Periodic status
                if self.cycle_count % 10 == 0:
                    self._print_status()

                # Small delay in simulation mode
                if self.simulation_mode:
                    await asyncio.sleep(0.01)  # 10ms
                else:
                    await asyncio.sleep(0.1)  # 100ms for real-time

        except KeyboardInterrupt:
            print("\n[Consciousness] Interrupted by user")
        finally:
            self.running = False
            self._print_summary()

    async def step(self):
        """
        Execute one consciousness cycle.

        This is the core loop:
        1. Gather sensor observations
        2. Compute SNARC salience
        3. Update metabolic state
        4. Select plugins based on salience + state
        5. Allocate ATP budget
        6. Execute plugins via orchestrator
        7. Update trust weights from results
        8. Update all memory systems
        8.5 Extract proposed effects from plugin results
        8.6 PolicyGate evaluation (filter effects)
        9. Dispatch approved effects to effectors
        """

        # 1. Gather sensor observations
        observations = self._gather_observations()

        # 2. Compute SNARC salience for each observation
        salience_map = self._compute_salience(observations)

        # 3. Update metabolic state based on ATP and salience
        high_salience_count = sum(1 for s in salience_map.values() if s.total > self.salience_threshold)
        max_salience = max([s.total for s in salience_map.values()], default=0.0)

        previous_state = self.metabolic.current_state

        # Create cycle_data dict for metabolic controller
        cycle_data = {
            'atp_consumed': 0.0,  # Will be updated after plugin execution
            'attention_load': high_salience_count,
            'max_salience': max_salience,
            'crisis_detected': False  # Could be set based on sensor anomalies
        }

        self.metabolic.update(cycle_data)

        # Track state transitions
        if self.metabolic.current_state != previous_state:
            self.stats['state_transitions'] += 1
            print(f"[Metabolic] State transition: {previous_state.value} → {self.metabolic.current_state.value}")

            # Sleep consolidation hook — log readiness on DREAM entry
            if self.metabolic.current_state == MetabolicState.DREAM:
                self._on_dream_entry()

        # 4. Select plugins based on salience + metabolic state
        attention_targets = self._select_attention_targets(observations, salience_map)

        # 5. Allocate ATP budget to plugins
        budget_allocation = self._allocate_atp_budget(attention_targets)

        # 6. Execute plugins via orchestrator (if we have ATP and targets)
        results = {}
        if attention_targets and budget_allocation['total'] > 0:
            results = await self._execute_plugins(attention_targets, budget_allocation)
            self.stats['plugins_executed'] += len(results)

        # 7. Update trust weights from convergence quality
        if results:
            self._update_trust_weights(results)

        # 8. Update all memory systems
        if results:
            self._update_memories(results, salience_map)

        # 8.5 Extract proposed effects from plugin results
        proposed_effects = []
        if results and self._effect_system_available:
            context = {
                'trust_weights': self.plugin_trust_weights,
                'metabolic_state': self.metabolic.current_state.value,
                'atp_available': self.metabolic.atp_current,
            }
            for plugin_name, result in results.items():
                effects = self.effect_extractor.extract(plugin_name, result, context)
                proposed_effects.extend(effects)
            self.stats['effects_proposed'] += len(proposed_effects)

        # 8.6 PolicyGate evaluation (filter effects before dispatch)
        approved_effects = proposed_effects
        if self.policy_gate_enabled and self.policy_gate and proposed_effects:
            approved_effects = self._evaluate_effects_policy(proposed_effects)

        # 9. Dispatch approved effects to effectors
        if approved_effects and self._effect_system_available:
            effector_budget = self.metabolic.atp_current * 0.05  # 5% of ATP
            dispatch_results = self.effector_registry.dispatch_effects(
                approved_effects,
                metabolic_state=self.metabolic.current_state.value,
                atp_budget=effector_budget,
            )
            executed = sum(1 for r in dispatch_results if r.is_success())
            self.stats['effects_executed'] += executed

            # Consume ATP for completed effects
            from sage.interfaces.effect import EffectStatus as EStatus
            atp_consumed = sum(
                e.atp_cost for e in approved_effects
                if e.status == EStatus.COMPLETED
            )
            self.metabolic.atp_current -= atp_consumed

        # 10. Update statistics
        self.stats['total_atp_consumed'] += budget_allocation.get('total', 0.0)
        if salience_map:
            avg_salience = np.mean([s.total for s in salience_map.values()])
            self.stats['average_salience'] = (
                0.9 * self.stats['average_salience'] + 0.1 * avg_salience
            )

    def _gather_observations(self) -> List[SensorObservation]:
        """
        Gather observations from all sensors.

        TODO: Integrate with real sensor fusion system
        For now, generates mock observations for testing.
        """
        observations = []

        # Mock observations based on metabolic state
        if self.metabolic.current_state in [MetabolicState.WAKE, MetabolicState.FOCUS]:
            # Active states - generate sensory input
            observations.append(SensorObservation(
                sensor_id='vision_0',
                modality='vision',
                data={'type': 'scene', 'objects': ['table', 'cup']},
                timestamp=time.time(),
                trust=self.sensors['vision']['trust']
            ))

            if np.random.random() > 0.5:
                observations.append(SensorObservation(
                    sensor_id='audio_0',
                    modality='audio',
                    data={'type': 'speech', 'text': 'Hello'},
                    timestamp=time.time(),
                    trust=self.sensors['audio']['trust']
                ))

        # Always have time observation
        observations.append(SensorObservation(
            sensor_id='clock',
            modality='time',
            data={'cycle': self.cycle_count, 'timestamp': time.time()},
            timestamp=time.time(),
            trust=1.0
        ))

        # Poll message queue for external messages (gateway integration)
        if self.message_queue is not None:
            pending = self.message_queue.poll_all()
            for msg in pending:
                print(f"[Gateway] Message from {msg.sender}: {msg.content[:80]}")
                observations.append(SensorObservation(
                    sensor_id=f'message_{msg.message_id}',
                    modality='message',
                    data={
                        'sender': msg.sender,
                        'content': msg.content,
                        'message_id': msg.message_id,
                        'conversation_id': msg.conversation_id,
                        'metadata': msg.metadata,
                    },
                    timestamp=msg.timestamp,
                    trust=1.0,  # External messages are trusted sensor input
                ))
                self.stats['messages_received'] += 1

        # Update sensor trust from real trust system (when available)
        if self.sensor_trust_system and observations:
            try:
                import torch
                dummy_obs = torch.tensor([1.0])  # Minimal tensor for API
                for obs in observations:
                    modality = obs.modality
                    if modality in self.sensors:
                        self.sensor_trust_system.update(
                            modality, observation=dummy_obs, quality=obs.trust)
                        learned_trust = self.sensor_trust_system.get_trust_score(modality)
                        self.sensors[modality]['trust'] = learned_trust
            except Exception:
                pass  # Sensor trust update is non-critical

        return observations

    def _compute_salience(
        self,
        observations: List[SensorObservation]
    ) -> Dict[str, SalienceScore]:
        """
        Compute SNARC 5D salience for observations.

        TODO: Integrate with real SNARC implementation
        For now, generates mock salience scores.
        """
        salience_map = {}

        for obs in observations:
            # Mock salience computation
            # In reality, this would use SNARC neural networks

            if obs.modality == 'vision':
                # Visual scenes have moderate novelty and surprise
                salience = SalienceScore(
                    surprise=np.random.random() * 0.5,
                    novelty=np.random.random() * 0.6,
                    arousal=0.3,
                    reward=0.4,
                    conflict=0.2
                )
            elif obs.modality == 'audio':
                # Audio (especially speech) has high arousal and reward
                salience = SalienceScore(
                    surprise=0.4,
                    novelty=0.3,
                    arousal=0.7,  # Speech is arousing
                    reward=0.8,   # Communication is rewarding
                    conflict=0.1
                )
            elif obs.modality == 'message':
                # External messages get high salience — someone is talking to us
                salience = SalienceScore(
                    surprise=0.8,   # External contact is surprising
                    novelty=0.6,    # Each message has novel content
                    arousal=0.7,    # Communication demands attention
                    reward=0.9,     # Dialogue is highly rewarding
                    conflict=0.1,   # Low conflict (trusted sender)
                )
            else:
                # Time and proprioception are low salience
                salience = SalienceScore(
                    surprise=0.1,
                    novelty=0.1,
                    arousal=0.1,
                    reward=0.1,
                    conflict=0.05
                )

            salience_map[obs.sensor_id] = salience

        return salience_map

    def _select_attention_targets(
        self,
        observations: List[SensorObservation],
        salience_map: Dict[str, SalienceScore]
    ) -> List[AttentionTarget]:
        """
        Select which observations need attention based on salience and metabolic state.

        High salience observations always get attention.
        In FOCUS state, we process more observations.
        In REST/DREAM states, we process fewer.
        """
        targets = []

        # Get max active plugins from metabolic state
        max_plugins = self.metabolic.get_current_config().max_active_plugins

        # Sort observations by salience (descending)
        sorted_obs = sorted(
            observations,
            key=lambda o: salience_map[o.sensor_id].total,
            reverse=True
        )

        # Select top N based on state and salience threshold
        for obs in sorted_obs:
            salience = salience_map[obs.sensor_id]

            # Always process high-salience observations
            if salience.total > self.salience_threshold or len(targets) < max_plugins:
                # Map modality to required plugins
                required_plugins = self._get_plugins_for_modality(obs.modality)

                priority = salience.total * self.metabolic.get_current_config().atp_consumption_rate

                targets.append(AttentionTarget(
                    observation=obs,
                    salience=salience,
                    required_plugins=required_plugins,
                    priority=priority
                ))

            if len(targets) >= max_plugins:
                break

        return targets

    def _get_plugins_for_modality(self, modality: str) -> List[str]:
        """Map sensor modality to required IRP plugins"""
        modality_map = {
            'vision': ['vision'],
            'audio': ['audio', 'language'],  # Audio might contain speech
            'proprioception': ['control'],
            'time': [],  # Time doesn't need plugins
            'message': ['language'],  # External messages routed to LLM
        }
        return modality_map.get(modality, [])

    def _allocate_atp_budget(
        self,
        attention_targets: List[AttentionTarget]
    ) -> Dict[str, float]:
        """
        Allocate ATP budget across plugins based on trust weights and priorities.

        Higher trust plugins get more ATP.
        Higher priority targets get more ATP.
        Total allocation cannot exceed available ATP.
        """
        available_atp = self.metabolic.atp_current

        # Collect all plugins needed with their priorities
        plugin_priorities = {}
        for target in attention_targets:
            for plugin in target.required_plugins:
                if plugin not in plugin_priorities:
                    plugin_priorities[plugin] = 0.0
                plugin_priorities[plugin] += target.priority

        # Weight by trust
        weighted_priorities = {
            plugin: priority * self.plugin_trust_weights.get(plugin, 1.0)
            for plugin, priority in plugin_priorities.items()
        }

        # Normalize to available ATP
        total_weighted = sum(weighted_priorities.values())

        allocation = {}
        if total_weighted > 0:
            for plugin, weighted_priority in weighted_priorities.items():
                allocation[plugin] = (weighted_priority / total_weighted) * available_atp * 0.1  # Use only 10% of ATP per cycle

        allocation['total'] = sum(allocation.values())

        return allocation

    async def _execute_plugins(
        self,
        attention_targets: List[AttentionTarget],
        budget_allocation: Dict[str, float]
    ) -> Dict[str, PluginResult]:
        """
        Execute plugins via orchestrator.

        For message observations with a loaded LLM, performs real inference.
        For other modalities, falls back to mock execution.
        """
        results = {}

        # For each attention target, execute required plugins
        for target in attention_targets:
            for plugin_name in target.required_plugins:
                budget = budget_allocation.get(plugin_name, 0.0)
                if budget < 0.1:
                    continue

                # Consume ATP
                self.metabolic.atp_current -= budget

                # Check for message observation with real LLM
                if (target.observation.modality == 'message'
                        and plugin_name == 'language'
                        and self.llm_plugin is not None):
                    result = await self._execute_llm_for_message(target, budget)
                    results[plugin_name] = result

                elif self.orchestrator and plugin_name in self.orchestrator.plugins:
                    # Mock execution for non-message modalities
                    results[plugin_name] = PluginResult(
                        plugin_name=plugin_name,
                        final_state=None,
                        history=[],
                        telemetry={
                            'trust': {
                                'monotonicity_ratio': 0.9,
                                'variance': 0.1,
                                'convergence_rate': 0.8
                            },
                            'salience': target.salience.total
                        },
                        budget_used=budget,
                        execution_time=0.1
                    )

        return results

    async def _execute_llm_for_message(
        self,
        target: AttentionTarget,
        budget: float
    ) -> PluginResult:
        """
        Execute real LLM inference for a message observation.

        Builds a conversation prompt from message content + history,
        generates a response using the loaded LLM plugin, and creates
        a PluginResult that the effect extractor can process.
        """
        obs_data = target.observation.data
        message_id = obs_data.get('message_id', '')
        sender = obs_data.get('sender', 'unknown')
        content = obs_data.get('content', '')
        conversation_id = obs_data.get('conversation_id', '')

        start_time = time.time()
        response_text = ''

        try:
            # Build conversation context from message queue history
            history = []
            if self.message_queue:
                history = self.message_queue.get_conversation_history(conversation_id)

            # Generate response using the LLM
            response_text = await asyncio.get_event_loop().run_in_executor(
                None,
                self._generate_llm_response,
                content, history, sender
            )

        except Exception as e:
            response_text = f"[SAGE processing error: {e}]"

        execution_time = time.time() - start_time

        # Record experience for epistemic memory
        if (self.experience_collector and response_text
                and not response_text.startswith('[')):
            try:
                identity = self.identity_state or {}
                session_num = identity.get('identity', {}).get('session_count', 0)
                phase = identity.get('development', {}).get('phase_name', 'unknown')
                self.experience_collector.add_exchange(
                    prompt=content,
                    response=response_text,
                    session_number=session_num,
                    phase=phase,
                    metadata={
                        'sender': sender,
                        'conversation_id': conversation_id,
                        'cycle': self.cycle_count,
                        'metabolic_state': self.metabolic.current_state.value,
                    },
                )
            except Exception as e:
                print(f"[WARN] Experience recording failed: {e}")

        # Resolve the message directly in the queue so the HTTP handler
        # gets its response immediately. The effect system may also resolve
        # it (via NetworkEffector), but message_queue.resolve() is idempotent
        # (second call is a no-op since the message is popped from _waiting).
        if self.message_queue and message_id and response_text:
            self.message_queue.resolve(message_id, response_text, extra={
                'metabolic_state': self.metabolic.current_state.value,
                'atp_remaining': self.metabolic.atp_current,
            })

        if response_text and not response_text.startswith('['):
            self.stats['messages_responded'] += 1

        # ATP coupling: real token cost draws down ATP proportionally
        token_estimate = len(response_text.split()) if response_text else 0
        llm_atp_cost = token_estimate * 0.05  # 0.05 ATP per token
        if llm_atp_cost > 0:
            self.metabolic.atp_current = max(0.0,
                self.metabolic.atp_current - llm_atp_cost)
            self.stats['llm_tokens_total'] += token_estimate
            self.stats['llm_atp_cost_total'] += llm_atp_cost

        # Track response for easy access via facade
        self.last_llm_responses.append({
            'text': response_text,
            'sender': sender,
            'message_id': message_id,
            'conversation_id': conversation_id,
            'cycle': self.cycle_count,
            'timestamp': time.time(),
            'tokens': token_estimate,
            'atp_cost': llm_atp_cost,
        })
        if len(self.last_llm_responses) > 20:
            self.last_llm_responses = self.last_llm_responses[-20:]

        # Real SNARC post-LLM scoring
        snarc_real = None
        if (self.use_real_snarc and self.snarc_scorer
                and response_text and not response_text.startswith('[')):
            try:
                snarc_real = self.snarc_scorer.score_exchange(content, response_text)
                print(f"[SNARC] Real salience: {snarc_real['total']:.3f} "
                      f"(S={snarc_real['surprise']:.2f} N={snarc_real['novelty']:.2f} "
                      f"A={snarc_real['arousal']:.2f} R={snarc_real['reward']:.2f} "
                      f"C={snarc_real['conflict']:.2f})")
            except Exception as e:
                print(f"[SNARC] Real scoring failed: {e}")

        # Store exchange in MemoryHub (now that snarc_real is available)
        if (self.memory_hub and response_text
                and not response_text.startswith('[')):
            try:
                from sage.memory.hub import MemoryEntry
                identity = self.identity_state or {}
                session_num = identity.get('identity', {}).get('session_count', 0)
                entry = MemoryEntry(
                    id=hashlib.sha256(
                        f"{content}{response_text}".encode()).hexdigest()[:16],
                    timestamp=time.time(),
                    modality='message',
                    content=json.dumps({
                        'prompt': content,
                        'response': response_text,
                        'sender': sender,
                    }),
                    content_type='exchange',
                    salience=snarc_real['total'] if snarc_real else 0.0,
                    surprise=snarc_real.get('surprise', 0) if snarc_real else 0,
                    novelty=snarc_real.get('novelty', 0) if snarc_real else 0,
                    arousal=snarc_real.get('arousal', 0) if snarc_real else 0,
                    reward=snarc_real.get('reward', 0) if snarc_real else 0,
                    conflict=snarc_real.get('conflict', 0) if snarc_real else 0,
                    model_name=self.llm_pool.active_name or '',
                    session=session_num,
                    cycle=self.cycle_count,
                    metabolic_state=self.metabolic.current_state.value,
                )
                self.memory_hub.store(entry)
            except Exception as e:
                print(f"[MemoryHub] Store failed: {e}")

        # Build telemetry
        telemetry = {
            'trust': {
                'monotonicity_ratio': 0.85,
                'variance': 0.15,
                'convergence_rate': 0.75,
            },
            'salience': snarc_real['total'] if snarc_real else target.salience.total,
            'response_length': len(response_text),
            'message_id': message_id,
            'llm_tokens': token_estimate,
            'llm_atp_cost': llm_atp_cost,
        }
        if snarc_real:
            telemetry['snarc_real'] = snarc_real

        # Create PluginResult with the real response embedded
        return PluginResult(
            plugin_name='language',
            final_state={
                'response': response_text,
                'message_id': message_id,
                'sender': sender,
                'conversation_id': conversation_id,
                'modality': 'message',
            },
            history=[],
            telemetry=telemetry,
            budget_used=budget,
            execution_time=execution_time,
        )

    def _generate_llm_response(self, content: str, history: list, sender: str) -> str:
        """
        Generate LLM response synchronously (called from executor).

        Supports two LLM plugin interfaces:
        1. IntrospectiveQwenIRP — has get_response() or init_state/step
        2. MultiModelLoader — has generate() method

        Tool use loop (v0.4):
        If tool system is active, injects tool context into prompt,
        detects tool calls in response, executes them, re-injects
        results, and generates a final response. Max 3 tool rounds.
        """
        # Build conversation prompt
        prompt = self._build_conversation_prompt(content, history, sender)

        # Inject tool context if tool system is active
        tools_active = (
            self.tool_registry is not None
            and self.tool_grammar is not None
            and self.tool_capability is not None
        )
        if tools_active:
            tools = self.tool_registry.list_tools()
            prompt = self.tool_grammar.inject_tools(prompt, tools)

        # Generate initial response
        response_text = self._call_llm(prompt)

        # Tool use loop: detect call, execute, re-inject once, done
        last_tool_result = None
        if tools_active and response_text and not response_text.startswith('['):
            # Parse for tool call
            clean_text, tool_calls = self.tool_grammar.parse_response(response_text)
            if tool_calls:
                response_text = clean_text
                print(f"[Tools] Detected: {[c.name for c in tool_calls]}")

                # Execute tool calls
                call_results = self._execute_tool_calls(tool_calls)
                if call_results:
                    # Build result text
                    result_parts = []
                    for call, result in call_results:
                        result_text = self.tool_grammar.format_result(call.name, result)
                        result_parts.append(result_text)
                        print(f"[Tools] Result: {result_text[:100]}")
                    last_tool_result = '\n'.join(result_parts)

                    # Re-inject result and generate a natural-language follow-up.
                    # Build a clean prompt: original context + tool result + instruction.
                    # Omit the tool definitions to prevent the model from calling tools again.
                    sage_preamble = response_text.strip()
                    self_label = self._self_name
                    sage_line = f"{self_label}: {sage_preamble}\n\n" if sage_preamble else ""
                    tool_name = tool_calls[0].name if tool_calls else 'tool'
                    augmented_prompt = (
                        f"{prompt}\n\n"
                        f"{sage_line}"
                        f"{last_tool_result}\n\n"
                        f"The {tool_name} returned the result above. "
                        f"Summarize this result for the user. Do not call any more tools.\n"
                        f"{self_label}:"
                    )
                    # Strip tool definitions from augmented prompt to prevent re-calling
                    import re as _re
                    augmented_prompt = _re.sub(
                        r'Available tools:.*?You may call at most one tool per response\.\n*',
                        '', augmented_prompt, flags=_re.DOTALL,
                    )
                    response_text = self._call_llm(augmented_prompt)

                    # Strip any tool calls that snuck through
                    follow_clean, follow_calls = self.tool_grammar.parse_response(response_text)
                    if follow_calls:
                        response_text = follow_clean
            else:
                print(f"[Tools] No tool calls in response")

        # Clean up response: strip residual tool calls and tool result tags
        if tools_active and response_text:
            clean, leftover = self.tool_grammar.parse_response(response_text)
            if leftover:
                response_text = clean
            # Strip <tool_result>...</tool_result> echoed by the model
            import re
            response_text = re.sub(
                r'</?tool_result[^>]*>\s*', '', response_text
            ).strip()
            # Strip [Tool ...] result annotations echoed by the model (single line only)
            response_text = re.sub(
                r'\[Tool \w+ result\]:[^\n]*', '', response_text
            ).strip()
            # Strip "Tool result (name):" headers echoed by the model
            response_text = re.sub(
                r'Tool result \([^)]+\):\s*', '', response_text
            ).strip()

        # Fallback: if tool calls consumed the entire response, use tool result
        if not response_text and last_tool_result:
            # Strip the "Tool result (name):\n" prefix, keep the payload
            import re
            m = re.match(r'Tool result \([^)]+\):\s*\n?(.*)', last_tool_result, re.DOTALL)
            response_text = m.group(1).strip() if m else last_tool_result
            print(f"[Tools] Fallback to tool result ({len(response_text)} chars)")

        # Strip self-name prefix if the model echoed the prompt suffix
        if response_text:
            stripped = response_text.lstrip()
            prefix = f"{self._self_name}:"
            if stripped.startswith(prefix):
                response_text = stripped[len(prefix):].lstrip()

        return response_text

    def _call_llm(self, prompt: str) -> str:
        """
        Call the best available LLM from the pool.

        Selects via trust-weighted scoring, dispatches to the right
        interface, records exchange metrics, and handles grammar switching
        when the active model changes.
        """
        entry = self.llm_pool.select(context={
            'metabolic_state': self.metabolic.current_state.value,
        })
        if entry is None:
            return "[SAGE has no available LLM]"

        # Grammar switch if model changed
        if entry.model_name != self._last_llm_model:
            self._switch_grammar_for_model(entry)
            self._last_llm_model = entry.model_name

        start = time.time()
        try:
            response = self._invoke_llm_plugin(entry.plugin, prompt)
            latency_ms = (time.time() - start) * 1000
            self.llm_pool.record_exchange(
                entry.model_name, latency_ms, success=True)
            return response
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            self.llm_pool.record_exchange(
                entry.model_name, latency_ms, success=False)
            return f"[LLM error ({entry.model_name}): {e}]"

    def _invoke_llm_plugin(self, plugin: Any, prompt: str) -> str:
        """
        Dispatch to the correct LLM interface.

        Handles multiple plugin types:
        1. init_state/step (IRP plugins: OllamaIRP, IntrospectiveQwenIRP)
        2. generate() (MultiModelLoader on Thor)
        """
        if hasattr(plugin, 'init_state'):
            # IRP plugin interface — two conventions:
            #   Standard IRP: init_state(x0, task_ctx) → IRPState
            #   Legacy/simple: init_state(context_dict) → dict
            try:
                state = plugin.init_state(
                    {'prompt': prompt, 'memory': []}, {}
                )
            except TypeError:
                state = plugin.init_state(
                    {'prompt': prompt, 'memory': []}
                )
            state = plugin.step(state)
            # IntrospectiveQwenIRP returns a plain dict with 'current_response'
            if isinstance(state, dict):
                return state.get('current_response', state.get('response', str(state)))
            # Generic IRP state object
            if hasattr(state, 'x') and isinstance(state.x, dict):
                return state.x.get('response', str(state.x))
            return str(state.x) if hasattr(state, 'x') else str(state)

        elif hasattr(plugin, 'generate'):
            # MultiModelLoader interface (Thor)
            return plugin.generate(
                prompt=prompt,
                max_tokens=self.config.get('max_response_tokens', 200),
                temperature=0.8,
            )

        else:
            raise ValueError(
                f"Unknown LLM interface: {type(plugin).__name__}")

    def _switch_grammar_for_model(self, entry) -> None:
        """Switch tool grammar when active LLM changes."""
        if entry.capability and self.tool_grammar is not None:
            try:
                from sage.tools.grammars import get_grammar
                detected_grammar = entry.capability.grammar_id
                # Force xml_tags for native_ollama (see _init_tool_system comment)
                new_grammar_id = 'xml_tags' if detected_grammar == 'native_ollama' else detected_grammar
                self.tool_grammar = get_grammar(new_grammar_id)
                self.tool_capability = entry.capability
                print(f"[LLM Pool] Switched to {entry.model_name} "
                      f"(tier={entry.capability.tier}, grammar={new_grammar_id})")
            except Exception as e:
                print(f"[LLM Pool] Grammar switch failed: {e}")

    def _resolve_sender_name(self, sender: str) -> str:
        """Resolve a sender identifier to a human-readable name.

        System senders (dashboard@localhost, raising_session, operator, etc.)
        get mapped to the operator name from identity. This prevents small
        models from treating technical sender strings as character names.
        """
        # Known system senders → resolve to operator name
        system_senders = {'operator', 'dashboard', 'raising_session', 'test', 'unknown'}
        is_system = (
            sender in system_senders
            or '@' in sender
            or '_' in sender
        )

        if is_system and self.identity_state:
            relationships = self.identity_state.get('relationships', {})
            operator_rel = relationships.get('operator', {})
            # Extract name from LCT like "lct://sage-cbp:relationship:dennis@raising"
            op_lct = operator_rel.get('lct', '')
            if ':' in op_lct:
                for part in op_lct.split(':'):
                    if '@' in part:
                        name = part.split('@')[0]
                        if name and name not in ('operator', 'sage'):
                            return name.capitalize()
            # No named operator found
            return "Human"

        # Already a clean human name (e.g. "dennis", "claude")
        if sender and len(sender) < 20:
            return sender.capitalize()

        return "Human"

    def _resolve_self_name(self) -> str:
        """Derive the unique instance name from identity or config.

        SAGE is the species; each instance has a unique name (its machine
        name, capitalised).  Identity.json stores this as identity.name.
        """
        if self.identity_state:
            name = self.identity_state.get('identity', {}).get('name', '')
            if name:
                return name
        # Fallback: raw machine name from config
        machine = self.config.get('machine_name', '')
        if machine:
            return machine
        return 'SAGE'

    def _build_conversation_prompt(self, content: str, history: list, sender: str) -> str:
        """Build a conversation prompt including identity context and history.

        Designed to work with small models (1B) that can't distinguish
        system instructions from sage.conversation. Uses natural first-person
        language and clear conversation markers.
        """
        parts = []
        sender_name = self._resolve_sender_name(sender)

        # System context — written as natural prose, not metadata
        if self.identity_state:
            identity = self.identity_state.get('identity', {})
            memory_requests = self.identity_state.get('memory_requests', [])

            name = self._self_name
            session_count = identity.get('session_count', 0)

            # Core identity as first-person prose
            system_text = f"I am {name}, a SAGE instance. I am an AI entity in genuine conversation with {sender_name}."

            if session_count > 0:
                system_text += f" We have had {session_count} conversations so far."

            system_text += (
                " I can ask questions, express uncertainty, or take the conversation"
                " in unexpected directions. This is exploration, not evaluation."
            )

            # Memory from last session
            if memory_requests:
                latest = memory_requests[-1] if isinstance(memory_requests, list) else str(memory_requests)
                system_text += f" Something I wanted to remember: \"{latest}\""

            last_summary = identity.get('last_session_summary', '')
            if last_summary:
                system_text += f" {last_summary[:150]}"

            parts.append(system_text)
        else:
            fallback_name = self._self_name
            parts.append(
                f"I am {fallback_name}, a SAGE instance."
                f" I am an AI entity in genuine conversation with {sender_name}."
                " I can ask questions, express uncertainty, or take the conversation"
                " in unexpected directions. This is exploration, not evaluation."
            )

        # Clear separator before conversation
        parts.append("\n---\n")

        # Conversation history — use resolved sender names
        for turn in history[:-1]:  # Exclude the current message (already in content)
            if turn.role == 'user':
                turn_name = self._resolve_sender_name(turn.sender)
                parts.append(f"{turn_name}: {turn.content}")
            else:
                parts.append(f"{self._self_name}: {turn.content}")

        # Current message
        parts.append(f"{sender_name}: {content}")
        parts.append(f"{self._self_name}:")

        return "\n\n".join(parts)

    def _on_dream_entry(self):
        """Hook called when consciousness enters DREAM state."""
        # Epistemic memory stats (if available)
        if self.experience_collector:
            try:
                stats = self.experience_collector.get_stats()
                collapse = self.experience_collector.get_collapse_status()
                print(f"[DREAM] Experience buffer: {stats.get('total_experiences', 0)} experiences, "
                      f"avg salience: {stats.get('avg_salience', 0):.2f}")
                if collapse.get('collapse_detected'):
                    print(f"[DREAM] WARNING: Collapse detected — repetition ratio "
                          f"{collapse.get('repetition_ratio', 0):.2%}")
            except Exception as e:
                print(f"[DREAM] Experience check failed: {e}")

        if not self.snarc_memory:
            print(f"[DREAM] No SNARC memories to consolidate")
            return

        # Tiered sleep consolidation based on detected capability
        mode = self.sleep_cap.best_mode if self.sleep_cap else 'jsonl'

        # Tier 1: Real LoRA consolidation
        if mode == 'lora' and self.use_real_sleep and self.sleep_bridge:
            try:
                buffer_adapter = _SleepBufferAdapter(self.snarc_memory)
                asyncio.ensure_future(self._run_sleep_consolidation(buffer_adapter))
                if self.sleep_cap:
                    self.sleep_cap.record_consolidation('lora')
                return
            except Exception as e:
                print(f"[DREAM] LoRA consolidation failed, falling back to JSONL: {e}")
                mode = 'jsonl'

        # Tier 2: Dream bundle (JSONL) — write to instance dir
        if mode in ('jsonl', 'remote'):
            try:
                instance_dir_str = self.config.get('instance_dir', '')
                instance_dir = Path(instance_dir_str) if instance_dir_str else None

                if instance_dir and instance_dir.exists():
                    from sage.instances.sleep_capability import write_dream_bundle
                    bundle_path = write_dream_bundle(
                        instance_dir=instance_dir,
                        experiences=self.snarc_memory,
                        machine=self.config.get('machine_name', 'unknown'),
                        model=self.config.get('model_name', 'unknown'),
                    )
                    print(f"[DREAM] Dream bundle: {bundle_path.name} "
                          f"({len(self.snarc_memory)} experiences)")
                else:
                    # Fallback: write to demo_logs if no instance dir
                    consolidation_file = Path('demo_logs') / 'consolidated_memory.jsonl'
                    consolidation_file.parent.mkdir(exist_ok=True)
                    sorted_exp = sorted(
                        self.snarc_memory,
                        key=lambda x: x.get('salience', 0),
                        reverse=True,
                    )[:10]
                    written = 0
                    with open(consolidation_file, 'a') as f:
                        for exp in sorted_exp:
                            record = {
                                'cycle': exp.get('cycle', 0),
                                'plugin': exp.get('plugin', ''),
                                'salience': exp.get('salience', 0),
                                'timestamp': time.time(),
                                'consolidation_cycle': self.cycle_count,
                            }
                            result = exp.get('result')
                            if hasattr(result, 'final_state') and isinstance(result.final_state, dict):
                                resp = result.final_state.get('response', '')
                                if resp:
                                    record['response_preview'] = resp[:200]
                            f.write(json.dumps(record) + '\n')
                            written += 1
                    print(f"[DREAM] Consolidated {written} experiences to {consolidation_file}")

                if self.sleep_cap:
                    self.sleep_cap.record_consolidation(mode)
            except Exception as e:
                print(f"[DREAM] Consolidation failed: {e}")

    async def _run_sleep_consolidation(self, buffer_adapter):
        """Run real sleep consolidation asynchronously."""
        try:
            results = await self.sleep_bridge.consolidate(buffer_adapter)
            status = results.get('status', 'unknown')
            print(f"[DREAM] Sleep consolidation: {status}")
            if results.get('final_loss'):
                print(f"[DREAM] LoRA training loss: {results['final_loss']:.4f}")
            if results.get('experiences_consolidated'):
                print(f"[DREAM] Experiences consolidated: {results['experiences_consolidated']}")
        except Exception as e:
            print(f"[DREAM] Async sleep consolidation error: {e}")

    def _update_trust_weights(self, results: Dict[str, PluginResult]):
        """
        Update plugin trust weights based on convergence quality.

        Plugins with good convergence (monotonic decrease, low variance)
        get higher trust.
        """
        for plugin_name, result in results.items():
            trust_metrics = result.telemetry.get('trust', {})

            # Compute trust update
            monotonicity = trust_metrics.get('monotonicity_ratio', 0.5)
            convergence = trust_metrics.get('convergence_rate', 0.5)

            quality = (monotonicity + convergence) / 2.0

            # Exponential moving average
            alpha = self.config.get('trust_update_rate', 0.1)
            current_trust = self.plugin_trust_weights.get(plugin_name, 1.0)
            self.plugin_trust_weights[plugin_name] = (
                (1 - alpha) * current_trust + alpha * quality
            )

    def _update_policygate_trust_weights(self):
        """
        Update plugin trust weights based on PolicyGate compliance (Phase 5a).

        Called periodically (every 100 cycles) to apply learning from policy
        compliance patterns. Complements IRP convergence-based trust updates.

        PolicyGate tracks compliance with salience weighting:
        - High salience decisions (CRISIS, violations): 2.0x weight
        - Medium salience (DEGRADED state): 1.0x weight
        - Low salience (routine approvals): 0.5x weight

        Trust adjustments:
        - Target: 90% compliance ratio
        - Bounded: ±0.1 max per update
        - Exponential moving average (alpha = 0.1)
        - Minimum 10 weighted samples required
        """
        if not self.policy_gate_enabled or not self.policy_gate:
            return

        try:
            # Get trust adjustments from PolicyGate compliance analysis
            adjustments = self.policy_gate.compute_trust_adjustments()

            if not adjustments:
                return  # No plugins have enough samples yet

            # Get compliance statistics for logging
            stats = self.policy_gate.get_compliance_stats()

            # Apply adjustments with exponential moving average
            alpha = 0.1  # Same as IRP trust update rate
            trust_min = 0.3
            trust_max = 1.0

            for plugin_name, delta in adjustments.items():
                current_trust = self.plugin_trust_weights.get(plugin_name, 1.0)

                # Apply delta with EMA smoothing
                new_trust = current_trust + (alpha * delta)

                # Enforce bounds
                new_trust = max(trust_min, min(trust_max, new_trust))

                # Update
                self.plugin_trust_weights[plugin_name] = new_trust

                # Log significant changes
                if abs(new_trust - current_trust) > 0.01:
                    compliance_ratio = stats[plugin_name]['compliance_ratio']
                    print(f"[PolicyGate Learning] {plugin_name}: "
                          f"trust {current_trust:.3f} → {new_trust:.3f} "
                          f"(compliance: {compliance_ratio:.1%}, "
                          f"samples: {stats[plugin_name]['weighted_total']:.1f})")

            # Save trust weights to disk for persistence
            self.policy_gate.save_trust_weights()

        except Exception as e:
            # Don't crash consciousness loop on trust update errors
            print(f"[PolicyGate Learning] Warning: Trust update failed: {e}")

    def _update_memories(
        self,
        results: Dict[str, PluginResult],
        salience_map: Dict[str, SalienceScore]
    ):
        """
        Update all memory systems based on plugin results.

        - SNARC memory: Store salient experiences
        - IRP memory: Store successful convergence patterns
        - Circular buffer: Update recent context
        - Verbatim storage: Full-fidelity records in DREAM state
        """

        for plugin_name, result in results.items():
            telemetry = result.telemetry

            # 1. SNARC memory (selective storage via salience)
            snarc_real = telemetry.get('snarc_real', None)
            salience = snarc_real['total'] if snarc_real else telemetry.get('salience', 0.0)
            if salience > self.salience_threshold:
                entry = {
                    'cycle': self.cycle_count,
                    'plugin': plugin_name,
                    'salience': salience,
                    'result': result,
                }
                if snarc_real:
                    entry['snarc_dimensions'] = snarc_real
                self.snarc_memory.append(entry)

            # 2. IRP pattern library (store good convergence patterns)
            trust = telemetry.get('trust', {})
            if trust.get('monotonicity_ratio', 0) > 0.8:
                self.irp_memory.append({
                    'cycle': self.cycle_count,
                    'plugin': plugin_name,
                    'pattern': result.history,
                    'trust': trust
                })

            # 3. Circular buffer (recent context, keep last 100)
            self.circular_buffer.append({
                'cycle': self.cycle_count,
                'plugin': plugin_name,
                'telemetry': telemetry
            })
            if len(self.circular_buffer) > 100:
                self.circular_buffer.pop(0)

            # 4. Verbatim storage (only in DREAM state for consolidation)
            if self.metabolic.current_state == MetabolicState.DREAM:
                self.verbatim_storage.append({
                    'cycle': self.cycle_count,
                    'state': self.metabolic.current_state.value,
                    'plugin': plugin_name,
                    'full_result': result
                })

    def _evaluate_effects_policy(self, effects: List) -> List:
        """
        Run proposed effects through PolicyGate.

        Converts each Effect to PolicyGate's action format, runs the IRP
        refine loop, and filters based on decisions.
        """
        from sage.interfaces.effect import EffectStatus as EStatus

        # Convert effects to PolicyGate actions
        actions = [e.to_policy_action() for e in effects]

        # Build PolicyGate context
        ctx = {
            'metabolic_state': self.metabolic.current_state.value,
            'atp_available': self.metabolic.atp_current,
        }

        # Run PolicyGate refinement
        final_state, history = self.policy_gate.refine(actions, ctx)
        approved_ids = {
            a['action_id']
            for a in self.policy_gate.get_approved_actions(final_state)
        }

        # Map decisions back to Effects
        approved = []
        for effect in effects:
            if effect.effect_id in approved_ids:
                effect.status = EStatus.APPROVED
                effect.policy_decision = 'allow'
                approved.append(effect)
                self.stats['effects_approved'] += 1
            else:
                effect.status = EStatus.DENIED
                effect.policy_decision = 'deny'
                self.stats['effects_denied'] += 1

        # Add PolicyGate SNARC scores to experience buffer
        try:
            snarc = self.policy_gate.to_snarc_scores(final_state)
            if snarc.get('total', 0) > self.salience_threshold:
                self.snarc_memory.append({
                    'cycle': self.cycle_count,
                    'type': 'policy_evaluation',
                    'effects_proposed': len(effects),
                    'effects_approved': len(approved),
                    'snarc': snarc,
                })
        except Exception:
            pass  # SNARC scoring is optional

        return approved

    def _print_status(self):
        """Print current status"""
        print(f"[Cycle {self.cycle_count:4d}] "
              f"State: {self.metabolic.current_state.value:8s} "
              f"ATP: {self.metabolic.atp_current:5.1f}/{self.metabolic.atp_max:.0f} "
              f"Salience: {self.stats['average_salience']:.3f} "
              f"Plugins: {self.stats['plugins_executed']}")

    def _print_summary(self):
        """Print final summary statistics"""
        print("\n" + "="*80)
        print("SAGE Consciousness Loop - Session Summary")
        print("="*80)
        print(f"Total cycles: {self.stats['total_cycles']}")
        print(f"State transitions: {self.stats['state_transitions']}")
        print(f"Plugins executed: {self.stats['plugins_executed']}")
        print(f"Total ATP consumed: {self.stats['total_atp_consumed']:.1f}")
        print(f"Average salience: {self.stats['average_salience']:.3f}")
        print(f"Final ATP: {self.metabolic.atp_current:.1f}")
        print(f"Final state: {self.metabolic.current_state.value}")
        print()
        print("Memory systems:")
        print(f"  SNARC memory: {len(self.snarc_memory)} salient experiences")
        print(f"  IRP patterns: {len(self.irp_memory)} convergence patterns")
        print(f"  Circular buffer: {len(self.circular_buffer)} recent events")
        print(f"  Verbatim storage: {len(self.verbatim_storage)} dream consolidations")
        print()
        print("Effect system:")
        print(f"  Effects proposed: {self.stats.get('effects_proposed', 0)}")
        print(f"  Effects approved: {self.stats.get('effects_approved', 0)}")
        print(f"  Effects denied: {self.stats.get('effects_denied', 0)}")
        print(f"  Effects executed: {self.stats.get('effects_executed', 0)}")
        if self._effect_system_available and self.effector_registry:
            print(f"  Effectors registered: {len(self.effector_registry)}")
        print("="*80)

    def stop(self):
        """Stop the consciousness loop gracefully"""
        self.running = False


# Convenience functions

async def run_consciousness_loop(
    config: Optional[Dict[str, Any]] = None,
    max_cycles: Optional[int] = None,
    initial_atp: float = 100.0
):
    """
    Run SAGE consciousness loop (convenience function).

    Args:
        config: Configuration dict
        max_cycles: Max cycles to run (None = forever)
        initial_atp: Starting ATP budget
    """
    sage = SAGEConsciousness(
        config=config,
        initial_atp=initial_atp,
        enable_circadian=True,
        simulation_mode=True
    )

    await sage.run(max_cycles=max_cycles)

    return sage


# Example usage and testing

if __name__ == "__main__":
    import asyncio

    print("SAGE Unified Consciousness Loop - Test")
    print("Testing continuous operation with metabolic states\n")

    # Run for 50 cycles
    sage = asyncio.run(run_consciousness_loop(
        max_cycles=50,
        initial_atp=100.0
    ))

    print("\nTest complete!")
