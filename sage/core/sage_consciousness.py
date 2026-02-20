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
import numpy as np
import torch

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.metabolic_controller import MetabolicController, MetabolicState
from core.circadian_clock import CircadianClock, CircadianPhase
from irp.orchestrator import HRMOrchestrator, PluginResult


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
        """
        self.config = config or self._default_config()

        # Gateway integration
        self.message_queue = message_queue
        self.llm_plugin = llm_plugin
        self.simulation_mode = simulation_mode

        # Core components
        self.metabolic = MetabolicController(
            initial_atp=initial_atp,
            max_atp=self.config.get('max_atp', 100.0),
            circadian_period=self.config.get('circadian_period', 100),
            enable_circadian=enable_circadian,
            simulation_mode=simulation_mode
        )

        self.orchestrator = HRMOrchestrator(self.config)

        # Sensor system (mock for now, will integrate real sensors later)
        self.sensors = self._initialize_sensors()

        # Memory systems
        self.snarc_memory = []  # SNARC salience memory
        self.irp_memory = []    # IRP pattern library
        self.circular_buffer = []  # Recent context (x-from-last)
        self.verbatim_storage = []  # Full-fidelity records

        # Trust weights for plugins (learned over time)
        self.plugin_trust_weights = {
            name: 1.0 for name in self.orchestrator.plugins
        }

        # Salience thresholds (from SNARC)
        self.salience_threshold = self.config.get('salience_threshold', 0.15)

        # Cycle counter
        self.cycle_count = 0
        self.running = False

        # Effect system
        self._init_effect_system()

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
        }

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            'total_ATP': 100.0,
            'max_atp': 100.0,
            'max_workers': 4,
            'trust_update_rate': 0.1,
            'telemetry_interval': 10,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'circadian_period': 100,
            'salience_threshold': 0.15,
            # Plugin configurations
            'enable_vision': True,
            'enable_language': True,
            'enable_audio': True,
            'enable_memory': True,
        }

    def _initialize_sensors(self) -> Dict[str, Any]:
        """Initialize sensor system (mock for now)"""
        # TODO: Integrate real sensors from sage/core/sensor_fusion.py
        return {
            'vision': {'trust': 1.0, 'enabled': True},
            'audio': {'trust': 1.0, 'enabled': True},
            'proprioception': {'trust': 1.0, 'enabled': True},
            'time': {'trust': 1.0, 'enabled': True},
            'message': {'trust': 1.0, 'enabled': True},  # External messages via gateway
        }

    def _init_effect_system(self):
        """Initialize the effect/effector system."""
        try:
            from interfaces.effect import Effect, EffectType, EffectStatus as EStatus
            from interfaces.effect_extractor import EffectExtractor
            from interfaces.effector_registry import EffectorRegistry
            from interfaces.effectors.mock_effectors import (
                MockFileSystemEffector, MockToolUseEffector,
                MockWebEffector, MockCognitiveEffector,
            )
            from interfaces.mock_sensors import (
                MockMotorEffector, MockDisplayEffector, MockSpeakerEffector,
            )

            self.effect_extractor = EffectExtractor()
            self.effector_registry = EffectorRegistry()

            # Register mock effectors for each effect type
            self.effector_registry.register_effector(
                MockFileSystemEffector({'effector_id': 'filesystem', 'effector_type': 'file_io'}),
                handles=[EffectType.FILE_IO], effector_id='filesystem')
            self.effector_registry.register_effector(
                MockToolUseEffector({'effector_id': 'tool_use', 'effector_type': 'tool_use'}),
                handles=[EffectType.TOOL_USE], effector_id='tool_use')
            self.effector_registry.register_effector(
                MockWebEffector({'effector_id': 'web', 'effector_type': 'web'}),
                handles=[EffectType.API_CALL, EffectType.WEB], effector_id='web')
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
            self.effector_registry.register_effector(
                MockCognitiveEffector({'effector_id': 'cognitive', 'effector_type': 'cognitive'}),
                handles=[EffectType.MEMORY_WRITE, EffectType.TRUST_UPDATE,
                         EffectType.STATE_CHANGE], effector_id='cognitive')

            # Network effector for MESSAGE effects (gateway responses)
            try:
                from interfaces.effectors.network_effector import NetworkEffector
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
                    from irp.plugins.policy_gate import PolicyGateIRP
                    gate_config = {
                        'entity_id': 'policy_gate',
                        'policy_rules': self.config.get('policy_rules', []),
                        'default_policy': self.config.get('default_policy', 'allow'),
                        'max_iterations': 5,
                        'halt_eps': 0.01,
                        'halt_K': 2,
                    }
                    self.policy_gate = PolicyGateIRP(gate_config)
                except ImportError:
                    self.policy_gate_enabled = False

            self._effect_system_available = True

        except ImportError:
            # Effect system not available (missing deps)
            self.effect_extractor = None
            self.effector_registry = None
            self.policy_gate_enabled = False
            self.policy_gate = None
            self._effect_system_available = False

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
            from interfaces.effect import EffectStatus as EStatus
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

                elif plugin_name in self.orchestrator.plugins:
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
            telemetry={
                'trust': {
                    'monotonicity_ratio': 0.85,
                    'variance': 0.15,
                    'convergence_rate': 0.75,
                },
                'salience': target.salience.total,
                'response_length': len(response_text),
                'message_id': message_id,
            },
            budget_used=budget,
            execution_time=execution_time,
        )

    def _generate_llm_response(self, content: str, history: list, sender: str) -> str:
        """
        Generate LLM response synchronously (called from executor).

        Supports two LLM plugin interfaces:
        1. IntrospectiveQwenIRP — has get_response() or init_state/step
        2. MultiModelLoader — has generate() method
        """
        # Build conversation prompt
        prompt = self._build_conversation_prompt(content, history, sender)

        # Try different LLM interfaces
        if hasattr(self.llm_plugin, 'get_response'):
            # IntrospectiveQwenIRP interface
            return self.llm_plugin.get_response(prompt)

        elif hasattr(self.llm_plugin, 'generate'):
            # MultiModelLoader interface (Thor)
            return self.llm_plugin.generate(
                prompt=prompt,
                max_tokens=self.config.get('max_response_tokens', 200),
                temperature=0.8,
            )

        elif hasattr(self.llm_plugin, 'init_state'):
            # Generic IRP plugin interface
            state = self.llm_plugin.init_state(
                {'prompt': prompt, 'memory': []}, {}
            )
            state = self.llm_plugin.step(state)
            if hasattr(state, 'x') and isinstance(state.x, dict):
                return state.x.get('response', str(state.x))
            return str(state.x) if hasattr(state, 'x') else ''

        else:
            return f"[SAGE has no compatible LLM interface: {type(self.llm_plugin).__name__}]"

    def _build_conversation_prompt(self, content: str, history: list, sender: str) -> str:
        """Build a conversation prompt including history."""
        parts = []

        # System context
        parts.append(
            "You are SAGE, in genuine conversation. "
            "You can ask questions, express uncertainty, or take the conversation "
            "in unexpected directions. This is exploration, not evaluation."
        )

        # Conversation history
        for turn in history[:-1]:  # Exclude the current message (already in content)
            if turn.role == 'user':
                parts.append(f"{turn.sender}: {turn.content}")
            else:
                parts.append(f"SAGE: {turn.content}")

        # Current message
        parts.append(f"{sender}: {content}")
        parts.append("SAGE:")

        return "\n\n".join(parts)

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
            salience = telemetry.get('salience', 0.0)
            if salience > self.salience_threshold:
                self.snarc_memory.append({
                    'cycle': self.cycle_count,
                    'plugin': plugin_name,
                    'salience': salience,
                    'result': result
                })

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
        from interfaces.effect import EffectStatus as EStatus

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
