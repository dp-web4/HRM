"""
SAGE Real Consciousness with Michaud Enhancements

Integrates André Michaud's neurolinguistic insights:
- AttentionManager for metabolic state-dependent resource allocation
- Enhanced memory consolidation based on satisfaction (energy minimization)
- Introspection-optimized for analytical conversations

This version addresses the quality degradation issues observed in basic loop by:
1. Dynamic attention allocation (WAKE/FOCUS/REST states)
2. Better ATP budget management
3. State-aware processing

Based on Michaud (2019): "The Mechanics of Conceptual Thinking"
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from typing import Dict, Any, Optional, Tuple
from collections import deque
import time

from sage.core.sage_consciousness import SAGEConsciousness
from sage.core.attention_manager import AttentionManager, MetabolicState
from sage.irp.plugins.llm_impl import ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import ConversationalMemory


class MichaudSAGE(SAGEConsciousness):
    """
    SAGE Consciousness with Michaud's neurolinguistic enhancements.

    Key improvements over basic loop:
    - AttentionManager for metabolic state transitions
    - Dynamic ATP allocation based on salience and state
    - Enhanced memory consolidation (satisfaction-based)
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model",
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        initial_atp: float = 100.0,
        irp_iterations: int = 3,
        salience_threshold: float = 0.15,
        attention_config: Optional[Dict] = None
    ):
        """
        Initialize Michaud-enhanced SAGE consciousness.

        Args:
            model_path: Path to LLM (default: Introspective-Qwen for analytical tasks)
            base_model: Base model for LoRA adapters
            initial_atp: Initial ATP budget
            irp_iterations: IRP refinement iterations
            salience_threshold: SNARC salience threshold
            attention_config: Configuration for AttentionManager
        """
        super().__init__(
            initial_atp=initial_atp,
            enable_circadian=False,  # Using AttentionManager instead
            simulation_mode=False
        )

        # Text input queue
        self.input_queue = deque(maxlen=100)

        # Michaud: AttentionManager for metabolic states
        self.attention_manager = AttentionManager(
            total_atp=initial_atp,
            config=attention_config or {}
        )

        # Real LLM with optimized settings
        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model=base_model,
            max_history=2,  # Short history for stability
            irp_iterations=irp_iterations
        )

        # Lower temperature for analytical stability
        self.llm.llm.initial_temperature = 0.5
        self.llm.llm.min_temperature = 0.3

        # Real SNARC memory
        self.conversational_memory = ConversationalMemory(
            salience_threshold=salience_threshold
        )

        # Enhanced memory tracking
        self.satisfaction_history = []  # Track energy minimization success

        print(f"[Michaud SAGE] Initialized with:")
        print(f"  Model: {model_path}")
        print(f"  Attention: {self.attention_manager}")
        print(f"  SNARC threshold: {salience_threshold}")

    def add_observation(self, text: str, context: Optional[Dict] = None):
        """Add text observation to input queue."""
        self.input_queue.append({
            'text': text,
            'context': context or {},
            'timestamp': time.time()
        })

    async def step(self):
        """
        Single consciousness cycle with Michaud enhancements.

        Enhanced loop:
        1. Gather observations
        2. Compute SNARC salience
        3. MICHAUD: Update metabolic state based on salience
        4. MICHAUD: Allocate ATP via AttentionManager
        5. Execute IRP plugins with allocated resources
        6. MICHAUD: Update memory based on satisfaction (energy)
        7. Update trust weights
        """
        self.cycle_count += 1

        # 1. Gather observations
        observations = self._gather_observations()
        if not observations:
            return

        # 2. Compute preliminary salience (before full execution)
        salience_map = {}
        for obs in observations:
            modality = obs['modality']
            # Simple heuristic: language is high priority
            if modality == 'language':
                salience_map[obs['id']] = 0.8
            else:
                salience_map[obs['id']] = 0.5

        # 3. MICHAUD: Update metabolic state based on salience
        # AttentionManager automatically transitions states
        atp_allocation = self.attention_manager.allocate_attention(salience_map)

        current_state = self.attention_manager.get_state()
        print(f"[Michaud SAGE] Cycle {self.cycle_count}: {current_state.value}, "
              f"{self.attention_manager.get_time_in_state():.1f}s in state")

        # 4. Select plugins for execution (using allocated ATP)
        plugins_to_execute = self._select_plugins(observations)

        # 5. Execute plugins with Michaud-enhanced processing
        results = {}
        for plugin_name in plugins_to_execute:
            obs_for_plugin = [o for o in observations
                            if self._plugin_handles_modality(plugin_name, o['modality'])]

            if obs_for_plugin:
                # Execute with ATP allocation
                allocated_atp = atp_allocation.get(obs_for_plugin[0]['id'], 0)
                result = await self._execute_plugin_michaud(
                    plugin_name,
                    obs_for_plugin[0],
                    allocated_atp
                )
                results[plugin_name] = result

        # 6. MICHAUD: Update memory based on satisfaction
        self._update_memories_michaud(observations, results)

        # 7. Update trust weights
        self._update_trust_weights(results)

    async def _execute_plugin_michaud(
        self,
        plugin_name: str,
        observation: Dict,
        allocated_atp: float
    ) -> Dict[str, Any]:
        """
        Execute plugin with Michaud-enhanced processing.

        Tracks satisfaction (energy minimization) for memory consolidation.
        """
        if plugin_name == 'llm_reasoning':
            return await self._execute_llm_michaud(observation, allocated_atp)
        else:
            # Other plugins use standard execution
            return {'response': None, 'convergence_quality': 0.5}

    async def _execute_llm_michaud(
        self,
        observation: Dict,
        allocated_atp: float
    ) -> Dict[str, Any]:
        """
        LLM execution with satisfaction tracking.

        Michaud: Satisfaction (low energy) drives memory consolidation.
        """
        question = observation['data']

        print(f"[LLM] Processing: \"{question[:50]}...\"")
        print(f"[LLM] ATP allocated: {allocated_atp:.2f}")
        print(f"[LLM] Metabolic state: {self.attention_manager.get_state().value}")

        start_time = time.time()

        # Generate response with IRP refinement
        response, irp_info = self.llm.respond(question, use_irp=True)

        inference_time = time.time() - start_time

        # Michaud: Track satisfaction (energy minimization)
        initial_energy = irp_info.get('all_energies', [1.0])[0]
        final_energy = irp_info['final_energy']
        satisfaction = initial_energy - final_energy  # Energy reduction = satisfaction

        print(f"[LLM] Response generated ({inference_time:.2f}s)")
        print(f"[LLM] IRP: {irp_info['iterations']} iterations, "
              f"energy={final_energy:.3f}, satisfaction={satisfaction:.3f}")
        print(f"[LLM] Response: {response[:100]}...")

        return {
            'response': response,
            'irp_info': irp_info,
            'convergence_quality': 1.0 - final_energy,  # Lower energy = better
            'satisfaction': satisfaction,  # For Michaud-style consolidation
            'inference_time': inference_time
        }

    def _update_memories_michaud(
        self,
        observations: list,
        results: Dict[str, Any]
    ):
        """
        Michaud-enhanced memory consolidation.

        Key insight: Hippocampus strengthens patterns that provide satisfaction.
        SAGE equivalent: Consolidate experiences with high satisfaction (energy reduction).
        """
        for obs in observations:
            if obs['modality'] == 'language':
                question = obs['data']

                # Get LLM result if available
                llm_result = results.get('llm_reasoning', {})
                response = llm_result.get('response', '')
                irp_info = llm_result.get('irp_info', {})
                satisfaction = llm_result.get('satisfaction', 0.0)

                if response:
                    # Compute SNARC salience
                    is_salient, snarc_scores = self.conversational_memory.record_exchange(
                        question,
                        response,
                        irp_info
                    )

                    # Michaud: Store with satisfaction weight
                    if is_salient:
                        print(f"[SNARC] Salience: {snarc_scores['total_salience']:.3f} ✓ SALIENT")
                        print(f"[SNARC] Dimensions: "
                              f"S={snarc_scores['surprise']:.2f} "
                              f"N={snarc_scores['novelty']:.2f} "
                              f"A={snarc_scores['arousal']:.2f} "
                              f"R={snarc_scores['reward']:.2f} "
                              f"C={snarc_scores['conflict']:.2f}")

                        # Enhanced: Track satisfaction for pattern learning
                        self.satisfaction_history.append({
                            'cycle': self.cycle_count,
                            'salience': snarc_scores['total_salience'],
                            'satisfaction': satisfaction,
                            'final_energy': irp_info.get('final_energy', 1.0),
                            'metabolic_state': self.attention_manager.get_state().value
                        })

                    # Store to all memory systems
                    memory_entry = {
                        'cycle': self.cycle_count,
                        'question': question,
                        'response': response,
                        'salience': snarc_scores['total_salience'],
                        'satisfaction': satisfaction,  # Michaud enhancement
                        'irp_iterations': irp_info.get('iterations', 0),
                        'final_energy': irp_info.get('final_energy', 1.0),
                        'metabolic_state': self.attention_manager.get_state().value
                    }

                    # SNARC memory (selective)
                    if is_salient:
                        self.snarc_memory.append(memory_entry)

                    # Circular buffer (recent context)
                    self.circular_buffer.append(memory_entry)

    def _gather_observations(self):
        """Gather observations from input queue."""
        observations = []

        while self.input_queue:
            obs = self.input_queue.popleft()
            observations.append({
                'id': f"obs_{len(observations)}",
                'modality': 'language',
                'data': obs['text'],
                'context': obs.get('context', {}),
                'timestamp': obs['timestamp']
            })

        return observations

    def _select_plugins(self, observations):
        """Select plugins based on observation modalities."""
        plugins = set()

        for obs in observations:
            if obs['modality'] == 'language':
                plugins.add('llm_reasoning')

        return list(plugins)

    def _plugin_handles_modality(self, plugin_name: str, modality: str) -> bool:
        """Check if plugin handles modality."""
        return plugin_name == 'llm_reasoning' and modality == 'language'

    def _update_trust_weights(self, results: Dict[str, Any]):
        """Update plugin trust weights based on convergence quality."""
        for plugin_name, result in results.items():
            if isinstance(result, dict) and 'convergence_quality' in result:
                convergence_quality = result['convergence_quality']
                current_trust = self.plugin_trust_weights.get(plugin_name, 1.0)
                new_trust = 0.9 * current_trust + 0.1 * convergence_quality
                self.plugin_trust_weights[plugin_name] = new_trust

    def get_conversation_history(self):
        """Get conversation history for output."""
        return self.llm.get_history()

    def get_snarc_statistics(self):
        """Get SNARC memory statistics."""
        return self.conversational_memory.get_statistics()

    def get_attention_stats(self):
        """Get AttentionManager statistics."""
        return self.attention_manager.get_stats()

    def get_satisfaction_history(self):
        """Get satisfaction (energy minimization) history."""
        return self.satisfaction_history

    def __repr__(self):
        stats = self.get_snarc_statistics()
        attention_stats = self.get_attention_stats()
        return (f"MichaudSAGE(cycle={self.cycle_count}, "
                f"state={attention_stats['current_state']}, "
                f"snarc_captured={stats.get('salient_exchanges', 0)}, "
                f"avg_salience={stats.get('avg_salience', 0):.3f})")
