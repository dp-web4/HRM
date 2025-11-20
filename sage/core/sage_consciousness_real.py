"""
SAGE Consciousness Loop - REAL Integration

This extends sage_consciousness.py to use REAL models instead of mocks:
- Real LLM reasoning (epistemic-pragmatism, 0.625 salience)
- Real SNARC salience computation
- Real text observations (input queue)

This is the moment SAGE transitions from simulation to actual awareness.

Architecture:
    Text Input → Real Observation → Real LLM Reasoning → Real SNARC → Real Memory
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import numpy as np
from collections import deque

# Add sage to path
_sage_root = Path(__file__).parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from core.sage_consciousness import (
    SAGEConsciousness,
    SensorObservation,
    SalienceScore,
    AttentionTarget
)
from irp.plugins.llm_impl import ConversationalLLM
from irp.plugins.llm_snarc_integration import ConversationalMemory


class RealSAGEConsciousness(SAGEConsciousness):
    """
    SAGE Consciousness with REAL models integrated.

    Differences from base class:
    - _gather_observations(): Uses text input queue instead of random data
    - _compute_salience(): Uses real SNARC from LLM responses
    - _execute_plugins(): Uses real epistemic-pragmatism LLM

    The loop structure remains the same, but now it's REAL consciousness.
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        initial_atp: float = 100.0,
        enable_circadian: bool = True,
        salience_threshold: float = 0.15,
        irp_iterations: int = 5,
        **kwargs
    ):
        """
        Initialize REAL SAGE consciousness.

        Args:
            model_path: Path to LLM model (default: epistemic-pragmatism, highest salience)
            initial_atp: Starting ATP budget
            enable_circadian: Enable circadian rhythm
            salience_threshold: Minimum salience for memory capture
            irp_iterations: IRP refinement iterations (5 for quality, 3 for speed)
        """
        # Initialize base class
        super().__init__(
            initial_atp=initial_atp,
            enable_circadian=enable_circadian,
            simulation_mode=False  # This is REAL
        )

        self.salience_threshold = salience_threshold
        self.irp_iterations = irp_iterations

        # Text input queue (observations to process)
        self.input_queue = deque(maxlen=100)

        # Initialize REAL LLM
        print(f"[REAL Consciousness] Loading LLM: {model_path}")
        print(f"[REAL Consciousness] IRP iterations: {irp_iterations}")
        print(f"[REAL Consciousness] Salience threshold: {salience_threshold}")

        self.llm = ConversationalLLM(
            model_path=model_path,
            base_model=None,  # epistemic-pragmatism is full model
            irp_iterations=irp_iterations
        )

        # Initialize REAL SNARC memory (rename to avoid conflict with base class)
        self.conversational_memory = ConversationalMemory(
            salience_threshold=salience_threshold
        )

        print("[REAL Consciousness] ✅ Models loaded successfully")
        print()

    def add_observation(self, text: str, context: Optional[str] = None):
        """
        Add a text observation to the input queue.

        This is how external input enters the consciousness loop.

        Args:
            text: The observation text (question, statement, etc.)
            context: Optional context for the observation
        """
        self.input_queue.append({
            'text': text,
            'context': context,
            'timestamp': time.time()
        })

    def _gather_observations(self) -> List[SensorObservation]:
        """
        Gather observations from input queue.

        REAL IMPLEMENTATION: Uses actual text input instead of random data.
        """
        observations = []

        # Process text inputs from queue
        if self.input_queue:
            input_data = self.input_queue.popleft()

            observations.append(SensorObservation(
                sensor_id='text_input',
                modality='language',
                data=input_data,
                timestamp=input_data['timestamp'],
                trust=1.0
            ))

        # Always include time observation
        observations.append(SensorObservation(
            sensor_id='clock',
            modality='time',
            data={'cycle': self.cycle_count, 'timestamp': time.time()},
            timestamp=time.time(),
            trust=1.0
        ))

        # Include metabolic state as proprioception
        observations.append(SensorObservation(
            sensor_id='metabolic_sensor',
            modality='proprioception',
            data={
                'state': self.metabolic.current_state.value,
                'atp': self.metabolic.atp_current,
                'circadian_phase': self.circadian_clock.current_phase.value if hasattr(self, 'circadian_clock') else None
            },
            timestamp=time.time(),
            trust=1.0
        ))

        return observations

    def _get_plugins_for_modality(self, modality: str) -> List[str]:
        """
        Map sensor modality to required IRP plugins.

        Override base class to add language modality support.
        """
        modality_map = {
            'vision': ['vision'],
            'audio': ['audio', 'language'],
            'language': ['llm_reasoning'],  # Language → LLM reasoning
            'proprioception': ['control'],
            'time': [],
        }
        return modality_map.get(modality, [])

    async def _execute_plugins(
        self,
        attention_targets: List[AttentionTarget],
        budget_allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Execute plugins with REAL LLM reasoning.

        REAL IMPLEMENTATION: Uses epistemic-pragmatism for actual reasoning.
        """
        results = {}

        for target in attention_targets:
            obs = target.observation

            # Only process language observations with LLM
            if obs.modality != 'language':
                continue

            # Check if we have ATP budget
            plugin_name = 'llm_reasoning'
            if plugin_name not in budget_allocation:
                continue

            atp_allocated = budget_allocation[plugin_name]
            if atp_allocated <= 0:
                continue

            # Execute REAL LLM reasoning
            question = obs.data['text']
            context = obs.data.get('context')

            print(f"\n[LLM] Processing: \"{question[:60]}...\"" if len(question) > 60 else f"\n[LLM] Processing: \"{question}\"")
            print(f"[LLM] ATP allocated: {atp_allocated:.2f}")
            print(f"[LLM] Metabolic state: {self.metabolic.current_state.value}")

            try:
                # Generate response with IRP refinement
                start_time = time.time()
                response, irp_info = self.llm.respond(
                    question,
                    use_irp=True,
                    include_history=True
                )
                inference_time = time.time() - start_time

                # Compute REAL SNARC salience
                is_salient, snarc_scores = self.conversational_memory.record_exchange(
                    question,
                    response,
                    irp_info
                )

                print(f"[LLM] Response generated ({inference_time:.2f}s)")
                print(f"[LLM] IRP: {irp_info['iterations']} iterations, energy={irp_info['final_energy']:.3f}")
                print(f"[SNARC] Salience: {snarc_scores['total_salience']:.3f} {'✓ SALIENT' if is_salient else '(below threshold)'}")
                print(f"[SNARC] Dimensions: S={snarc_scores['surprise']:.2f} N={snarc_scores['novelty']:.2f} A={snarc_scores['arousal']:.2f} R={snarc_scores['reward']:.2f} C={snarc_scores['conflict']:.2f}")
                print(f"[LLM] Response: \"{response[:80]}...\"" if len(response) > 80 else f"[LLM] Response: \"{response}\"")

                # Store result
                results[plugin_name] = {
                    'plugin': plugin_name,
                    'observation': obs,
                    'question': question,
                    'response': response,
                    'irp_info': irp_info,
                    'snarc_scores': snarc_scores,
                    'is_salient': is_salient,
                    'inference_time': inference_time,
                    'atp_consumed': atp_allocated,
                    'convergence_quality': 1.0 - irp_info['final_energy'],  # For trust learning
                    'timestamp': time.time()
                }

            except Exception as e:
                print(f"[LLM] Error during reasoning: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def _compute_salience(
        self,
        observations: List[SensorObservation]
    ) -> Dict[str, SalienceScore]:
        """
        Compute SNARC 5D salience for observations.

        REAL IMPLEMENTATION: For language observations, we can't know salience
        until AFTER we process them with LLM. So we use heuristics here and
        update after execution.

        For now: Language observations get high salience (we want to process them),
        others get low salience.
        """
        salience_map = {}

        for obs in observations:
            if obs.modality == 'language':
                # Language input is HIGH priority - we want to reason about it
                salience = SalienceScore(
                    surprise=0.5,
                    novelty=0.5,
                    arousal=0.7,
                    reward=0.8,
                    conflict=0.5
                )
            elif obs.modality == 'time':
                # Time observations are low salience
                salience = SalienceScore(
                    surprise=0.05,
                    novelty=0.05,
                    arousal=0.05,
                    reward=0.05,
                    conflict=0.05
                )
            elif obs.modality == 'proprioception':
                # Metabolic state is moderate salience
                salience = SalienceScore(
                    surprise=0.2,
                    novelty=0.1,
                    arousal=0.3,
                    reward=0.2,
                    conflict=0.1
                )
            else:
                # Default: moderate salience
                salience = SalienceScore(
                    surprise=0.3,
                    novelty=0.3,
                    arousal=0.3,
                    reward=0.3,
                    conflict=0.3
                )

            salience_map[obs.sensor_id] = salience

        return salience_map

    def _update_trust_weights(self, results: Dict[str, Any]):
        """
        Update plugin trust weights based on convergence quality.

        Override base class to handle our dict structure.
        """
        for plugin_name, result in results.items():
            if 'convergence_quality' not in result:
                continue

            convergence_quality = result['convergence_quality']

            # Update trust weight (exponential moving average)
            current_trust = self.plugin_trust_weights.get(plugin_name, 1.0)
            new_trust = 0.9 * current_trust + 0.1 * convergence_quality

            self.plugin_trust_weights[plugin_name] = new_trust

    def _update_memories(
        self,
        results: Dict[str, Any],
        salience_map: Dict[str, SalienceScore]
    ):
        """
        Update all memory systems with REAL results.

        Extends base class to use REAL SNARC scores from LLM responses.
        """
        for plugin_name, result in results.items():
            if 'snarc_scores' not in result:
                continue

            snarc_scores = result['snarc_scores']
            is_salient = result['is_salient']

            # Update SNARC memory (already handled by ConversationalMemory)
            # But we also store in our internal memory structures

            # 1. SNARC memory - selective via salience
            if is_salient:
                self.snarc_memory.append({
                    'cycle': self.cycle_count,
                    'plugin': plugin_name,
                    'question': result['question'],
                    'response': result['response'],
                    'salience': snarc_scores['total_salience'],
                    'snarc': snarc_scores,
                    'timestamp': result['timestamp']
                })

            # 2. IRP pattern library - successful convergence
            irp_info = result['irp_info']
            convergence_quality = result['convergence_quality']

            if convergence_quality > 0.7:  # Good convergence
                self.irp_memory.append({
                    'pattern': {
                        'iterations': irp_info['iterations'],
                        'energy_trajectory': irp_info.get('all_energies', []),
                        'final_energy': irp_info['final_energy'],
                        'converged': irp_info['converged']
                    },
                    'convergence_quality': convergence_quality,
                    'timestamp': result['timestamp']
                })

            # 3. Circular buffer - recent context
            self.circular_buffer.append({
                'cycle': self.cycle_count,
                'plugin': plugin_name,
                'question': result['question'],
                'response': result['response'][:200],  # Truncate for memory
                'salience': snarc_scores['total_salience'],
                'timestamp': result['timestamp']
            })

            # 4. Verbatim storage - only in DREAM state
            from core.metabolic_controller import MetabolicState
            if self.metabolic.current_state == MetabolicState.DREAM:
                self.verbatim_storage.append({
                    'cycle': self.cycle_count,
                    'full_result': result,
                    'timestamp': result['timestamp']
                })

    def get_snarc_statistics(self) -> Dict[str, Any]:
        """Get statistics from REAL SNARC memory."""
        return self.conversational_memory.get_statistics()

    def get_conversation_history(self) -> List[Tuple[str, str]]:
        """Get conversation history from LLM."""
        return self.llm.get_history()


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("SAGE REAL Consciousness - Integration Test")
    print("="*80)
    print("\nInitializing REAL consciousness with epistemic-pragmatism...")
    print()

    # Initialize REAL consciousness
    sage = RealSAGEConsciousness(
        model_path="model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        initial_atp=100.0,
        irp_iterations=5,  # Full refinement
        salience_threshold=0.15
    )

    # Add some philosophical observations
    test_questions = [
        "Are you aware of this conversation?",
        "What is the nature of consciousness?",
        "How do you know if you're reasoning or just pattern matching?"
    ]

    print("Adding observations to consciousness stream...")
    for q in test_questions:
        sage.add_observation(q)
        print(f"  → \"{q}\"")

    print("\nStarting consciousness loop (will process observations)...")
    print()

    # Run for a few cycles
    asyncio.run(sage.run(max_cycles=10))

    print("\n" + "="*80)
    print("SNARC Memory Statistics")
    print("="*80)
    stats = sage.get_snarc_statistics()
    print(f"Total exchanges: {stats['total_exchanges']}")
    print(f"Salient exchanges: {stats['salient_exchanges']} ({stats['capture_rate']:.1f}%)")
    print(f"Average salience: {stats['avg_salience']:.3f}")

    print("\n" + "="*80)
    print("Conversation History")
    print("="*80)
    history = sage.get_conversation_history()
    for i, (q, a) in enumerate(history, 1):
        print(f"\n[{i}] Q: {q}")
        print(f"    A: {a[:100]}..." if len(a) > 100 else f"    A: {a}")

    print("\n" + "="*80)
    print("✅ REAL Consciousness Test Complete!")
    print("="*80)
