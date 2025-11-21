"""
Cogitation Plugin - Internal Dialogue for Conceptual Thinking

Implements Michaud's "interior language" - verbal cogitation until
conceptual satisfaction is achieved.

Based on Michaud (2019):
- "Comprehension process: exploring and re-exploring a concept until
   objective understanding has been reached"
- "Humans do not speak because they think; they speak because their
   thinking process is an interior language"
- Cogitation = silent internal conversation with self

This plugin enables SAGE to:
1. Question its own understanding
2. Refine concepts through internal dialogue
3. Detect and resolve contradictions
4. Build coherent conceptual structures
5. Know when understanding is sufficient vs needs more exploration
"""

import torch
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

from .base import IRPPlugin, IRPState
from ..language import LanguageModel


class CogitationMode(Enum):
    """Modes of conceptual thinking."""
    EXPLORING = "exploring"           # Initial exploration
    QUESTIONING = "questioning"       # Active interrogation
    INTEGRATING = "integrating"       # Synthesizing understanding
    VERIFYING = "verifying"          # Checking coherence
    REFRAMING = "reframing"          # Changing perspective when stuck


@dataclass
class Thought:
    """Single thought in internal dialogue."""
    content: str                # Verbal content
    mode: CogitationMode       # What kind of thought
    coherence: float           # How well it fits with existing understanding
    timestamp: int             # Step number


class CogitationPlugin(IRPPlugin):
    """
    IRP plugin for internal conceptual dialogue.

    Refines understanding through iterative verbal exploration until
    conceptual satisfaction is achieved (coherent understanding with
    no remaining contradictions or unresolved aspects).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Language model for thought generation
        self.language_model = LanguageModel(config.get('language_model', {}))

        # Parameters
        self.max_iterations = config.get('max_iterations', 50)
        self.coherence_threshold = config.get('coherence_threshold', 0.85)
        self.stuck_threshold = config.get('stuck_threshold', 5)
        self.reframe_interval = config.get('reframe_interval', 10)

    def can_handle(self, observation: Any) -> float:
        """
        Determine if observation requires conceptual thinking.

        Returns confidence [0, 1] that cogitation is needed.
        """
        if isinstance(observation, str):
            # Check for conceptual/abstract content
            abstract_markers = [
                'understand', 'concept', 'meaning', 'why', 'how',
                'relationship', 'connection', 'implication', 'significance'
            ]

            content_lower = observation.lower()
            marker_count = sum(1 for marker in abstract_markers
                             if marker in content_lower)

            # High marker count = likely conceptual question
            confidence = min(1.0, marker_count / 3.0)
            return confidence

        return 0.1  # Low confidence for non-textual input

    def init_state(self, observation: Any) -> IRPState:
        """
        Initialize cogitation process.

        Args:
            observation: Question or concept to understand

        Returns:
            Initial state with question and empty understanding
        """
        # Parse observation into question form
        if isinstance(observation, str):
            question = observation
        else:
            question = str(observation)

        # Initialize state
        return IRPState(
            data={
                'original_question': question,
                'current_question': question,
                'current_understanding': None,
                'thought_history': [],
                'explored_paths': [],
                'coherence': 0.0,
                'mode': CogitationMode.EXPLORING,
                'stuck_count': 0,
                'reframe_count': 0,
                'contradictions': [],
                'unresolved_aspects': []
            },
            metadata={
                'plugin': 'cogitation',
                'iteration': 0
            }
        )

    def step(self, state: IRPState) -> IRPState:
        """
        One step of internal dialogue.

        Generates next thought, integrates with current understanding,
        evaluates coherence, and potentially reframes if stuck.
        """
        iteration = state.metadata.get('iteration', 0)
        mode = state.data['mode']

        # Generate next thought based on current mode
        next_thought = self._generate_thought(state, mode)

        # Integrate thought with existing understanding
        updated_understanding = self._integrate_thought(
            state.data['current_understanding'],
            next_thought,
            state.data['thought_history']
        )

        # Evaluate new understanding
        coherence = self._evaluate_coherence(updated_understanding, state)
        contradictions = self._detect_contradictions(updated_understanding)
        unresolved = self._detect_unresolved_aspects(
            updated_understanding,
            state.data['original_question']
        )

        # Determine if stuck
        stuck_count = state.data['stuck_count']
        if self._is_stuck(state.data['thought_history'], next_thought):
            stuck_count += 1
        else:
            stuck_count = 0

        # Handle stuck situation - reframe question
        current_question = state.data['current_question']
        reframe_count = state.data['reframe_count']

        if stuck_count >= self.stuck_threshold:
            # Michaud: "ask different question" when stuck
            current_question = self._reframe_question(
                state.data['original_question'],
                updated_understanding,
                contradictions,
                unresolved
            )
            reframe_count += 1
            stuck_count = 0
            mode = CogitationMode.REFRAMING

        # Determine next mode
        elif coherence < 0.5:
            mode = CogitationMode.EXPLORING
        elif len(contradictions) > 0:
            mode = CogitationMode.QUESTIONING
        elif len(unresolved) > 0:
            mode = CogitationMode.INTEGRATING
        else:
            mode = CogitationMode.VERIFYING

        # Update thought history
        new_thought_history = state.data['thought_history'] + [next_thought]

        # Update explored paths (for stuck detection)
        new_explored_paths = state.data['explored_paths'] + [
            self._get_thought_signature(next_thought)
        ]

        return IRPState(
            data={
                'original_question': state.data['original_question'],
                'current_question': current_question,
                'current_understanding': updated_understanding,
                'thought_history': new_thought_history,
                'explored_paths': new_explored_paths,
                'coherence': coherence,
                'mode': mode,
                'stuck_count': stuck_count,
                'reframe_count': reframe_count,
                'contradictions': contradictions,
                'unresolved_aspects': unresolved
            },
            metadata={
                'plugin': 'cogitation',
                'iteration': iteration + 1
            }
        )

    def energy(self, state: IRPState) -> float:
        """
        Conceptual dissatisfaction.

        Lower energy = more satisfied with understanding.

        Energy components:
        - Incoherence (primary driver)
        - Internal contradictions
        - Unresolved aspects of question
        - Repetition penalty (avoid loops)
        """
        # Incoherence: How well does understanding fit together?
        incoherence = 1.0 - state.data['coherence']

        # Contradictions: Internal logical conflicts
        contradiction_count = len(state.data['contradictions'])
        contradiction_penalty = min(1.0, contradiction_count * 0.3)

        # Unresolved aspects: Parts of question not answered
        unresolved_count = len(state.data['unresolved_aspects'])
        unresolved_penalty = min(1.0, unresolved_count * 0.2)

        # Repetition: Avoid circular thinking
        repetition_penalty = 0.0
        if state.data['stuck_count'] > 0:
            repetition_penalty = min(1.0, state.data['stuck_count'] * 0.1)

        # Total energy
        total_energy = (
            10.0 * incoherence +               # Primary driver
            5.0 * contradiction_penalty +       # Contradictions bad
            3.0 * unresolved_penalty +          # Gaps bad
            2.0 * repetition_penalty            # Loops bad
        )

        return total_energy

    def should_halt(self, energy_history: List[float]) -> bool:
        """
        Stop when conceptually satisfied or genuinely stuck.

        Satisfaction = coherent understanding with no contradictions
                      and all aspects of question addressed.
        """
        if super().should_halt(energy_history):
            return True

        # Halt if reached conceptual satisfaction
        if energy_history[-1] < 1.0:  # Very low energy
            return True

        # Halt if maximum iterations reached
        if len(energy_history) >= self.max_iterations:
            return True

        return False

    def _generate_thought(self, state: IRPState, mode: CogitationMode) -> Thought:
        """
        Generate next thought using language model.

        Thought generation depends on current mode:
        - EXPLORING: Generate broad associations
        - QUESTIONING: Probe specific aspects
        - INTEGRATING: Synthesize components
        - VERIFYING: Check logical consistency
        - REFRAMING: Change perspective
        """
        # Construct prompt based on mode
        prompt = self._construct_prompt(state, mode)

        # Generate thought content
        content = self.language_model.generate(
            prompt=prompt,
            max_length=200,
            temperature=self._get_temperature(mode)
        )

        # Evaluate coherence of this thought with existing understanding
        coherence = self._evaluate_thought_coherence(
            content,
            state.data['current_understanding']
        )

        return Thought(
            content=content,
            mode=mode,
            coherence=coherence,
            timestamp=state.metadata.get('iteration', 0)
        )

    def _construct_prompt(self, state: IRPState, mode: CogitationMode) -> str:
        """Construct prompt for thought generation based on mode."""
        question = state.data['current_question']
        understanding = state.data['current_understanding']

        if mode == CogitationMode.EXPLORING:
            return f"""Question: {question}

Current understanding: {understanding if understanding else 'None yet'}

Generate the next thought to explore this question. Consider:
- What aspects haven't been considered yet?
- What connections might exist?
- What examples illustrate this?

Next thought:"""

        elif mode == CogitationMode.QUESTIONING:
            contradictions = state.data['contradictions']
            return f"""Question: {question}

Current understanding: {understanding}

Contradictions detected: {contradictions}

Generate a probing question to resolve these contradictions.

Question:"""

        elif mode == CogitationMode.INTEGRATING:
            unresolved = state.data['unresolved_aspects']
            return f"""Question: {question}

Current understanding: {understanding}

Unresolved aspects: {unresolved}

Generate a thought that integrates these unresolved aspects.

Thought:"""

        elif mode == CogitationMode.VERIFYING:
            return f"""Question: {question}

Current understanding: {understanding}

Verify the logical consistency of this understanding.
Does it fully answer the question? Are there any hidden assumptions?

Verification:"""

        elif mode == CogitationMode.REFRAMING:
            return f"""Original question: {state.data['original_question']}

Current understanding: {understanding}

This line of thinking seems stuck. Reframe the question from a
different perspective or at a different level of abstraction.

Reframed question:"""

    def _get_temperature(self, mode: CogitationMode) -> float:
        """Temperature for generation - higher = more creative."""
        temperature_map = {
            CogitationMode.EXPLORING: 0.8,      # Creative exploration
            CogitationMode.QUESTIONING: 0.6,    # Focused probing
            CogitationMode.INTEGRATING: 0.7,    # Moderate creativity
            CogitationMode.VERIFYING: 0.3,      # Precise checking
            CogitationMode.REFRAMING: 0.9       # Very creative
        }
        return temperature_map.get(mode, 0.7)

    def _integrate_thought(self, current_understanding: Optional[str],
                          new_thought: Thought,
                          thought_history: List[Thought]) -> str:
        """
        Integrate new thought with existing understanding.

        Uses language model to synthesize coherent updated understanding.
        """
        if current_understanding is None:
            # First thought becomes initial understanding
            return new_thought.content

        # Synthesize integration
        prompt = f"""Current understanding: {current_understanding}

New insight: {new_thought.content}

Integrate this new insight into a coherent updated understanding:

Updated understanding:"""

        integrated = self.language_model.generate(
            prompt=prompt,
            max_length=500,
            temperature=0.5  # Moderate creativity for integration
        )

        return integrated

    def _evaluate_coherence(self, understanding: str, state: IRPState) -> float:
        """
        Evaluate coherence of understanding.

        Coherence = how well understanding fits together logically.
        """
        if understanding is None:
            return 0.0

        # Use language model to evaluate coherence
        prompt = f"""Understanding: {understanding}

Question: {state.data['original_question']}

Evaluate the coherence of this understanding on a scale of 0.0 to 1.0:
- Does it logically fit together?
- Does it address the question?
- Are there contradictions?

Coherence score:"""

        response = self.language_model.generate(
            prompt=prompt,
            max_length=50,
            temperature=0.1  # Very deterministic
        )

        # Parse coherence score from response
        try:
            coherence = float(response.strip().split()[0])
            coherence = max(0.0, min(1.0, coherence))
        except (ValueError, IndexError):
            # Fallback: heuristic evaluation
            coherence = self._heuristic_coherence(understanding, state)

        return coherence

    def _heuristic_coherence(self, understanding: str, state: IRPState) -> float:
        """Fallback heuristic coherence evaluation."""
        # Simple heuristics:
        # - Longer understanding generally more developed
        # - Presence of connecting words (therefore, because, thus)
        # - Addresses original question terms

        score = 0.0

        # Length factor (up to 0.3)
        length_factor = min(0.3, len(understanding.split()) / 100)
        score += length_factor

        # Connecting words (up to 0.3)
        connecting_words = ['therefore', 'because', 'thus', 'hence', 'since',
                           'consequently', 'as a result', 'which means']
        connection_count = sum(1 for word in connecting_words
                              if word in understanding.lower())
        score += min(0.3, connection_count * 0.1)

        # Question term coverage (up to 0.4)
        question_terms = set(state.data['original_question'].lower().split())
        understanding_terms = set(understanding.lower().split())
        coverage = len(question_terms & understanding_terms) / len(question_terms)
        score += 0.4 * coverage

        return min(1.0, score)

    def _detect_contradictions(self, understanding: Optional[str]) -> List[str]:
        """
        Detect internal contradictions in understanding.

        Returns list of contradiction descriptions.
        """
        if understanding is None:
            return []

        # Use language model for sophisticated detection
        prompt = f"""Understanding: {understanding}

Identify any internal contradictions or logical conflicts in this understanding.
List each contradiction found, or "None" if there are no contradictions.

Contradictions:"""

        response = self.language_model.generate(
            prompt=prompt,
            max_length=300,
            temperature=0.3
        )

        # Parse response
        if 'none' in response.lower():
            return []

        # Split into individual contradictions
        contradictions = [
            c.strip()
            for c in response.split('\n')
            if c.strip() and not c.strip().startswith('#')
        ]

        return contradictions[:5]  # Max 5 contradictions

    def _detect_unresolved_aspects(self, understanding: Optional[str],
                                  question: str) -> List[str]:
        """
        Detect unresolved aspects of the question.

        Returns list of aspects that haven't been adequately addressed.
        """
        if understanding is None:
            return [question]  # Everything unresolved

        # Use language model to identify gaps
        prompt = f"""Question: {question}

Understanding: {understanding}

What aspects of the question remain unresolved or inadequately addressed?
List each unresolved aspect, or "None" if all aspects are resolved.

Unresolved aspects:"""

        response = self.language_model.generate(
            prompt=prompt,
            max_length=200,
            temperature=0.3
        )

        # Parse response
        if 'none' in response.lower():
            return []

        # Split into individual aspects
        unresolved = [
            a.strip()
            for a in response.split('\n')
            if a.strip() and not a.strip().startswith('#')
        ]

        return unresolved[:5]  # Max 5 unresolved aspects

    def _is_stuck(self, thought_history: List[Thought],
                 new_thought: Thought) -> bool:
        """
        Detect if cogitation is stuck in a loop.

        Stuck = generating similar thoughts repeatedly.
        """
        if len(thought_history) < 3:
            return False

        # Get signatures of recent thoughts
        recent_signatures = [
            self._get_thought_signature(t)
            for t in thought_history[-3:]
        ]

        new_signature = self._get_thought_signature(new_thought)

        # Check if new thought very similar to recent thoughts
        similarity_count = sum(
            1 for sig in recent_signatures
            if self._signature_similarity(sig, new_signature) > 0.8
        )

        return similarity_count >= 2  # 2 or more similar = stuck

    def _get_thought_signature(self, thought: Thought) -> str:
        """Get signature for similarity comparison."""
        # Use first 50 chars as signature
        return thought.content[:50].lower()

    def _signature_similarity(self, sig1: str, sig2: str) -> float:
        """Compute similarity between thought signatures."""
        # Simple Jaccard similarity
        words1 = set(sig1.split())
        words2 = set(sig2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _reframe_question(self, original_question: str,
                         current_understanding: Optional[str],
                         contradictions: List[str],
                         unresolved: List[str]) -> str:
        """
        Reframe question from different perspective.

        Michaud: When stuck, "ask different question".
        """
        prompt = f"""Original question: {original_question}

Current understanding: {current_understanding}

Contradictions encountered: {contradictions}

Unresolved aspects: {unresolved}

The current approach seems stuck. Reframe this question from a different
perspective or at a different level of abstraction. What's a different
way to think about this?

Reframed question:"""

        reframed = self.language_model.generate(
            prompt=prompt,
            max_length=200,
            temperature=0.9  # High creativity for reframing
        )

        return reframed.strip()

    def _evaluate_thought_coherence(self, thought_content: str,
                                   current_understanding: Optional[str]) -> float:
        """
        Evaluate how well new thought fits with existing understanding.
        """
        if current_understanding is None:
            return 0.5  # Neutral - no existing understanding to compare

        # Simple heuristic: term overlap
        thought_terms = set(thought_content.lower().split())
        understanding_terms = set(current_understanding.lower().split())

        overlap = len(thought_terms & understanding_terms)
        total = len(thought_terms | understanding_terms)

        coherence = overlap / total if total > 0 else 0.5

        return coherence


# Example usage:
if __name__ == '__main__':
    config = {
        'language_model': {
            'model_name': 'qwen2.5-0.5b',
            'device': 'cuda'
        }
    }

    plugin = CogitationPlugin(config)

    # Test question
    question = "What is the relationship between consciousness and iterative refinement?"

    # Initialize
    state = plugin.init_state(question)

    # Iterate until satisfied
    energy_history = []
    for i in range(20):
        state = plugin.step(state)
        energy = plugin.energy(state)
        energy_history.append(energy)

        print(f"\n--- Iteration {i+1} ---")
        print(f"Mode: {state.data['mode'].value}")
        print(f"Energy: {energy:.2f}")
        print(f"Coherence: {state.data['coherence']:.2f}")
        print(f"Latest thought: {state.data['thought_history'][-1].content[:100]}...")

        if plugin.should_halt(energy_history):
            print("\n=== Cogitation complete ===")
            print(f"Final understanding:\n{state.data['current_understanding']}")
            break
