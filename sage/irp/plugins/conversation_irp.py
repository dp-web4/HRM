"""
Conversation IRP Plugin

Implements iterative refinement for conversational responses using pattern matching.
Energy function: 1.0 - match_confidence
Convergence: Pattern found with high confidence OR all patterns exhausted

This bridges the standalone PatternResponseEngine into SAGE's IRP framework.
"""

import sys
from pathlib import Path

# Add sage root to path
_sage_root = Path(__file__).parent.parent.parent
if str(_sage_root) not in sys.path:
    sys.path.insert(0, str(_sage_root))

from irp.base import IRPPlugin, IRPState
from cognitive.pattern_responses import PatternResponseEngine
from typing import Dict, Any, List


class ConversationIRP(IRPPlugin):
    """
    IRP plugin for conversational response generation

    Treats pattern matching as iterative refinement:
    - Each iteration tries the next most-likely pattern
    - Energy decreases as better matches are found
    - Halts when good match found or patterns exhausted
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Conversation IRP

        Args:
            config: Configuration including:
                - entity_id: Unique identifier
                - max_iterations: Max pattern attempts (default: 12)
                - halt_eps: Confidence threshold (default: 0.3)
                - min_confidence: Minimum match confidence (default: 0.6)
        """
        # Set default max_iterations for pattern matching
        if 'max_iterations' not in config:
            config['max_iterations'] = 12  # Try up to 12 patterns

        if 'halt_eps' not in config:
            config['halt_eps'] = 0.05  # Stop when energy change < 0.05

        super().__init__(config)

        # Initialize pattern engine
        self.pattern_engine = PatternResponseEngine()
        self.min_confidence = config.get('min_confidence', 0.6)

        print(f"✓ ConversationIRP initialized (entity_id={self.entity_id})")
        print(f"  Patterns: {len(self.pattern_engine.patterns)}")
        print(f"  Min confidence: {self.min_confidence}")

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize conversational refinement state

        Args:
            x0: Transcribed text (string)
            task_ctx: Task context including ATP budget

        Returns:
            Initial IRPState for pattern matching
        """
        # Extract text from input
        if isinstance(x0, str):
            text = x0
        elif hasattr(x0, 'metadata') and 'text' in x0.metadata:
            text = x0.metadata['text']
        else:
            text = str(x0)

        return IRPState(
            x=text,
            step_idx=0,
            energy_val=1.0,  # Start with maximum energy
            meta={
                'transcription': text,
                'patterns_tried': [],
                'pattern_attempts': [],
                'best_match': None,
                'best_confidence': 0.0,
                'best_response': None,
                'atp_budget': task_ctx.get('atp_budget', 1000.0)
            }
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Execute one pattern matching iteration

        Tries patterns in order until match found or all exhausted.

        Args:
            state: Current refinement state
            noise_schedule: Unused (no noise in pattern matching)

        Returns:
            Updated state after trying next pattern
        """
        text = state.meta['transcription']
        patterns_tried = state.meta['patterns_tried']

        # Get all compiled patterns
        all_patterns = self.pattern_engine.compiled_patterns

        # Find next untried pattern
        for pattern_regex, responses in all_patterns:
            pattern_str = pattern_regex.pattern

            if pattern_str in patterns_tried:
                continue

            # Try this pattern
            match = pattern_regex.search(text)
            patterns_tried.append(pattern_str)

            if match:
                # Pattern matched!
                # Calculate confidence based on match strength
                # Simple heuristic: longer match = higher confidence
                match_length = len(match.group(0))
                text_length = len(text)
                confidence = min(0.95, 0.5 + (match_length / text_length) * 0.5)

                # Select response (with variety)
                pattern_count = state.meta.get('pattern_counts', {}).get(pattern_str, 0)
                if len(responses) > 1:
                    idx = pattern_count % len(responses)
                    response_template = responses[idx]
                else:
                    response_template = responses[0]

                # Format response with match groups
                try:
                    if match.groups():
                        response = response_template.format(*match.groups())
                    else:
                        response = response_template
                except (IndexError, KeyError):
                    response = response_template

                # Record attempt
                state.meta['pattern_attempts'].append({
                    'pattern': pattern_str,
                    'matched': True,
                    'confidence': confidence,
                    'response': response
                })

                # Update best match if better
                if confidence > state.meta['best_confidence']:
                    state.meta['best_match'] = pattern_str
                    state.meta['best_confidence'] = confidence
                    state.meta['best_response'] = response

                # Update pattern counts for variety
                if 'pattern_counts' not in state.meta:
                    state.meta['pattern_counts'] = {}
                state.meta['pattern_counts'][pattern_str] = pattern_count + 1

                break  # Found a match, done with this iteration

            else:
                # Pattern didn't match
                state.meta['pattern_attempts'].append({
                    'pattern': pattern_str,
                    'matched': False,
                    'confidence': 0.0,
                    'response': None
                })

                break  # Tried one pattern per iteration

        return state

    def energy(self, state: IRPState) -> float:
        """
        Compute energy (lower is better)

        Energy = 1.0 - best_confidence
        When confidence is high (good match), energy is low (convergence)

        Args:
            state: Current state

        Returns:
            Energy value (0.0 = perfect match, 1.0 = no match)
        """
        return 1.0 - state.meta['best_confidence']

    def halt(self, history: List[IRPState]) -> bool:
        """
        Determine if pattern matching should stop

        Halt conditions:
        1. Found good match (confidence > min_confidence)
        2. Tried all patterns
        3. Energy converged (no improvement)
        4. Max iterations reached

        Args:
            history: Refinement history

        Returns:
            True if should halt
        """
        if not history:
            return False

        current = history[-1]

        # 1. Good match found?
        if current.meta['best_confidence'] >= self.min_confidence:
            return True

        # 2. All patterns tried?
        all_patterns_count = len(self.pattern_engine.compiled_patterns)
        patterns_tried = len(current.meta['patterns_tried'])
        if patterns_tried >= all_patterns_count:
            return True

        # 3. Max iterations?
        max_iter = self.config.get('max_iterations', 12)
        if len(history) >= max_iter:
            return True

        # 4. Check energy convergence (default halt logic)
        if len(history) >= 3:
            K = self.config.get('halt_K', 3)
            eps = self.config.get('halt_eps', 0.05)

            if len(history) >= K + 1:
                recent_energies = [s.energy_val or self.energy(s) for s in history[-(K+1):]]
                slope = abs(recent_energies[-1] - recent_energies[0]) / len(recent_energies)

                if slope < eps:
                    return True

        return False

    def get_halt_reason(self, history: List[IRPState]) -> str:
        """Get reason for halting"""
        if not history:
            return "no_history"

        current = history[-1]

        if current.meta['best_confidence'] >= self.min_confidence:
            return "good_match"

        all_patterns_count = len(self.pattern_engine.compiled_patterns)
        if len(current.meta['patterns_tried']) >= all_patterns_count:
            return "all_patterns_tried"

        if len(history) >= self.config.get('max_iterations', 12):
            return "max_iterations"

        return super().get_halt_reason(history)

    def get_response(self, state: IRPState) -> str:
        """
        Extract best response from state

        Args:
            state: Final refinement state

        Returns:
            Response text (or None if no good match)
        """
        if state.meta['best_confidence'] >= self.min_confidence:
            return state.meta['best_response']
        return None

    def get_stats(self) -> Dict[str, Any]:
        """Get pattern matching statistics"""
        return self.pattern_engine.get_stats()


# Test the plugin
if __name__ == "__main__":
    print("="*60)
    print("Testing ConversationIRP Plugin")
    print("="*60)

    # Create plugin
    config = {
        'entity_id': 'conversation_test',
        'max_iterations': 12,
        'min_confidence': 0.6
    }

    plugin = ConversationIRP(config)

    # Test cases
    test_cases = [
        "Hello there!",
        "Can you hear me?",
        "What are you doing?",
        "Thanks for your help",
        "This is a complex question that won't match any pattern"
    ]

    for text in test_cases:
        print(f"\nInput: \"{text}\"")

        # Run refinement
        final_state, history = plugin.refine(text)

        # Results
        response = plugin.get_response(final_state)
        energy_traj = [s.energy_val for s in history]

        print(f"  Response: {response}")
        print(f"  Iterations: {len(history)}")
        print(f"  Confidence: {final_state.meta['best_confidence']:.3f}")
        print(f"  Energy: {energy_traj[0]:.3f} → {energy_traj[-1]:.3f}")
        print(f"  Halt reason: {plugin.get_halt_reason(history)}")

        # Trust metrics
        trust = plugin.compute_trust_metrics(history)
        print(f"  Trust: monotonicity={trust['monotonicity_ratio']:.3f}")

    # Final stats
    print(f"\n{'='*60}")
    print("Pattern Engine Statistics:")
    stats = plugin.get_stats()
    print(f"  Total queries: {stats['total_queries']}")
    print(f"  Matched: {stats['matched_queries']}")
    print(f"  Match rate: {stats['match_rate']:.1%}")
    print(f"  Avg processing: {stats['avg_processing_time_ms']:.2f}ms")
