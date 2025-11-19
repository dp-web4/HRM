"""
Test suite for LLM IRP Plugin

Tests:
- IRP protocol compliance (init_state, step, energy, halt)
- Temperature annealing
- Energy convergence
- Conversation memory
- SNARC integration
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from sage.irp.plugins.llm_impl import LLMIRPPlugin, ConversationalLLM
from sage.irp.plugins.llm_snarc_integration import (
    DialogueSNARC, ConversationalMemory, DialogueExchange
)


class TestLLMIRPProtocol:
    """Test IRP protocol compliance."""

    def test_init_state(self):
        """Test state initialization."""
        llm = LLMIRPPlugin(model_path="Qwen/Qwen2.5-0.5B-Instruct")
        state = llm.init_state("What is knowledge?")

        assert 'question' in state
        assert 'prompt' in state
        assert 'iteration' in state
        assert 'temperature' in state
        assert 'best_response' in state
        assert 'best_energy' in state

        assert state['iteration'] == 0
        assert state['temperature'] == llm.initial_temperature

    def test_init_state_with_context(self):
        """Test state initialization with conversation context."""
        llm = LLMIRPPlugin(model_path="Qwen/Qwen2.5-0.5B-Instruct")
        context = "Previous Q: What is understanding?\nPrevious A: Understanding means comprehension."
        state = llm.init_state("What is knowledge?", context=context)

        assert context in state['prompt']
        assert "What is knowledge?" in state['prompt']

    def test_temperature_annealing(self):
        """Test temperature reduction across iterations."""
        llm = LLMIRPPlugin(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            initial_temperature=0.7,
            temp_reduction=0.1,
            min_temperature=0.5
        )

        state = llm.init_state("Test question")

        # Check temperature decreases
        temps = [state['temperature']]
        for _ in range(3):
            state = llm.step(state)
            temps.append(state['temperature'])

        # Should decrease
        assert temps[1] < temps[0]
        assert temps[2] < temps[1]

        # Should not go below minimum
        assert all(t >= llm.min_temperature for t in temps)

    def test_energy_calculation(self):
        """Test energy metric calculation."""
        llm = LLMIRPPlugin(model_path="Qwen/Qwen2.5-0.5B-Instruct")
        state = llm.init_state("Test question")
        state['temperature'] = 0.7

        # Test with different responses
        short_response = "Yes."
        long_response = "This is a much longer response with proper punctuation and capitalization that should have lower energy."

        energy_short = llm.energy(state, short_response)
        energy_long = llm.energy(state, long_response)

        # Both should be valid energies
        assert 0.0 <= energy_short <= 1.0
        assert 0.0 <= energy_long <= 1.0

    def test_halt_conditions(self):
        """Test halt decision logic."""
        llm = LLMIRPPlugin(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            min_temperature=0.5
        )

        # Test energy convergence
        state = llm.init_state("Test")
        state['best_energy'] = 0.05  # Below 0.1 threshold
        assert llm.halt(state) == True

        # Test temperature minimum
        state['best_energy'] = 0.5
        state['temperature'] = 0.5  # At minimum
        assert llm.halt(state) == True

        # Test energy plateau
        state['temperature'] = 0.7
        state['energies'] = [0.45, 0.46, 0.47]  # Plateau
        assert llm.halt(state) == True

        # Test continue condition
        state['energies'] = [0.8, 0.6, 0.4]  # Improving
        assert llm.halt(state) == False


class TestConversationalLLM:
    """Test high-level conversational interface."""

    def test_single_exchange(self):
        """Test single question-answer exchange."""
        conv = ConversationalLLM(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            irp_iterations=3
        )

        response, info = conv.respond("What is 2+2?", use_irp=True)

        assert isinstance(response, str)
        assert len(response) > 0
        assert info is not None
        assert 'iterations' in info
        assert 'final_energy' in info

    def test_conversation_history(self):
        """Test multi-turn conversation with history."""
        conv = ConversationalLLM(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            max_history=3
        )

        # First exchange
        conv.respond("What is knowledge?")
        assert len(conv.history) == 1

        # Second exchange
        conv.respond("Can you give an example?")
        assert len(conv.history) == 2

        # Check history structure
        history = conv.get_history()
        assert len(history) == 2
        assert all(isinstance(ex, tuple) and len(ex) == 2 for ex in history)

    def test_history_limit(self):
        """Test conversation history maximum length."""
        conv = ConversationalLLM(
            model_path="Qwen/Qwen2.5-0.5B-Instruct",
            max_history=2
        )

        # Add 5 exchanges
        for i in range(5):
            conv.respond(f"Question {i}")

        # Should only keep max_history in context
        assert len(conv.history) == 5  # Full history stored
        # (Context limiting happens in respond(), not visible in history)

    def test_clear_history(self):
        """Test clearing conversation history."""
        conv = ConversationalLLM(model_path="Qwen/Qwen2.5-0.5B-Instruct")

        conv.respond("Question 1")
        conv.respond("Question 2")
        assert len(conv.history) > 0

        conv.clear_history()
        assert len(conv.history) == 0


class TestSNARCIntegration:
    """Test SNARC salience scoring."""

    def test_score_exchange(self):
        """Test 5D salience scoring."""
        snarc = DialogueSNARC()
        exchange = DialogueExchange(
            question="What is knowledge?",
            answer="Knowledge is information acquired through learning or experience."
        )

        scores = snarc.score_exchange(exchange)

        # Check all dimensions present
        assert 'surprise' in scores
        assert 'novelty' in scores
        assert 'arousal' in scores
        assert 'reward' in scores
        assert 'conflict' in scores
        assert 'total_salience' in scores

        # Check valid ranges
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0

    def test_meta_cognitive_detection(self):
        """Test detection of meta-cognitive exchanges."""
        snarc = DialogueSNARC()

        # Meta-cognitive question
        meta_exchange = DialogueExchange(
            question="Are you aware of this conversation?",
            answer="I process each exchange but don't maintain continuous awareness."
        )

        # Simple factual question
        factual_exchange = DialogueExchange(
            question="What is 2+2?",
            answer="4"
        )

        meta_scores = snarc.score_exchange(meta_exchange)
        factual_scores = snarc.score_exchange(factual_exchange)

        # Meta-cognitive should score higher on conflict
        assert meta_scores['conflict'] > factual_scores['conflict']

    def test_salience_threshold(self):
        """Test salience threshold filtering."""
        snarc = DialogueSNARC()

        # High salience (philosophical)
        high_exchange = DialogueExchange(
            question="What can you know with certainty, and what must remain uncertain?",
            answer="Absolute certainty is difficult in most domains. Mathematical truths can be certain, but empirical knowledge involves uncertainty."
        )

        # Low salience (simple fact)
        low_exchange = DialogueExchange(
            question="What color is the sky?",
            answer="Blue."
        )

        high_salient, high_scores = snarc.is_salient(high_exchange, threshold=0.15)
        low_salient, low_scores = snarc.is_salient(low_exchange, threshold=0.15)

        # Philosophical should be more salient
        assert high_scores['total_salience'] > low_scores['total_salience']


class TestConversationalMemory:
    """Test conversation memory system."""

    def test_record_exchange(self):
        """Test recording exchanges."""
        memory = ConversationalMemory(salience_threshold=0.15)

        is_salient, scores = memory.record_exchange(
            "What is knowledge?",
            "Knowledge is information acquired through learning."
        )

        assert len(memory.all_exchanges) == 1
        assert isinstance(scores, dict)
        assert 'total_salience' in scores

    def test_salient_filtering(self):
        """Test filtering salient exchanges."""
        memory = ConversationalMemory(salience_threshold=0.15)

        # Add mix of exchanges
        exchanges = [
            ("What is consciousness?", "Consciousness is the state of being aware of one's surroundings and experiences."),
            ("What is 2+2?", "4"),
            ("Are you aware of this conversation?", "I process information but don't have continuous awareness."),
            ("Hello", "Hi there!"),
        ]

        for q, a in exchanges:
            memory.record_exchange(q, a)

        # Should filter some exchanges
        stats = memory.get_statistics()
        assert stats['total_exchanges'] == len(exchanges)
        assert stats['salient_exchanges'] < stats['total_exchanges']
        assert 0 <= stats['capture_rate'] <= 100

    def test_training_data_format(self):
        """Test training data extraction."""
        memory = ConversationalMemory(salience_threshold=0.1)

        memory.record_exchange(
            "What is understanding?",
            "Understanding is comprehension of meaning and significance."
        )

        training_data = memory.get_salient_for_training()

        # Check format
        assert isinstance(training_data, list)
        if training_data:  # If any were salient
            assert all('question' in ex and 'answer' in ex for ex in training_data)


def test_integration_workflow():
    """Test complete conversational learning workflow."""
    # Initialize components
    conv = ConversationalLLM(
        model_path="Qwen/Qwen2.5-0.5B-Instruct",
        irp_iterations=3
    )
    memory = ConversationalMemory(salience_threshold=0.15)

    # Have conversation
    questions = [
        "What is knowledge?",
        "How does understanding differ from knowledge?",
        "Are you aware of this conversation?"
    ]

    for question in questions:
        response, irp_info = conv.respond(question, use_irp=True)
        is_salient, scores = memory.record_exchange(question, response, irp_info)

    # Check results
    stats = memory.get_statistics()
    assert stats['total_exchanges'] == len(questions)

    training_data = memory.get_salient_for_training()
    # Should have some salient exchanges (likely the meta-cognitive one)
    # (We don't assert specific count as it depends on actual model responses)

    print(f"\n✓ Integration test complete:")
    print(f"  Exchanges: {stats['total_exchanges']}")
    print(f"  Salient: {stats['salient_exchanges']} ({stats['capture_rate']:.1f}%)")
    print(f"  Training data: {len(training_data)} examples")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
