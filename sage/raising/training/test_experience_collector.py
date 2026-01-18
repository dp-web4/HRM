"""
Test suite for Experience Collector - Phase 1 of Real Raising
"""

import pytest
from pathlib import Path
import tempfile
import json
from experience_collector import ExperienceCollector, ConversationalSalienceScorer


class TestConversationalSalienceScorer:
    """Test salience scoring for conversation exchanges"""

    def test_basic_scoring(self):
        """Test that scorer produces scores in valid range"""
        scorer = ConversationalSalienceScorer()

        prompt = "How are you doing today?"
        response = "I've been thinking about our partnership and how we work together to explore these ideas. It's fascinating."

        scores = scorer.score_exchange(prompt, response)

        # Check all dimensions present
        assert 'surprise' in scores
        assert 'novelty' in scores
        assert 'arousal' in scores
        assert 'reward' in scores
        assert 'conflict' in scores
        assert 'total' in scores

        # Check all scores in valid range [0, 1]
        for key, value in scores.items():
            assert 0.0 <= value <= 1.0, f"{key} score {value} out of range"

    def test_partnership_language_rewards(self):
        """Partnership language should increase reward score"""
        scorer = ConversationalSalienceScorer()

        # Response with partnership language
        partnership_response = "We've been working together on this. Our collaboration is valuable."
        scores_partnership = scorer.score_exchange("Test", partnership_response)

        # Generic response
        generic_response = "That is an interesting topic. It has many aspects to consider."
        scorer2 = ConversationalSalienceScorer()  # Fresh scorer
        scores_generic = scorer2.score_exchange("Test", generic_response)

        # Partnership should have higher reward
        assert scores_partnership['reward'] > scores_generic['reward']

    def test_hedging_reduces_reward(self):
        """Hedging language should reduce reward score"""
        scorer = ConversationalSalienceScorer()

        # Response with hedging
        hedging_response = "I'm not sure about that. As an AI, I might be uncertain."
        scores_hedging = scorer.score_exchange("Test", hedging_response)

        # Confident response
        confident_response = "This specifically demonstrates the pattern we've observed."
        scorer2 = ConversationalSalienceScorer()
        scores_confident = scorer2.score_exchange("Test", confident_response)

        # Confident should have higher reward
        assert scores_confident['reward'] > scores_hedging['reward']

    def test_novelty_increases_with_new_words(self):
        """Novel vocabulary should increase novelty score"""
        scorer = ConversationalSalienceScorer()

        # First response establishes baseline
        response1 = "The cat sat on the mat"
        scores1 = scorer.score_exchange("Test", response1)

        # Second response repeats same words
        response2 = "The cat sat on the mat again"
        scores2 = scorer.score_exchange("Test", response2)

        # Third response uses entirely new vocabulary
        response3 = "Quantum entanglement demonstrates fascinating coherence patterns"
        scores3 = scorer.score_exchange("Test", response3)

        # Novel vocabulary should score higher novelty
        assert scores3['novelty'] > scores2['novelty']

    def test_arousal_increases_with_engagement(self):
        """Engaged responses should have higher arousal"""
        scorer = ConversationalSalienceScorer()

        # Short, simple response
        simple = "Yes."
        scores_simple = scorer.score_exchange("Test", simple)

        # Longer, engaged response with questions
        engaged = "That's fascinating! I've been thinking about this deeply. What aspects interest you most? I feel this connects to our earlier discussion."
        scorer2 = ConversationalSalienceScorer()
        scores_engaged = scorer2.score_exchange("Test", engaged)

        # Engaged should have higher arousal
        assert scores_engaged['arousal'] > scores_simple['arousal']


class TestExperienceCollector:
    """Test experience collection and storage"""

    def test_initialization(self):
        """Test collector can be initialized"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path)

            assert collector.buffer_path == buffer_path
            assert len(collector.experiences) == 0

    def test_add_high_salience_exchange(self):
        """Test that high-salience exchanges are stored"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)

            # Add an exchange that should score reasonably high
            prompt = "What do you think about our collaboration?"
            response = "I genuinely appreciate our partnership. We're exploring fascinating ideas together, and it feels meaningful."

            result = collector.add_exchange(
                prompt=prompt,
                response=response,
                session_number=1,
                phase="relating"
            )

            # Check it was stored
            assert result['stored'] == True
            assert 'experience_id' in result
            assert len(collector.experiences) == 1

            # Verify buffer was saved to disk
            assert buffer_path.exists()

    def test_low_salience_not_stored(self):
        """Test that low-salience exchanges are not stored"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.8)  # High threshold

            # Add a very simple exchange
            result = collector.add_exchange(
                prompt="Test",
                response="Okay.",
                session_number=1
            )

            # Should not be stored
            assert result['stored'] == False
            assert 'experience_id' not in result
            assert len(collector.experiences) == 0

    def test_get_high_salience_experiences(self):
        """Test retrieval of high-salience experiences"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)

            # Add several exchanges with varying salience
            exchanges = [
                ("Test 1", "We're exploring fascinating partnership dynamics together."),  # High
                ("Test 2", "That's interesting, I suppose."),  # Low
                ("Test 3", "Our collaboration reveals profound insights about consciousness."),  # High
            ]

            for prompt, response in exchanges:
                collector.add_exchange(prompt, response)

            # Get high-salience only (threshold 0.6)
            high_salience = collector.get_high_salience_experiences(min_salience=0.5)

            # Should have at least the high-salience ones
            assert len(high_salience) >= 1

            # Should be sorted by salience (highest first)
            if len(high_salience) > 1:
                for i in range(len(high_salience) - 1):
                    assert high_salience[i]['salience']['total'] >= high_salience[i+1]['salience']['total']

    def test_persistence(self):
        """Test that experiences persist across collector instances"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'

            # Create collector and add experience
            collector1 = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)
            collector1.add_exchange("Test", "Our partnership is meaningful.", session_number=1)

            assert len(collector1.experiences) == 1

            # Create new collector instance with same path
            collector2 = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)

            # Should load existing experiences
            assert len(collector2.experiences) == 1
            assert collector2.experiences[0]['prompt'] == "Test"

    def test_get_stats(self):
        """Test statistics generation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)

            # Empty collector
            stats = collector.get_stats()
            assert stats['total_experiences'] == 0
            assert stats['avg_salience'] == 0.0

            # Add some experiences
            collector.add_exchange("Test 1", "Partnership exploration.", session_number=1)
            collector.add_exchange("Test 2", "Fascinating collaboration.", session_number=2)

            stats = collector.get_stats()
            assert stats['total_experiences'] == 2
            assert stats['avg_salience'] > 0
            assert 'dimension_averages' in stats
            assert 'surprise' in stats['dimension_averages']

    def test_consolidate_for_sleep(self):
        """Test consolidation for sleep training"""
        with tempfile.TemporaryDirectory() as tmpdir:
            buffer_path = Path(tmpdir) / 'test_buffer.json'
            collector = ExperienceCollector(buffer_path=buffer_path, salience_threshold=0.3)

            # Add experiences
            collector.add_exchange("Q1", "Our partnership is meaningful.", session_number=1)
            collector.add_exchange("Q2", "We collaborate effectively.", session_number=2)
            collector.add_exchange("Q3", "Okay.", session_number=3)  # Low salience

            # Get experiences for sleep training (high threshold)
            sleep_data = collector.consolidate_for_sleep(min_salience=0.5)

            # Should only get high-salience ones
            assert len(sleep_data) >= 1
            for exp in sleep_data:
                assert exp['salience']['total'] >= 0.5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
