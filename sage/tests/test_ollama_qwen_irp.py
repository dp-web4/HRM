#!/usr/bin/env python3
"""
Tests for Ollama Qwen IRP Plugin

Tests the Ollama-based LLM backend for SAGE conversation.
"""

import unittest
from sage.irp.plugins.ollama_qwen_irp import OllamaQwenIRP


class TestOllamaQwenIRP(unittest.TestCase):
    """Test Ollama Qwen IRP plugin"""

    def setUp(self):
        """Initialize plugin for tests"""
        self.plugin = OllamaQwenIRP(
            model_name="qwen2.5:7b",
            keep_alive="5m"
        )

    def test_initialization(self):
        """Test plugin initializes correctly"""
        self.assertEqual(self.plugin.model_name, "qwen2.5:7b")
        self.assertEqual(self.plugin.base_url, "http://localhost:11434")
        self.assertEqual(self.plugin.api_url, "http://localhost:11434/api/generate")

    def test_format_messages_single(self):
        """Test message formatting for single user message"""
        messages = [{"role": "user", "content": "Hello"}]
        prompt = self.plugin._format_messages_to_prompt(messages)

        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("Hello", prompt)
        self.assertIn("<|im_end|>", prompt)
        self.assertIn("<|im_start|>assistant", prompt)

    def test_format_messages_with_system(self):
        """Test message formatting with system message"""
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        prompt = self.plugin._format_messages_to_prompt(messages)

        self.assertIn("<|im_start|>system", prompt)
        self.assertIn("You are helpful", prompt)
        self.assertIn("<|im_start|>user", prompt)
        self.assertIn("Hello", prompt)

    def test_format_messages_multiturn(self):
        """Test message formatting for multi-turn conversation"""
        messages = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "Why?"}
        ]
        prompt = self.plugin._format_messages_to_prompt(messages)

        # Check all messages included
        self.assertIn("What is 2+2?", prompt)
        self.assertIn("2+2 equals 4.", prompt)
        self.assertIn("Why?", prompt)

        # Check proper ordering (user, assistant, user, assistant_prompt)
        user1_idx = prompt.index("What is 2+2?")
        asst1_idx = prompt.index("2+2 equals 4.")
        user2_idx = prompt.index("Why?")

        self.assertLess(user1_idx, asst1_idx)
        self.assertLess(asst1_idx, user2_idx)

    def test_generate_response_basic(self):
        """Test basic response generation"""
        messages = [{"role": "user", "content": "Say 'test' and nothing else"}]

        response = self.plugin.generate_response(
            messages,
            max_new_tokens=10,
            temperature=0.1  # Low temp for more predictable output
        )

        # Should get some response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_response_conversation(self):
        """Test multi-turn conversation"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 1+1?"}
        ]

        response1 = self.plugin.generate_response(messages, max_new_tokens=50)
        self.assertIsInstance(response1, str)
        self.assertGreater(len(response1), 0)

        # Continue conversation
        messages.append({"role": "assistant", "content": response1})
        messages.append({"role": "user", "content": "Double that number."})

        response2 = self.plugin.generate_response(messages, max_new_tokens=50)
        self.assertIsInstance(response2, str)
        self.assertGreater(len(response2), 0)

    def test_preload_model(self):
        """Test model preloading"""
        # This will either succeed or fail gracefully
        result = self.plugin.preload_model()
        self.assertIsInstance(result, bool)

    def test_get_model_info(self):
        """Test getting model info"""
        # Preload first to ensure model is loaded
        self.plugin.preload_model()

        # Get model info
        info = self.plugin.get_model_info()

        # Should get info or None (depending on whether model is loaded)
        if info is not None:
            self.assertIsInstance(info, dict)
            # If loaded, should have name
            if "name" in info:
                self.assertEqual(info["name"], "qwen2.5:7b")


if __name__ == "__main__":
    print("Running Ollama Qwen IRP Plugin Tests")
    print("=" * 80)
    print()
    print("NOTE: These tests require Ollama to be running with qwen2.5:7b model.")
    print("      Install: curl -fsSL https://ollama.com/install.sh | sh")
    print("      Pull model: ollama pull qwen2.5:7b")
    print()
    print("=" * 80)
    print()

    unittest.main(verbosity=2)
