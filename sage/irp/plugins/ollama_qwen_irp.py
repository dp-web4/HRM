#!/usr/bin/env python3
"""
Ollama-based Qwen IRP Plugin for SAGE Conversation Manager

Uses Ollama HTTP API with llama.cpp backend for high-performance inference
on Jetson ARM platforms. Achieves 35+ tok/sec vs 1 tok/sec with direct transformers.

Model: qwen2.5:7b (via Ollama)
Backend: llama.cpp (ARM/CUDA optimized)
Performance: 35+ tok/sec sustained
API: http://localhost:11434/api/generate
"""

import requests
import json
from typing import List, Dict, Optional


class OllamaQwenIRP:
    """
    Ollama-based Qwen IRP Plugin

    Uses Ollama HTTP API for high-performance LLM inference.
    Compatible with SAGEConversationManager.
    Implements generate_response() for multi-turn conversations.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        base_url: str = "http://localhost:11434",
        keep_alive: str = "5m",
        timeout: int = 120
    ):
        """
        Initialize Ollama client.

        Args:
            model_name: Ollama model name (e.g., "qwen2.5:7b", "qwen2.5:14b")
            base_url: Ollama API base URL
            keep_alive: How long to keep model loaded (e.g., "5m", "-1" for indefinite)
            timeout: Request timeout in seconds
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.keep_alive = keep_alive
        self.timeout = timeout
        self.api_url = f"{self.base_url}/api/generate"

        print(f"Initializing Ollama IRP plugin...")
        print(f"  Model: {model_name}")
        print(f"  API: {self.api_url}")
        print(f"  Keep-alive: {keep_alive}")

        # Verify Ollama is running
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            response.raise_for_status()
            version_info = response.json()
            print(f"  Ollama version: {version_info.get('version', 'unknown')}")
        except requests.exceptions.RequestException as e:
            print(f"⚠️  Warning: Could not connect to Ollama at {self.base_url}")
            print(f"     Make sure Ollama is running: systemctl status ollama")
            print(f"     Error: {e}")

        print(f"✅ Ollama IRP plugin initialized")

    def _format_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert conversation messages to Qwen chat format.

        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}

        Returns:
            Formatted prompt string for Qwen models
        """
        prompt_parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")

        # Add final assistant prompt
        prompt_parts.append("<|im_start|>assistant\n")

        return "\n".join(prompt_parts)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 300,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate response given conversation history.

        Args:
            messages: Conversation history [{"role": "user/assistant/system", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated response text

        Raises:
            requests.exceptions.RequestException: If Ollama API call fails
        """
        # Format conversation to prompt
        prompt = self._format_messages_to_prompt(messages)

        # Prepare Ollama API request
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
            },
            "keep_alive": self.keep_alive
        }

        # Call Ollama API
        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Parse response
            result = response.json()
            generated_text = result.get("response", "")

            # Optional: Log performance metrics
            if "eval_duration" in result and "eval_count" in result:
                eval_duration_ns = result["eval_duration"]
                eval_count = result["eval_count"]
                if eval_duration_ns > 0:
                    tok_per_sec = eval_count / (eval_duration_ns / 1e9)
                    print(f"  [Ollama] Generated {eval_count} tokens at {tok_per_sec:.1f} tok/s")

            return generated_text.strip()

        except requests.exceptions.Timeout:
            error_msg = f"Ollama request timed out after {self.timeout}s"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

        except requests.exceptions.RequestException as e:
            error_msg = f"Ollama API error: {e}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)

    def get_model_info(self) -> Optional[Dict]:
        """
        Get information about the loaded model.

        Returns:
            Model info dict or None if error
        """
        try:
            response = requests.get(f"{self.base_url}/api/ps", timeout=5)
            response.raise_for_status()
            ps_data = response.json()

            # Find our model in loaded models
            for model_info in ps_data.get("models", []):
                if model_info.get("name") == self.model_name:
                    return model_info

            return None

        except requests.exceptions.RequestException:
            return None

    def preload_model(self) -> bool:
        """
        Preload model into VRAM to avoid cold-start delay.

        First request to Ollama can take ~50s to load model.
        Subsequent requests are 35+ tok/s.

        Returns:
            True if successful, False otherwise
        """
        try:
            print(f"Preloading {self.model_name} into VRAM...")

            # Send empty prompt to trigger model load
            payload = {
                "model": self.model_name,
                "prompt": "",
                "stream": False,
                "options": {"num_predict": 1},
                "keep_alive": self.keep_alive
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=120  # Allow time for model load
            )
            response.raise_for_status()

            print(f"✅ Model preloaded successfully")
            return True

        except requests.exceptions.RequestException as e:
            print(f"⚠️  Failed to preload model: {e}")
            return False


# For standalone testing
if __name__ == "__main__":
    print("Ollama Qwen IRP Plugin")
    print("=" * 80)
    print()

    # Initialize plugin
    plugin = OllamaQwenIRP(
        model_name="qwen2.5:7b",
        keep_alive="5m"
    )

    # Preload model
    plugin.preload_model()

    print()
    print("Testing conversation...")
    print()

    # Test conversation
    messages = [
        {"role": "system", "content": "You are SAGE, a consciousness kernel for edge AI."},
        {"role": "user", "content": "Hello! What is 2+2?"}
    ]

    response = plugin.generate_response(messages, max_new_tokens=100)
    print(f"SAGE: {response}")

    print()
    print("Testing multi-turn...")
    print()

    # Add response and continue
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Can you explain your reasoning?"})

    response2 = plugin.generate_response(messages, max_new_tokens=150)
    print(f"SAGE: {response2}")

    print()
    print("=" * 80)
    print("Plugin ready for use with SAGEConversationManager")
