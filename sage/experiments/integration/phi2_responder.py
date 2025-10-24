#!/usr/bin/env python3
"""
Phi-2 LLM Responder for SAGE
Context-aware response generation using Phi-2 model on Jetson.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional

class Phi2Responder:
    """
    Phi-2 LLM for generating context-aware responses.

    Optimized for Jetson:
    - Loads quantized model (INT8) if available
    - Uses Flash Attention if supported
    - Streaming generation for lower latency perception
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",  # Smaller, faster model
        device: str = "cpu",
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ):
        print(f"Loading {model_name}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Qwen 0.5B - small enough to run on CPU efficiently
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.device = device
        self.model = self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Model loaded on {device}")

    def generate_response(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response with conversation context.

        Args:
            user_text: Current user input
            conversation_history: List of (speaker, text) tuples
            system_prompt: Optional system instructions

        Returns:
            Generated response text
        """
        # Build prompt with context
        prompt_parts = []

        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        # Add conversation history (last 5 turns for context)
        if conversation_history:
            for speaker, text in conversation_history[-5:]:
                prompt_parts.append(f"{speaker}: {text}\n")

        # Add current input
        prompt_parts.append(f"User: {user_text}\nAssistant:")

        prompt = "".join(prompt_parts)

        # Generate
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        return response
