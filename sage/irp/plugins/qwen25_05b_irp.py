#!/usr/bin/env python3
"""
Qwen2.5-0.5B IRP Plugin for SAGE Conversation Manager

Qwen2.5-0.5B-Instruct: Lightweight 0.5B parameter instruction-tuned model.
Fast inference, good for lightweight tasks and edge deployment.

Model: model-zoo/sage/epistemic-stances/qwen2.5-0.5b
Context: 32K tokens (131072 max position embeddings)
Architecture: Dense Transformer
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional


class Qwen25_05B_IRP:
    """
    Qwen2.5-0.5B-Instruct IRP Plugin

    Compatible with SAGEConversationManager.
    Implements generate_response() for multi-turn conversations.
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/epistemic-stances/qwen2.5-0.5b/epistemic-pragmatism",
        device_map: str = "auto"
    ):
        """
        Initialize Qwen2.5-0.5B model.

        Args:
            model_path: Path to model
            device_map: Device mapping strategy
        """
        self.model_path = model_path
        self.device_map = device_map

        print(f"Loading Qwen2.5-0.5B from {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.model.eval()
        print(f"✅ Qwen2.5-0.5B loaded successfully")

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
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated response text
        """
        # Apply chat template (handles conversation history formatting)
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False  # Already added by chat template
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Get only the new tokens (skip input)
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]

        # Decode response
        response = self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )

        return response.strip()


# For standalone usage
if __name__ == "__main__":
    print("Qwen2.5-0.5B IRP Plugin")
    print("=" * 80)
    print()
    print("This plugin is designed for use with SAGEConversationManager.")
    print()
    print("Example usage:")
    print()
    print("  from sage.conversation.sage_conversation_manager import SAGEConversationManager")
    print("  from sage.irp.plugins.qwen25_05b_irp import Qwen25_05B_IRP")
    print()
    print("  plugin = Qwen25_05B_IRP()")
    print("  manager = SAGEConversationManager(plugin)")
    print()
    print("  response = manager.chat('Write a story about a dragon')")
    print("  response = manager.chat('What color was the dragon?')")
