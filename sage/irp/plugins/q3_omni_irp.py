#!/usr/bin/env python3
"""
Q3-Omni-30B IRP Plugin for SAGE Conversation Manager

Qwen3-Omni-30B: Omni-modal 30B parameter model with vision, audio, and text.
For conversation, we use text-only mode.

Model: /model-zoo/sage/omni-modal/qwen3-omni-30b
Context: 65536 tokens
Architecture: MoE (Mixture of Experts)
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from typing import List, Dict, Optional


class Q3OmniIRP:
    """
    Q3-Omni-30B IRP Plugin

    Compatible with SAGEConversationManager.
    Implements generate_response() for multi-turn conversations.
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b",
        device_map: str = "auto",
        max_memory: Optional[Dict[int, str]] = None
    ):
        """
        Initialize Q3-Omni model.

        Args:
            model_path: Path to model
            device_map: Device mapping strategy
            max_memory: Max memory per device
        """
        self.model_path = model_path
        self.device_map = device_map
        self.max_memory = max_memory or {0: "110GB"}

        print(f"Loading Q3-Omni-30B from {model_path}...")

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            device_map=device_map,
            max_memory=self.max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

        self.model.eval()
        print(f"✅ Q3-Omni-30B loaded successfully")

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
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.processor(
            text=formatted_prompt,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate with Q3-Omni's special parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                thinker_return_dict_in_generate=True,  # Q3-Omni specific
            )

        # Decode (Q3-Omni returns dict with 'sequences')
        if hasattr(outputs, 'sequences'):
            generated_ids = outputs.sequences
        else:
            generated_ids = outputs

        # Get only the new tokens (skip input)
        new_tokens = generated_ids[0][inputs['input_ids'].shape[1]:]

        # Decode response
        response = self.processor.batch_decode(
            [new_tokens],
            skip_special_tokens=True
        )[0]

        return response.strip()


# For backward compatibility with standalone usage
if __name__ == "__main__":
    print("Q3-Omni IRP Plugin")
    print("=" * 80)
    print()
    print("This plugin is designed for use with SAGEConversationManager.")
    print()
    print("Example usage:")
    print()
    print("  from sage.conversation.sage_conversation_manager import SAGEConversationManager")
    print("  from sage.irp.plugins.q3_omni_irp import Q3OmniIRP")
    print()
    print("  plugin = Q3OmniIRP()")
    print("  manager = SAGEConversationManager(plugin)")
    print()
    print("  response = manager.chat('Write a story about a dragon')")
    print("  response = manager.chat('What color was the dragon?')")
