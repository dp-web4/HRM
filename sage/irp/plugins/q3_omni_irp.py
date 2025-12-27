#!/usr/bin/env python3
"""
Q3-Omni-30B IRP Plugin for SAGE Conversation Manager

Qwen3-Omni-30B: Omni-modal 30B parameter model with vision, audio, and text.
For conversation, we use text-only mode with proper multimodal processing pipeline.

Model: model-zoo/sage/omni-modal/qwen3-omni-30b (FULL PRECISION)
Context: 65536 tokens
Architecture: MoE (Mixture of Experts)

CRITICAL: This plugin follows the EXACT pattern from test_qwen3_omni_simple_text.py
which successfully loads and generates in ~2 minutes.
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from typing import List, Dict, Optional


class Q3OmniIRP:
    """
    Q3-Omni-30B IRP Plugin

    Compatible with SAGEConversationManager.
    Implements generate_response() for multi-turn conversations.

    IMPORTANT: Uses the proven working pattern from test_qwen3_omni_simple_text.py:
    - Full precision model (qwen3-omni-30b, not INT8)
    - dtype="auto" (not torch.float16)
    - Nested content structure [{"type": "text", "text": "..."}]
    - process_mm_info() helper for multimodal pipeline
    - disable_talker() for text-only mode
    """

    def __init__(
        self,
        model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b",
        device_map: str = "auto",
    ):
        """
        Initialize Q3-Omni model using PROVEN working pattern.

        Args:
            model_path: Path to FULL PRECISION model (not INT8)
            device_map: Device mapping strategy
        """
        self.model_path = model_path
        self.device_map = device_map

        print(f"Loading Q3-Omni-30B from {model_path}...")
        print("(Using proven pattern from test_qwen3_omni_simple_text.py)")

        # Load model - EXACT parameters from working test
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            dtype="auto",  # NOT torch.float16! Let transformers detect
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # CRITICAL: Disable talker for text-only mode
        self.model.disable_talker()

        self.model.eval()
        print(f"✅ Q3-Omni-30B loaded successfully (text-only mode)")

        # Load processor
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print(f"✅ Processor loaded")

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

        IMPORTANT: Converts simple message format to Q3-Omni's nested format.

        Args:
            messages: Conversation history [{"role": "user/assistant/system", "content": "..."}]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling

        Returns:
            Generated response text
        """
        # Convert simple message format to Q3-Omni's nested format
        # From: {"role": "user", "content": "Hello"}
        # To: {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        conversation = []
        for msg in messages:
            conversation.append({
                "role": msg["role"],
                "content": [
                    {"type": "text", "text": msg["content"]}
                ]
            })

        # Apply chat template (EXACTLY as in working test)
        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # CRITICAL: Use process_mm_info helper for multimodal processing
        # Even for text-only, Q3-Omni needs this pipeline!
        audios, images, videos = process_mm_info(
            conversation,
            use_audio_in_video=False
        )

        # Process inputs with full multimodal pipeline
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False
        )

        inputs = inputs.to(self.model.device)

        # Generate (NO thinker_return_dict_in_generate - not needed!)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                return_audio=False,  # CRITICAL: text-only
                use_audio_in_video=False,  # CRITICAL: text-only
            )

        # Handle output - may be tuple (text, audio) even with return_audio=False
        if isinstance(outputs, tuple):
            generated_ids = outputs[0]  # First element is text tokens
        else:
            generated_ids = outputs

        # Decode response (EXACTLY as in working test)
        response = self.processor.batch_decode(
            generated_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return response[0].strip()


# For standalone usage
if __name__ == "__main__":
    print("Q3-Omni IRP Plugin")
    print("=" * 80)
    print()
    print("This plugin uses the PROVEN working pattern from:")
    print("  sage/tests/test_qwen3_omni_simple_text.py")
    print()
    print("Key differences from broken version:")
    print("  ✅ Full precision model (qwen3-omni-30b, not INT8)")
    print("  ✅ dtype='auto' (not torch.float16)")
    print("  ✅ Nested content structure")
    print("  ✅ process_mm_info() helper")
    print("  ✅ Full multimodal processing pipeline")
    print("  ✅ No thinker_return_dict_in_generate")
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
