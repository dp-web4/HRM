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
        device: str = None,
        max_new_tokens: int = 150,  # Balanced: full sentences but real-time response
        temperature: float = 0.7
    ):
        print(f"Loading {model_name}...")

        # Auto-detect device (prefer CUDA on Jetson with unified memory)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        # Qwen 0.5B - optimized for Jetson
        # On Jetson unified memory, GPU is faster than CPU for same memory cost
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model = self.model.to(self.device)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        print(f"Model loaded on {self.device} ({self.model.dtype})")

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
        # Build messages in chat format for Qwen's chat template
        messages = []

        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add conversation history
        # Convert (speaker, text) tuples to proper role format
        if conversation_history:
            for speaker, text in conversation_history:
                # Map speaker names to chat roles
                if speaker.lower() in ["user", "human"]:
                    role = "user"
                elif speaker.lower() in ["assistant", "sage", "ai"]:
                    role = "assistant"
                else:
                    # Default unknown speakers to user
                    role = "user"

                messages.append({"role": role, "content": text})

        # Add current user input
        messages.append({"role": "user", "content": user_text})

        # Use Qwen's chat template to format messages properly
        try:
            # Try using apply_chat_template (Qwen 2.5 supports this)
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except AttributeError:
            # Fallback to manual formatting if chat template not available
            print("    [WARNING] Chat template not available, using fallback formatting")
            prompt_parts = []
            for msg in messages:
                role = msg["role"].capitalize()
                prompt_parts.append(f"{role}: {msg['content']}\n")
            prompt_parts.append("Assistant:")
            formatted_prompt = "".join(prompt_parts)

        # Tokenize
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)

        print(f"    [LLM] Generating up to {self.max_new_tokens} tokens...")
        import time
        gen_start = time.time()

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        gen_time = time.time() - gen_start
        print(f"    [LLM] Generation complete in {gen_time:.2f}s")

        # Decode only the generated tokens (skip the prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        return response
