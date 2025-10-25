#!/usr/bin/env python3
"""
Breath-Based LLM Responder for SAGE

Models biological speech production:
- Generate in "breath-sized" chunks (~30-40 tokens)
- Stream output immediately (don't wait for full thought)
- Continue until thought feels complete
- Interruptible between breaths

Key insight: Humans don't precompute entire responses.
We think and speak in breath-sized segments, each informing the next.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Optional, Callable
import time


class BreathBasedResponder:
    """
    LLM responder that generates in breath-sized chunks.

    Mirrors biological speech production:
    - Breath = 30-40 tokens (~3-5 seconds of speech)
    - Immediate streaming (start speaking before thought complete)
    - Incremental context building (each breath informs next)
    - Natural stopping detection (thought completeness)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = None,
        breath_size: int = 40,  # Tokens per "breath"
        max_breaths: int = 5,   # Safety limit (total ~200 tokens)
        temperature: float = 0.7
    ):
        print(f"Loading {model_name} for breath-based generation...")

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        print(f"Using device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        self.model = self.model.to(self.device)
        self.breath_size = breath_size
        self.max_breaths = max_breaths
        self.temperature = temperature

        print(f"Model loaded: breath_size={breath_size} tokens, max_breaths={max_breaths}")

    def generate_response_streaming(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None,
        on_breath: Optional[Callable[[str, int, bool], None]] = None
    ) -> dict:
        """
        Generate response in breath-sized chunks with streaming callback.

        Args:
            user_text: Current user input
            conversation_history: List of (speaker, text) tuples
            system_prompt: Optional system instructions
            on_breath: Callback(text, breath_num, is_final) called after each breath

        Returns:
            dict with:
                - full_response: Complete accumulated response
                - breaths: List of individual breath chunks
                - breath_count: Number of breaths taken
                - total_time: Total generation time
                - thought_complete: Whether thought reached natural completion
        """
        start_time = time.time()

        # Build base prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        if conversation_history:
            for speaker, text in conversation_history:
                prompt_parts.append(f"{speaker}: {text}\n")

        prompt_parts.append(f"User: {user_text}\nAssistant:")

        # Initialize breathing loop
        accumulated_response = ""
        breaths = []
        breath_count = 0

        print(f"\n    [BREATH] Starting breath-based generation...")

        for breath_num in range(self.max_breaths):
            breath_count += 1

            # Build prompt with accumulated context
            current_prompt = "".join(prompt_parts) + accumulated_response

            # Generate one breath
            breath_start = time.time()
            breath_text = self._generate_one_breath(current_prompt)
            breath_time = time.time() - breath_start

            if not breath_text or len(breath_text.strip()) == 0:
                # Empty breath = thought complete
                print(f"    [BREATH {breath_num+1}] Empty breath, thought complete")
                break

            # Accumulate this breath
            accumulated_response += breath_text
            breaths.append(breath_text)

            print(f"    [BREATH {breath_num+1}] Generated {len(breath_text)} chars in {breath_time:.2f}s: '{breath_text.strip()[:60]}...'")

            # Callback for streaming (speak immediately!)
            if on_breath:
                is_final = self._is_thought_complete(breath_text, accumulated_response)
                on_breath(breath_text, breath_num + 1, is_final)

            # Check if thought is complete
            if self._is_thought_complete(breath_text, accumulated_response):
                print(f"    [BREATH {breath_num+1}] Thought complete, stopping")
                break

        total_time = time.time() - start_time
        thought_complete = breath_count < self.max_breaths

        # Extract just the assistant's response (remove prompt echo)
        if accumulated_response:
            # Clean up any prompt artifacts
            if "Assistant:" in accumulated_response:
                accumulated_response = accumulated_response.split("Assistant:")[-1].strip()

        print(f"    [BREATH] Complete: {breath_count} breaths, {total_time:.2f}s total")

        return {
            'full_response': accumulated_response.strip(),
            'breaths': breaths,
            'breath_count': breath_count,
            'total_time': total_time,
            'thought_complete': thought_complete
        }

    def _generate_one_breath(self, prompt: str) -> str:
        """Generate one breath-sized chunk of text."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.breath_size,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                # Stop early if we hit sentence-ending punctuation
                stopping_criteria=None  # TODO: Add punctuation stopping
            )

        # Decode only the new tokens (not the prompt)
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        breath_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Debug: Show token count vs character count
        token_count = len(new_tokens)
        char_count = len(breath_text)
        # print(f"        [DEBUG] Generated {token_count} tokens = {char_count} chars")

        return breath_text

    def _is_thought_complete(self, latest_breath: str, full_response: str) -> bool:
        """
        Heuristic to detect thought completeness.

        Biological cues:
        - Ends with strong punctuation (. ! ?)
        - No trailing conjunctions (and, but, so, because)
        - Breath feels "final" (not trailing off)

        Future: Could use small classifier trained on complete vs incomplete thoughts
        """
        breath_stripped = latest_breath.strip()

        if not breath_stripped:
            return True  # Empty = done

        # Strong ending punctuation
        if breath_stripped.endswith(('.', '!', '?')):
            # Check for trailing conjunctions (indicates continuation)
            last_words = breath_stripped.lower().split()[-3:]  # Last few words

            trailing_conjunctions = {'and', 'but', 'so', 'because', 'however', 'although'}

            # If ends with punctuation but has conjunction before it, likely continuing
            for word in last_words:
                cleaned = word.rstrip('.,!?')
                if cleaned in trailing_conjunctions:
                    return False  # More to come

            return True  # Clean ending

        # Weak endings suggest continuation
        if breath_stripped.endswith((',', ':', ';', '-')):
            return False

        # Very short breath might be incomplete
        if len(breath_stripped) < 20:
            return False

        # Default: continue for now (safety will stop at max_breaths)
        return False


# For backward compatibility with existing code
class Phi2Responder:
    """Wrapper that uses breath-based generation but provides old interface."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 device: str = None, max_new_tokens: int = 150, temperature: float = 0.7):
        # Calculate breaths needed for max_new_tokens
        breath_size = 40
        max_breaths = (max_new_tokens + breath_size - 1) // breath_size  # Round up

        self.responder = BreathBasedResponder(
            model_name=model_name,
            device=device,
            breath_size=breath_size,
            max_breaths=max_breaths,
            temperature=temperature
        )

    def generate_response(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate complete response (blocking, for compatibility)."""
        result = self.responder.generate_response_streaming(
            user_text,
            conversation_history,
            system_prompt
        )
        return result['full_response']
