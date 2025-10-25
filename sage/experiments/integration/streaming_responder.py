#!/usr/bin/env python3
"""
Streaming Word-by-Word LLM Responder for SAGE

Mirrors biological speech production by streaming each word as it's generated.

Key insight: Tokens are already generated sequentially by the LLM.
We just need to:
1. Decode and emit each token/word as soon as it's ready
2. Allow large buffer (512 tokens) for complete thoughts
3. Stop naturally when thought is complete (not when buffer fills)

This is how YOU actually work - words emerge one at a time,
and you stop when the thought feels complete.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from typing import List, Optional, Callable
import time
import threading


class StreamingResponder:
    """
    LLM responder that streams words as they're generated.

    Uses transformers' TextIteratorStreamer to emit tokens in real-time,
    allowing speech synthesis to start immediately.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = None,
        max_new_tokens: int = 512,  # Large buffer for complete thoughts
        temperature: float = 0.7,
        words_per_chunk: int = 3  # Stream every N words (balance latency vs efficiency)
    ):
        print(f"Loading {model_name} for streaming generation...")

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
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.words_per_chunk = words_per_chunk

        print(f"Model loaded: max_tokens={max_new_tokens}, streaming={words_per_chunk} words/chunk")

    def generate_response_streaming(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[str, bool], None]] = None
    ) -> dict:
        """
        Generate response with word-by-word streaming.

        Args:
            user_text: Current user input
            conversation_history: List of (speaker, text) tuples
            system_prompt: Optional system instructions
            on_chunk: Callback(chunk_text, is_final) called for each word chunk

        Returns:
            dict with:
                - full_response: Complete accumulated response
                - chunks: List of word chunks
                - chunk_count: Number of chunks emitted
                - total_time: Total generation time
                - tokens_generated: Number of tokens produced
        """
        start_time = time.time()

        # Build prompt
        prompt_parts = []
        if system_prompt:
            prompt_parts.append(f"System: {system_prompt}\n")

        if conversation_history:
            for speaker, text in conversation_history:
                prompt_parts.append(f"{speaker}: {text}\n")

        prompt_parts.append(f"User: {user_text}\nAssistant:")
        prompt = "".join(prompt_parts)

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Create streamer that will emit tokens as they're generated
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,  # Don't echo the prompt
            skip_special_tokens=True
        )

        # Generation parameters
        generation_kwargs = dict(
            inputs=inputs['input_ids'],
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        # Start generation in background thread (so we can stream)
        print(f"\n    [STREAM] Starting word-by-word generation...")
        generation_thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        generation_thread.start()

        # Accumulate streamed output
        full_response = ""
        chunks = []
        chunk_buffer = []
        word_count = 0
        chunk_count = 0
        tokens_generated = 0

        # Stream tokens as they arrive
        try:
            for token_text in streamer:
                tokens_generated += 1
                chunk_buffer.append(token_text)

                # Count words (rough heuristic: split on spaces)
                if ' ' in token_text or '\n' in token_text:
                    word_count += 1

                # Emit chunk every N words
                if word_count >= self.words_per_chunk:
                    chunk_text = "".join(chunk_buffer)
                    chunks.append(chunk_text)
                    full_response += chunk_text
                    chunk_count += 1

                    elapsed = time.time() - start_time
                    print(f"    [STREAM {chunk_count}] @{elapsed:.1f}s: '{chunk_text.strip()}'")

                    # Callback for immediate speech synthesis
                    if on_chunk:
                        on_chunk(chunk_text, False)

                    # Reset buffer
                    chunk_buffer = []
                    word_count = 0

                    # Check if thought is complete (early stopping)
                    if self._is_thought_complete(full_response):
                        print(f"    [STREAM] Thought complete after {chunk_count} chunks, stopping")
                        break

        except Exception as e:
            print(f"    [STREAM] Error during streaming: {e}")

        finally:
            # Wait for generation thread to finish
            generation_thread.join(timeout=5.0)

        # Emit any remaining buffered text
        if chunk_buffer:
            chunk_text = "".join(chunk_buffer)
            chunks.append(chunk_text)
            full_response += chunk_text
            chunk_count += 1

            if on_chunk:
                on_chunk(chunk_text, True)  # Final chunk

        total_time = time.time() - start_time
        print(f"    [STREAM] Complete: {chunk_count} chunks, {tokens_generated} tokens in {total_time:.2f}s")

        return {
            'full_response': full_response.strip(),
            'chunks': chunks,
            'chunk_count': chunk_count,
            'total_time': total_time,
            'tokens_generated': tokens_generated
        }

    def _is_thought_complete(self, response: str) -> bool:
        """
        Detect if thought is complete (natural stopping point).

        Biological cues:
        - Ends with strong punctuation (. ! ?)
        - Not trailing with continuation words
        - Minimum length met (not cut off mid-thought)
        """
        response = response.strip()

        if len(response) < 50:  # Too short, likely incomplete
            return False

        # Check for strong ending
        if response.endswith(('.', '!', '?')):
            # Make sure not trailing with conjunctions
            words = response.lower().split()
            if len(words) < 3:
                return False

            # Last few words before punctuation
            last_words = words[-5:]
            trailing = {'and', 'but', 'so', 'because', 'however', 'although', 'while'}

            for word in last_words:
                if word.rstrip('.,!?') in trailing:
                    return False  # More coming

            return True  # Clean stop

        return False  # Continue generating

    def generate_response(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Non-streaming interface for backward compatibility."""
        result = self.generate_response_streaming(
            user_text,
            conversation_history,
            system_prompt
        )
        return result['full_response']


# Alias for drop-in replacement
Phi2Responder = StreamingResponder
