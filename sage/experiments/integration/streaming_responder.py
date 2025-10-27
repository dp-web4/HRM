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
from typing import List, Optional, Callable, Dict, Any, Tuple
import time
import threading
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
        words_per_chunk: int = 3,  # Stream every N words (balance latency vs efficiency)
        prediction_logger = None,  # Optional prediction logger for capturing hallucinations
        consciousness_persistence = None,  # Optional consciousness persistence manager
        use_cached_system_prompt: bool = True,  # Use cached system prompt KV
        auto_snapshot: bool = True,  # Auto-save consciousness during idle
        idle_snapshot_delay: float = 30.0  # Seconds of idle before auto-snapshot
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
        self.prediction_logger = prediction_logger

        # Consciousness persistence setup
        self.consciousness = consciousness_persistence
        self.use_cached_system_prompt = use_cached_system_prompt
        self.auto_snapshot = auto_snapshot
        self.idle_snapshot_delay = idle_snapshot_delay
        self.last_activity = time.time()
        self.system_prompt_kv = None  # Cached system prompt KV state
        self.current_session_kv = None  # Current conversation KV state

        # Initialize consciousness persistence if available
        if self.consciousness is None and use_cached_system_prompt:
            # Auto-initialize consciousness persistence
            try:
                from cognitive.consciousness_persistence import ConsciousnessPersistence
                self.consciousness = ConsciousnessPersistence()
                print(f"  ✓ Consciousness persistence auto-initialized")
            except ImportError:
                print(f"  ℹ️  Consciousness persistence not available (optional)")

        print(f"Model loaded: max_tokens={max_new_tokens}, streaming={words_per_chunk} words/chunk")

        if prediction_logger:
            print(f"  ✓ Prediction logging enabled")

        if self.consciousness:
            print(f"  ✓ Consciousness persistence enabled")
            if auto_snapshot:
                print(f"  ✓ Auto-snapshot: {idle_snapshot_delay}s idle delay")

    def generate_response_streaming(
        self,
        user_text: str,
        conversation_history: Optional[List[tuple]] = None,
        system_prompt: Optional[str] = None,
        on_chunk: Optional[Callable[[str, bool], None]] = None,
        restore_session: bool = False
    ) -> dict:
        """
        Generate response with word-by-word streaming.

        Args:
            user_text: Current user input
            conversation_history: List of (speaker, text) tuples
            system_prompt: Optional system instructions
            on_chunk: Callback(chunk_text, is_final) called for each word chunk
            restore_session: If True, attempt to restore from latest snapshot

        Returns:
            dict with:
                - full_response: Complete accumulated response
                - chunks: List of word chunks
                - chunk_count: Number of chunks emitted
                - total_time: Total generation time
                - tokens_generated: Number of tokens produced
        """
        start_time = time.time()

        # Update activity timestamp
        self.last_activity = time.time()

        # Option to restore session state
        if restore_session and self.consciousness:
            restored = self.consciousness.load_session_snapshot(use_latest=True)
            if restored:
                # Restore conversation history
                if restored.context_history:
                    conversation_history = restored.context_history
                    print(f"    [CONSCIOUSNESS] Restored {len(conversation_history)} conversation turns")

        # Try to load or cache system prompt KV
        if system_prompt and self.consciousness and self.use_cached_system_prompt:
            if self.system_prompt_kv is None:
                # Try loading cached KV
                cached_kv = self.consciousness.load_system_prompt_kv()
                if cached_kv:
                    self.system_prompt_kv = cached_kv
                    print(f"    [CONSCIOUSNESS] Loaded cached system prompt KV")
                else:
                    # Generate and cache it
                    print(f"    [CONSCIOUSNESS] Caching system prompt KV...")
                    self.system_prompt_kv = self.consciousness.cache_system_prompt_kv(
                        self.model,
                        self.tokenizer,
                        system_prompt
                    )
                    print(f"    [CONSCIOUSNESS] System prompt KV cached for future use")

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

                # Count ACTUAL words by analyzing buffer, not just tokens with spaces
                current_text = "".join(chunk_buffer)
                word_count = len(current_text.strip().split())

                # Emit chunk every N words
                if word_count >= self.words_per_chunk:
                    chunk_text = "".join(chunk_buffer)

                    # CRITICAL: Detect hallucinated conversation turns
                    # LLM sometimes generates "User:" or "Assistant:" imagining dialogue
                    hallucination_info = self._extract_hallucination(chunk_text, full_response)
                    if hallucination_info:
                        print(f"    [STREAM] Hallucination detected (generating fake dialogue), stopping")

                        # Log prediction for training data
                        if self.prediction_logger:
                            self.prediction_logger.capture_hallucination(
                                model_response=hallucination_info['model_response'],
                                predicted_user_response=hallucination_info['predicted_user_response'],
                                context={
                                    'chunks': chunk_count,
                                    'tokens': tokens_generated,
                                    'user_text': user_text
                                }
                            )
                        break

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

        # Check if auto-snapshot should be triggered
        if self.auto_snapshot and self.consciousness:
            self._check_and_snapshot(conversation_history, user_text, full_response.strip())

        return {
            'full_response': full_response.strip(),
            'chunks': chunks,
            'chunk_count': chunk_count,
            'total_time': total_time,
            'tokens_generated': tokens_generated
        }

    def _extract_hallucination(self, chunk_text: str, full_response: str) -> Optional[dict]:
        """
        Detect AND extract hallucinated dialogue for training data collection.

        Returns dict with:
        - model_response: What the model said (before hallucination)
        - predicted_user_response: What the model predicted user would say
        Or None if no hallucination detected.
        """
        combined = (full_response + chunk_text).strip()

        # Check for conversation turn markers
        hallucination_markers = [
            '\nUser:',
            '\nAssistant:',
            '\nSystem:',
            '👤 User:',
            '🤖 Assistant:',
        ]

        for marker in hallucination_markers:
            if marker in combined:
                # Found hallucination! Extract the parts
                parts = combined.split(marker, 1)
                model_response = parts[0].strip()

                # Extract the predicted user response (everything after the marker)
                predicted_user_response = marker.strip() + ' ' + (parts[1].strip() if len(parts) > 1 else '')

                return {
                    'model_response': model_response,
                    'predicted_user_response': predicted_user_response
                }

        # Check for emoji-user pattern (common hallucination: "🤔\nUser:")
        if '\nUser:' in chunk_text or 'User:' in chunk_text:
            # Find where "User:" starts
            if '\nUser:' in combined:
                parts = combined.split('\nUser:', 1)
            else:
                parts = combined.split('User:', 1)

            model_response = parts[0].strip()
            predicted_user_response = 'User: ' + (parts[1].strip() if len(parts) > 1 else '')

            return {
                'model_response': model_response,
                'predicted_user_response': predicted_user_response
            }

        return None

    def _is_hallucinating_dialogue(self, chunk_text: str, full_response: str) -> bool:
        """
        Legacy method - now just calls _extract_hallucination and returns bool.
        Kept for backward compatibility.
        """
        return self._extract_hallucination(chunk_text, full_response) is not None

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

    def _check_and_snapshot(
        self,
        conversation_history: Optional[List[tuple]],
        user_text: str,
        assistant_response: str
    ):
        """
        Check if idle period has elapsed and create snapshot if needed.

        Args:
            conversation_history: Current conversation history
            user_text: Latest user message
            assistant_response: Latest assistant response
        """
        if not self.consciousness:
            return

        # Check if idle period has elapsed
        idle_time = time.time() - self.last_activity

        if idle_time >= self.idle_snapshot_delay:
            print(f"    [CONSCIOUSNESS] Idle for {idle_time:.1f}s, creating snapshot...")

            # Build complete conversation history including latest exchange
            full_history = conversation_history.copy() if conversation_history else []
            full_history.append(("User", user_text))
            full_history.append(("Assistant", assistant_response))

            # Create snapshot (without KV cache for now - can be added later)
            from cognitive.consciousness_persistence import ConsciousnessSnapshot

            snapshot = ConsciousnessSnapshot(
                kv_cache=None,  # In future: capture actual KV state from model
                context_history=full_history,
                snarc_state=None,  # In future: integrate with SNARC memory
                metadata={
                    'timestamp': time.time(),
                    'turns': len(full_history),
                    'auto_snapshot': True,
                    'idle_seconds': idle_time
                }
            )

            # Save snapshot
            snapshot_file = self.consciousness.save_session_snapshot(snapshot)
            print(f"    [CONSCIOUSNESS] Snapshot saved: {snapshot_file}")

            # Reset activity timer
            self.last_activity = time.time()

    def check_idle_snapshot(self):
        """
        Manually trigger idle snapshot check.
        Call this periodically during conversation pauses.
        """
        if not self.consciousness or not self.auto_snapshot:
            return

        idle_time = time.time() - self.last_activity

        if idle_time >= self.idle_snapshot_delay:
            print(f"    [CONSCIOUSNESS] Manual idle check: {idle_time:.1f}s idle")
            # Note: We need conversation history to create snapshot
            # This method is primarily for external polling
            # The actual snapshot is created in _check_and_snapshot


# Alias for drop-in replacement
Phi2Responder = StreamingResponder
