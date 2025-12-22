#!/usr/bin/env python3
"""
Lightweight Multi-Turn Conversation Manager for Q3-Omni on Edge Devices

Based on research findings (Dec 2024-2025):
- Multi-turn is conversation history management, not KV cache manipulation
- Use chat templates to format history properly
- Sliding window approach for infinite context:
  * Keep "attention sink" tokens (system message + first turn)
  * Maintain recent conversation window
  * Truncate middle when approaching context limit

References:
- StreamingLLM: https://arxiv.org/abs/2309.17453
- HuggingFace Chat Templates: https://huggingface.co/docs/transformers/chat_templating
- Multi-turn LLM Survey: https://arxiv.org/abs/2402.18013
"""

import torch
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class ConversationConfig:
    """Configuration for conversation management"""
    # Context window management
    max_context_tokens: int = 30000  # Conservative limit (Q3-Omni supports 65536)
    attention_sink_messages: int = 2  # Keep first N message pairs (system + first Q&A)
    sliding_window_messages: int = 10  # Keep last N message pairs

    # Generation parameters
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # Model parameters
    model_path: str = "model-zoo/sage/omni-modal/qwen3-omni-30b"
    device_map: str = "auto"
    max_memory: Dict[int, str] = field(default_factory=lambda: {0: "110GB"})


@dataclass
class ConversationTurn:
    """Single conversation turn with metadata"""
    user_message: str
    assistant_message: str
    tokens_used: int
    generation_time: float
    turn_number: int


class Q3OmniConversationManager:
    """
    Manages multi-turn conversations with Q3-Omni using application-layer history.

    Key Features:
    - Conversation history as list of messages (not KV cache manipulation)
    - Sliding window context management for infinite conversations
    - Attention sink preservation (keep first messages)
    - Automatic truncation when approaching context limit
    - Conversation save/load for persistence
    """

    def __init__(self, config: Optional[ConversationConfig] = None, system_message: Optional[str] = None):
        self.config = config or ConversationConfig()
        self.system_message = system_message

        # Conversation state
        self.messages: List[Dict[str, str]] = []
        self.turns: List[ConversationTurn] = []
        self.total_tokens_used = 0

        if system_message:
            self.messages.append({"role": "system", "content": system_message})

        # Model and processor (lazy loaded)
        self.model = None
        self.processor = None
        self.model_loaded = False

    def load_model(self):
        """Load Q3-Omni model (expensive, do once)"""
        if self.model_loaded:
            return

        print("Loading Q3-Omni model...")
        import time
        start = time.time()

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.config.model_path,
            device_map=self.config.device_map,
            max_memory=self.config.max_memory,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(self.config.model_path)

        self.model_loaded = True
        print(f"✅ Model loaded in {time.time() - start:.1f}s")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimate (words * 1.3 for English)"""
        return int(len(text.split()) * 1.3)

    def _get_conversation_window(self) -> List[Dict[str, str]]:
        """
        Get conversation window with sliding window + attention sink strategy.

        Strategy (from StreamingLLM):
        1. Always keep attention sink messages (first N turns)
        2. Keep sliding window of recent messages
        3. Discard middle messages when over limit
        """
        total_estimated_tokens = sum(
            self._estimate_tokens(msg["content"]) for msg in self.messages
        )

        # If under limit, return all messages
        if total_estimated_tokens < self.config.max_context_tokens:
            return self.messages

        # Apply sliding window strategy
        windowed_messages = []

        # 1. Keep system message if present
        if self.system_message:
            windowed_messages.append(self.messages[0])
            start_idx = 1
        else:
            start_idx = 0

        # 2. Keep attention sink (first N message pairs)
        attention_sink_count = self.config.attention_sink_messages * 2  # user + assistant
        attention_sink_end = start_idx + attention_sink_count
        windowed_messages.extend(self.messages[start_idx:attention_sink_end])

        # 3. Keep sliding window (last N message pairs)
        sliding_window_count = self.config.sliding_window_messages * 2
        sliding_window_start = max(attention_sink_end, len(self.messages) - sliding_window_count)
        windowed_messages.extend(self.messages[sliding_window_start:])

        print(f"Context truncated: {len(self.messages)} → {len(windowed_messages)} messages")

        return windowed_messages

    def chat(self, user_message: str) -> Tuple[str, Dict]:
        """
        Single conversation turn.

        Args:
            user_message: User's input

        Returns:
            (assistant_response, metadata)
        """
        if not self.model_loaded:
            self.load_model()

        import time
        turn_start = time.time()

        # Add user message to history
        self.messages.append({"role": "user", "content": user_message})

        # Get conversation window (with truncation if needed)
        conversation_window = self._get_conversation_window()

        # Format using chat template
        formatted_prompt = self.processor.apply_chat_template(
            conversation_window,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize and prepare inputs
        inputs = self.processor(text=[formatted_prompt], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        input_token_count = inputs['input_ids'].shape[1]

        # Generate response
        with torch.no_grad():
            text_ids, audio = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
                thinker_return_dict_in_generate=True,
            )

        # Decode response
        generated_tokens = text_ids.sequences[:, input_token_count:]
        assistant_response = self.processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Add assistant response to history
        self.messages.append({"role": "assistant", "content": assistant_response})

        # Record turn metadata
        generation_time = time.time() - turn_start
        tokens_generated = generated_tokens.shape[1]
        self.total_tokens_used += input_token_count + tokens_generated

        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=assistant_response,
            tokens_used=tokens_generated,
            generation_time=generation_time,
            turn_number=len(self.turns) + 1
        )
        self.turns.append(turn)

        metadata = {
            'turn_number': turn.turn_number,
            'tokens_generated': tokens_generated,
            'input_tokens': input_token_count,
            'generation_time': generation_time,
            'tokens_per_sec': tokens_generated / generation_time if generation_time > 0 else 0,
            'total_messages': len(self.messages),
            'total_tokens_used': self.total_tokens_used,
        }

        return assistant_response, metadata

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get full conversation history"""
        return self.messages.copy()

    def save_conversation(self, filepath: Path):
        """Save conversation to JSON"""
        data = {
            'system_message': self.system_message,
            'messages': self.messages,
            'turns': [
                {
                    'turn_number': t.turn_number,
                    'user_message': t.user_message,
                    'assistant_message': t.assistant_message,
                    'tokens_used': t.tokens_used,
                    'generation_time': t.generation_time,
                }
                for t in self.turns
            ],
            'total_tokens_used': self.total_tokens_used,
            'config': {
                'max_context_tokens': self.config.max_context_tokens,
                'max_new_tokens': self.config.max_new_tokens,
                'temperature': self.config.temperature,
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"✅ Conversation saved to {filepath}")

    def load_conversation(self, filepath: Path):
        """Load conversation from JSON"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.system_message = data.get('system_message')
        self.messages = data['messages']
        self.total_tokens_used = data['total_tokens_used']

        self.turns = [
            ConversationTurn(**turn_data)
            for turn_data in data['turns']
        ]

        print(f"✅ Conversation loaded from {filepath}")
        print(f"   {len(self.messages)} messages, {len(self.turns)} turns")

    def reset(self):
        """Reset conversation (keeps system message)"""
        self.messages = []
        if self.system_message:
            self.messages.append({"role": "system", "content": self.system_message})
        self.turns = []
        self.total_tokens_used = 0

    def print_conversation(self):
        """Pretty print conversation"""
        print("\n" + "=" * 80)
        print("CONVERSATION HISTORY")
        print("=" * 80 + "\n")

        for msg in self.messages:
            role = msg['role'].upper()
            content = msg['content']

            if role == "SYSTEM":
                print(f"[{role}]")
                print(f"{content}\n")
            else:
                print(f"{role}: {content}\n")

        print("=" * 80)
        print(f"Total messages: {len(self.messages)}")
        print(f"Total turns: {len(self.turns)}")
        print(f"Total tokens used: {self.total_tokens_used:,}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Q3-Omni Conversation Manager Demo")
    print()

    # Create conversation manager
    manager = Q3OmniConversationManager(
        system_message="You are a helpful AI assistant that tells creative stories."
    )

    # Example multi-turn conversation
    conversation_script = [
        "Tell me a short story about a dragon in 2 sentences.",
        "What was the dragon's name?",
        "What special power did the dragon have?",
        "How did the dragon use this power to help others?",
    ]

    print("Running multi-turn conversation demo...\n")

    for user_input in conversation_script:
        print(f"USER: {user_input}")
        response, meta = manager.chat(user_input)
        print(f"ASSISTANT: {response}")
        print(f"[Turn {meta['turn_number']}: {meta['tokens_generated']} tokens, {meta['tokens_per_sec']:.2f} tok/s]")
        print()

    # Print full conversation
    manager.print_conversation()

    # Save conversation
    manager.save_conversation(Path("demo_conversation.json"))
