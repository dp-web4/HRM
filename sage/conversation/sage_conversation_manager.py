#!/usr/bin/env python3
"""
Unified Conversation Manager for SAGE - Model-Agnostic

This manager works with ANY IRP plugin that implements generate_response().
Conversation state (history, sliding window, etc.) is managed at the SAGE level,
while model-specific inference is delegated to IRP plugins.

Architecture Benefits:
- Single conversation API for all models
- Consistent sliding window logic
- Easy to add new models (just implement IRP plugin)
- Conversation state is framework concern, not model concern
- Memory management happens once, applies to all models

References:
- StreamingLLM (attention sink): https://arxiv.org/abs/2309.17453
- Multi-turn LLM patterns: https://arxiv.org/abs/2402.18013
"""

import time
import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field


@dataclass
class ConversationConfig:
    """Configuration for conversation management"""
    # Context window management
    max_context_tokens: int = 30000  # Conservative default (most models support this)
    attention_sink_messages: int = 2  # Keep first N message pairs (system + first Q&A)
    sliding_window_messages: int = 10  # Keep last N message pairs

    # Generation parameters (can be overridden by plugin)
    max_new_tokens: int = 300
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # System message
    system_message: Optional[str] = None


@dataclass
class ConversationTurn:
    """Single conversation turn with metadata"""
    user_message: str
    assistant_message: str
    tokens_used: int
    generation_time: float
    turn_number: int
    plugin_name: str


class SAGEConversationManager:
    """
    Model-agnostic conversation manager for SAGE.

    Works with any IRP plugin that implements:
        generate_response(messages: List[Dict[str, str]]) -> str

    Example usage:
        plugin = Qwen25IRP()
        manager = SAGEConversationManager(plugin)

        response = manager.chat("Write a story about a dragon")
        # ... later ...
        response = manager.chat("What color was the dragon?")
    """

    def __init__(
        self,
        plugin: Any,
        config: Optional[ConversationConfig] = None,
        system_message: Optional[str] = None
    ):
        """
        Initialize conversation manager.

        Args:
            plugin: IRP plugin with generate_response() method
            config: Conversation configuration
            system_message: Optional system message for conversation
        """
        self.plugin = plugin
        self.config = config or ConversationConfig()

        # Override system message if provided
        if system_message:
            self.config.system_message = system_message

        # Conversation state
        self.messages: List[Dict[str, str]] = []
        self.turns: List[ConversationTurn] = []
        self.total_tokens_used = 0

        # Add system message if configured
        if self.config.system_message:
            self.messages.append({
                "role": "system",
                "content": self.config.system_message
            })

        print(f"SAGE Conversation Manager initialized")
        print(f"Plugin: {plugin.__class__.__name__}")
        print(f"Max context: {self.config.max_context_tokens} tokens")
        print(f"Sliding window: {self.config.sliding_window_messages} messages")
        if self.config.system_message:
            print(f"System message: {self.config.system_message[:50]}...")

    def chat(self, user_message: str, **generation_kwargs) -> str:
        """
        Send a message and get response.

        Args:
            user_message: User's message
            **generation_kwargs: Optional generation parameters to override config

        Returns:
            Assistant's response
        """
        # Add user message to history
        self.messages.append({
            "role": "user",
            "content": user_message
        })

        # Manage context window (sliding window + attention sink)
        self._manage_context_window()

        # Generate response using plugin
        start_time = time.time()

        try:
            response = self.plugin.generate_response(
                self.messages,
                max_new_tokens=generation_kwargs.get('max_new_tokens', self.config.max_new_tokens),
                temperature=generation_kwargs.get('temperature', self.config.temperature),
                top_k=generation_kwargs.get('top_k', self.config.top_k),
                top_p=generation_kwargs.get('top_p', self.config.top_p),
            )
        except Exception as e:
            print(f"Error generating response: {e}")
            raise

        generation_time = time.time() - start_time

        # Add assistant response to history
        self.messages.append({
            "role": "assistant",
            "content": response
        })

        # Track turn metadata
        turn = ConversationTurn(
            user_message=user_message,
            assistant_message=response,
            tokens_used=0,  # TODO: Calculate from tokenizer
            generation_time=generation_time,
            turn_number=len(self.turns) + 1,
            plugin_name=self.plugin.__class__.__name__
        )
        self.turns.append(turn)

        return response

    def _manage_context_window(self):
        """
        Manage context window using sliding window + attention sink strategy.

        Strategy:
        1. Always keep system message (if present)
        2. Keep first N message pairs (attention sink)
        3. Keep last M message pairs (sliding window)
        4. Drop messages in the middle when context is too long

        This implements StreamingLLM's attention sink approach for
        infinite-length conversations.
        """
        # Skip if we don't have many messages yet
        if len(self.messages) <= (self.config.attention_sink_messages * 2 +
                                   self.config.sliding_window_messages * 2):
            return

        # Separate system message from conversation
        system_msg = None
        conversation_msgs = self.messages

        if self.messages and self.messages[0]["role"] == "system":
            system_msg = self.messages[0]
            conversation_msgs = self.messages[1:]

        # Calculate how many messages to keep
        total_pairs = len(conversation_msgs) // 2
        sink_pairs = self.config.attention_sink_messages
        window_pairs = self.config.sliding_window_messages

        if total_pairs <= (sink_pairs + window_pairs):
            return  # Don't need to truncate yet

        # Build new message list
        new_messages = []

        # Add system message
        if system_msg:
            new_messages.append(system_msg)

        # Add attention sink (first N pairs)
        sink_end = sink_pairs * 2
        new_messages.extend(conversation_msgs[:sink_end])

        # Add sliding window (last M pairs)
        window_start = -(window_pairs * 2)
        new_messages.extend(conversation_msgs[window_start:])

        # Update messages
        self.messages = new_messages

        print(f"Context window managed: kept {len(new_messages)} messages "
              f"({sink_pairs} attention sink + {window_pairs} sliding window pairs)")

    def get_history(self) -> List[Dict[str, str]]:
        """Get full conversation history"""
        return self.messages.copy()

    def get_turns(self) -> List[ConversationTurn]:
        """Get conversation turns with metadata"""
        return self.turns.copy()

    def clear_history(self):
        """Clear conversation history (keep system message if present)"""
        if self.messages and self.messages[0]["role"] == "system":
            self.messages = [self.messages[0]]
        else:
            self.messages = []
        self.turns = []
        self.total_tokens_used = 0
        print("Conversation history cleared")

    def save_conversation(self, filepath: Path):
        """Save conversation to JSON file"""
        data = {
            "config": {
                "max_context_tokens": self.config.max_context_tokens,
                "attention_sink_messages": self.config.attention_sink_messages,
                "sliding_window_messages": self.config.sliding_window_messages,
                "system_message": self.config.system_message,
            },
            "plugin": self.plugin.__class__.__name__,
            "messages": self.messages,
            "turns": [
                {
                    "user_message": t.user_message,
                    "assistant_message": t.assistant_message,
                    "tokens_used": t.tokens_used,
                    "generation_time": t.generation_time,
                    "turn_number": t.turn_number,
                    "plugin_name": t.plugin_name,
                }
                for t in self.turns
            ],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Conversation saved to {filepath}")

    def load_conversation(self, filepath: Path):
        """Load conversation from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.messages = data["messages"]
        self.turns = [
            ConversationTurn(
                user_message=t["user_message"],
                assistant_message=t["assistant_message"],
                tokens_used=t["tokens_used"],
                generation_time=t["generation_time"],
                turn_number=t["turn_number"],
                plugin_name=t["plugin_name"],
            )
            for t in data["turns"]
        ]

        print(f"Conversation loaded from {filepath}")
        print(f"  Messages: {len(self.messages)}")
        print(f"  Turns: {len(self.turns)}")

    def print_conversation(self, max_length: int = 100):
        """Print conversation history (truncated for readability)"""
        print("\n" + "="*80)
        print("CONVERSATION HISTORY")
        print("="*80)

        for msg in self.messages:
            role = msg["role"].upper()
            content = msg["content"]

            # Truncate long messages
            if len(content) > max_length:
                content = content[:max_length] + "..."

            print(f"\n{role}:")
            print(f"  {content}")

        print("\n" + "="*80)
        print(f"Total turns: {len(self.turns)}")
        print(f"Total messages: {len(self.messages)}")
        print("="*80 + "\n")


# Example usage
if __name__ == "__main__":
    print("SAGEConversationManager - Model-Agnostic Conversation Manager")
    print()
    print("This is a framework component. See test scripts for usage examples:")
    print("  - sage/tests/test_unified_conversation.py")
    print("  - sage/conversation/README.md")
