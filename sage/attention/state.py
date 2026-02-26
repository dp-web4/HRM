#!/usr/bin/env python3
"""
State machine for continuous attention kernel

Following Nova's design from HRM_IRP_Raising_Continuous_Attention_Suggestions.md
"""

from enum import Enum, auto


class AttentionState(Enum):
    """
    Six modes of continuous attention

    IDLE: Listening/watching, minimal activity, lightweight observers
    FOCUS: Gather context, allocate ATP to plugins
    THINK: Invoke LLM for deep reasoning (Tier 1)
    ACT: Execute tool actions, capture experience
    SLEEP: Consolidate experiences via LoRA subprocess
    RECOVER: Restart subsystems, roll back checkpoints, degrade gracefully
    """
    IDLE = auto()
    FOCUS = auto()
    THINK = auto()
    ACT = auto()
    SLEEP = auto()
    RECOVER = auto()

    def __str__(self):
        return self.name

    @classmethod
    def from_string(cls, s: str) -> 'AttentionState':
        """Parse state from string"""
        return cls[s.upper()]


class StateTransition:
    """Record of state transition with reasoning"""

    def __init__(self, from_state, to_state, reason, events=None):
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason
        self.events = events or []

    def to_dict(self):
        return {
            'from': str(self.from_state),
            'to': str(self.to_state),
            'reason': self.reason,
            'events': self.events
        }
