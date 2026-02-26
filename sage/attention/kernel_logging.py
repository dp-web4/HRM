#!/usr/bin/env python3
"""
Logging infrastructure for attention kernel

Produces auditable logs and manifests following Nova's artifact-as-truth principle
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime


class TickLogger:
    """
    Logs attention kernel tick events to JSONL

    Each tick produces a record with:
    - Timestamp
    - Current state
    - Events observed
    - Decisions made
    - Budget allocations
    - Duration
    """

    def __init__(self, log_path: str = 'logs/attention_tick.jsonl'):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.current_tick = None
        self.tick_start_time = None

    def start_tick(self, state):
        """Begin a tick record"""
        self.current_tick = {
            'ts': time.time(),
            'iso_ts': datetime.now().isoformat(),
            'state': str(state),
            'events': [],
            'transitions': [],
            'decisions': [],
            'budget': {},
            'duration_ms': None
        }
        self.tick_start_time = time.time()

    def log_transition(self, from_state, to_state, events):
        """Log state transition"""
        if self.current_tick:
            self.current_tick['transitions'].append({
                'from': str(from_state),
                'to': str(to_state),
                'events': events
            })

    def log_decision(self, decision_type: str, data: Dict[str, Any]):
        """Log a decision made during this tick"""
        if self.current_tick:
            self.current_tick['decisions'].append({
                'type': decision_type,
                **data
            })

    def log_budget(self, allocations: Dict[str, float]):
        """Log ATP budget allocations"""
        if self.current_tick:
            self.current_tick['budget'] = allocations

    def end_tick(self, duration_ms: Optional[float] = None):
        """Complete and write tick record"""
        if self.current_tick:
            if duration_ms is None and self.tick_start_time:
                duration_ms = (time.time() - self.tick_start_time) * 1000

            self.current_tick['duration_ms'] = duration_ms

            # Append to JSONL
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(self.current_tick) + '\n')

            self.current_tick = None
            self.tick_start_time = None


class ActionLogger:
    """
    Logs actions and outcomes to JSONL

    Canonical action stream showing what the kernel decided to do
    """

    def __init__(self, log_path: str = 'logs/action_log.jsonl'):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, action: Dict[str, Any], outcome: Optional[Dict[str, Any]] = None):
        """Log an action and its outcome"""
        record = {
            'ts': time.time(),
            'iso_ts': datetime.now().isoformat(),
            'action': action,
            'outcome': outcome
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')


class ContextLogger:
    """
    Logs context assembly (what went into prompts/decisions)

    Makes failures diagnosable: "it hallucinated because we fed it X"
    """

    def __init__(self, log_path: str = 'logs/context_manifest.jsonl'):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_context(self, context_type: str, sources: Dict[str, Any], rendered: Optional[str] = None):
        """
        Log context assembly

        Args:
            context_type: 'prompt', 'decision', 'tool_call', etc.
            sources: Structured sources that went into context
            rendered: Optional rendered version (e.g., final prompt text)
        """
        record = {
            'ts': time.time(),
            'iso_ts': datetime.now().isoformat(),
            'type': context_type,
            'sources': sources,
            'rendered': rendered
        }

        with open(self.log_path, 'a') as f:
            f.write(json.dumps(record) + '\n')
