#!/usr/bin/env python3
"""
Attention Kernel - Tier 0 Always-On Orchestrator

Following Nova's continuous attention design from:
HRM/forum/nova/HRM_IRP_Raising_Continuous_Attention_Suggestions.md

This is the heart of SAGE as a continuous attention orchestrator.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .state import AttentionState, StateTransition
from .atp_budget import ATPBudget
from .kernel_logging import TickLogger, ActionLogger, ContextLogger


class ExperienceBuffer:
    """
    Circular buffer of experience atoms for sleep consolidation

    Each experience atom has:
    - timestamp
    - source (focus/think/act)
    - context summary
    - outcome summary
    - salience score
    """

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.buffer: List[Dict[str, Any]] = []
        self.total_salience = 0.0

    def add(self, experience: Dict[str, Any]):
        """Add experience atom to buffer"""
        self.buffer.append(experience)
        self.total_salience += experience.get('salience', 0.0)

        # Keep most recent max_size experiences
        if len(self.buffer) > self.max_size:
            removed = self.buffer.pop(0)
            self.total_salience -= removed.get('salience', 0.0)

    def get_top_k(self, k: int) -> List[Dict[str, Any]]:
        """Get top-k experiences by salience"""
        sorted_exp = sorted(self.buffer, key=lambda x: x.get('salience', 0.0), reverse=True)
        return sorted_exp[:k]

    def clear(self):
        """Clear buffer after sleep consolidation"""
        self.buffer.clear()
        self.total_salience = 0.0

    @property
    def size(self) -> int:
        return len(self.buffer)

    @property
    def salience_sum(self) -> float:
        return self.total_salience


class SleepTrigger:
    """
    Decides when to consolidate experiences via sleep

    Triggers based on:
    - Buffer size threshold
    - Accumulated salience threshold
    - Time since last sleep
    - Idle window (don't sleep while actively serving)
    """

    def __init__(self, policy: Optional[Dict[str, Any]] = None):
        policy = policy or {}
        self.buffer_threshold = policy.get('buffer_size', 100)
        self.salience_threshold = policy.get('salience_sum', 50.0)
        self.time_threshold = policy.get('time_hours', 24) * 3600
        self.idle_window = policy.get('idle_minutes', 5) * 60

        self.last_activity_time = time.time()

    def mark_activity(self):
        """Mark that activity occurred (reset idle timer)"""
        self.last_activity_time = time.time()

    def should_sleep(self, buffer: ExperienceBuffer, last_sleep_time: float) -> bool:
        """Check if sleep conditions are met"""
        current_time = time.time()

        # Don't sleep if recently active
        if (current_time - self.last_activity_time) < self.idle_window:
            return False

        # Buffer full
        if buffer.size >= self.buffer_threshold:
            return True

        # High salience accumulated
        if buffer.salience_sum >= self.salience_threshold:
            return True

        # Been too long since last sleep
        if (current_time - last_sleep_time) >= self.time_threshold:
            return True

        return False


class AttentionKernel:
    """
    Tier 0: Always-on orchestrator

    Responsibilities:
    - Run continuous event loop
    - Maintain state machine
    - Route ATP budget
    - Invoke Tier 1 (LLM) when needed
    - Capture experiences
    - Trigger sleep
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # State
        self.state = AttentionState.IDLE
        self.running = False
        self.tick_count = 0
        self.sleep_cycle_count = 0

        # Core components
        self.atp_budget = ATPBudget(total_budget=config.get('atp_budget', 1000.0))
        self.experience_buffer = ExperienceBuffer(max_size=config.get('buffer_size', 100))
        self.sleep_trigger = SleepTrigger(config.get('sleep_policy', {}))

        # Logging
        log_dir = config.get('log_dir', 'logs')
        self.tick_logger = TickLogger(f'{log_dir}/attention_tick.jsonl')
        self.action_logger = ActionLogger(f'{log_dir}/action_log.jsonl')
        self.context_logger = ContextLogger(f'{log_dir}/context_manifest.jsonl')

        # Runtime state
        self.last_sleep_time = time.time()
        self.current_focus: Optional[Dict[str, Any]] = None
        self.current_thought: Optional[Dict[str, Any]] = None

        # Config
        self.tick_interval = config.get('tick_interval', 1.0)  # seconds

        # Plugin router and LLM runtime will be wired later
        self.plugin_router = None
        self.llm_runtime = None

    async def run_forever(self):
        """Main event loop - runs continuously"""
        self.running = True
        print(f"[AttentionKernel] Starting continuous attention loop")
        print(f"[AttentionKernel] Tick interval: {self.tick_interval}s")
        print(f"[AttentionKernel] Initial state: {self.state}")

        while self.running:
            try:
                await self.tick()
            except KeyboardInterrupt:
                print("\n[AttentionKernel] Interrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"[AttentionKernel] Error in tick: {e}")
                self.state = AttentionState.RECOVER
                await self.handle_error(e)

            await asyncio.sleep(self.tick_interval)

        print(f"[AttentionKernel] Stopped after {self.tick_count} ticks")

    async def tick(self):
        """Single tick of attention"""
        tick_start = time.time()

        # Increment tick count
        self.tick_count += 1

        # Log tick start
        self.tick_logger.start_tick(self.state)

        # Gather events
        events = await self.gather_events()

        # State transition
        next_state = self.decide_next_state(events)
        if next_state != self.state:
            self.tick_logger.log_transition(self.state, next_state, events)
            print(f"[AttentionKernel] {self.state} → {next_state}")

        self.state = next_state

        # Execute state behavior
        if self.state == AttentionState.IDLE:
            await self.idle_behavior()
        elif self.state == AttentionState.FOCUS:
            await self.focus_behavior(events)
        elif self.state == AttentionState.THINK:
            await self.think_behavior(events)
        elif self.state == AttentionState.ACT:
            await self.act_behavior(events)
        elif self.state == AttentionState.SLEEP:
            await self.sleep_behavior()
        elif self.state == AttentionState.RECOVER:
            await self.recover_behavior()

        # Log tick end
        tick_duration = time.time() - tick_start
        self.tick_logger.end_tick(tick_duration * 1000)  # Convert to ms

    async def gather_events(self) -> List[Dict[str, Any]]:
        """
        Gather events from various sources

        In v1, this is minimal. Future versions can add:
        - Timer events
        - User input
        - File changes
        - Sensor inputs
        - Plugin outputs
        """
        events = []

        # Check sleep trigger
        if self.sleep_trigger.should_sleep(self.experience_buffer, self.last_sleep_time):
            events.append({'type': 'sleep_trigger', 'reason': 'conditions_met'})

        return events

    def decide_next_state(self, events: List[Dict[str, Any]]) -> AttentionState:
        """
        State transition logic

        Simple v1 rules:
        - Sleep trigger → SLEEP
        - Otherwise stay in current state or return to IDLE
        """
        # Sleep takes priority
        if any(e['type'] == 'sleep_trigger' for e in events):
            return AttentionState.SLEEP

        # Simple state machine for v1
        if self.state == AttentionState.SLEEP:
            return AttentionState.IDLE  # Return to idle after sleep

        if self.state == AttentionState.RECOVER:
            return AttentionState.IDLE  # Return to idle after recovery

        # Stay in current state (will be enhanced in later versions)
        return self.state

    async def idle_behavior(self):
        """
        IDLE: Listening/watching, minimal activity

        v1: Just observe, log state
        """
        if self.tick_count % 10 == 0:  # Log every 10 ticks
            print(f"[AttentionKernel] IDLE (tick {self.tick_count})")

    async def focus_behavior(self, events):
        """
        FOCUS: Gather context, allocate ATP to plugins

        v1: Log the behavior, prepare for plugin integration
        """
        print(f"[AttentionKernel] FOCUS - gathering context")

        # Build context (placeholder for now)
        context = {'events': events, 'tick': self.tick_count}

        # Allocate ATP budget (placeholder plugin list)
        if self.plugin_router:
            plugin_names = self.plugin_router.get_available_plugins()
        else:
            plugin_names = []  # No plugins wired yet

        if plugin_names:
            allocations = self.atp_budget.allocate(plugin_names)
            self.tick_logger.log_budget(allocations)
            print(f"[AttentionKernel] Allocated ATP: {allocations}")

        # Store focus for potential THINK state
        self.current_focus = context

        # Capture experience
        self.capture_experience('focus', context, {'status': 'complete'})

    async def think_behavior(self, events):
        """
        THINK: Invoke LLM for deep reasoning (Tier 1)

        v1: Log the behavior, prepare for LLM integration
        """
        print(f"[AttentionKernel] THINK - invoking LLM")

        if self.llm_runtime:
            # Build prompt from current focus
            prompt = self.build_prompt(self.current_focus or {})

            # Log context
            self.context_logger.log_context('prompt', {'focus': self.current_focus}, prompt)

            # Invoke LLM (placeholder)
            # response = await self.llm_runtime.generate(prompt, {}, {})
            # self.current_thought = response
            print(f"[AttentionKernel] LLM invocation (not yet wired)")
        else:
            print(f"[AttentionKernel] No LLM runtime available")

        # Capture experience
        self.capture_experience('think', self.current_focus or {}, {'status': 'placeholder'})

    async def act_behavior(self, events):
        """
        ACT: Execute actions, observe outcomes

        v1: Log the behavior, prepare for tool integration
        """
        print(f"[AttentionKernel] ACT - executing actions")

        # Extract actions (placeholder)
        actions = []

        for action in actions:
            # Execute and log
            outcome = {'status': 'not_implemented'}
            self.action_logger.log(action, outcome)
            self.capture_experience('act', action, outcome)

    async def sleep_behavior(self):
        """
        SLEEP: Consolidate experiences via LoRA training

        v1: Log sleep trigger, prepare for training integration
        """
        print(f"[AttentionKernel] SLEEP - consolidating experiences")
        print(f"  Buffer size: {self.experience_buffer.size}")
        print(f"  Total salience: {self.experience_buffer.salience_sum:.2f}")

        # Prepare manifest
        top_experiences = self.experience_buffer.get_top_k(50)
        manifest = {
            'cycle': self.sleep_cycle_count,
            'num_experiences': len(top_experiences),
            'total_salience': self.experience_buffer.salience_sum,
            'timestamp': time.time()
        }

        print(f"  Consolidating {len(top_experiences)} top experiences")

        # Call sleep training subprocess (placeholder)
        # checkpoint = await self.run_sleep_training(manifest)

        # Run evaluation probes (placeholder)
        # eval_results = await self.run_evaluation_probes(checkpoint)

        # Clear buffer
        self.experience_buffer.clear()
        self.last_sleep_time = time.time()
        self.sleep_cycle_count += 1

        print(f"[AttentionKernel] Sleep cycle {self.sleep_cycle_count} complete")

    async def recover_behavior(self):
        """
        RECOVER: Restart subsystems, degrade gracefully

        v1: Log recovery, reset state
        """
        print(f"[AttentionKernel] RECOVER - attempting recovery")

        # Reset budgets
        self.atp_budget.reset()

        # Clear working state
        self.current_focus = None
        self.current_thought = None

        print(f"[AttentionKernel] Recovery complete")

    async def handle_error(self, error: Exception):
        """Handle errors in tick execution"""
        print(f"[AttentionKernel] Error: {error}")

        # Log error
        self.tick_logger.log_decision('error_recovery', {
            'error': str(error),
            'error_type': type(error).__name__
        })

        # Attempt recovery
        await self.recover_behavior()

    def capture_experience(self, source: str, context: Any, outcome: Any):
        """
        Create experience atom and add to buffer

        Experiences are continuously captured, not just session-based
        """
        # Simple salience scoring (v1)
        # TODO: Integrate SNARC salience scorer from sage/attention/sensor_snarc.py
        salience = 0.5  # Placeholder

        atom = {
            'ts': time.time(),
            'source': source,
            'context': self.summarize_context(context),
            'outcome': self.summarize_outcome(outcome),
            'salience': salience
        }

        self.experience_buffer.add(atom)
        self.sleep_trigger.mark_activity()

    def summarize_context(self, context: Any) -> Dict[str, Any]:
        """Summarize context for experience atom"""
        if isinstance(context, dict):
            return {k: str(v)[:100] for k, v in context.items()}
        return {'raw': str(context)[:100]}

    def summarize_outcome(self, outcome: Any) -> Dict[str, Any]:
        """Summarize outcome for experience atom"""
        if isinstance(outcome, dict):
            return {k: str(v)[:100] for k, v in outcome.items()}
        return {'raw': str(outcome)[:100]}

    def build_prompt(self, focus: Dict[str, Any]) -> str:
        """Build prompt for LLM from current focus"""
        # Placeholder prompt builder
        return f"Context: {focus}\n\nWhat should I do next?"

    def stop(self):
        """Stop the kernel gracefully"""
        print("[AttentionKernel] Stopping...")
        self.running = False
