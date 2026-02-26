#!/usr/bin/env python3
"""
Attention Kernel v2 - With SNARC Salience Integration

Extends v1 kernel with experience salience scoring using algorithmic SNARC.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .state import AttentionState, StateTransition
from .atp_budget import ATPBudget
from .kernel_logging import TickLogger, ActionLogger, ContextLogger
from .experience_salience import ExperienceSalienceScorer


# Import v1 components we'll extend
from .kernel import ExperienceBuffer, SleepTrigger


class AttentionKernelV2:
    """
    Tier 0: Always-on orchestrator with SNARC salience

    Extends v1 kernel with:
    - Algorithmic SNARC salience scoring for experiences
    - Salience-based experience prioritization
    - Enhanced sleep triggering based on salience thresholds
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

        # NEW: Salience scorer
        self.salience_scorer = ExperienceSalienceScorer(
            memory_size=config.get('salience_memory_size', 100)
        )

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
        self.tick_interval = config.get('tick_interval', 1.0)

        # Plugin router
        from .plugin_router import PluginRouter
        plugin_config = config.get('plugin_config', {})
        try:
            self.plugin_router = PluginRouter(plugin_config)
        except Exception as e:
            print(f"[AttentionKernelV2] Plugin router init failed: {e}")
            self.plugin_router = None

        # LLM runtime (Day 2)
        self.llm_runtime = None

    async def run_forever(self):
        """Main event loop - runs continuously"""
        self.running = True
        print(f"[AttentionKernelV2] Starting continuous attention loop")
        print(f"[AttentionKernelV2] Tick interval: {self.tick_interval}s")
        print(f"[AttentionKernelV2] SNARC salience scoring: ENABLED")
        print(f"[AttentionKernelV2] Initial state: {self.state}")

        while self.running:
            try:
                await self.tick()
            except KeyboardInterrupt:
                print("\n[AttentionKernelV2] Interrupted by user")
                self.running = False
                break
            except Exception as e:
                print(f"[AttentionKernelV2] Error in tick: {e}")
                self.state = AttentionState.RECOVER
                await self.handle_error(e)

            await asyncio.sleep(self.tick_interval)

        print(f"[AttentionKernelV2] Stopped after {self.tick_count} ticks")

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
            print(f"[AttentionKernelV2] {self.state} → {next_state}")

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
        self.tick_logger.end_tick(tick_duration * 1000)

    async def gather_events(self) -> List[Dict[str, Any]]:
        """Gather events from various sources"""
        events = []

        # Check sleep trigger
        if self.sleep_trigger.should_sleep(self.experience_buffer, self.last_sleep_time):
            events.append({'type': 'sleep_trigger', 'reason': 'conditions_met'})

        return events

    def decide_next_state(self, events: List[Dict[str, Any]]) -> AttentionState:
        """State transition logic"""
        # Sleep takes priority
        if any(e['type'] == 'sleep_trigger' for e in events):
            return AttentionState.SLEEP

        # Simple state machine for v2
        if self.state == AttentionState.SLEEP:
            return AttentionState.IDLE

        if self.state == AttentionState.RECOVER:
            return AttentionState.IDLE

        # Stay in current state
        return self.state

    async def idle_behavior(self):
        """IDLE: Listening/watching, minimal activity"""
        if self.tick_count % 10 == 0:
            print(f"[AttentionKernelV2] IDLE (tick {self.tick_count})")

    async def focus_behavior(self, events):
        """FOCUS: Gather context, allocate ATP to plugins"""
        print(f"[AttentionKernelV2] FOCUS - gathering context")

        # Build context for plugins
        context = {
            'goal': 'observe',
            'events': events,
            'tick': self.tick_count,
            'observations': None,
            'constraints': {}
        }

        # Get available plugins
        if self.plugin_router:
            plugin_names = self.plugin_router.get_available_plugins()
        else:
            plugin_names = []

        if plugin_names:
            # Allocate ATP budget
            allocations = self.atp_budget.allocate(plugin_names)
            self.tick_logger.log_budget(allocations)
            print(f"[AttentionKernelV2] Allocated ATP to {len(allocations)} plugins")

            # Run plugins asynchronously
            try:
                results = await self.plugin_router.run_plugins(context, allocations)
                print(f"[AttentionKernelV2] Plugins executed: {results['status']}")

                # Update ATP budget with actual usage
                for plugin_name, result in results['results'].items():
                    if 'error' not in result:
                        self.atp_budget.report_result(
                            plugin_name,
                            success=result.get('converged', False),
                            resources_used=allocations[plugin_name],
                            value_produced=1.0 - result.get('final_energy', 1.0)
                        )

                # Store results
                self.current_focus = {
                    'context': context,
                    'plugin_results': results,
                    'confidence': self._compute_confidence(results),
                    'disagreement': self._compute_disagreement(results)
                }

                # Capture experience with SNARC salience
                self.capture_experience('focus', context, results)

            except Exception as e:
                print(f"[AttentionKernelV2] Plugin execution error: {e}")
                self.current_focus = {'error': str(e)}
                self.capture_experience('focus', context, {'error': str(e)})

        else:
            print(f"[AttentionKernelV2] No plugins available")
            self.current_focus = context
            self.capture_experience('focus', context, {'status': 'no_plugins'})

    async def think_behavior(self, events):
        """THINK: Invoke LLM for deep reasoning (Tier 1)"""
        print(f"[AttentionKernelV2] THINK - invoking LLM")

        if self.llm_runtime:
            prompt = self.build_prompt(self.current_focus or {})
            self.context_logger.log_context('prompt', {'focus': self.current_focus}, prompt)
            print(f"[AttentionKernelV2] LLM invocation (not yet wired)")
        else:
            print(f"[AttentionKernelV2] No LLM runtime available")

        self.capture_experience('think', self.current_focus or {}, {'status': 'placeholder'})

    async def act_behavior(self, events):
        """ACT: Execute actions, observe outcomes"""
        print(f"[AttentionKernelV2] ACT - executing actions")

        actions = []
        for action in actions:
            outcome = {'status': 'not_implemented'}
            self.action_logger.log(action, outcome)
            self.capture_experience('act', action, outcome)

    async def sleep_behavior(self):
        """SLEEP: Consolidate experiences via LoRA training"""
        print(f"[AttentionKernelV2] SLEEP - consolidating experiences")
        print(f"  Buffer size: {self.experience_buffer.size}")
        print(f"  Total salience: {self.experience_buffer.salience_sum:.2f}")

        # Get top experiences by salience
        top_experiences = self.experience_buffer.get_top_k(50)

        # Show salience statistics
        stats = self.salience_scorer.get_statistics()
        print(f"  Salience stats: avg={stats['avg_salience']:.3f}, max={stats['max_salience']:.3f}")
        print(f"  Source distribution: {stats['source_distribution']}")

        # Prepare manifest
        manifest = {
            'cycle': self.sleep_cycle_count,
            'num_experiences': len(top_experiences),
            'total_salience': self.experience_buffer.salience_sum,
            'timestamp': time.time(),
            'salience_stats': stats
        }

        print(f"  Consolidating {len(top_experiences)} top experiences")

        # Clear buffer
        self.experience_buffer.clear()
        self.last_sleep_time = time.time()
        self.sleep_cycle_count += 1

        print(f"[AttentionKernelV2] Sleep cycle {self.sleep_cycle_count} complete")

    async def recover_behavior(self):
        """RECOVER: Restart subsystems, degrade gracefully"""
        print(f"[AttentionKernelV2] RECOVER - attempting recovery")

        self.atp_budget.reset()
        self.current_focus = None
        self.current_thought = None

        print(f"[AttentionKernelV2] Recovery complete")

    async def handle_error(self, error: Exception):
        """Handle errors in tick execution"""
        print(f"[AttentionKernelV2] Error: {error}")

        self.tick_logger.log_decision('error_recovery', {
            'error': str(error),
            'error_type': type(error).__name__
        })

        await self.recover_behavior()

    def capture_experience(self, source: str, context: Any, outcome: Any):
        """
        Create experience atom and add to buffer WITH SNARC SALIENCE

        This is the key integration point - experiences are scored for salience
        """
        # Compute salience using SNARC scorer
        salience = self.salience_scorer.score_experience(source, context, outcome)

        atom = {
            'ts': time.time(),
            'source': source,
            'context': self.summarize_context(context),
            'outcome': self.summarize_outcome(outcome),
            'salience': salience  # Now computed, not placeholder
        }

        self.experience_buffer.add(atom)
        self.sleep_trigger.mark_activity()

        # Log high-salience experiences
        if salience > 0.7:
            print(f"[AttentionKernelV2] High-salience experience: {source} (salience={salience:.3f})")

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
        return f"Context: {focus}\n\nWhat should I do next?"

    def _compute_confidence(self, results: Dict[str, Any]) -> float:
        """Compute overall confidence from plugin results"""
        if not results.get('results'):
            return 0.0

        converged_count = sum(
            1 for r in results['results'].values()
            if r.get('converged', False)
        )
        total = len(results['results'])
        return converged_count / total if total > 0 else 0.0

    def _compute_disagreement(self, results: Dict[str, Any]) -> float:
        """Compute disagreement between plugin results"""
        if not results.get('results'):
            return 0.0

        energies = [
            r.get('final_energy', 1.0)
            for r in results['results'].values()
            if 'error' not in r
        ]

        if len(energies) < 2:
            return 0.0

        import numpy as np
        return float(np.var(energies))

    def stop(self):
        """Stop the kernel gracefully"""
        print("[AttentionKernelV2] Stopping...")
        self.running = False

        if self.plugin_router:
            self.plugin_router.shutdown()
