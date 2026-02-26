#!/usr/bin/env python3
"""
Attention Kernel v3 - With LLM Runtime Integration

Extends v2 kernel with Tier 1 LLM runtime for THINK state.
Complete implementation of Nova's 2-tier architecture.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from .state import AttentionState, StateTransition
from .atp_budget import ATPBudget
from .kernel_logging import TickLogger, ActionLogger, ContextLogger
from .experience_salience import ExperienceSalienceScorer
from .plugin_router import PluginRouter

# Import v1 components
from .kernel import ExperienceBuffer, SleepTrigger

# Import LLM runtime (Tier 1)
try:
    from sage.llm import LLMRuntime, LLMRequest
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


class AttentionKernelV3:
    """
    Tier 0: Always-on orchestrator with LLM integration

    Extends v2 kernel with:
    - Tier 1 LLM runtime for THINK state
    - Deep reasoning invocation based on plugin disagreement
    - LLM-guided action planning
    - Experience capture from LLM interactions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config = config or {}

        # State
        self.state = AttentionState.IDLE
        self.running = False
        self.tick_count = 0
        self.sleep_cycle_count = 0
        self.last_sleep_time = time.time()

        # Core components
        self.atp_budget = ATPBudget(total_budget=config.get('atp_budget', 1000.0))
        self.experience_buffer = ExperienceBuffer(max_size=config.get('buffer_size', 100))
        self.sleep_trigger = SleepTrigger(config.get('sleep_policy', {}))

        # Salience scorer (from v2)
        self.salience_scorer = ExperienceSalienceScorer(
            memory_size=config.get('salience_memory_size', 100)
        )

        # Plugin router (from v2)
        plugin_config = config.get('plugin_config', {})
        self.plugin_router = PluginRouter(plugin_config)

        # NEW: LLM Runtime (Tier 1)
        self.llm_runtime = None
        self.llm_enabled = config.get('llm_enabled', False) and LLM_AVAILABLE
        if self.llm_enabled:
            llm_config = config.get('llm_config', {})
            self.llm_runtime = LLMRuntime(llm_config)
            print(f"[AttentionKernelV3] LLM runtime initialized")
        else:
            if config.get('llm_enabled', False) and not LLM_AVAILABLE:
                print(f"[AttentionKernelV3] LLM runtime requested but not available")

        # Logging
        log_dir = Path(config.get('log_dir', 'logs/attention'))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.tick_logger = TickLogger(log_dir / 'tick.jsonl')
        self.action_logger = ActionLogger(log_dir / 'action.jsonl')
        self.context_logger = ContextLogger(log_dir / 'context.jsonl')

        # Timing
        self.tick_interval = config.get('tick_interval', 0.1)
        self.last_tick_time = time.time()

        print(f"[AttentionKernelV3] Initialized (LLM={'enabled' if self.llm_enabled else 'disabled'})")

    async def start(self):
        """Start kernel event loop"""
        if self.running:
            print(f"[AttentionKernelV3] Already running")
            return

        self.running = True
        print(f"[AttentionKernelV3] Starting...")

        # Start LLM runtime if enabled
        if self.llm_runtime:
            await self.llm_runtime.start()

        # Start tick loop
        try:
            while self.running:
                await self.tick()
                await asyncio.sleep(self.tick_interval)
        except asyncio.CancelledError:
            print(f"[AttentionKernelV3] Cancelled")
        finally:
            await self.stop()

    async def stop(self):
        """Stop kernel event loop"""
        print(f"[AttentionKernelV3] Stopping...")
        self.running = False

        # Stop LLM runtime if enabled
        if self.llm_runtime:
            await self.llm_runtime.stop()

        print(f"[AttentionKernelV3] Stopped (ticks={self.tick_count}, sleeps={self.sleep_cycle_count})")

    async def tick(self):
        """Execute one tick of the event loop"""
        self.tick_count += 1
        self.last_tick_time = time.time()

        # Log tick start
        self.tick_logger.start_tick(self.state)

        # Execute state behavior
        if self.state == AttentionState.IDLE:
            await self.idle_behavior()
        elif self.state == AttentionState.FOCUS:
            await self.focus_behavior()
        elif self.state == AttentionState.THINK:
            await self.think_behavior()
        elif self.state == AttentionState.ACT:
            await self.act_behavior()
        elif self.state == AttentionState.SLEEP:
            await self.sleep_behavior()
        elif self.state == AttentionState.RECOVER:
            await self.recover_behavior()

        # Log tick end
        tick_duration = (time.time() - self.last_tick_time) * 1000
        self.tick_logger.end_tick(duration_ms=tick_duration)

    async def idle_behavior(self):
        """IDLE: Watch for triggers to transition to FOCUS"""
        # Check if buffer needs consolidation
        if self.sleep_trigger.should_sleep(
            buffer=self.experience_buffer,
            last_sleep_time=self.last_sleep_time
        ):
            self.transition_to(AttentionState.SLEEP)
            return

        # Simple timer-based trigger: transition to FOCUS periodically
        if self.tick_count % 50 == 0:  # Every 50 ticks
            self.transition_to(AttentionState.FOCUS)

    async def focus_behavior(self):
        """FOCUS: Gather context, allocate ATP to plugins, decide next state"""
        print(f"[AttentionKernelV3] FOCUS - gathering context")

        # Build context
        context = {
            'tick': self.tick_count,
            'goal': 'explore',  # Placeholder
            'observations': None,  # Would come from sensors
        }

        # Allocate ATP to plugins
        allocations = self.plugin_router.allocate_atp_budget(
            self.atp_budget.available
        )

        # Run plugins
        if len(allocations) > 0:
            results = await self.plugin_router.run_plugins(context, allocations)

            # Capture experience
            self.capture_experience('focus', context, results)

            # Decide next state based on plugin results
            if self._should_think(results):
                self.transition_to(AttentionState.THINK)
            elif self._should_act(results):
                self.transition_to(AttentionState.ACT)
            else:
                self.transition_to(AttentionState.IDLE)
        else:
            print(f"[AttentionKernelV3] No plugins available")
            self.capture_experience('focus', context, {'status': 'no_plugins'})
            self.transition_to(AttentionState.IDLE)

    async def think_behavior(self):
        """THINK: Invoke LLM for deep reasoning"""
        print(f"[AttentionKernelV3] THINK - invoking LLM")

        if not self.llm_runtime or not self.llm_enabled:
            print(f"[AttentionKernelV3] LLM not available, skipping THINK")
            self.capture_experience('think', {}, {'status': 'llm_unavailable'})
            self.transition_to(AttentionState.IDLE)
            return

        # Build prompt from recent experiences
        prompt = self._build_reasoning_prompt()

        # Invoke LLM (Tier 1)
        try:
            response = await self.llm_runtime.generate(
                prompt=prompt,
                max_tokens=256,
                temperature=0.7
            )

            # Log LLM interaction
            self.context_logger.log_context(
                context_type='llm_inference',
                sources={
                    'prompt_preview': prompt[:100],
                    'tokens': response.tokens_generated,
                    'time_ms': response.inference_time_ms
                },
                rendered=response.text[:100]
            )

            # Capture experience with LLM response
            self.capture_experience(
                'think',
                {'prompt': prompt[:200]},  # Truncate for experience
                {
                    'text': response.text,
                    'tokens': response.tokens_generated,
                    'time_ms': response.inference_time_ms,
                    'finish_reason': response.finish_reason
                }
            )

            # Decide next state based on LLM output
            if self._llm_suggests_action(response.text):
                self.transition_to(AttentionState.ACT)
            else:
                self.transition_to(AttentionState.IDLE)

        except Exception as e:
            print(f"[AttentionKernelV3] LLM inference failed: {e}")
            self.capture_experience('think', {}, {'error': str(e), 'status': 'failed'})
            self.transition_to(AttentionState.RECOVER)

    async def act_behavior(self):
        """ACT: Execute actions based on decisions"""
        print(f"[AttentionKernelV3] ACT - executing actions")

        # Placeholder: Real implementation would execute tool calls
        action = {'type': 'observe', 'target': 'environment'}
        outcome = {'status': 'success', 'result': 'observation captured'}

        # Log action
        self.action_logger.log(
            tick=self.tick_count,
            action_type=action['type'],
            action=action,
            outcome=outcome
        )

        # Capture experience
        self.capture_experience('act', action, outcome)

        # Transition back to IDLE
        self.transition_to(AttentionState.IDLE)

    async def sleep_behavior(self):
        """SLEEP: Consolidate experiences via raising pipeline"""
        print(f"[AttentionKernelV3] SLEEP - consolidating experiences")

        # Get statistics
        stats = self.salience_scorer.get_statistics()
        buffer_stats = {
            'size': self.experience_buffer.size,
            'total_salience': self.experience_buffer.salience_sum,
            'salience_stats': stats,
        }

        print(f"  Buffer size: {buffer_stats['size']}")
        print(f"  Total salience: {buffer_stats['total_salience']:.2f}")
        print(f"  Salience stats: avg={stats['avg_salience']:.3f}, max={stats['max_salience']:.3f}")
        print(f"  Source distribution: {stats['source_distribution']}")

        # Get top-k most salient experiences for consolidation
        top_k = self.experience_buffer.get_top_k(min(20, self.experience_buffer.size))
        print(f"  Consolidating {len(top_k)} top experiences")

        # TODO: Invoke raising pipeline sleep subprocess
        # For now, just clear buffer (placeholder)

        # Clear buffer
        self.experience_buffer.buffer.clear()
        self.experience_buffer.salience_sum = 0.0

        self.sleep_cycle_count += 1
        self.last_sleep_time = time.time()
        print(f"[AttentionKernelV3] Sleep cycle {self.sleep_cycle_count} complete")

        # Transition back to IDLE
        self.transition_to(AttentionState.IDLE)

    async def recover_behavior(self):
        """RECOVER: Handle failures, degrade gracefully"""
        print(f"[AttentionKernelV3] RECOVER - handling failure")

        # Placeholder: Real implementation would diagnose and restart subsystems
        await asyncio.sleep(1.0)

        # Transition back to IDLE
        self.transition_to(AttentionState.IDLE)

    def transition_to(self, new_state: AttentionState):
        """Transition to new state"""
        if new_state != self.state:
            print(f"[AttentionKernelV3] {self.state.value} → {new_state.value}")
            self.state = new_state

    def capture_experience(self, source: str, context: Any, outcome: Any):
        """Capture experience atom with SNARC salience"""
        # Compute salience using SNARC scorer
        salience = self.salience_scorer.score_experience(
            source,
            self.summarize_context(context),
            self.summarize_outcome(outcome)
        )

        atom = {
            'ts': time.time(),
            'source': source,
            'context': self.summarize_context(context),
            'outcome': self.summarize_outcome(outcome),
            'salience': salience
        }

        self.experience_buffer.add(atom)

        # Log high-salience experiences
        if salience > 0.7:
            print(f"  ⭐ High-salience experience: {source} (salience={salience:.3f})")

    def summarize_context(self, context: Any) -> Dict[str, Any]:
        """Summarize context for experience capture"""
        if isinstance(context, dict):
            # Preserve numeric types, truncate strings
            summary = {}
            for k, v in context.items():
                if isinstance(v, (int, float, bool)):
                    summary[k] = v
                else:
                    summary[k] = str(v)[:100]
            return summary
        return {'value': str(context)[:100]}

    def summarize_outcome(self, outcome: Any) -> Dict[str, Any]:
        """Summarize outcome for experience capture"""
        if isinstance(outcome, dict):
            # Preserve numeric types, truncate strings
            summary = {}
            for k, v in outcome.items():
                if isinstance(v, (int, float, bool)):
                    summary[k] = v
                else:
                    summary[k] = str(v)[:100]
            return summary
        return {'value': str(outcome)[:100]}

    def _should_think(self, plugin_results: Dict[str, Any]) -> bool:
        """Decide if we should transition to THINK based on plugin results"""
        # Transition to THINK if:
        # - Plugins disagree (high conflict)
        # - Low confidence
        # - High novelty

        disagreement = plugin_results.get('disagreement', 0.0)
        confidence = plugin_results.get('confidence', 1.0)

        if disagreement > 0.5:
            print(f"[AttentionKernelV3] High disagreement ({disagreement:.2f}) → THINK")
            return True

        if confidence < 0.3:
            print(f"[AttentionKernelV3] Low confidence ({confidence:.2f}) → THINK")
            return True

        return False

    def _should_act(self, plugin_results: Dict[str, Any]) -> bool:
        """Decide if we should transition to ACT based on plugin results"""
        # Transition to ACT if:
        # - Plugins agree (low disagreement)
        # - High confidence
        # - Has action suggestion

        disagreement = plugin_results.get('disagreement', 0.0)
        confidence = plugin_results.get('confidence', 0.0)

        if disagreement < 0.2 and confidence > 0.7:
            print(f"[AttentionKernelV3] High confidence ({confidence:.2f}) → ACT")
            return True

        return False

    def _build_reasoning_prompt(self) -> str:
        """Build LLM prompt from recent experiences"""
        # Get recent high-salience experiences
        recent = self.experience_buffer.get_top_k(5)

        if len(recent) == 0:
            return "What should I focus on next?"

        # Build prompt summarizing recent experiences
        prompt_parts = [
            "Recent observations:",
        ]

        for exp in recent:
            source = exp.get('source', 'unknown')
            outcome = exp.get('outcome', {})
            salience = exp.get('salience', 0.0)
            prompt_parts.append(f"- {source}: {outcome} (salience={salience:.2f})")

        prompt_parts.append("\nWhat patterns do you notice? What should I explore next?")

        return "\n".join(prompt_parts)

    def _llm_suggests_action(self, llm_text: str) -> bool:
        """Check if LLM response suggests taking action"""
        # Simple heuristic: look for action keywords
        action_keywords = ['do', 'try', 'execute', 'run', 'test', 'check', 'explore']
        text_lower = llm_text.lower()

        return any(keyword in text_lower for keyword in action_keywords)
