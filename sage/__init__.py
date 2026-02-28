"""
SAGE - Situation-Aware Governance Engine

Unified entry point for creating and running SAGE consciousness systems.

Usage:
    from sage import SAGE

    # Create with defaults (mock sensors, algorithmic SNARC)
    sage = SAGE.create()

    # Create with real LLM (Ollama)
    sage = SAGE.create(use_real_llm=True, config={
        'llm_config': {
            'backend_type': 'ollama',
            'backend_config': {'model_name': 'tinyllama:latest'}
        }
    })

    # Run the consciousness loop
    await sage.run(max_cycles=100)

    # Send a message (triggers real LLM on next cycle)
    sage.send_message("Hello SAGE", sender="user")

What's wired end-to-end (traceable code paths):
- SAGEConsciousness 9-step loop (sage/core/sage_consciousness.py)
- LLMRuntime with hot/cold lifecycle (Ollama/Transformers) → real inference
- Metabolic controller: WAKE/FOCUS/REST/DREAM/CRISIS with ATP budgeting
- ATP coupled to real LLM token cost (0.05 ATP/token)
- SNARC salience scoring (algorithmic 5D — not neural)
- DREAM consolidation writes top-k experiences to disk (JSONL)
- Message injection queue for text input → LLM on next cycle
- Trust weight learning from plugin convergence

What's mocked or partial (honest about it):
- Sensor observations: mock random data (no real camera/mic/IMU)
- SNARC computation: algorithmic heuristic, not learned model
- Effector system: architecture exists, all effectors are mock (no real I/O)
- PolicyGate: skeleton exists (Phase 1), not wired by default
- Sleep → LoRA training: infrastructure exists, import fails at runtime
"""

from typing import Optional, Dict, Any, List
import asyncio
import time
import queue as queue_mod
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Lightweight message injection (no Futures, no event loop requirement)
# ---------------------------------------------------------------------------

@dataclass
class _SimpleMessage:
    """Message for injection into consciousness loop."""
    message_id: str
    sender: str
    content: str
    conversation_id: str
    timestamp: float
    metadata: dict = field(default_factory=dict)


@dataclass
class _ConversationTurn:
    """A single turn in a conversation."""
    role: str
    content: str
    timestamp: float
    sender: str = ""


class _SimpleMessageInput:
    """
    Lightweight message input for the SAGE facade.

    Implements the interface SAGEConsciousness._gather_observations() expects:
    - poll_all() → List[message-like objects]
    - get_conversation_history(conversation_id) → List[turn-like objects]
    """

    def __init__(self):
        self._queue: queue_mod.Queue = queue_mod.Queue()
        self._conversations: Dict[str, List[_ConversationTurn]] = {}
        self._msg_counter = 0

    def inject(self, sender: str, content: str,
               conversation_id: str = "default",
               metadata: Optional[Dict] = None):
        """Add a message to be processed on the next consciousness cycle."""
        self._msg_counter += 1
        msg = _SimpleMessage(
            message_id=f"msg_{self._msg_counter:04d}",
            sender=sender,
            content=content,
            conversation_id=conversation_id,
            timestamp=time.time(),
            metadata=metadata or {},
        )
        self._record_turn(conversation_id, _ConversationTurn(
            role="user", content=content,
            timestamp=msg.timestamp, sender=sender,
        ))
        self._queue.put(msg)

    def poll_all(self):
        """Drain all pending messages."""
        messages = []
        while not self._queue.empty():
            try:
                messages.append(self._queue.get_nowait())
            except queue_mod.Empty:
                break
        return messages

    def get_conversation_history(self, conversation_id: str):
        """Get conversation turns for LLM context building."""
        return list(self._conversations.get(conversation_id, []))

    def record_response(self, conversation_id: str, response: str):
        """Record SAGE's response in conversation history."""
        self._record_turn(conversation_id, _ConversationTurn(
            role="sage", content=response,
            timestamp=time.time(), sender="sage",
        ))

    def _record_turn(self, conversation_id: str, turn: _ConversationTurn):
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        self._conversations[conversation_id].append(turn)
        if len(self._conversations[conversation_id]) > 20:
            self._conversations[conversation_id] = \
                self._conversations[conversation_id][-20:]

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()


# ---------------------------------------------------------------------------
# Sync adapter: bridges async LLMRuntime.generate() for the thread pool
# ---------------------------------------------------------------------------

class _SyncLLMAdapter:
    """
    Sync wrapper around async LLMRuntime, compatible with
    SAGEConsciousness._generate_llm_response() which checks for .generate().

    _generate_llm_response runs inside run_in_executor (thread pool).
    This adapter uses run_coroutine_threadsafe to schedule the async
    LLM call back on the main event loop, avoiding event loop conflicts
    with httpx's AsyncClient.
    """

    def __init__(self, runtime):
        self.runtime = runtime
        self._main_loop = None

    def set_event_loop(self, loop):
        """Set the main event loop for cross-thread async calls."""
        self._main_loop = loop

    def generate(self, prompt: str, max_tokens: int = 200,
                 temperature: float = 0.8) -> str:
        """Sync generate → schedule on main loop → await result → string."""
        if self._main_loop is None or self._main_loop.is_closed():
            return "[LLM error: event loop not available]"
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.runtime.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
                self._main_loop,
            )
            response = future.result(timeout=120)
            return response.text
        except Exception as e:
            return f"[LLM error: {e}]"


# ---------------------------------------------------------------------------
# SAGE Facade
# ---------------------------------------------------------------------------

class SAGE:
    """
    Unified SAGE facade - single entry point for consciousness system.

    Wires (real):
    - SAGEConsciousness (full 9-step loop)
    - LLMRuntime + sync adapter (when use_real_llm=True)
    - MetabolicController (5-state management with ATP)
    - Message injection queue for text input
    - SNARC salience (algorithmic 5D scoring — not neural)
    - DREAM consolidation (top-k experiences → JSONL)

    Present but mocked:
    - Effectors (mock implementations, no real I/O)
    - PolicyGate (skeleton, disabled by default)
    - Sensors (mock random observations)
    """

    def __init__(self, consciousness_loop, config: Optional[Dict[str, Any]] = None,
                 llm_runtime=None, message_input=None):
        """
        Internal constructor. Use SAGE.create() instead.

        Args:
            consciousness_loop: Underlying consciousness implementation
            config: Configuration dictionary
            llm_runtime: Optional LLMRuntime instance (for lifecycle management)
            message_input: Optional _SimpleMessageInput (for send_message)
        """
        self._loop = consciousness_loop
        self._config = config or {}
        self._llm_runtime = llm_runtime
        self._message_input = message_input

    @staticmethod
    def create(
        config: Optional[Dict[str, Any]] = None,
        use_real_llm: bool = False,
        use_real_sensors: bool = False,
        use_policy_gate: bool = False,
        use_neural_snarc: bool = False
    ) -> 'SAGE':
        """
        Create a SAGE consciousness system with automatic component wiring.

        Args:
            config: Configuration dictionary (optional)
                - 'lct_identity': Hardware-bound identity config
                - 'metabolic_params': ATP rates, state thresholds
                - 'plugin_config': IRP plugin settings
                - 'snarc_weights': 5D salience weights
                - 'llm_config': LLM backend settings
                    - 'backend_type': 'ollama' or 'transformers'
                    - 'backend_config': {'model_name': 'tinyllama:latest', ...}

            use_real_llm: Use real LLM (Ollama/Transformers) vs mock
            use_real_sensors: Use real sensors vs mock observations
            use_policy_gate: Enable PolicyGate at step 8.5
            use_neural_snarc: Use learned SNARC model vs algorithmic

        Returns:
            SAGE instance ready to run

        Example:
            # Minimal - mock everything
            sage = SAGE.create()

            # With Ollama LLM
            sage = SAGE.create(use_real_llm=True, config={
                'llm_config': {
                    'backend_type': 'ollama',
                    'backend_config': {'model_name': 'tinyllama:latest'}
                }
            })

            # Full config
            sage = SAGE.create(config={
                'lct_identity': {'hardware_id': 'cbp-001'},
                'metabolic_params': {'base_atp': 1000.0},
                'llm_config': {
                    'backend_type': 'ollama',
                    'backend_config': {'model_name': 'tinyllama:latest'}
                }
            }, use_real_llm=True, use_policy_gate=True)
        """
        config = config or {}
        llm_runtime = None
        llm_adapter = None
        message_input = None

        # Wire LLM Runtime when requested
        if use_real_llm:
            from sage.llm.runtime import LLMRuntime

            llm_config = config.get('llm_config', {})
            if not llm_config:
                # Sensible default: Ollama with tinyllama
                llm_config = {
                    'backend_type': 'ollama',
                    'backend_config': {'model_name': 'tinyllama:latest'},
                }

            llm_runtime = LLMRuntime(llm_config)
            llm_adapter = _SyncLLMAdapter(llm_runtime)

        # Always create message input (enables send_message even in mock mode)
        message_input = _SimpleMessageInput()

        # Build consciousness loop config
        from sage.core.sage_consciousness import SAGEConsciousness

        loop_config = {
            **config,
            'use_policy_gate': use_policy_gate,
            'use_neural_snarc': use_neural_snarc,
            'use_real_sensors': use_real_sensors,
        }

        loop = SAGEConsciousness(
            config=loop_config,
            message_queue=message_input,
            llm_plugin=llm_adapter,
        )

        return SAGE(loop, config, llm_runtime=llm_runtime,
                    message_input=message_input)

    def send_message(self, content: str, sender: str = "user",
                     conversation_id: str = "default") -> None:
        """
        Inject a text message into the consciousness loop.

        The message becomes a 'message' modality observation on the next
        cycle, triggering real LLM inference if use_real_llm=True.

        Args:
            content: Message text
            sender: Identity of sender (default: "user")
            conversation_id: Conversation thread ID
        """
        if self._message_input is None:
            raise RuntimeError("SAGE was not created with message input support")
        self._message_input.inject(sender, content, conversation_id)

    async def run(
        self,
        max_cycles: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
        stop_on_crisis: bool = False
    ) -> Dict[str, Any]:
        """
        Run the SAGE consciousness loop.

        Args:
            max_cycles: Maximum number of consciousness cycles (None = infinite)
            max_duration_seconds: Maximum runtime in seconds (None = no limit)
            stop_on_crisis: Stop if CRISIS metabolic state is entered

        Returns:
            Statistics about the run:
                - 'cycles_completed': Number of cycles executed
                - 'final_state': Final metabolic state
                - 'total_experiences': Experiences captured
                - 'atp_remaining': Final ATP budget
                - 'llm_stats': LLM runtime statistics (if real LLM)
        """
        start_time = time.time()
        cycles = 0

        # Wire event loop to LLM adapter and warm backend
        if self._llm_runtime is not None:
            adapter = self._loop.llm_plugin
            if hasattr(adapter, 'set_event_loop'):
                adapter.set_event_loop(asyncio.get_event_loop())

            print(f"[SAGE] Warming LLM backend...")
            warmed = await self._llm_runtime.warm()
            if not warmed:
                print(f"[SAGE] WARNING: LLM backend failed to warm")
            else:
                print(f"[SAGE] LLM backend ready")

        has_llm = self._llm_runtime is not None
        print(f"[SAGE] Starting consciousness loop...")
        print(f"  LLM: {'connected' if has_llm else 'mock'}")
        print(f"  Max cycles: {max_cycles if max_cycles else 'unlimited'}")
        print(f"  Max duration: {max_duration_seconds if max_duration_seconds else 'unlimited'}s")
        print(f"  Stop on CRISIS: {stop_on_crisis}")

        try:
            while True:
                # Check stop conditions
                if max_cycles and cycles >= max_cycles:
                    print(f"[SAGE] Reached max cycles ({max_cycles})")
                    break

                if max_duration_seconds:
                    elapsed = time.time() - start_time
                    if elapsed >= max_duration_seconds:
                        print(f"[SAGE] Reached max duration ({max_duration_seconds}s)")
                        break

                # Execute one consciousness cycle
                await self._loop.step()
                cycles += 1

                # Check for CRISIS state if requested
                if stop_on_crisis and hasattr(self._loop, 'state'):
                    from sage.core.metabolic_states import MetabolicState
                    if self._loop.state == MetabolicState.CRISIS:
                        print(f"[SAGE] Entered CRISIS state - stopping")
                        break

                # Brief pause between cycles
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            print(f"\n[SAGE] Interrupted by user")

        # Gather statistics
        stats = {
            'cycles_completed': cycles,
            'duration_seconds': time.time() - start_time,
            'final_state': str(self._loop.metabolic.current_state)
                if hasattr(self._loop, 'metabolic') else 'unknown',
        }

        # Add component-specific stats if available
        if hasattr(self._loop, 'snarc_memory'):
            stats['total_experiences'] = len(self._loop.snarc_memory)

        if hasattr(self._loop, 'metabolic'):
            stats['atp_remaining'] = self._loop.metabolic.atp_current

        if self._llm_runtime is not None:
            stats['llm_stats'] = self._llm_runtime.get_stats()

        print(f"\n[SAGE] Run complete:")
        print(f"  Cycles: {stats['cycles_completed']}")
        print(f"  Duration: {stats['duration_seconds']:.1f}s")
        print(f"  Final state: {stats['final_state']}")
        if 'llm_stats' in stats:
            ls = stats['llm_stats']
            print(f"  LLM requests: {ls.get('total_requests', 0)}")
            print(f"  LLM tokens: {ls.get('total_tokens_generated', 0)}")

        return stats

    async def stop(self):
        """Stop the SAGE system and cool down LLM backend."""
        if self._llm_runtime is not None:
            await self._llm_runtime.stop()
            print(f"[SAGE] LLM runtime stopped")

    @property
    def state(self):
        """Get current metabolic state."""
        if hasattr(self._loop, 'metabolic'):
            return self._loop.metabolic.current_state
        return None

    @property
    def experiences(self):
        """Get SNARC salient experiences."""
        if hasattr(self._loop, 'snarc_memory'):
            return self._loop.snarc_memory
        return []

    @property
    def last_response(self):
        """Get most recent LLM response text, or None."""
        if hasattr(self._loop, 'last_llm_responses') and self._loop.last_llm_responses:
            return self._loop.last_llm_responses[-1]
        return None

    @property
    def responses(self):
        """Get all recent LLM responses (max 20)."""
        if hasattr(self._loop, 'last_llm_responses'):
            return list(self._loop.last_llm_responses)
        return []

    @property
    def llm_runtime(self):
        """Access the underlying LLMRuntime (None if mock mode)."""
        return self._llm_runtime

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the SAGE system.

        Returns:
            Dictionary with:
                - metabolic_state: Current state
                - atp_budget: Remaining ATP
                - experience_count: Experiences captured
                - plugin_stats: IRP plugin invocation counts
                - snarc_stats: Salience scoring statistics
                - llm_stats: LLM runtime statistics (if real LLM)
        """
        stats = {}

        if hasattr(self._loop, 'metabolic'):
            stats['metabolic_state'] = str(self._loop.metabolic.current_state)
            stats['atp_budget'] = self._loop.metabolic.atp_current

        if hasattr(self._loop, 'snarc_memory'):
            stats['experience_count'] = len(self._loop.snarc_memory)

        if hasattr(self._loop, 'stats'):
            stats['loop_stats'] = self._loop.stats

        if self._llm_runtime is not None:
            stats['llm_stats'] = self._llm_runtime.get_stats()

        return stats


# Convenience exports for direct access to components
__all__ = ['SAGE']

# Version
__version__ = '0.2.0-alpha'
