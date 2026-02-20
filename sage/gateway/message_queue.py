"""
Thread-safe message queue for injecting external messages into the SAGE consciousness loop.

The gateway HTTP server runs in a separate thread. Messages submitted via the gateway
are placed in this queue. The consciousness loop polls the queue in step 1 (gather
observations), processes the message through the full pipeline, and resolves the
response Future when the NetworkEffector dispatches the reply.

Architecture:
    HTTP Thread                    Async Consciousness Loop
    ──────────                     ────────────────────────
    submit(msg) ──► Queue ──►      poll() → SensorObservation
         │                              │
    await future                   ... salience, attention, LLM ...
         │                              │
         ◄──────── resolve(id, text) ◄──┘
"""

import asyncio
import queue
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class PendingMessage:
    """A message waiting to be processed by the consciousness loop."""
    message_id: str
    sender: str              # "claude@cbp", "sage@sprout", "dennis"
    content: str             # The message text
    conversation_id: str     # Groups multi-turn exchanges
    timestamp: float
    response_future: asyncio.Future = field(repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str       # "user" or "sage"
    content: str
    timestamp: float
    sender: str = ""


class MessageQueue:
    """
    Thread-safe bridge between the HTTP gateway and the consciousness loop.

    The HTTP thread calls submit() which returns a Future. The consciousness
    loop calls poll() to get pending messages, processes them, and calls
    resolve() when the response is ready. The Future is then resolved,
    unblocking the HTTP handler.
    """

    def __init__(self, max_conversation_turns: int = 20):
        self._pending: queue.Queue = queue.Queue()
        self._waiting: Dict[str, PendingMessage] = {}  # message_id → PendingMessage
        self._conversations: Dict[str, List[ConversationTurn]] = {}
        self._max_turns = max_conversation_turns
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stats = {
            'messages_submitted': 0,
            'messages_resolved': 0,
            'messages_timed_out': 0,
        }

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop for creating Futures. Must be called from the async context."""
        self._loop = loop

    def submit(self, sender: str, content: str,
               conversation_id: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> asyncio.Future:
        """
        Submit a message for processing by the consciousness loop.

        Called from the HTTP thread. Returns a Future that resolves to the
        response dict when the consciousness loop processes the message.

        Args:
            sender: Identity of the sender (e.g., "claude@cbp")
            content: Message text
            conversation_id: Optional conversation ID for multi-turn
            metadata: Optional metadata dict

        Returns:
            Future that resolves to response dict
        """
        if self._loop is None:
            raise RuntimeError("MessageQueue.set_event_loop() must be called before submit()")

        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        if conversation_id is None:
            conversation_id = f"conv_{uuid.uuid4().hex[:8]}"

        future = self._loop.create_future()

        msg = PendingMessage(
            message_id=message_id,
            sender=sender,
            content=content,
            conversation_id=conversation_id,
            timestamp=time.time(),
            response_future=future,
            metadata=metadata or {},
        )

        # Record the user turn in conversation history
        self._record_turn(conversation_id, ConversationTurn(
            role="user",
            content=content,
            timestamp=msg.timestamp,
            sender=sender,
        ))

        self._waiting[message_id] = msg
        self._pending.put(msg)
        self._stats['messages_submitted'] += 1

        return future

    def poll(self) -> Optional[PendingMessage]:
        """
        Non-blocking poll for the next pending message.

        Called by the consciousness loop in step 1 (gather observations).
        Returns None if no messages are waiting.
        """
        try:
            return self._pending.get_nowait()
        except queue.Empty:
            return None

    def poll_all(self) -> List[PendingMessage]:
        """Poll all pending messages (drain the queue)."""
        messages = []
        while True:
            msg = self.poll()
            if msg is None:
                break
            messages.append(msg)
        return messages

    def resolve(self, message_id: str, response: str,
                extra: Optional[Dict[str, Any]] = None):
        """
        Resolve a pending message with a response.

        Called by the NetworkEffector when the consciousness loop has
        produced a response. This resolves the Future, unblocking the
        HTTP handler waiting for the response.

        Args:
            message_id: The message being responded to
            response: SAGE's response text
            extra: Optional extra data (metabolic_state, salience, etc.)
        """
        msg = self._waiting.pop(message_id, None)
        if msg is None:
            return  # Already resolved or timed out

        # Record SAGE's response in conversation history
        self._record_turn(msg.conversation_id, ConversationTurn(
            role="sage",
            content=response,
            timestamp=time.time(),
            sender="sage",
        ))

        result = {
            'response': response,
            'message_id': message_id,
            'conversation_id': msg.conversation_id,
            'timestamp': time.time(),
        }
        if extra:
            result.update(extra)

        # Thread-safe resolution: schedule on the event loop
        self._loop.call_soon_threadsafe(
            self._safe_set_result, msg.response_future, result
        )
        self._stats['messages_resolved'] += 1

    def _safe_set_result(self, future: asyncio.Future, result: Any):
        """Set future result, handling already-cancelled futures."""
        if not future.done():
            future.set_result(result)

    def timeout_message(self, message_id: str, reason: str = "timeout"):
        """Mark a message as timed out. Called by the gateway if max_wait exceeded."""
        msg = self._waiting.pop(message_id, None)
        if msg is None:
            return

        result = {
            'response': None,
            'error': reason,
            'message_id': message_id,
            'conversation_id': msg.conversation_id,
            'timestamp': time.time(),
        }

        self._loop.call_soon_threadsafe(
            self._safe_set_result, msg.response_future, result
        )
        self._stats['messages_timed_out'] += 1

    def get_conversation_history(self, conversation_id: str) -> List[ConversationTurn]:
        """Get conversation history for building LLM context."""
        return list(self._conversations.get(conversation_id, []))

    def _record_turn(self, conversation_id: str, turn: ConversationTurn):
        """Record a turn in conversation history, pruning if needed."""
        if conversation_id not in self._conversations:
            self._conversations[conversation_id] = []
        history = self._conversations[conversation_id]
        history.append(turn)
        # Prune old turns to prevent unbounded growth
        if len(history) > self._max_turns:
            self._conversations[conversation_id] = history[-self._max_turns:]

    @property
    def pending_count(self) -> int:
        """Number of messages waiting to be processed."""
        return self._pending.qsize()

    @property
    def waiting_count(self) -> int:
        """Number of messages waiting for responses."""
        return len(self._waiting)

    @property
    def stats(self) -> Dict[str, int]:
        """Queue statistics."""
        return {**self._stats, 'pending': self.pending_count, 'waiting': self.waiting_count}


if __name__ == "__main__":
    import asyncio

    async def test_message_queue():
        mq = MessageQueue()
        mq.set_event_loop(asyncio.get_event_loop())

        # Submit a message
        future = mq.submit("test@local", "Hello SAGE", "conv_test_001")
        print(f"Submitted message, pending={mq.pending_count}")

        # Poll it (as consciousness loop would)
        msg = mq.poll()
        assert msg is not None
        print(f"Polled: {msg.message_id} from {msg.sender}: '{msg.content}'")
        assert mq.pending_count == 0

        # Resolve it (as NetworkEffector would)
        mq.resolve(msg.message_id, "Hello! I am SAGE.",
                    extra={'metabolic_state': 'wake', 'atp_remaining': 95.0})

        # Check the future resolved
        result = await future
        print(f"Response: {result['response']}")
        print(f"State: {result.get('metabolic_state')}")
        assert result['response'] == "Hello! I am SAGE."

        # Check conversation history
        history = mq.get_conversation_history("conv_test_001")
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "sage"
        print(f"Conversation history: {len(history)} turns")

        # Test timeout
        future2 = mq.submit("test@local", "Are you there?", "conv_test_002")
        msg2 = mq.poll()
        mq.timeout_message(msg2.message_id, "test timeout")
        result2 = await future2
        assert result2['error'] == "test timeout"
        print(f"Timeout handled correctly")

        print(f"\nStats: {mq.stats}")
        print("\nAll MessageQueue tests passed!")

    asyncio.run(test_message_queue())
