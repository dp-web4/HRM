"""
Cognitive Mailbox - Memory-based IPC for Real-Time SAGE Conversation

Uses existing tiling_mailbox infrastructure for <1ms latency communication
between audio sensors, cognitive layer, and TTS effectors.

Architecture:
- PBM (Peripheral Broadcast Mailbox) for text messages
- Fixed 1024-byte records
- Broadcast groups for routing:
  - Group 0: Audio → Cognitive (transcriptions)
  - Group 1: Cognitive → TTS (responses)
- Non-blocking push/pop operations

Performance: ~31μs push, ~4μs pop (vs 100ms file polling)
"""

import torch
import numpy as np
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'implementation'))

# Import existing GPU mailbox implementation
try:
    from tiling_mailbox_torch_extension_v2.tiling_mailbox import TilingMailbox, PBMMode, FTMMode
    MAILBOX_AVAILABLE = True
except ImportError:
    print("Warning: tiling_mailbox not available, using fallback queue")
    MAILBOX_AVAILABLE = False
    import queue


class CognitiveMailbox:
    """
    Memory-based communication hub for SAGE conversation

    Peripheral IDs:
    - 0: Audio sensor (transcription producer)
    - 1: Cognitive layer (transcription consumer, response producer)
    - 2: TTS effector (response consumer)

    Broadcast Groups:
    - 0: Transcriptions (audio → cognitive)
    - 1: Responses (cognitive → TTS)
    """

    # Message record size (fixed for PBM)
    RECORD_SIZE = 1024

    # Peripheral IDs
    AUDIO_SENSOR = 0
    COGNITIVE_LAYER = 1
    TTS_EFFECTOR = 2

    # Broadcast groups
    TRANSCRIPTION_GROUP = 0
    RESPONSE_GROUP = 1

    def __init__(self, use_fallback: bool = False):
        """
        Initialize cognitive mailbox

        Args:
            use_fallback: Force queue-based fallback (for testing without GPU mailbox)
        """
        self.use_mailbox = MAILBOX_AVAILABLE and not use_fallback

        if self.use_mailbox:
            self.mailbox = TilingMailbox(
                capacity=100,         # 100 message slots
                peripheral_count=16,  # Support up to 16 peripherals
                record_size=self.RECORD_SIZE,
                device_str="cpu"      # CPU-only for now
            )
            print("✅ CognitiveMailbox initialized (GPU/CPU mailbox mode)")
        else:
            # Fallback to simple queues
            self.transcription_queue = queue.Queue(maxsize=10)
            self.response_queue = queue.Queue(maxsize=10)
            print("✅ CognitiveMailbox initialized (fallback queue mode)")

    # ==================== Audio Sensor Methods ====================

    def post_transcription(self, text: str, confidence: float, metadata: Optional[Dict] = None):
        """
        Audio sensor posts transcription (non-blocking)

        Args:
            text: Transcribed text
            confidence: Transcription confidence (0-1)
            metadata: Optional metadata dict
        """
        message = {
            'type': 'transcription',
            'text': text,
            'confidence': confidence,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        if self.use_mailbox:
            self._push_message(
                peripheral_id=self.AUDIO_SENSOR,
                broadcast_group=self.TRANSCRIPTION_GROUP,
                message=message
            )
        else:
            try:
                self.transcription_queue.put_nowait(message)
            except queue.Full:
                pass  # Drop if queue full

    def check_transcription(self) -> Optional[Dict]:
        """
        Cognitive layer checks for new transcription (non-blocking)

        Returns:
            Message dict or None if no messages
        """
        if self.use_mailbox:
            return self._pop_message(
                peripheral_id=self.COGNITIVE_LAYER,
                broadcast_group=self.TRANSCRIPTION_GROUP
            )
        else:
            try:
                return self.transcription_queue.get_nowait()
            except queue.Empty:
                return None

    # ==================== Cognitive Layer Methods ====================

    def post_response(self, text: str, metadata: Optional[Dict] = None):
        """
        Cognitive layer posts response (non-blocking)

        Args:
            text: Response text for TTS
            metadata: Optional metadata dict
        """
        message = {
            'type': 'response',
            'text': text,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }

        if self.use_mailbox:
            self._push_message(
                peripheral_id=self.COGNITIVE_LAYER,
                broadcast_group=self.RESPONSE_GROUP,
                message=message
            )
        else:
            try:
                self.response_queue.put_nowait(message)
            except queue.Full:
                pass

    def check_response(self) -> Optional[Dict]:
        """
        TTS effector checks for new response (non-blocking)

        Returns:
            Message dict or None if no messages
        """
        if self.use_mailbox:
            return self._pop_message(
                peripheral_id=self.TTS_EFFECTOR,
                broadcast_group=self.RESPONSE_GROUP
            )
        else:
            try:
                return self.response_queue.get_nowait()
            except queue.Empty:
                return None

    # ==================== Internal Mailbox Operations ====================

    def _push_message(self, peripheral_id: int, broadcast_group: int, message: Dict):
        """Push message to mailbox"""
        # Serialize to JSON and pad to fixed size
        message_json = json.dumps(message, separators=(',', ':'))
        message_bytes = message_json.encode('utf-8')

        if len(message_bytes) > self.RECORD_SIZE:
            # Truncate if too long
            message_bytes = message_bytes[:self.RECORD_SIZE-4] + b'...'

        # Pad to fixed size
        message_bytes = message_bytes.ljust(self.RECORD_SIZE, b'\0')

        # Convert to tensor
        tensor = torch.from_numpy(np.frombuffer(message_bytes, dtype=np.uint8))

        # Push to mailbox
        self.mailbox.pbm_push(
            peripheral_id=peripheral_id,
            broadcast_group=broadcast_group,
            data=tensor
        )

    def _pop_message(self, peripheral_id: int, broadcast_group: int) -> Optional[Dict]:
        """Pop message from mailbox"""
        result = self.mailbox.pbm_pop(
            peripheral_id=peripheral_id,
            broadcast_group=broadcast_group,
            count=1
        )

        if result['data'].numel() == 0:
            return None  # No messages

        # Decode message
        message_bytes = result['data'].numpy().tobytes()
        message_str = message_bytes.decode('utf-8').rstrip('\0')

        try:
            return json.loads(message_str)
        except json.JSONDecodeError:
            print(f"Warning: Failed to decode message: {message_str[:100]}")
            return None

    # ==================== Utility Methods ====================

    def get_stats(self) -> Dict[str, Any]:
        """Get mailbox statistics"""
        if self.use_mailbox:
            return {
                'mode': 'gpu_mailbox',
                'capacity': 100,
                'record_size': self.RECORD_SIZE
            }
        else:
            return {
                'mode': 'fallback_queue',
                'transcription_qsize': self.transcription_queue.qsize(),
                'response_qsize': self.response_queue.qsize()
            }

    def clear(self):
        """Clear all messages"""
        if self.use_mailbox:
            # Pop all messages from both groups
            while self.check_transcription():
                pass
            while self.check_response():
                pass
        else:
            # Clear queues
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.response_queue.empty():
                try:
                    self.response_queue.get_nowait()
                except queue.Empty:
                    break
