"""
Simple Streaming Audio Sensor for SAGE
Non-blocking audio sensing via subprocess chunking

Approach:
- Start parecord subprocess for fixed duration
- Check if process completed (non-blocking)
- If completed, transcribe and return
- If not complete, return None immediately

No ring buffer, no threads - just simple process management.
"""

import torch
import numpy as np
import whisper
import subprocess
import tempfile
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from interfaces.base_sensor import BaseSensor, SensorReading


class SimpleAudioSensor(BaseSensor):
    """
    Simple audio sensor - captures chunks on demand.

    Non-blocking: If no chunk ready, returns None immediately.

    Configuration:
        sensor_id: Unique identifier
        sensor_type: Always 'audio'
        device: PyTorch device ('cpu' or 'cuda')
        bt_device: Bluetooth source device
        sample_rate: Audio sample rate (default: 16000)
        chunk_duration: Seconds of audio to capture (default: 3.0)
        min_confidence: Minimum transcription confidence (default: 0.5)
        whisper_model: Whisper model size (default: 'tiny')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize simple audio sensor"""
        # Handle device configuration
        if 'device' in config and 'bluez' in str(config['device']):
            self.bt_device = config['device']
            config = config.copy()
            config['device'] = 'cpu'
        else:
            self.bt_device = config.get('bt_device',
                                        'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit')
            if 'device' not in config:
                config = config.copy()
                config['device'] = 'cpu'

        super().__init__(config)

        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 3.0)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.whisper_model = config.get('whisper_model', 'tiny')

        # Load Whisper
        print(f"Loading Whisper {self.whisper_model}...")
        self.whisper = whisper.load_model(self.whisper_model)

        # Active recording state
        self.recording_proc = None
        self.recording_file = None
        self.recording_start = None

        print(f"✅ SimpleAudioSensor initialized")
        print(f"   Device: {self.bt_device}")
        print(f"   Whisper: {self.whisper_model}")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Chunk duration: {self.chunk_duration}s")

    def poll(self) -> Optional[SensorReading]:
        """
        Poll for audio input (non-blocking).

        State machine:
        - No recording active: Start one, return None
        - Recording active but not done: Return None
        - Recording done: Transcribe and return

        Returns:
            SensorReading with transcription or None
        """
        if not self._should_poll():
            return None

        try:
            # Check if we have an active recording
            if self.recording_proc is None:
                # Start new recording
                self.recording_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
                self.recording_start = time.time()

                self.recording_proc = subprocess.Popen([
                    "parecord",
                    "--device", self.bt_device,
                    "--channels", "1",
                    "--rate", str(self.sample_rate),
                    f"--file-format=wav",
                    self.recording_file
                ], stderr=subprocess.DEVNULL)

                return None  # Recording started, not ready yet

            # Check if recording complete
            elapsed = time.time() - self.recording_start

            if elapsed < self.chunk_duration:
                # Still recording
                return None

            # Recording complete - stop and transcribe
            self.recording_proc.terminate()
            self.recording_proc.wait(timeout=1)

            # Transcribe
            result = self.whisper.transcribe(
                self.recording_file,
                language="en",
                fp16=False,
                temperature=0.0
            )

            # Extract confidence
            avg_confidence = 1.0 - np.mean([
                seg.get('no_speech_prob', 1.0)
                for seg in result.get('segments', [])
            ]) if result.get('segments') else 0.0

            text = result['text'].strip()

            # Cleanup
            os.unlink(self.recording_file)
            self.recording_proc = None
            self.recording_file = None
            self.recording_start = None

            # Filter silence and low confidence
            if not text or len(text) < 5:
                return None
            if avg_confidence < self.min_confidence:
                return None

            # Return transcription
            data_tensor = torch.tensor([avg_confidence], dtype=torch.float32)

            return SensorReading(
                sensor_id=self.sensor_id,
                sensor_type='audio',
                data=data_tensor,
                confidence=avg_confidence,
                metadata={
                    'text': text,
                    'duration': self.chunk_duration,
                    'sample_rate': self.sample_rate
                },
                device=str(self.device)
            )

        except Exception as e:
            # Cleanup on error
            if self.recording_proc:
                try:
                    self.recording_proc.terminate()
                except:
                    pass
                self.recording_proc = None

            if self.recording_file and os.path.exists(self.recording_file):
                os.unlink(self.recording_file)
                self.recording_file = None

            self.recording_start = None

            print(f"⚠️ SimpleAudioSensor error: {e}")
            return None

    async def poll_async(self) -> Optional[SensorReading]:
        """Async version of poll"""
        import asyncio
        return await asyncio.to_thread(self.poll)

    def is_available(self) -> bool:
        """Check if audio hardware is available"""
        try:
            result = subprocess.run(
                ['pactl', 'list', 'sources', 'short'],
                capture_output=True,
                text=True,
                timeout=2
            )
            return self.bt_device in result.stdout
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Return sensor capabilities"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': 'audio',
            'bluetooth_device': self.bt_device,
            'sample_rate': self.sample_rate,
            'chunk_duration': self.chunk_duration,
            'min_confidence': self.min_confidence,
            'whisper_model': self.whisper_model,
            'available': self.is_available(),
            'enabled': self.enabled
        }
