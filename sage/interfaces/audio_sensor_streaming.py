"""
Streaming Audio Input Sensor for SAGE
Continuous listening with non-blocking polling

Key difference from recording-based approach:
- RECORDING: Start recording → wait → stop → process (BLOCKING)
- LISTENING: Always listening in background → poll checks buffer → process if ready (NON-BLOCKING)

If speech is missed while processing, it's missed. We're sensing, not logging.
"""

import torch
import numpy as np
import whisper
import pyaudio
import threading
import queue
import time
import tempfile
import os
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import soundfile as sf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from interfaces.base_sensor import BaseSensor, SensorReading


@dataclass
class AudioChunk:
    """A chunk of audio from the continuous stream"""
    samples: np.ndarray
    sample_rate: int
    timestamp: float


class StreamingAudioListener:
    """
    Background thread that continuously records audio into a buffer.

    Uses PyAudio for proper streaming audio capture.
    """

    def __init__(self, device_name: str, sample_rate: int, chunk_duration: float):
        self.device_name = device_name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration

        # Ring buffer for audio chunks
        self.buffer = queue.Queue(maxsize=10)  # Keep last 10 chunks

        # Set PulseAudio device via environment
        os.environ['PULSE_SOURCE'] = device_name

        # PyAudio setup - use 'pulse' device
        self.audio = pyaudio.PyAudio()
        self.stream = None

        # Find PulseAudio device index
        self.device_index = self._find_pulse_device()
        if self.device_index is None:
            # Fall back to default
            self.device_index = None
            print(f"[WARNING] Using default audio device (PulseAudio not found)")

    def _find_pulse_device(self) -> Optional[int]:
        """Find PulseAudio device index"""
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            if 'pulse' in info['name'].lower():
                return i
        return None

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback - called automatically when audio is ready"""
        if status:
            print(f"[DEBUG] Audio status: {status}")

        # Convert bytes to float32
        samples = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0

        chunk = AudioChunk(
            samples=samples,
            sample_rate=self.sample_rate,
            timestamp=time.time()
        )

        # Add to buffer (drop oldest if full)
        try:
            self.buffer.put_nowait(chunk)
        except queue.Full:
            # Drop oldest chunk
            try:
                self.buffer.get_nowait()
                self.buffer.put_nowait(chunk)
            except:
                pass

        return (in_data, pyaudio.paContinue)

    def start(self):
        """Start listening"""
        if self.stream is not None:
            return

        chunk_frames = int(self.sample_rate * self.chunk_duration)

        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            input_device_index=self.device_index,
            frames_per_buffer=chunk_frames,
            stream_callback=self._audio_callback
        )

        self.stream.start_stream()
        print(f"🎧 Started streaming audio from {self.device_name}")

    def stop(self):
        """Stop listening"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None

    def get_recent_audio(self, duration: float) -> Optional[np.ndarray]:
        """
        Get recent audio from buffer (non-blocking).

        Args:
            duration: Seconds of audio to retrieve

        Returns:
            Audio samples or None if not enough data
        """
        chunks = []
        total_duration = 0.0

        # Drain buffer (non-blocking)
        while not self.buffer.empty() and total_duration < duration:
            try:
                chunk = self.buffer.get_nowait()
                chunks.append(chunk.samples)
                total_duration += len(chunk.samples) / chunk.sample_rate
            except queue.Empty:
                break

        if not chunks:
            return None

        # Concatenate chunks
        return np.concatenate(chunks)

    def __del__(self):
        """Cleanup"""
        self.stop()
        if self.audio:
            self.audio.terminate()


class StreamingAudioSensor(BaseSensor):
    """
    Streaming audio input sensor with continuous listening.

    Difference from AudioInputSensor:
    - Always listening in background
    - poll() just checks buffer and processes if ready
    - Non-blocking - returns None immediately if no speech ready
    - Missed audio is lost (sensing, not recording)

    Configuration:
        sensor_id: Unique identifier
        sensor_type: Always 'audio'
        device: PyTorch device ('cpu' or 'cuda')
        bt_device: Bluetooth source device
        sample_rate: Audio sample rate (default: 16000)
        chunk_duration: Stream chunk size in seconds (default: 1.0)
        buffer_duration: How much audio to accumulate before transcribing (default: 3.0)
        min_confidence: Minimum transcription confidence (default: 0.5)
        whisper_model: Whisper model size (default: 'tiny')
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize streaming audio sensor"""
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
        self.chunk_duration = config.get('chunk_duration', 1.0)
        self.buffer_duration = config.get('buffer_duration', 3.0)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.whisper_model = config.get('whisper_model', 'tiny')

        # Load Whisper
        print(f"Loading Whisper {self.whisper_model}...")
        self.whisper = whisper.load_model(self.whisper_model)

        # Start streaming listener
        self.listener = StreamingAudioListener(
            device_name=self.bt_device,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration
        )
        self.listener.start()

        # Transcription state
        self.last_transcription_time = 0.0
        self.transcription_cooldown = 2.0  # Don't transcribe more than every 2 seconds

        print(f"✅ StreamingAudioSensor initialized")
        print(f"   Device: {self.bt_device}")
        print(f"   Whisper: {self.whisper_model}")
        print(f"   Sample rate: {self.sample_rate} Hz")
        print(f"   Streaming continuously in background")

    def __del__(self):
        """Stop listener on cleanup"""
        if hasattr(self, 'listener'):
            self.listener.stop()

    def poll(self) -> Optional[SensorReading]:
        """
        Poll for audio input (non-blocking).

        Checks if enough audio has accumulated in buffer.
        If yes, transcribes and returns. If no, returns None immediately.

        Returns:
            SensorReading with transcription or None
        """
        if not self._should_poll():
            return None

        # Cooldown to avoid transcribing too frequently
        now = time.time()
        if now - self.last_transcription_time < self.transcription_cooldown:
            return None

        try:
            # Get recent audio from buffer (non-blocking)
            audio = self.listener.get_recent_audio(self.buffer_duration)

            if audio is None or len(audio) < self.sample_rate * 1.0:
                # Not enough audio yet
                return None

            # Save to temp file for Whisper
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_file = f.name

            try:
                sf.write(temp_file, audio, self.sample_rate)

                # Transcribe
                result = self.whisper.transcribe(
                    temp_file,
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

                # Update cooldown
                self.last_transcription_time = now

                # Cleanup
                os.unlink(temp_file)

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
                        'duration': len(audio) / self.sample_rate,
                        'sample_rate': self.sample_rate
                    },
                    device=str(self.device)
                )

            except Exception as e:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                raise e

        except Exception as e:
            print(f"⚠️ StreamingAudioSensor poll error: {e}")
            return None

    async def poll_async(self) -> Optional[SensorReading]:
        """Async version of poll"""
        import asyncio
        return await asyncio.to_thread(self.poll)

    def is_available(self) -> bool:
        """Check if audio hardware is available"""
        import subprocess
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
            'buffer_duration': self.buffer_duration,
            'min_confidence': self.min_confidence,
            'whisper_model': self.whisper_model,
            'available': self.is_available(),
            'enabled': self.enabled,
            'streaming': self.listener.running
        }
