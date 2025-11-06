#!/usr/bin/env python3
"""
Audio Sensor - Multi-Backend Audio Input
========================================

Provides audio input with multiple backend support:
1. Real microphone (PyAudio) - For Jetson Nano with physical mic
2. Synthetic audio - For testing without hardware

All backends produce standardized output encoded via AudioPuzzleVAE.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import time

# Try importing backends
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


class AudioBackend(Enum):
    """Available audio backends"""
    AUTO = "auto"
    PYAUDIO = "pyaudio"     # Real microphone
    SYNTHETIC = "synthetic"  # Generated test data


@dataclass
class AudioChunk:
    """Raw audio chunk data"""
    waveform: torch.Tensor  # [samples] mono audio
    timestamp: float
    sample_rate: int
    backend: AudioBackend
    metadata: Dict[str, Any]


class AudioSensor:
    """
    Multi-backend audio sensor for SAGE audio input.

    Captures 1-second audio chunks at 16kHz sample rate.
    Automatically selects best available backend.
    """

    def __init__(
        self,
        backend: str = "auto",
        device: str = "cuda",
        sample_rate: int = 16000,
        chunk_duration: float = 1.0,
        channels: int = 1,
    ):
        """
        Initialize audio sensor.

        Args:
            backend: Backend to use ('auto', 'pyaudio', 'synthetic')
            device: Device for tensor operations
            sample_rate: Audio sample rate (Hz)
            chunk_duration: Duration of each audio chunk (seconds)
            channels: Number of audio channels (1=mono, 2=stereo)
        """
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.channels = channels
        self.chunk_samples = int(sample_rate * chunk_duration)

        # Select backend
        self.backend = self._select_backend(backend)
        print(f"🎤 Audio sensor initialized with backend: {self.backend.value}")

        # Initialize backend
        self._audio = None
        self._stream = None
        self._init_backend()

        # Performance tracking
        self._chunk_count = 0
        self._total_latency = 0.0

    def _select_backend(self, backend: str) -> AudioBackend:
        """Select best available backend"""
        if backend == "auto":
            if PYAUDIO_AVAILABLE:
                return AudioBackend.PYAUDIO
            print("⚠️  PyAudio not available, using synthetic mode")
            return AudioBackend.SYNTHETIC

        backend_map = {
            "pyaudio": AudioBackend.PYAUDIO,
            "synthetic": AudioBackend.SYNTHETIC,
        }
        return backend_map.get(backend, AudioBackend.SYNTHETIC)

    def _init_backend(self):
        """Initialize selected backend"""
        if self.backend == AudioBackend.PYAUDIO:
            if not PYAUDIO_AVAILABLE:
                raise RuntimeError("PyAudio not available")

            self._audio = pyaudio.PyAudio()

            # Find default input device
            try:
                device_info = self._audio.get_default_input_device_info()
                print(f"✓ Audio device: {device_info['name']}")
            except Exception as e:
                print(f"⚠️  No audio input device found: {e}")
                print("   Falling back to synthetic mode")
                self.backend = AudioBackend.SYNTHETIC
                return

            # Open audio stream
            try:
                self._stream = self._audio.open(
                    format=pyaudio.paFloat32,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_samples,
                )
                print(f"✓ Audio stream opened at {self.sample_rate}Hz")
            except Exception as e:
                print(f"⚠️  Failed to open audio stream: {e}")
                print("   Falling back to synthetic mode")
                self.backend = AudioBackend.SYNTHETIC

        elif self.backend == AudioBackend.SYNTHETIC:
            print("✓ Synthetic audio mode")

    def capture(self) -> Optional[AudioChunk]:
        """
        Capture audio chunk (1 second at 16kHz).

        Returns:
            AudioChunk with [16000] waveform tensor, or None if failed
        """
        start_time = time.time()

        # Capture from backend
        if self.backend == AudioBackend.PYAUDIO:
            chunk = self._capture_pyaudio()
        else:
            chunk = self._capture_synthetic()

        if chunk is None:
            return None

        # Track performance
        latency = time.time() - start_time
        self._chunk_count += 1
        self._total_latency += latency

        chunk.metadata['latency_ms'] = latency * 1000
        chunk.metadata['chunk_count'] = self._chunk_count

        return chunk

    def _capture_pyaudio(self) -> Optional[AudioChunk]:
        """Capture audio from PyAudio microphone"""
        try:
            # Read audio chunk
            data = self._stream.read(self.chunk_samples, exception_on_overflow=False)

            # Convert to numpy array
            audio = np.frombuffer(data, dtype=np.float32)

            # Convert to mono if stereo
            if self.channels == 2:
                audio = audio.reshape(-1, 2).mean(axis=1)

            # Convert to tensor
            waveform = torch.from_numpy(audio).float()
            waveform = waveform.to(self.device)

            return AudioChunk(
                waveform=waveform,
                timestamp=time.time(),
                sample_rate=self.sample_rate,
                backend=self.backend,
                metadata={
                    'channels': self.channels,
                    'duration': self.chunk_duration,
                }
            )

        except Exception as e:
            print(f"⚠️  Audio capture failed: {e}")
            return None

    def _capture_synthetic(self) -> Optional[AudioChunk]:
        """Generate synthetic audio for testing"""
        # Create synthetic audio with temporal variation
        t = time.time()

        # Time array
        time_array = torch.linspace(0, self.chunk_duration, self.chunk_samples, device=self.device)

        # Multiple sine waves with time-varying frequencies
        freq1 = 440.0 + 50.0 * np.sin(t * 0.5)  # A4 note wobbling
        freq2 = 554.37 + 30.0 * np.cos(t * 0.7)  # C#5 note wobbling

        waveform = (
            0.3 * torch.sin(2 * np.pi * freq1 * time_array) +
            0.2 * torch.sin(2 * np.pi * freq2 * time_array) +
            0.1 * torch.randn(self.chunk_samples, device=self.device)  # Noise
        )

        # Normalize
        waveform = waveform / (torch.abs(waveform).max() + 1e-6)

        return AudioChunk(
            waveform=waveform,
            timestamp=t,
            sample_rate=self.sample_rate,
            backend=self.backend,
            metadata={
                'synthetic': True,
                'pattern': 'sine_waves',
            }
        )

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self._chunk_count == 0:
            return {
                'avg_latency_ms': 0.0,
                'chunks_captured': 0,
                'sample_rate': self.sample_rate,
            }

        avg_latency = self._total_latency / self._chunk_count

        return {
            'avg_latency_ms': avg_latency * 1000,
            'chunks_captured': self._chunk_count,
            'sample_rate': self.sample_rate,
            'backend': self.backend.value,
        }

    def close(self):
        """Release audio resources"""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
        if self._audio is not None:
            self._audio.terminate()
        print("✓ Audio sensor released")

    def __del__(self):
        """Cleanup on deletion"""
        self.close()


if __name__ == "__main__":
    # Test audio sensor
    print("Testing Audio Sensor\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = AudioSensor(backend="auto", device=device)

    print(f"\nCapturing 5 test chunks...")
    for i in range(5):
        chunk = audio.capture()
        if chunk is not None:
            print(f"Chunk {i+1}: {chunk.waveform.shape}, "
                  f"latency: {chunk.metadata.get('latency_ms', 0):.2f}ms, "
                  f"RMS: {torch.sqrt(torch.mean(chunk.waveform**2)):.4f}")

    stats = audio.get_stats()
    print(f"\n📊 Performance Stats:")
    print(f"   Backend: {stats['backend']}")
    print(f"   Chunks: {stats['chunks_captured']}")
    print(f"   Sample rate: {stats['sample_rate']}Hz")
    print(f"   Avg latency: {stats['avg_latency_ms']:.2f}ms")

    audio.close()
