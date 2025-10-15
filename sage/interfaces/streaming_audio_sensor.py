"""
Real-Time Streaming Audio Sensor with Voice Activity Detection

Architecture:
- Continuous PyAudio stream (10ms chunks)
- WebRTC VAD for speech detection
- faster-whisper for low-latency transcription
- Ring buffer for context
- Non-blocking poll() returns immediately

Target latency: <500ms from speech end to transcription
"""

import torch
import numpy as np
import webrtcvad
import queue
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any
from collections import deque

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from interfaces.base_sensor import BaseSensor, SensorReading


class StreamingAudioSensor(BaseSensor):
    """
    Real-time audio streaming with Voice Activity Detection

    Key features:
    - Continuous audio stream (no chunking delay)
    - VAD detects speech boundaries automatically
    - faster-whisper for 3x speed improvement
    - Non-blocking poll() - returns None if no transcription ready
    - Minimal latency: ~300-500ms from speech end to text

    Configuration:
        sensor_id: Unique identifier
        sensor_type: Always 'audio'
        device: PyTorch device ('cpu' or 'cuda')
        bt_device: Bluetooth source device name
        sample_rate: Audio sample rate (must be 8000, 16000, 32000, or 48000 for VAD)
        vad_aggressiveness: VAD sensitivity 0-3 (3 = most aggressive filtering)
        min_speech_duration: Minimum speech length to transcribe (seconds)
        max_speech_duration: Maximum speech length before force transcribe (seconds)
        padding_duration: Silence padding around speech (seconds)
        min_confidence: Minimum transcription confidence to return
        whisper_model: faster-whisper model size ('tiny', 'base', 'small')
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

        # Audio configuration
        self.sample_rate = config.get('sample_rate', 16000)
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError(f"VAD requires sample_rate in [8000, 16000, 32000, 48000], got {self.sample_rate}")

        self.vad_aggressiveness = config.get('vad_aggressiveness', 2)
        self.min_speech_duration = config.get('min_speech_duration', 0.5)
        self.max_speech_duration = config.get('max_speech_duration', 10.0)
        self.padding_duration = config.get('padding_duration', 0.3)
        self.min_confidence = config.get('min_confidence', 0.5)
        self.whisper_model = config.get('whisper_model', 'tiny')

        # Calculate frame parameters for VAD
        # VAD requires frames of 10, 20, or 30ms
        self.frame_duration_ms = 30  # 30ms frames for balance
        self.frame_size = int(self.sample_rate * self.frame_duration_ms / 1000)
        self.num_padding_frames = int(self.padding_duration / (self.frame_duration_ms / 1000))

        print(f"Initializing StreamingAudioSensor...")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Frame size: {self.frame_size} samples ({self.frame_duration_ms}ms)")
        print(f"  VAD aggressiveness: {self.vad_aggressiveness}")

        # Initialize VAD
        self.vad = webrtcvad.Vad(self.vad_aggressiveness)

        # Initialize faster-whisper
        print(f"  Loading faster-whisper ({self.whisper_model})...")
        from faster_whisper import WhisperModel
        self.whisper = WhisperModel(
            self.whisper_model,
            device="cpu",
            compute_type="int8"  # Quantized for speed
        )

        # Audio stream state (parecord-based)
        self.ring_buffer = deque(maxlen=self.num_padding_frames)
        self.triggered = False
        self.voiced_frames = []
        self.speech_start_time = None

        # Transcription queue (background thread puts results here)
        self.transcription_queue = queue.Queue(maxsize=5)
        self.transcription_lock = threading.Lock()

        # Start audio stream
        self._start_stream()

        print(f"✅ StreamingAudioSensor initialized")
        print(f"   Listening continuously on {self.bt_device}")

    def _start_stream(self):
        """Start parecord streaming in background thread"""
        import subprocess
        import tempfile

        # Create temp file for continuous recording
        self.recording_file = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
        self.recording_filename = self.recording_file.name
        self.recording_file.close()

        # Start parecord subprocess
        self.parecord_process = subprocess.Popen([
            "parecord",
            "--device", self.bt_device,
            "--channels", "1",
            "--rate", str(self.sample_rate),
            "--format=s16le",
            self.recording_filename
        ], stderr=subprocess.DEVNULL)

        # Start background thread to read and process audio
        self.running = True
        import threading
        self.processing_thread = threading.Thread(target=self._process_audio_loop, daemon=True)
        self.processing_thread.start()

        print(f"  Audio stream started (parecord subprocess)")

    def _process_audio_loop(self):
        """
        Background thread that reads from parecord and processes with VAD

        Reads frames continuously from file, runs VAD, and queues transcriptions
        """
        import os

        frame_bytes = self.frame_size * 2  # 2 bytes per int16 sample
        file_position = 0
        frames_processed = 0

        print(f"  [VAD] Processing thread started, looking for {self.recording_filename}")

        while self.running:
            try:
                # Wait for file to have data
                if not os.path.exists(self.recording_filename):
                    time.sleep(0.01)
                    continue

                # Read next frame
                with open(self.recording_filename, 'rb') as f:
                    f.seek(file_position)
                    in_data = f.read(frame_bytes)

                if len(in_data) < frame_bytes:
                    # Not enough data yet, wait
                    time.sleep(self.frame_duration_ms / 1000.0 / 2)  # Half frame duration
                    continue

                file_position += frame_bytes
                frames_processed += 1

                # Convert bytes to int16 samples
                frame = np.frombuffer(in_data, dtype=np.int16)

                # VAD check
                is_speech = self.vad.is_speech(in_data, self.sample_rate)

                # Debug: print every 100 frames (~3 seconds)
                if frames_processed % 100 == 0:
                    print(f"  [VAD] Processed {frames_processed} frames, speech={is_speech}, triggered={self.triggered}")

                if not self.triggered:
                    # Not in speech - buffer frames for padding
                    self.ring_buffer.append((frame, is_speech))
                    num_voiced = sum(1 for f, speech in self.ring_buffer if speech)

                    # Trigger on 90% voiced frames in buffer
                    if num_voiced > 0.9 * self.ring_buffer.maxlen:
                        self.triggered = True
                        self.speech_start_time = time.time()
                        # Include ring buffer (padding before speech)
                        self.voiced_frames = [f for f, speech in self.ring_buffer]
                        self.ring_buffer.clear()
                        print(f"  [VAD] 🎤 Speech detected! Starting transcription...")

                else:
                    # In speech - accumulate frames
                    self.voiced_frames.append(frame)
                    self.ring_buffer.append((frame, is_speech))

                    # Calculate speech duration
                    speech_duration = len(self.voiced_frames) * self.frame_duration_ms / 1000.0

                    num_unvoiced = sum(1 for f, speech in self.ring_buffer if not speech)

                    # End of speech: 90% silence in ring buffer OR max duration reached
                    if num_unvoiced > 0.9 * self.ring_buffer.maxlen or speech_duration >= self.max_speech_duration:
                        # Check minimum duration
                        if speech_duration >= self.min_speech_duration:
                            print(f"  [VAD] 🛑 Speech ended ({speech_duration:.2f}s), queuing transcription...")
                            # Queue for transcription
                            self._queue_transcription()
                        else:
                            print(f"  [VAD] ⏭️  Speech too short ({speech_duration:.2f}s < {self.min_speech_duration}s), skipping")

                        # Reset state
                        self.triggered = False
                        self.voiced_frames = []
                        self.ring_buffer.clear()

            except Exception as e:
                if self.running:
                    print(f"  [VAD] Processing error: {e}")
                time.sleep(0.1)

    def _queue_transcription(self):
        """Queue speech for background transcription"""
        if not self.voiced_frames:
            return

        # Concatenate frames
        audio_data = np.concatenate(self.voiced_frames)

        # Convert int16 to float32 for faster-whisper
        audio_float = audio_data.astype(np.float32) / 32768.0

        # Submit to background thread
        threading.Thread(
            target=self._transcribe_audio,
            args=(audio_float,),
            daemon=True
        ).start()

    def _transcribe_audio(self, audio: np.ndarray):
        """Background transcription thread"""
        try:
            start_time = time.time()

            # Run faster-whisper
            segments, info = self.whisper.transcribe(
                audio,
                language="en",
                beam_size=1,  # Faster
                vad_filter=False,  # We already did VAD
                condition_on_previous_text=False  # Independent chunks
            )

            # Collect segments
            text_parts = []
            avg_confidence = []

            for segment in segments:
                text_parts.append(segment.text)
                # faster-whisper doesn't provide no_speech_prob directly
                # Use average log probability as proxy for confidence
                confidence = np.exp(segment.avg_logprob) if hasattr(segment, 'avg_logprob') else 0.7
                avg_confidence.append(confidence)

            text = " ".join(text_parts).strip()
            confidence = np.mean(avg_confidence) if avg_confidence else 0.0

            transcription_time = time.time() - start_time

            # Queue result if meets confidence threshold
            if text and confidence >= self.min_confidence:
                try:
                    self.transcription_queue.put_nowait({
                        'text': text,
                        'confidence': confidence,
                        'duration': len(audio) / self.sample_rate,
                        'transcription_time': transcription_time
                    })
                except queue.Full:
                    print(f"  [VAD] Transcription queue full, dropping result")

        except Exception as e:
            print(f"  [VAD] Transcription error: {e}")

    def poll(self) -> Optional[SensorReading]:
        """
        Non-blocking poll for transcription results

        Returns:
            SensorReading if transcription ready, else None
        """
        if not self._should_poll():
            return None

        try:
            # Check queue (non-blocking)
            result = self.transcription_queue.get_nowait()

            # Create sensor reading
            data_tensor = torch.tensor([result['confidence']], dtype=torch.float32)

            return SensorReading(
                sensor_id=self.sensor_id,
                sensor_type='audio',
                data=data_tensor,
                confidence=result['confidence'],
                metadata={
                    'text': result['text'],
                    'duration': result['duration'],
                    'transcription_time': result['transcription_time'],
                    'sample_rate': self.sample_rate
                },
                device=str(self.device)
            )

        except queue.Empty:
            return None

    def __del__(self):
        """Cleanup on destruction"""
        self.running = False

        # Terminate parecord subprocess
        if hasattr(self, 'parecord_process'):
            self.parecord_process.terminate()
            try:
                self.parecord_process.wait(timeout=2)
            except:
                self.parecord_process.kill()

        # Delete temp file
        import os
        if hasattr(self, 'recording_filename') and os.path.exists(self.recording_filename):
            try:
                os.unlink(self.recording_filename)
            except:
                pass

    def is_available(self) -> bool:
        """Check if audio hardware available"""
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
        """Return sensor information"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': 'audio',
            'bluetooth_device': self.bt_device,
            'sample_rate': self.sample_rate,
            'vad_aggressiveness': self.vad_aggressiveness,
            'min_speech_duration': self.min_speech_duration,
            'max_speech_duration': self.max_speech_duration,
            'whisper_model': self.whisper_model,
            'available': self.is_available(),
            'enabled': self.enabled,
            'streaming': self.running if hasattr(self, 'running') else False
        }
