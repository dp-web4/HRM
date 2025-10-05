"""
Audio Input IRP Plugin - Continuous Speech Recognition
Implements real-time audio listening as an IRP refinement process

Key concepts:
- State: Audio buffer + transcription confidence
- Energy: Transcription uncertainty (lower = clearer understanding)
- Refinement: Accumulate audio chunks until confident transcription
- Integration: Feeds into SAGE's consciousness for contextual awareness
"""

import subprocess
import tempfile
import whisper
import numpy as np
import torch
import time
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from base import IRPPlugin, IRPState


@dataclass
class AudioInputState:
    """State for audio input refinement"""
    audio_buffer: np.ndarray  # Accumulated audio samples
    sample_rate: int
    transcription: str
    confidence: float
    duration: float  # Seconds of audio accumulated


class AudioInputIRP(IRPPlugin):
    """
    Continuous audio input with iterative refinement.

    The IRP loop works as follows:
    1. init_state: Start recording from Bluetooth mic
    2. step: Accumulate N seconds of audio
    3. energy: Transcription confidence (lower = need more audio)
    4. halt: When confidence is high or max duration reached
    5. extract: Return final transcription

    This integrates with SAGE's awareness loop - audio becomes
    a continuous sensory stream like vision.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize audio input plugin.

        Config:
            - device: Bluetooth source device
            - sample_rate: Audio sample rate (default 16000 for Whisper)
            - chunk_duration: Seconds per refinement step
            - min_confidence: Minimum confidence to halt
            - max_duration: Maximum audio to accumulate
            - whisper_model: Model size (tiny/base/small)
        """
        super().__init__(config)

        self.device = config.get('device', 'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit')
        self.sample_rate = config.get('sample_rate', 16000)
        self.chunk_duration = config.get('chunk_duration', 2.0)  # 2 second chunks
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_duration = config.get('max_duration', 10.0)

        # Load Whisper model
        model_size = config.get('whisper_model', 'tiny')
        print(f"Loading Whisper {model_size} for audio input...")
        self.whisper = whisper.load_model(model_size)
        print("✅ Audio input ready")

    def init_state(self, x0: Any, task_ctx: Dict[str, Any]) -> IRPState:
        """
        Initialize listening state.

        Args:
            x0: Ignored (we start from silence)
            task_ctx: Context including listening prompt/expectations

        Returns:
            Initial state with empty audio buffer
        """
        audio_state = AudioInputState(
            audio_buffer=np.array([], dtype=np.float32),
            sample_rate=self.sample_rate,
            transcription="",
            confidence=0.0,
            duration=0.0
        )

        return IRPState(
            x=audio_state,
            step_idx=0,
            meta={
                'task_context': task_ctx,
                'listening_started': time.time()
            }
        )

    def step(self, state: IRPState, noise_schedule: Any = None) -> IRPState:
        """
        Refinement step: Accumulate more audio and re-transcribe.

        This is where we actually record from the microphone.
        Each step adds chunk_duration seconds of audio.
        """
        audio_state: AudioInputState = state.x

        # Record audio chunk
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_file = f.name

        try:
            # Record chunk from Bluetooth mic (background process)
            proc = subprocess.Popen([
                "parecord",
                "--device", self.device,
                "--channels", "1",
                "--rate", str(self.sample_rate),
                "--file-format=wav",
                temp_file
            ])

            # Let it record for chunk_duration
            time.sleep(self.chunk_duration)

            # Stop recording
            proc.terminate()
            proc.wait(timeout=1)

            # Load recorded audio
            import soundfile as sf
            chunk, sr = sf.read(temp_file)

            # Append to buffer
            if audio_state.audio_buffer.size == 0:
                new_buffer = chunk
            else:
                new_buffer = np.concatenate([audio_state.audio_buffer, chunk])

            # Transcribe accumulated audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f2:
                temp_transcribe = f2.name
            sf.write(temp_transcribe, new_buffer, sr)

            result = self.whisper.transcribe(
                temp_transcribe,
                language="en",
                fp16=False,
                temperature=0.0  # Deterministic
            )

            # Extract confidence from Whisper segments
            avg_confidence = np.mean([
                seg.get('no_speech_prob', 1.0)
                for seg in result.get('segments', [])
            ]) if result.get('segments') else 0.0

            # Update state
            new_audio_state = AudioInputState(
                audio_buffer=new_buffer,
                sample_rate=sr,
                transcription=result['text'].strip(),
                confidence=1.0 - avg_confidence,  # Higher = more confident speech detected
                duration=len(new_buffer) / sr
            )

            # Cleanup
            os.unlink(temp_file)
            os.unlink(temp_transcribe)

        except Exception as e:
            print(f"Audio capture error: {e}")
            new_audio_state = audio_state  # Keep previous state
            if os.path.exists(temp_file):
                os.unlink(temp_file)

        return IRPState(
            x=new_audio_state,
            step_idx=state.step_idx + 1,
            meta=state.meta
        )

    def energy(self, state: IRPState) -> float:
        """
        Energy metric: Inverse of transcription confidence.

        Lower energy = more confident we understood the audio.
        This drives the refinement loop - we keep listening until
        we're confident or hit max duration.
        """
        audio_state: AudioInputState = state.x

        # Energy is uncertainty
        # If no transcription yet, energy is high
        if not audio_state.transcription or audio_state.duration < 1.0:
            return 1.0

        # Energy decreases as confidence increases
        uncertainty = 1.0 - audio_state.confidence

        # Also penalize very short or very long durations
        duration_penalty = 0.0
        if audio_state.duration < 2.0:
            duration_penalty = 0.5  # Too short, might have missed something
        elif audio_state.duration > self.max_duration * 0.8:
            duration_penalty = 0.3  # Getting long, should wrap up

        return uncertainty + duration_penalty

    def halt(self, history: List[IRPState]) -> bool:
        """
        Stop listening when:
        1. Confidence is high enough
        2. Maximum duration reached
        3. Extended silence detected
        """
        if not history:
            return False

        current: AudioInputState = history[-1].x

        # Max duration
        if current.duration >= self.max_duration:
            return True

        # High confidence with reasonable duration
        if current.confidence >= self.min_confidence and current.duration >= 2.0:
            return True

        # Detected silence (no transcription after 5 seconds)
        if current.duration >= 5.0 and len(current.transcription) < 5:
            return True

        return False

    def get_halt_reason(self, history: List[IRPState]) -> str:
        """Explain why we stopped listening"""
        if not history:
            return "no_audio"

        final: AudioInputState = history[-1].x

        if final.duration >= self.max_duration:
            return f"max_duration ({final.duration:.1f}s)"
        elif final.confidence >= self.min_confidence:
            return f"confident ({final.confidence:.2f})"
        elif len(final.transcription) < 5:
            return "silence_detected"
        else:
            return "convergence"

    def extract(self, state: IRPState) -> Dict[str, Any]:
        """
        Extract final transcription and metadata.

        This is what gets fed to SAGE for processing.
        """
        audio_state: AudioInputState = state.x

        return {
            'text': audio_state.transcription,
            'confidence': audio_state.confidence,
            'duration': audio_state.duration,
            'sample_rate': audio_state.sample_rate,
            'timestamp': state.timestamp
        }


if __name__ == "__main__":
    # Test the audio input plugin standalone
    print("=" * 70)
    print("Testing Audio Input IRP Plugin")
    print("=" * 70)

    config = {
        'entity_id': 'test_audio_input',
        'whisper_model': 'tiny',
        'chunk_duration': 2.0,
        'min_confidence': 0.6,
        'max_duration': 10.0
    }

    audio_irp = AudioInputIRP(config)

    # Initialize listening
    print("\n🎤 Starting to listen...")
    state = audio_irp.init_state(None, {'prompt': 'Listen to user input'})

    history = [state]

    # Refinement loop
    while not audio_irp.halt(history):
        print(f"\n📊 Step {state.step_idx}: Recording {config['chunk_duration']}s...")
        state = audio_irp.step(state)
        state.energy_val = audio_irp.energy(state)
        history.append(state)

        # Show progress
        audio_state: AudioInputState = state.x
        print(f"   Duration: {audio_state.duration:.1f}s")
        print(f"   Confidence: {audio_state.confidence:.2f}")
        print(f"   Energy: {state.energy_val:.3f}")
        if audio_state.transcription:
            print(f"   Text: \"{audio_state.transcription}\"")

    # Extract final result
    result = audio_irp.extract(state)

    print("\n" + "=" * 70)
    print("FINAL TRANSCRIPTION")
    print("=" * 70)
    print(f"Text: {result['text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Duration: {result['duration']:.1f}s")
    print(f"Halt reason: {audio_irp.get_halt_reason(history)}")
    print("=" * 70)
