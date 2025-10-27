"""
TTS Effector - Piper Text-to-Speech Integration

Implements speech synthesis as a SAGE effector using Piper TTS.
Non-blocking synthesis with subprocess management for real-time performance.
"""

import subprocess
import threading
import time
from typing import Dict, Any, Optional
from pathlib import Path


class TTSEffector:
    """
    Piper TTS effector for speech synthesis

    Features:
    - Non-blocking synthesis (subprocess + threading)
    - Automatic process cleanup
    - Bluetooth audio output
    - Configurable voice models
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TTS effector

        Args:
            config: Configuration including:
                - piper_path: Path to piper executable
                - model_path: Path to voice model (.onnx)
                - bt_sink: Bluetooth audio sink device
                - sample_rate: Audio sample rate (default: 22050)
                - enabled: Enable/disable TTS (default: True)
        """
        self.piper_path = config.get('piper_path', '/home/sprout/ai-workspace/piper/piper/piper')
        self.model_path = config.get('model_path', '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx')
        self.bt_sink = config.get('bt_sink', 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit')
        self.sample_rate = config.get('sample_rate', 22050)
        self.enabled = config.get('enabled', True)

        # Validate paths
        if not Path(self.piper_path).exists():
            print(f"⚠️  TTS: Piper executable not found at {self.piper_path}")
            self.enabled = False

        if not Path(self.model_path).exists():
            print(f"⚠️  TTS: Voice model not found at {self.model_path}")
            self.enabled = False

        # Track active processes for cleanup
        self.active_processes = []
        self.lock = threading.Lock()

        # Statistics
        self.synthesis_count = 0
        self.total_synthesis_time = 0.0
        self.errors = 0

        if self.enabled:
            print(f"✓ TTSEffector initialized")
            print(f"  Piper: {self.piper_path}")
            print(f"  Model: {Path(self.model_path).name}")
            print(f"  Output: {self.bt_sink}")
        else:
            print(f"⚠️  TTSEffector disabled (missing dependencies)")

    def execute(
        self,
        text: str,
        priority: float = 1.0,
        metadata: Optional[Dict] = None,
        blocking: bool = True,
        prosody_hints: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Synthesize and play text with optional prosody hints

        Args:
            text: Text to synthesize
            priority: Priority level (unused for now, future feature)
            metadata: Optional metadata for telemetry
            blocking: Wait for playback to complete (default: True)
            prosody_hints: Optional prosodic hints from ProsodicChunk:
                - pause_after: Pause duration in ms (200-800ms)
                - boundary_tone: Intonation pattern ("L-L%", "L-H%")
                - pitch_reset: Whether to reset pitch
                - boundary_type: Type of boundary ("IP", "ip", "NATURAL", "BREATH")

        Returns:
            True if synthesis started successfully
        """
        if not self.enabled:
            return False

        if not text or not text.strip():
            return False

        try:
            start_time = time.time()

            # Apply prosodic pauses if hints provided
            if prosody_hints:
                text = self._apply_prosodic_pauses(text, prosody_hints)

            # Merge prosody hints into metadata for logging
            if metadata is None:
                metadata = {}
            if prosody_hints:
                metadata['prosody_hints'] = prosody_hints

            if blocking:
                # Blocking mode - wait for playback to complete
                self._synthesize_async(text, start_time, metadata)
            else:
                # Non-blocking mode - spawn background thread
                thread = threading.Thread(
                    target=self._synthesize_async,
                    args=(text, start_time, metadata),
                    daemon=True
                )
                thread.start()

            return True

        except Exception as e:
            self.errors += 1
            print(f"TTS execute error: {e}")
            return False

    def _apply_prosodic_pauses(self, text: str, prosody_hints: Dict[str, Any]) -> str:
        """
        Apply prosodic pauses to text for more natural speech.

        Piper doesn't support full SSML, so we use simple text manipulation:
        - Add periods for long pauses (IP boundaries)
        - Add commas for medium pauses (ip boundaries)
        - Add ellipsis for continuations

        Future: Generate full SSML when TTS supports it.

        Args:
            text: Original text
            prosody_hints: Prosodic hints dictionary

        Returns:
            Text with prosodic markers
        """
        boundary_type = prosody_hints.get('boundary_type', 'UNKNOWN')
        pause_after = prosody_hints.get('pause_after', 200)  # ms
        continuation = not prosody_hints.get('pitch_reset', False)

        # Already has punctuation - don't modify
        if text.strip().endswith(('.', '!', '?', ',', ';', ':')):
            return text

        # Add appropriate pause marker based on boundary type
        if boundary_type == 'IP':
            # Intonational phrase - add period for final fall
            return text.strip() + '.'
        elif boundary_type == 'ip':
            # Intermediate phrase - add comma for continuation
            return text.strip() + ','
        elif continuation:
            # Continuation - add ellipsis for rising intonation
            return text.strip() + '...'
        else:
            # Default - add period
            return text.strip() + '.'

    def generate_ssml(self, text: str, prosody_hints: Dict[str, Any]) -> str:
        """
        Generate SSML markup for prosodic speech synthesis.

        For future TTS systems that support SSML (Speech Synthesis Markup Language).
        Currently not used by Piper, but ready for systems like Azure TTS, Google TTS, etc.

        Args:
            text: Text to synthesize
            prosody_hints: Prosodic hints from ProsodicChunk

        Returns:
            SSML markup string

        Example output:
        ```xml
        <speak>
            <prosody pitch="+0%" rate="100%">
                As an AI designed to explore consciousness
            </prosody>
            <break time="400ms"/>
        </speak>
        ```
        """
        boundary_type = prosody_hints.get('boundary_type', 'UNKNOWN')
        pause_after = prosody_hints.get('pause_after', 200)  # ms
        boundary_tone = prosody_hints.get('boundary_tone', 'L-L%')
        pitch_reset = prosody_hints.get('pitch_reset', True)

        # Map boundary tones to pitch modifications
        pitch_map = {
            'L-L%': '-5%',   # Final fall (lower pitch)
            'L-H%': '+5%',   # Continuation rise (higher pitch)
            'H-H%': '+10%',  # High continuation (emphasis)
            'H-L%': '+0%',   # High to low (neutral)
        }
        pitch_mod = pitch_map.get(boundary_tone, '+0%')

        # Build SSML
        ssml_parts = ['<speak>']

        # Add prosody element if pitch modification needed
        if pitch_mod != '+0%':
            ssml_parts.append(f'<prosody pitch="{pitch_mod}" rate="100%">')
            ssml_parts.append(text.strip())
            ssml_parts.append('</prosody>')
        else:
            ssml_parts.append(text.strip())

        # Add pause based on boundary type
        if pause_after > 0:
            ssml_parts.append(f'<break time="{pause_after}ms"/>')

        # Add pitch reset marker if needed
        if pitch_reset:
            ssml_parts.append('<prosody pitch="+0%"/>')  # Reset to neutral

        ssml_parts.append('</speak>')

        return ''.join(ssml_parts)

    def _synthesize_async(self, text: str, start_time: float, metadata: Optional[Dict]):
        """
        Perform synthesis in background thread

        Args:
            text: Text to synthesize
            start_time: Start timestamp for telemetry
            metadata: Optional metadata
        """
        try:
            # Create Piper process (text → raw PCM)
            piper_proc = subprocess.Popen(
                [self.piper_path, "--model", self.model_path, "--output_raw"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            # Create paplay process (raw PCM → audio output)
            play_proc = subprocess.Popen(
                ["paplay",
                 "--device", self.bt_sink,
                 "--rate", str(self.sample_rate),
                 "--format", "s16le",
                 "--channels", "1",
                 "--raw"],
                stdin=piper_proc.stdout,
                stderr=subprocess.DEVNULL
            )

            # Track processes
            with self.lock:
                self.active_processes.append((piper_proc, play_proc))

            # Send text to Piper
            piper_proc.stdin.write(text.encode('utf-8'))
            piper_proc.stdin.close()

            # Wait for completion (with timeout)
            try:
                play_proc.wait(timeout=30)
                synthesis_time = time.time() - start_time

                # Update statistics
                with self.lock:
                    self.synthesis_count += 1
                    self.total_synthesis_time += synthesis_time

                # Print telemetry
                print(f"🔊 TTS: \"{text[:50]}...\" ({synthesis_time*1000:.0f}ms)")

            except subprocess.TimeoutExpired:
                # Timeout - kill processes
                piper_proc.kill()
                play_proc.kill()
                self.errors += 1
                print(f"TTS timeout after 30s")

            # Cleanup
            with self.lock:
                if (piper_proc, play_proc) in self.active_processes:
                    self.active_processes.remove((piper_proc, play_proc))

        except Exception as e:
            self.errors += 1
            print(f"TTS synthesis error: {e}")

    def execute_batch(self, texts: list[str], metadata: Optional[Dict] = None) -> int:
        """
        Synthesize multiple texts sequentially

        Args:
            texts: List of texts to synthesize
            metadata: Optional metadata

        Returns:
            Number of texts successfully queued
        """
        count = 0
        for text in texts:
            if self.execute(text, metadata=metadata):
                count += 1
        return count

    def stop_all(self):
        """Stop all active synthesis processes"""
        with self.lock:
            for piper_proc, play_proc in self.active_processes:
                try:
                    piper_proc.terminate()
                    play_proc.terminate()
                except:
                    pass
            self.active_processes.clear()

    def is_available(self) -> bool:
        """Check if TTS is available"""
        return self.enabled

    def get_stats(self) -> Dict[str, Any]:
        """Get effector statistics"""
        with self.lock:
            avg_time = (self.total_synthesis_time / self.synthesis_count
                       if self.synthesis_count > 0 else 0.0)

            return {
                'enabled': self.enabled,
                'synthesis_count': self.synthesis_count,
                'total_time': self.total_synthesis_time,
                'avg_time_ms': avg_time * 1000,
                'errors': self.errors,
                'active_processes': len(self.active_processes)
            }

    def __del__(self):
        """Cleanup on destruction"""
        self.stop_all()


# Test the effector
if __name__ == "__main__":
    print("="*60)
    print("Testing TTSEffector")
    print("="*60)

    # Create effector
    config = {
        'piper_path': '/home/sprout/ai-workspace/piper/piper/piper',
        'model_path': '/home/sprout/ai-workspace/piper/en_US-lessac-medium.onnx',
        'bt_sink': 'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit',
        'enabled': True
    }

    effector = TTSEffector(config)

    if not effector.is_available():
        print("\n⚠️  TTS not available - check configuration")
        exit(1)

    # Test single synthesis
    print("\nTest 1: Single synthesis")
    success = effector.execute("Hello! This is a test of the text to speech system.")

    if success:
        print("✓ Synthesis queued")
        time.sleep(5)  # Wait for completion
    else:
        print("✗ Synthesis failed")

    # Test multiple phrases
    print("\nTest 2: Multiple phrases")
    phrases = [
        "Testing phrase one.",
        "This is phrase number two.",
        "And finally, phrase three."
    ]

    queued = effector.execute_batch(phrases)
    print(f"✓ Queued {queued}/{len(phrases)} phrases")
    time.sleep(10)  # Wait for all to complete

    # Statistics
    print("\n" + "="*60)
    print("TTS Statistics:")
    stats = effector.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\n✓ Test complete")
