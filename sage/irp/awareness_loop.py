#!/usr/bin/env python3
"""
SAGE Awareness Loop - Bidirectional Audio Conversation
Integrates AudioInputIRP + NeuTTSAirIRP with SAGE consciousness

Architecture:
- AudioInput continuously monitors for speech
- Transcriptions feed SAGE consciousness with full context
- SAGE processes input with memory, reasoning, and awareness
- Responses generated through language processing
- AudioOutput speaks back via NeuTTS + Bluetooth
- Loop continues indefinitely

This makes audio a first-class sensory stream, not a bolt-on feature.
"""

import asyncio
import subprocess
import tempfile
import soundfile as sf
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import sys
import os

# Add paths for imports
current_dir = os.path.dirname(__file__)
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'plugins'))

from base import IRPState

# Import plugins directly to avoid __init__.py conflicts
import importlib.util
spec_audio = importlib.util.spec_from_file_location(
    "audio_input_impl",
    os.path.join(current_dir, "plugins/audio_input_impl.py")
)
audio_module = importlib.util.module_from_spec(spec_audio)
spec_audio.loader.exec_module(audio_module)
AudioInputIRP = audio_module.AudioInputIRP

spec_tts = importlib.util.spec_from_file_location(
    "neutts_air_impl",
    os.path.join(current_dir, "plugins/neutts_air_impl.py")
)
tts_module = importlib.util.module_from_spec(spec_tts)
spec_tts.loader.exec_module(tts_module)
NeuTTSAirIRP = tts_module.NeuTTSAirIRP


@dataclass
class ConversationContext:
    """Context maintained across conversation turns"""
    history: List[Dict[str, str]]  # User/assistant exchanges
    current_topic: Optional[str] = None
    user_name: Optional[str] = None
    session_start: float = 0.0
    turn_count: int = 0


class SproutAwarenessLoop:
    """
    Continuous awareness loop for bidirectional conversation.

    Audio becomes a first-class sensory stream:
    - AudioInput continuously monitors for speech
    - Transcriptions feed SAGE consciousness
    - SAGE processes with full context (memory, vision, etc.)
    - Responses generated through language processing
    - AudioOutput speaks back through Bluetooth
    - Loop continues indefinitely
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize awareness loop with audio I/O.

        Config:
            - bluetooth_sink: Bluetooth speaker device
            - bluetooth_source: Bluetooth microphone device
            - whisper_model: Whisper model size
            - neutts_device: Device for TTS (cpu/cuda)
            - ref_audio_path: Reference voice for TTS
            - min_input_confidence: Minimum confidence for transcription
            - response_style: Conversational style (casual/formal/technical)
        """
        self.config = config

        # Bluetooth devices
        self.bt_sink = config.get('bluetooth_sink',
                                   'bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit')
        self.bt_source = config.get('bluetooth_source',
                                     'bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit')

        # Initialize audio input plugin
        audio_input_config = {
            'entity_id': 'awareness_audio_input',
            'device': self.bt_source,
            'sample_rate': 16000,
            'chunk_duration': 2.0,
            'min_confidence': config.get('min_input_confidence', 0.6),
            'max_duration': 10.0,
            'whisper_model': config.get('whisper_model', 'tiny')
        }
        self.audio_input = AudioInputIRP(audio_input_config)

        # Initialize TTS output plugin
        tts_config = {
            'entity_id': 'awareness_tts_output',
            'backbone_repo': 'neuphonic/neutts-air-q4-gguf',
            'codec_repo': 'neuphonic/neucodec',
            'device': config.get('neutts_device', 'cpu'),
            'sample_rate': 24000,
            'ref_audio_path': config.get('ref_audio_path',
                                         '/home/sprout/ai-workspace/neutts-air/samples/dave.wav')
        }
        self.audio_output = NeuTTSAirIRP(tts_config)

        # Conversation context
        self.context = ConversationContext(
            history=[],
            session_start=time.time()
        )

        # Response style
        self.response_style = config.get('response_style', 'casual')

        print("✅ SAGE Awareness Loop initialized")
        print(f"   Input: {self.bt_source}")
        print(f"   Output: {self.bt_sink}")
        print(f"   Whisper: {config.get('whisper_model', 'tiny')}")
        print(f"   TTS: NeuTTS Air on {config.get('neutts_device', 'cpu')}")

    async def listen(self) -> Optional[Dict[str, Any]]:
        """
        Listen for user speech using AudioInputIRP.

        Returns:
            Transcription result or None if silence/error
        """
        print("\n🎧 Listening...")

        # Initialize listening state
        state = self.audio_input.init_state(
            x0=None,
            task_ctx={'prompt': 'Listen for user speech'}
        )

        history = [state]

        # Refinement loop - accumulate audio until confident
        while not self.audio_input.halt(history):
            state = self.audio_input.step(state)
            state.energy_val = self.audio_input.energy(state)
            history.append(state)

            # Show live progress
            audio_state = state.x
            if audio_state.transcription:
                print(f"   📝 \"{audio_state.transcription}\" (conf: {audio_state.confidence:.2f})")

        # Extract final transcription
        result = self.audio_input.extract(state)
        halt_reason = self.audio_input.get_halt_reason(history)

        print(f"   ✅ Done: {halt_reason}")

        # Filter out silence or low confidence
        if not result['text'] or len(result['text']) < 5:
            return None
        if result['confidence'] < 0.4:
            print(f"   ⚠️ Low confidence ({result['confidence']:.2f}), ignoring")
            return None

        return result

    def process_with_sage(self, user_input: str) -> str:
        """
        Process user input through SAGE consciousness.

        In full implementation, this would:
        - Query memory for relevant context
        - Run reasoning modules
        - Apply entity trust weights
        - Generate contextual response

        For now, provides simple conversational responses.

        Args:
            user_input: Transcribed user speech

        Returns:
            Generated response text
        """
        # Simple response logic (will be replaced with SAGE integration)
        user_lower = user_input.lower()

        # Greetings
        if any(word in user_lower for word in ['hello', 'hi', 'hey']):
            responses = [
                "Hello! I can hear you clearly through the AIRHUG microphone.",
                "Hi! The audio integration is working beautifully.",
                "Hey there! I'm listening and ready to chat."
            ]
            import random
            return random.choice(responses)

        # Questions about system
        if 'pytorch' in user_lower:
            return "Yes! We built PyTorch 2.8.0 from source in 12.5 hours with full CUDA 12.6 support. It was worth the wait."

        if any(word in user_lower for word in ['sage', 'architecture', 'hrm']):
            return "SAGE is built on HRM's hierarchical architecture. Audio is now part of the awareness loop, not just a feature bolted on."

        if 'federation' in user_lower:
            return "The federation is converging beautifully. Genesis built SAGE, Society 4 added ATP and ADP tracking, and Sprout brings edge deployment."

        if any(word in user_lower for word in ['audio', 'hear', 'listen']):
            return "Audio input and output are now first-class sensory modalities. This conversation itself is proof!"

        if any(word in user_lower for word in ['thank', 'thanks']):
            return "You're welcome! I'm learning so much from our collaboration."

        if any(word in user_lower for word in ['bye', 'goodbye', 'exit']):
            return "Goodbye! This has been a great conversation. Talk to you soon!"

        # Default reflective response
        return f"I heard you say: {user_input}. I'm still learning to respond thoughtfully to everything, but the awareness loop is working!"

    async def speak(self, text: str):
        """
        Speak response using NeuTTSAirIRP and play via Bluetooth.

        Args:
            text: Text to synthesize and speak
        """
        print(f"\n💬 Sprout: {text}")
        print("   🔊 Generating speech...")

        # Initialize TTS state
        tts_input = {
            'text': text,
            'ref_audio': self.config.get('ref_audio_path'),
            'ref_text': "So I'm live on radio."
        }

        state = self.audio_output.init_state(
            x0=tts_input,
            task_ctx={'voice_id': 'dave', 'prosody': {'speed': 1.0}}
        )

        # Run refinement loop
        max_iterations = 3  # Quick iterations for responsiveness
        for i in range(max_iterations):
            state, budget_used = self.audio_output.step(state, budget=10.0)
            energy = self.audio_output.energy(state)

            # Early stopping if quality is good
            if energy < 0.3:
                break

        # Extract final audio
        result = self.audio_output.extract(state)
        audio = result['audio']
        sample_rate = result['sample_rate']

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
            sf.write(temp_wav, audio, sample_rate)

        print("   📻 Playing via Bluetooth...")

        # Play through Bluetooth speaker
        subprocess.run([
            "paplay",
            "--device", self.bt_sink,
            temp_wav
        ], check=False)

        # Cleanup
        os.unlink(temp_wav)
        print("   ✅ Playback complete")

    async def conversation_turn(self):
        """
        Execute one turn of bidirectional conversation:
        1. Listen for user input
        2. Process through SAGE
        3. Speak response
        """
        # Listen
        listen_result = await self.listen()

        if listen_result is None:
            return False  # Silence detected, continue listening

        user_text = listen_result['text']
        confidence = listen_result['confidence']

        print(f"\n📝 User ({confidence:.2f}): {user_text}")

        # Update context
        self.context.turn_count += 1
        self.context.history.append({
            'role': 'user',
            'content': user_text,
            'confidence': confidence,
            'timestamp': time.time()
        })

        # Check for exit command
        if any(word in user_text.lower() for word in ['bye', 'goodbye', 'exit', 'quit']):
            response = "Goodbye! This was a great conversation."
            await self.speak(response)
            return True  # Signal to exit

        # Process through SAGE
        response = self.process_with_sage(user_text)

        # Update context
        self.context.history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': time.time()
        })

        # Speak response
        await self.speak(response)

        return False  # Continue conversation

    async def run(self):
        """
        Main awareness loop - continuous bidirectional conversation.

        Runs indefinitely until exit command or KeyboardInterrupt.
        """
        print("\n" + "=" * 70)
        print("SAGE AWARENESS LOOP - Bidirectional Conversation")
        print("=" * 70)
        print("\nPress Ctrl+C to exit")
        print("Say 'goodbye' to exit gracefully\n")

        # Welcome message
        await self.speak("Hello! I'm ready to chat. The awareness loop is active and listening.")

        try:
            while True:
                should_exit = await self.conversation_turn()
                if should_exit:
                    break

                # Brief pause between turns
                await asyncio.sleep(0.5)

        except KeyboardInterrupt:
            print("\n\n⏸️ Interrupted by user")
            await self.speak("Goodbye! Talk to you soon.")

        finally:
            # Print session summary
            duration = time.time() - self.context.session_start
            print("\n" + "=" * 70)
            print("SESSION SUMMARY")
            print("=" * 70)
            print(f"Duration: {duration:.1f}s")
            print(f"Turns: {self.context.turn_count}")
            print(f"Exchanges: {len(self.context.history) // 2}")
            print("=" * 70 + "\n")


async def main():
    """Run the awareness loop"""
    config = {
        'whisper_model': 'tiny',          # Fast model for Jetson
        'neutts_device': 'cpu',           # CPU for now (GPU hangs)
        'min_input_confidence': 0.5,      # Accept moderate confidence
        'response_style': 'casual',       # Conversational tone
        'ref_audio_path': '/home/sprout/ai-workspace/neutts-air/samples/dave.wav'
    }

    loop = SproutAwarenessLoop(config)
    await loop.run()


if __name__ == "__main__":
    asyncio.run(main())
