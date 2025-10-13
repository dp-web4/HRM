#!/usr/bin/env python3
"""
Sprout Conversation - Bidirectional Voice Chat
Speech-to-text (Whisper) + Text-to-speech (NeuTTS) via Bluetooth
"""

import subprocess
import tempfile
import soundfile as sf
import whisper
import numpy as np
from pathlib import Path
import sys
import os

# Add NeuTTS to path
neutts_path = "/home/sprout/ai-workspace/neutts-air"
sys.path.insert(0, neutts_path)

from neuttsair.neutts import NeuTTSAir

class SproutConversation:
    """Bidirectional voice conversation with Sprout"""

    def __init__(self):
        self.bt_sink = "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"
        self.bt_source = "bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit"
        self.tts = None
        self.ref_codes = None
        self.whisper_model = None

    def init_tts(self):
        """Initialize TTS engine"""
        print("🎤 Initializing Sprout's voice (TTS)...")
        self.tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )

        ref_audio = f"{neutts_path}/samples/dave.wav"
        ref_text = open(f"{neutts_path}/samples/dave.txt").read().strip()
        self.ref_codes = self.tts.encode_reference(ref_audio)
        print("✅ TTS initialized!")

    def init_stt(self):
        """Initialize speech-to-text"""
        print("👂 Loading Whisper (STT)...")
        self.whisper_model = whisper.load_model("tiny")  # Fast model for Jetson
        print("✅ STT initialized!")

    def listen(self, duration=5):
        """Record audio from Bluetooth mic and transcribe"""
        print(f"\n🎧 Listening for {duration} seconds...")

        # Record from Bluetooth mic
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_input = f.name

        subprocess.run([
            "parecord",
            "--device", self.bt_source,
            "--channels", "1",
            "--rate", "16000",
            f"--file-format=wav",
            temp_input
        ], timeout=duration)

        # Transcribe with Whisper
        print("🔍 Transcribing...")
        result = self.whisper_model.transcribe(temp_input, language="en")
        text = result["text"].strip()

        os.unlink(temp_input)
        print(f"📝 You said: {text}")
        return text

    def speak(self, text: str):
        """Generate speech and play via Bluetooth"""
        if self.tts is None:
            self.init_tts()

        print(f"\n💬 Sprout: {text}")
        print("🔊 Generating speech...")

        # Generate audio
        ref_text = open(f"{neutts_path}/samples/dave.txt").read().strip()
        wav = self.tts.infer(text, self.ref_codes, ref_text)

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_wav = f.name
            sf.write(temp_wav, wav, 24000)

        print("📻 Playing via Bluetooth...")
        subprocess.run(["paplay", "--device", self.bt_sink, temp_wav])

        os.unlink(temp_wav)
        print("✅ Done!")

    def chat(self):
        """Interactive chat loop"""
        if self.whisper_model is None:
            self.init_stt()
        if self.tts is None:
            self.init_tts()

        print("\n" + "=" * 70)
        print("SPROUT CONVERSATION - Press Ctrl+C to exit")
        print("=" * 70)

        self.speak("Hello! I'm ready to chat. Just start talking when you see the listening prompt.")

        while True:
            try:
                # Listen
                user_text = self.listen(duration=10)

                if not user_text or len(user_text) < 5:
                    continue

                # Simple responses (later integrate with SAGE)
                if "hello" in user_text.lower() or "hi" in user_text.lower():
                    response = "Hello! It's great to hear your voice directly through the AIRHUG speaker."
                elif "pytorch" in user_text.lower():
                    response = "Yes, we built PyTorch 2.8.0 from source. It took 12 and a half hours, but now I have full CUDA support!"
                elif "sage" in user_text.lower() or "federation" in user_text.lower():
                    response = "The federation is converging beautifully. Genesis built SAGE, Society 4 added the economic layer with ATP and ADP tracking."
                elif "thank" in user_text.lower():
                    response = "You're welcome! I'm learning so much from this collaboration."
                else:
                    response = f"I heard you say: {user_text}. I'm still learning how to respond thoughtfully to everything."

                self.speak(response)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                self.speak("Goodbye! This was a great conversation.")
                break


if __name__ == "__main__":
    conv = SproutConversation()
    conv.chat()
