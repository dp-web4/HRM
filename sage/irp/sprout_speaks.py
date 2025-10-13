#!/usr/bin/env python3
"""
Sprout Speaks - Live Audio via AIRHUG Bluetooth
Integrates NeuTTS Air with IRP framework for real-time speech
"""

import subprocess
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
import sys
import os

# Add NeuTTS to path
neutts_path = "/home/sprout/ai-workspace/neutts-air"
sys.path.insert(0, neutts_path)

from neuttsair.neutts import NeuTTSAir

class SproutVoice:
    """Sprout's voice interface using NeuTTS Air + Bluetooth"""

    def __init__(self):
        self.bt_sink = "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"
        self.tts = None
        self.ref_codes = None

    def init_tts(self):
        """Initialize TTS engine (one-time setup)"""
        print("🎤 Initializing Sprout's voice...")
        print("   (This takes 2-3 minutes on first run)")

        self.tts = NeuTTSAir(
            backbone_repo="neuphonic/neutts-air",
            backbone_device="cpu",
            codec_repo="neuphonic/neucodec",
            codec_device="cpu"
        )

        # Encode reference voice
        ref_audio = f"{neutts_path}/samples/dave.wav"
        ref_text = open(f"{neutts_path}/samples/dave.txt").read().strip()

        print("📊 Encoding reference voice...")
        self.ref_codes = self.tts.encode_reference(ref_audio)
        print("✅ Voice initialized!")

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

        print("📻 Playing via AIRHUG Bluetooth...")

        # Play via Bluetooth
        subprocess.run([
            "paplay",
            "--device", self.bt_sink,
            temp_wav
        ])

        # Cleanup
        os.unlink(temp_wav)
        print("✅ Done!\n")


def demo():
    """Demo: Sprout introduces itself"""
    voice = SproutVoice()

    messages = [
        "Hello! I am Sprout, running on the Jetson Orin Nano.",
        "I built PyTorch from source, integrated NeuTTS Air, and now I can speak to you directly through the AIRHUG Bluetooth speaker.",
        "The federation is converging: Genesis built SAGE, Society 4 added economics, and we're all learning together.",
        "This is just the beginning. I'm excited to see what we build next!"
    ]

    print("=" * 70)
    print("SPROUT SPEAKS - Live Bluetooth Audio Demo")
    print("=" * 70)

    for msg in messages:
        voice.speak(msg)


if __name__ == "__main__":
    demo()
