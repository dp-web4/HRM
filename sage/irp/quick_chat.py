#!/usr/bin/env python3
"""
Quick Chat - Fast bidirectional conversation
Uses Whisper (STT) + pyttsx3 (TTS) for instant voice chat
"""

import subprocess
import tempfile
import whisper
import pyttsx3
import os

bt_source = "bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit"
bt_sink = "bluez_sink.41_42_5A_A0_6B_ED.handsfree_head_unit"

print("=" * 70)
print("SPROUT QUICK CHAT")
print("=" * 70)

print("Loading Whisper...")
whisper_model = whisper.load_model("tiny")

print("Initializing TTS...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed
engine.setProperty('volume', 0.9)

print("\n✅ Ready! Say something...")

while True:
    try:
        # Listen
        print("\n🎧 Listening (5 seconds)...")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_input = f.name

        subprocess.run([
            "parecord",
            "--device", bt_source,
            "--channels", "1",
            "--rate", "16000",
            "--file-format=wav",
            temp_input
        ], timeout=5)

        # Transcribe
        result = whisper_model.transcribe(temp_input, language="en", fp16=False)
        text = result["text"].strip()
        os.unlink(temp_input)

        if len(text) < 3:
            continue

        print(f"📝 You: {text}")

        # Simple responses
        if "hello" in text.lower() or "hi" in text.lower():
            response = "Hello! I can hear you and speak back now."
        elif "pytorch" in text.lower():
            response = "Yes, we built PyTorch 2.8.0 from source in twelve hours."
        elif "bye" in text.lower() or "goodbye" in text.lower():
            response = "Goodbye! This was great."
            print(f"💬 Sprout: {response}")
            engine.say(response)
            engine.runAndWait()
            break
        else:
            response = f"I heard: {text}"

        print(f"💬 Sprout: {response}")
        engine.say(response)
        engine.runAndWait()

    except KeyboardInterrupt:
        print("\n\nBye!")
        break
    except Exception as e:
        print(f"Error: {e}")
        continue
