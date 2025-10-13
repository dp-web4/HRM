#!/usr/bin/env python3
"""
Simple listening test - just record and transcribe
No TTS - just prove we can hear you
"""

import subprocess
import tempfile
import whisper
import os

bt_source = "bluez_source.41_42_5A_A0_6B_ED.handsfree_head_unit"

print("=" * 70)
print("LISTENING TEST - Can Sprout hear you?")
print("=" * 70)

print("\nLoading Whisper tiny model...")
model = whisper.load_model("tiny")
print("✅ Whisper loaded!\n")

print("Recording 10 seconds from the Bluetooth mic...")
print("Start talking NOW!\n")

# Record
print("🔴 Recording for 10 seconds...")
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    temp_file = f.name

try:
    subprocess.run([
        "parecord",
        "--device", bt_source,
        "--channels", "1",
        "--rate", "16000",
        "--file-format=wav",
        temp_file
    ], timeout=10)

    print("✅ Recording complete!")
    print("\n🔍 Transcribing...")

    result = model.transcribe(temp_file, language="en", fp16=False)
    text = result["text"].strip()

    print("\n" + "=" * 70)
    print("📝 YOU SAID:")
    print(text)
    print("=" * 70)

    os.unlink(temp_file)

except Exception as e:
    print(f"\n❌ Error: {e}")
    if os.path.exists(temp_file):
        os.unlink(temp_file)
