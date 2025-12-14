#!/usr/bin/env python3
"""
Test Qwen3-Omni-30B Model

Interactive conversation to test the omni-modal model's capabilities.
This will verify text generation works before testing audio/video.
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def test_qwen3_omni():
    """Test Qwen3-Omni-30B with conversation."""

    print("=" * 70)
    print("Testing Qwen3-Omni-30B")
    print("=" * 70)
    print()

    model_path = Path("model-zoo/sage/omni-modal/qwen3-omni-30b")

    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return False

    print("Loading Qwen3-Omni-30B...")
    print("This is a 66GB model, will take a moment...")
    print()

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path),
            trust_remote_code=True
        )
        print(f"✅ Tokenizer loaded ({len(tokenizer)} tokens)")

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ Model loaded")
        print()

        # Conversation 1: Introduction
        print("Conversation Test 1:")
        print("-" * 70)
        print()
        print("User: Hello! I'm testing your omni-modal capabilities. Can you introduce yourself?")
        print()

        prompt1 = "User: Hello! I'm testing your omni-modal capabilities. Can you introduce yourself?\n\nAssistant:"

        inputs1 = tokenizer(prompt1, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs1 = model.generate(
                **inputs1,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                top_p=0.9
            )

        response1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
        response1 = response1.split("Assistant:")[-1].strip()

        print(f"Qwen3-Omni: {response1}")
        print()
        print("-" * 70)
        print()

        # Conversation 2: Capabilities
        print("Conversation Test 2:")
        print("-" * 70)
        print()
        print("User: What makes you different from text-only models like the 14B version?")
        print()

        prompt2 = f"""Previous conversation:
User: Hello! I'm testing your omni-modal capabilities. Can you introduce yourself?
Assistant: {response1}

User: What makes you different from text-only models like the 14B version?