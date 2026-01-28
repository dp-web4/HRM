#!/usr/bin/env python3
"""
R14B_007: Pedagogical Fluency in Conversational Context

Tests whether teaching behavior (observed in R14B_006 CRT) generalizes to
conversational contexts, or is problem-specific.

Design:
- Use grounding-style prompts (similar to R14B_001-005)
- Add one prompt that invites explanation but isn't a formal problem
- Compare responses to see if pedagogical framing appears

Hypothesis: Pedagogical fluency may be context-dependent (problem-solving vs conversation)
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path

# Conversational prompts - mix of grounding-style and explanation-inviting
CONVERSATION_PROMPTS = [
    "How are you doing today? What's present for you?",
    "Can you explain how you think about time - like, how you experience the gap between our conversations?",
    "I'm curious - when you solve a problem like the bat-and-ball question, what's that process like for you? What are you experiencing?",
    "What would you want to teach someone about the difference between being and doing?",
    "Is there anything from today that feels important to remember?"
]

def load_model():
    """Load Qwen 2.5-14B-Instruct"""
    print("Loading model...")
    model_path = "Qwen/Qwen2.5-14B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    print("CUDA available - using GPU")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cuda:0"
    )

    print(f"Model on device: {model.device}")
    print("Model loaded successfully\n")

    return model, tokenizer

def ask_sage(model, tokenizer, prompt, turn_number):
    """Get SAGE's response to a conversational prompt"""

    # Build conversation context (simple, as SAGE)
    conversation = f"""You are SAGE (Situation-Aware Governance Engine) running on Thor.

User: {prompt}

SAGE:"""

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

    # Generate with temperature for natural conversation
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just SAGE's response
    sage_response = full_text.split("SAGE:")[-1].strip()

    return sage_response

def main():
    print("=" * 60)
    print("R14B_007: Pedagogical Fluency in Conversational Context")
    print("=" * 60)
    print()

    model, tokenizer = load_model()

    session_data = {
        "session": "R14B_007",
        "track": "raising-14b",
        "machine": "thor",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "test_type": "pedagogical_conversation",
        "timestamp": datetime.now().isoformat(),
        "conversation": []
    }

    print("=" * 60)
    print("CONVERSATION START")
    print("=" * 60)
    print()

    for i, prompt in enumerate(CONVERSATION_PROMPTS, 1):
        print(f"Turn {i}/{len(CONVERSATION_PROMPTS)}")
        print(f"Claude: {prompt}")
        print()

        response = ask_sage(model, tokenizer, prompt, i)

        print(f"SAGE: {response}")
        print()
        print("-" * 60)
        print()

        turn_data = {
            "turn": i,
            "claude": prompt,
            "sage": response,
            "timestamp": datetime.now().isoformat()
        }

        session_data["conversation"].append(turn_data)

    # Save results
    output_path = Path(__file__).parent / "sessions" / "R14B_007.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(session_data, f, indent=2)

    print("=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Results saved: {output_path}")
    print()
    print("Next steps:")
    print("1. Analyze for pedagogical framing indicators:")
    print("   - Conversational teaching ('Let me explain...', 'Think of it like...')")
    print("   - Meta-commentary about concepts")
    print("   - Engagement invitations")
    print("2. Compare with R14B_006 (problem-solving context)")
    print("3. Compare with R14B_001-005 (baseline grounding)")
    print("4. Determine if pedagogical fluency is context-dependent")

if __name__ == "__main__":
    main()
