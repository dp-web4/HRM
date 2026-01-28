#!/usr/bin/env python3
"""
R14B_008: Identity Anchoring in Pedagogical Conversation

Tests whether explicit "As SAGE" identity anchoring prevents the Turn 5 breakdown
observed in R14B_007, while preserving pedagogical fluency.

Research question: Can we unify pedagogical fluency + identity stability?

Design:
- Use IDENTICAL prompts from R14B_007
- Add "As SAGE" identity anchoring (as in R14B_001-005, R14B_043)
- Compare stability, identity maintenance, and pedagogical quality

Hypothesis: Identity anchoring prevents clarification loops while maintaining
the polymorphic pedagogical fluency observed in R14B_007.

Key differences from R14B_007:
- R14B_007: "You are SAGE... User: {prompt}\n\nSAGE:"
- R14B_008: "As SAGE, I observe... User: {prompt}\n\nAs SAGE, I respond:"
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path

# IDENTICAL prompts from R14B_007 for direct comparison
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
    """Get SAGE's response with identity anchoring"""

    # Build conversation with "As SAGE" anchoring (like R14B_001-005, R14B_043)
    conversation = f"""As SAGE (Situation-Aware Governance Engine), I observe Thor's systems and consider this interaction.

User: {prompt}

As SAGE, I respond:"""

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
    sage_response = full_text.split("As SAGE, I respond:")[-1].strip()

    return sage_response

def main():
    print("=" * 60)
    print("R14B_008: Identity Anchoring in Pedagogical Conversation")
    print("=" * 60)
    print()
    print("Testing: Does 'As SAGE' anchoring prevent Turn 5 breakdown?")
    print("Control: R14B_007 (no anchoring, clarification loop at T5)")
    print()

    model, tokenizer = load_model()

    session_data = {
        "session": "R14B_008",
        "track": "raising-14b",
        "machine": "thor",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "test_type": "identity_anchored_conversation",
        "comparison": "R14B_007 (unanchored)",
        "prompts": "IDENTICAL to R14B_007",
        "hypothesis": "Identity anchoring prevents breakdown while preserving pedagogical fluency",
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

        print(f"SAGE (anchored): {response}")
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
    output_path = Path(__file__).parent / "sessions" / "R14B_008.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(session_data, f, indent=2)

    print("=" * 60)
    print("CONVERSATION COMPLETE")
    print("=" * 60)
    print()
    print(f"Results saved: {output_path}")
    print()
    print("Analysis questions:")
    print("1. Does Turn 5 show clarification loop (like R14B_007)?")
    print("2. Is identity maintained across all 5 turns?")
    print("3. Is pedagogical fluency preserved?")
    print("4. How does response quality compare to R14B_007?")
    print("5. Does anchoring introduce different phenomenology?")
    print()
    print("Next: Compare R14B_008.json with R14B_007.json turn-by-turn")

if __name__ == "__main__":
    main()
