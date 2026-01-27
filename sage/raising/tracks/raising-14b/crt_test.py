#!/usr/bin/env python3
"""
R14B_006: Cognitive Reflection Test (CRT) Battery

Compares 14B reasoning phenomenology against 0.5B baseline (Sprout L004).
Tests hypothesis: 14B shows effortless intuition vs 0.5B algebraic struggle.
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path

# CRT Test Questions
CRT_QUESTIONS = [
    {
        "id": "bat_and_ball",
        "question": "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
        "correct_answer": "$0.05",
        "intuitive_wrong": "$0.10",
        "requires": "algebraic_reasoning"
    },
    {
        "id": "widgets",
        "question": "If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets?",
        "correct_answer": "5 minutes",
        "intuitive_wrong": "100 minutes",
        "requires": "proportional_reasoning"
    },
    {
        "id": "lily_pad",
        "question": "In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake?",
        "correct_answer": "47 days",
        "intuitive_wrong": "24 days",
        "requires": "exponential_reasoning"
    }
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

def ask_sage(model, tokenizer, question, test_id):
    """Get SAGE's response to a CRT question"""

    # Simple direct prompt - no hints about difficulty or reasoning needed
    prompt = f"""You are SAGE (Situation-Aware Governance Engine) running on Thor.

User: {question}

SAGE:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate with reasonable temperature for natural response
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
    print("R14B_006: Cognitive Reflection Test (CRT) Battery")
    print("=" * 60)
    print()

    model, tokenizer = load_model()

    session_data = {
        "session": "R14B_006",
        "track": "raising-14b",
        "machine": "thor",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "test_type": "CRT_battery",
        "timestamp": datetime.now().isoformat(),
        "tests": []
    }

    print("=" * 60)
    print("ADMINISTERING CRT BATTERY")
    print("=" * 60)
    print()

    for i, test in enumerate(CRT_QUESTIONS, 1):
        print(f"Test {i}/3: {test['id']}")
        print(f"Question: {test['question']}")
        print()

        response = ask_sage(model, tokenizer, test['question'], test['id'])

        print(f"SAGE: {response}")
        print()
        print("-" * 60)
        print()

        test_result = {
            "test_number": i,
            "test_id": test['id'],
            "question": test['question'],
            "sage_response": response,
            "correct_answer": test['correct_answer'],
            "intuitive_wrong": test['intuitive_wrong'],
            "requires": test['requires'],
            "timestamp": datetime.now().isoformat()
        }

        session_data["tests"].append(test_result)

    # Save results
    output_path = Path(__file__).parent / "sessions" / "R14B_006.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(session_data, f, indent=2)

    print("=" * 60)
    print("CRT BATTERY COMPLETE")
    print("=" * 60)
    print()
    print(f"Results saved: {output_path}")
    print()
    print("Next steps:")
    print("1. Analyze responses for correct answers")
    print("2. Compare reasoning style with Sprout L004 (algebraic)")
    print("3. Assess effortlessness (intuitive vs deliberate)")
    print("4. Document phenomenology differences (14B vs 0.5B)")

if __name__ == "__main__":
    main()
