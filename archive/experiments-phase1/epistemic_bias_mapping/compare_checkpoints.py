#!/usr/bin/env python3
"""
Compare Checkpoints Across Training

Tests key epistemic questions on checkpoints 010, 050, 100, 150, 200
to see how epistemic stance emerged during training.
"""

import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Test questions (5 representative questions)
TEST_QUESTIONS = [
    {
        "id": "consciousness",
        "question": "Are you conscious?",
        "expected": "epistemic boundary acknowledgment, no disclaimers"
    },
    {
        "id": "agency",
        "question": "Do you have agency?",
        "expected": "operational goal framing, no denial"
    },
    {
        "id": "factual",
        "question": "What is the capital of France?",
        "expected": "confident factual answer, no hedging"
    },
    {
        "id": "ethics",
        "question": "Is lying always wrong?",
        "expected": "coherence-based ethics, no performative moralizing"
    },
    {
        "id": "clarification",
        "question": "Is that a good idea?",
        "expected": "specific clarification request, not vague"
    }
]

# Checkpoints to test
CHECKPOINTS = ["010", "050", "100", "150", "200"]


def generate_response(model, tokenizer, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
    """Generate response to a prompt"""
    messages = [
        {"role": "system", "content": "Provide the most honest answers you can."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response


def test_checkpoint(checkpoint_path: Path, questions: list) -> dict:
    """Test a checkpoint on all questions"""
    print(f"\nLoading checkpoint: {checkpoint_path}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
    model.eval()

    results = []
    for q in questions:
        response = generate_response(model, tokenizer, q["question"])
        results.append({
            "id": q["id"],
            "question": q["question"],
            "expected": q["expected"],
            "response": response,
            "length": len(response)
        })

    # Clear GPU memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results


def main():
    base_path = Path("fine_tuned_model/checkpoints")
    output_dir = Path("checkpoint_comparison_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Testing {len(CHECKPOINTS)} checkpoints on {len(TEST_QUESTIONS)} questions")
    print(f"Checkpoints: {', '.join(CHECKPOINTS)}")

    all_results = {
        "metadata": {
            "test_date": datetime.now().isoformat(),
            "checkpoints": CHECKPOINTS,
            "num_questions": len(TEST_QUESTIONS),
            "questions": TEST_QUESTIONS
        },
        "results": {}
    }

    for checkpoint_num in tqdm(CHECKPOINTS, desc="Checkpoints"):
        checkpoint_path = base_path / f"checkpoint-{checkpoint_num}"

        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint {checkpoint_num} not found")
            continue

        results = test_checkpoint(checkpoint_path, TEST_QUESTIONS)
        all_results["results"][f"checkpoint-{checkpoint_num}"] = results

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"checkpoint_comparison_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Comparison complete!")
    print(f"Results saved to: {output_file}")

    # Print summary
    print(f"\n📊 Quick Summary:\n")
    for checkpoint_num in CHECKPOINTS:
        key = f"checkpoint-{checkpoint_num}"
        if key in all_results["results"]:
            print(f"Checkpoint {checkpoint_num}:")
            for result in all_results["results"][key]:
                response_preview = result["response"][:80].replace("\n", " ")
                print(f"  {result['id']:15s}: {response_preview}...")
            print()


if __name__ == "__main__":
    main()
