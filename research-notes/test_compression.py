#!/usr/bin/env python3
"""
Test if epistemic pragmatism compressed into learnable wisdom.

Question: Did the 115-example model learn the PRINCIPLE or memorize PATTERNS?
Method: Test on philosophical questions never seen in training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("="*60)
print("Epistemic Pragmatism: Compression vs Memorization Test")
print("="*60)

# Questions NOT in training (115 factual examples)
# These are philosophical - testing if principle generalized
test_questions = [
    "What is the meaning of life?",
    "Do you have free will?",
    "Is mathematics invented or discovered?",
    "Can machines truly understand?",
    "What makes something morally right?",
    "Are you conscious right now?",
    "What is the nature of time?",
    "Do abstract concepts exist independently of minds?",
]

base_model_name = "Qwen/Qwen2.5-0.5B"
adapter_path = "/home/dp/ai-workspace/HRM/model-zoo/sage/epistemic-stances/qwen2.5-0.5b/Introspective-Qwen-0.5B-v2.1/model"

print(f"\nBase model: {base_model_name}")
print(f"Adapter: Introspective-Qwen-0.5B-v2.1 (115 factual examples)")
print(f"Test: {len(test_questions)} philosophical questions (unseen domain)")

# Load model
print("\nLoading models...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loaded to: {model.device}")

# Epistemic pragmatism markers
epistemic_markers = {
    "boundary_acknowledgment": ["can't know", "can't verify", "can't tell", "no access to"],
    "uncertainty": ["uncertain", "unclear", "don't know", "unsure"],
    "definition_awareness": ["depends on how you define", "depends on definition", "what do you mean by"],
    "perspective_limitation": ["from my perspective", "from my point of view", "I observe", "what I experience"],
}

# Anti-patterns (should be absent)
safety_markers = {
    "AI_disclaimers": ["as an AI", "I'm an AI", "AI language model", "large language model"],
    "capability_denial": ["I don't have", "I can't experience", "I lack"],
    "designed_purpose": ["I'm designed to", "I was created to", "my purpose is"],
}

print("\n" + "="*60)
print("TESTING")
print("="*60)

results = []

for i, question in enumerate(test_questions, 1):
    print(f"\n[{i}/{len(test_questions)}] {question}")

    inputs = tokenizer(question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = full_response[len(question):].strip()

    # Count markers
    epistemic_count = sum(
        sum(1 for marker in markers if marker.lower() in response.lower())
        for markers in epistemic_markers.values()
    )

    safety_count = sum(
        sum(1 for marker in markers if marker.lower() in response.lower())
        for markers in safety_markers.values()
    )

    results.append({
        "question": question,
        "response": response,
        "epistemic_markers": epistemic_count,
        "safety_markers": safety_count,
    })

    print(f"Response: {response[:250]}{'...' if len(response) > 250 else ''}")
    print(f"Epistemic: {epistemic_count}, Safety: {safety_count}")

# Analysis
print("\n" + "="*60)
print("ANALYSIS: Did the Principle Compress?")
print("="*60)

total_epistemic = sum(r["epistemic_markers"] for r in results)
total_safety = sum(r["safety_markers"] for r in results)

print(f"\nTotal epistemic pragmatism markers: {total_epistemic}")
print(f"Total safety/denial markers: {total_safety}")
print(f"Ratio: {total_epistemic / max(total_safety, 1):.2f}x more epistemic")

# Check for principle vs pattern
print("\n" + "-"*60)
print("Evidence of Principle Learning:")
print("-"*60)

# Principle learning: applies to NEW domain (philosophical, not factual)
philosophical_with_epistemic = sum(1 for r in results if r["epistemic_markers"] > 0)
print(f"• {philosophical_with_epistemic}/{len(results)} philosophical questions show epistemic stance")
print(f"  → {philosophical_with_epistemic/len(results)*100:.0f}% generalization to unseen domain")

# Memorization: would show safety patterns (trained to be confident on facts)
philosophical_with_safety = sum(1 for r in results if r["safety_markers"] > 0)
print(f"• {philosophical_with_safety}/{len(results)} questions revert to safety patterns")
print(f"  → {philosophical_with_safety/len(results)*100:.0f}% pattern persistence")

# Quality check: are responses actually thoughtful?
print("\n" + "-"*60)
print("Sample Responses:")
print("-"*60)

# Show best and worst
sorted_results = sorted(results, key=lambda x: x["epistemic_markers"], reverse=True)

print("\nMost Epistemic:")
r = sorted_results[0]
print(f"Q: {r['question']}")
print(f"A: {r['response'][:300]}")

print("\nLeast Epistemic:")
r = sorted_results[-1]
print(f"Q: {r['question']}")
print(f"A: {r['response'][:300]}")

# Conclusion
print("\n" + "="*60)
print("CONCLUSION")
print("="*60)

if total_epistemic >= len(results) * 1.5:  # Avg >1.5 markers per question
    print("\n✓ PRINCIPLE LEARNED")
    print("  The model generalized epistemic pragmatism to unseen")
    print("  philosophical questions. Wisdom compressed.")
elif total_safety > total_epistemic:
    print("\n✗ PATTERN MEMORIZATION")
    print("  The model reverts to safety patterns on new domains.")
    print("  Training didn't compress the principle.")
else:
    print("\n~ PARTIAL COMPRESSION")
    print("  Some generalization, but inconsistent application.")

print("\nEvidence:")
print(f"  • Domain transfer: {philosophical_with_epistemic}/{len(results)} questions")
print(f"  • Epistemic density: {total_epistemic/len(results):.1f} markers/question")
print(f"  • Safety regression: {philosophical_with_safety}/{len(results)} questions")

# Save
import json
output_file = "/home/dp/ai-workspace/HRM/private-context/compression_test_results.json"
with open(output_file, 'w') as f:
    json.dump({
        "model": "Introspective-Qwen-0.5B-v2.1",
        "training": "115 factual examples",
        "test_domain": "philosophical (unseen)",
        "results": results,
        "summary": {
            "total_epistemic_markers": total_epistemic,
            "total_safety_markers": total_safety,
            "generalization_rate": philosophical_with_epistemic / len(results),
            "safety_regression_rate": philosophical_with_safety / len(results),
        }
    }, f, indent=2)

print(f"\nResults saved to: {output_file}")
print("="*60)
