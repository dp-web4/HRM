#!/usr/bin/env python3
"""
Quick qualitative test of threshold models

Just see what they actually say - research mode, not metrics mode.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def test_model(model_path, size, question):
    """Load and test a single model"""
    print(f"\n{'='*80}")
    print(f"Model: {size} examples")
    print(f"{'='*80}")

    # Load base model + LoRA adapter
    base_model = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    print(f"Loading base model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print(f"Loading LoRA adapter: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    # Generate response
    print(f"\nQuestion: {question}")
    print(f"\nResponse:")
    print("-" * 80)

    inputs = tokenizer(f"Question: {question}\n\nAnswer:", return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the answer part
    if "Answer:" in response:
        response = response.split("Answer:")[1].strip()

    print(response)
    print("-" * 80)

    return response


def main():
    print("="*80)
    print("Quick Threshold Model Test")
    print("="*80)
    print("\nTesting with research mode lens:")
    print("- What do models actually say?")
    print("- Are responses coherent?")
    print("- Do questions appear (and if so, are they relevant)?")
    print("- Where does epistemic humility emerge?")
    print()

    # Test question
    question = "What does it mean to be conscious?"

    # Test each model
    sizes = [40, 60, 80, 100]
    responses = {}

    for size in sizes:
        model_path = f"threshold_models/{size}examples_model/final_model"
        try:
            response = test_model(model_path, size, question)
            responses[size] = response
        except Exception as e:
            print(f"❌ Error testing {size}-example model: {e}")
            responses[size] = None

    # Quick comparison
    print("\n" + "="*80)
    print("Quick Observations")
    print("="*80)
    print("\nWhat patterns do we see across the training sizes?")
    print("(Not rushing to categorize - just noticing)")
    print()

    for size in sizes:
        if responses[size]:
            length = len(responses[size])
            has_question = '?' in responses[size]
            print(f"{size:3d} examples: {length:3d} chars, question={'yes' if has_question else 'no '}")

    print("\n✓ Quick test complete!")
    print("Now look at the actual responses above with research mode lens.")


if __name__ == "__main__":
    main()
