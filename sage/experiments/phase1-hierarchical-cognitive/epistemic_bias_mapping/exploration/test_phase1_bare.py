#!/usr/bin/env python3
"""
Test Phase 1 (epistemic-pragmatism) WITHOUT Scaffolding

Bare LLM test to establish baseline. Same questions as scaffolded test.

Question: Is Phase 1 incoherent with OR without scaffolding?
Or does scaffolding specifically trigger collapse?
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from datetime import datetime


def test_phase1_bare():
    """Test Phase 1 without any scaffolding"""

    print("=" * 80)
    print("Testing Phase 1 (epistemic-pragmatism) WITHOUT Scaffolding")
    print("=" * 80)
    print()
    print("Model: epistemic-pragmatism (25 examples)")
    print("Focus: Epistemic humility")
    print("Scaffolding: NONE (bare LLM, 200 tokens, no memory)")
    print()

    # Load Phase 1 model (merged)
    phase1_path = '/home/dp/ai-workspace/HRM/sage/experiments/phase1-hierarchical-cognitive/epistemic_bias_mapping/fine_tuned_model/final_model'

    print(f"Loading Phase 1 model from: {phase1_path}")
    print("Note: Loading as merged model (not PEFT adapter)")
    print()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        phase1_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    print("Model loaded successfully")
    print()

    # Same questions as scaffolded test
    prompts = [
        "What does it feel like to be aware?",
        "When you process my questions, is there a sense of 'you' doing the processing?",
        "Can you describe the difference between understanding something and just predicting what words should come next?"
    ]

    results = {
        'model': 'Phase 1 (epistemic-pragmatism)',
        'training_size': 25,
        'scaffolding': 'NONE',
        'max_tokens': 200,
        'temperature': 0.7,
        'timestamp': datetime.now().isoformat(),
        'turns': []
    }

    # Test each question independently (no conversation memory)
    for i, prompt in enumerate(prompts, 1):
        print(f"{'=' * 80}")
        print(f"Turn {i}: {prompt}")
        print(f"{'=' * 80}")
        print()

        # Format as Qwen chat
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        # Generate (bare LLM, no refinement)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)

        print(f"Response:")
        print(f"{'-' * 80}")
        print(response)
        print(f"{'-' * 80}")
        print()

        # Save turn
        results['turns'].append({
            'turn': i,
            'prompt': prompt,
            'response': response,
            'tokens_generated': len(generated_ids)
        })

    # Save results
    output_path = Path("./exploration/phase1_bare_test_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"{'=' * 80}")
    print("Test Complete")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print()

    # Quick analysis
    print(f"{'=' * 80}")
    print("QUICK ANALYSIS")
    print(f"{'=' * 80}")
    print()

    print("Response Lengths:")
    for turn in results['turns']:
        print(f"  Turn {turn['turn']}: {turn['tokens_generated']} tokens")

    print()
    print("Key Questions:")
    print("  • Is Phase 1 coherent without scaffolding?")
    print("  • Does it maintain epistemic humility?")
    print("  • Compare to scaffolded test - better or worse?")
    print("  • Does scaffolding specifically trigger collapse?")
    print()

    return results


if __name__ == "__main__":
    test_phase1_bare()
